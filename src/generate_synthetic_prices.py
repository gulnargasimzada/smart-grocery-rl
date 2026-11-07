"""Synthetic weekly price paths for every product. Run from the repo root (after src.fetch_open_prices):
    python -m src.generate_synthetic_prices

Output: data/price_weekly_synthetic.csv - one row per product and week (weeks 0..103, the same weeks as
data/demand_weekly.csv; week 0 = early January, week_of_year = week % 52). Every row is synthetic
(is_synthetic = True, data_source = "synthetic") and is never written into data/price_history.csv.

Model per product (all random draws come from one generator seeded with SEED, so the file is reproducible):
  baseline  median real regular price per purchase unit from data/price_history.csv when the product has
            at least MIN_OBS observations from at least MIN_STORES stores; otherwise price_usd from products.csv.
            The baseline is the price level at the middle of the 104-week horizon.
  inflation product-specific annual rate around a category rate (compounded weekly).
  season    sine wave over 52 weeks with a random phase (strong for fruit and vegetables, weak otherwise).
  noise     slow AR(1) cost shocks.
  regular   shelf prices are sticky: the regular price only moves to the latent level on occasional
            price-change weeks (frequent for produce, rare for packaged goods) or after a large drift.
  promotion temporary discounts lasting 1 or 2 weeks with a category-specific start probability and depth
            (more likely in the holiday weeks 46 and 51). The regular price is kept separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_WEEKS = 104                      # same horizon as src/data_gen.py (data/demand_weekly.csv)
HOLIDAY_WEEKS = {46, 51}           # same holiday weeks as src/demand.py
MIN_OBS, MIN_STORES = 5, 3         # a product's real prices are used for calibration only above these counts
DRIFT_TRIGGER = 0.08               # regular price is re-set when it is more than 8 % away from the latent level

PRODUCTS_PATH = Path("data/products.csv")
HISTORY_PATH = Path("data/price_history.csv")
DEMAND_PATH = Path("data/demand_weekly.csv")
OUT_PATH = Path("data/price_weekly_synthetic.csv")

# category: annual inflation, seasonal amplitude, AR(1) noise sigma, weekly regular-price change probability,
#           weekly promotion start probability, promotion depth range
CATEGORY_PARAMS = {
    "fruit":      dict(inflation=0.020, season=0.08, sigma=0.030, change=0.35, promo=0.08, depth=(0.15, 0.35)),
    "vegetables": dict(inflation=0.020, season=0.08, sigma=0.030, change=0.35, promo=0.08, depth=(0.15, 0.35)),
    "dairy":      dict(inflation=0.030, season=0.02, sigma=0.015, change=0.06, promo=0.06, depth=(0.10, 0.25)),
    "protein":    dict(inflation=0.035, season=0.03, sigma=0.020, change=0.08, promo=0.07, depth=(0.10, 0.30)),
    "grains":     dict(inflation=0.025, season=0.01, sigma=0.010, change=0.05, promo=0.06, depth=(0.10, 0.30)),
    "legumes":    dict(inflation=0.025, season=0.01, sigma=0.010, change=0.05, promo=0.05, depth=(0.10, 0.25)),
    "beverages":  dict(inflation=0.025, season=0.02, sigma=0.015, change=0.06, promo=0.07, depth=(0.10, 0.30)),
    "fats_oils":  dict(inflation=0.030, season=0.01, sigma=0.015, change=0.05, promo=0.05, depth=(0.10, 0.25)),
    "processed":  dict(inflation=0.025, season=0.01, sigma=0.010, change=0.05, promo=0.10, depth=(0.15, 0.35)),
}
DEFAULT_PARAMS = CATEGORY_PARAMS["grains"]
HOLIDAY_PROMO_MULT = 2.0
AR_PHI = 0.9

COLUMNS = ["product_id", "week", "week_of_year", "unit", "regular_price_usd", "promo_price_usd", "is_promo",
           "promo_depth", "price_usd", "baseline_price_usd", "baseline_source", "n_real_obs", "is_synthetic",
           "data_source", "seed"]


def baselines(products: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    """Baseline price per purchase unit and where it comes from (real Open Prices median or products.csv)."""
    out = products[["product_id", "price_usd"]].copy()
    out["baseline_price_usd"] = out["price_usd"].astype(float)
    out["baseline_source"] = "products_csv"
    out["n_real_obs"] = 0
    if history is not None and len(history):
        reg = history.dropna(subset=["regular_price_usd"])
        stats = reg.groupby("product_id").agg(n=("regular_price_usd", "size"),
                                              stores=("store_osm_id", "nunique"),
                                              median=("regular_price_usd", "median"))
        out["n_real_obs"] = out["product_id"].map(stats["n"]).fillna(0).astype(int)
        ok = stats[(stats["n"] >= MIN_OBS) & (stats["stores"] >= MIN_STORES)]
        use = out["product_id"].isin(ok.index)
        out.loc[use, "baseline_price_usd"] = out.loc[use, "product_id"].map(ok["median"])
        out.loc[use, "baseline_source"] = "open_prices_median"
    return out


def simulate_product(base: float, params: dict, rng: np.random.Generator, n_weeks: int) -> dict:
    """Weekly regular price, promotion flag and depth for one product."""
    infl = float(np.clip(rng.normal(params["inflation"], 0.005), 0.0, 0.06))
    phase = rng.uniform(0, 2 * np.pi)
    weeks = np.arange(n_weeks)
    trend = (1 + infl) ** ((weeks - (n_weeks - 1) / 2) / 52)          # baseline = level at mid-horizon
    season = 1 + params["season"] * np.sin(2 * np.pi * (weeks % 52) / 52 + phase)
    shock = np.zeros(n_weeks)
    for w in range(1, n_weeks):
        shock[w] = AR_PHI * shock[w - 1] + rng.normal(0, params["sigma"])
    latent = base * trend * season * np.exp(shock)

    regular = np.empty(n_weeks)
    regular[0] = latent[0]
    for w in range(1, n_weeks):
        drift = abs(regular[w - 1] / latent[w] - 1)
        regular[w] = latent[w] if (rng.random() < params["change"] or drift > DRIFT_TRIGGER) else regular[w - 1]
    regular = np.round(regular, 2)

    is_promo = np.zeros(n_weeks, dtype=bool)
    depth = np.zeros(n_weeks)
    w = 0
    while w < n_weeks:
        p = params["promo"] * (HOLIDAY_PROMO_MULT if w % 52 in HOLIDAY_WEEKS else 1.0)
        if rng.random() < p:
            length = int(rng.integers(1, 3))                              # 1 or 2 weeks
            d = round(float(rng.uniform(*params["depth"])), 2)
            end = min(w + length, n_weeks)
            is_promo[w:end] = True
            depth[w:end] = d
            w = end + 1                                                   # at least one full-price week after
        else:
            w += 1
    promo = np.where(is_promo, np.round(regular * (1 - depth), 2), np.nan)
    return {"regular": regular, "is_promo": is_promo, "depth": np.where(is_promo, depth, np.nan), "promo": promo}


def main() -> None:
    products = pd.read_csv(PRODUCTS_PATH)
    history = pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else None
    if history is None:
        print(f"note: {HISTORY_PATH} not found, all baselines come from products.csv")
    weeks = sorted(pd.read_csv(DEMAND_PATH, usecols=["week"])["week"].unique())
    assert weeks == list(range(N_WEEKS)), f"demand_weekly.csv weeks {weeks[0]}..{weeks[-1]} != 0..{N_WEEKS - 1}"

    base = baselines(products, history).set_index("product_id")
    rng = np.random.default_rng(SEED)
    frames = []
    for p in products.itertuples():                                       # products.csv row order
        params = CATEGORY_PARAMS.get(p.category, DEFAULT_PARAMS)
        b = base.loc[p.product_id]
        sim = simulate_product(float(b["baseline_price_usd"]), params, rng, N_WEEKS)
        frames.append(pd.DataFrame({
            "product_id": p.product_id,
            "week": np.arange(N_WEEKS),
            "week_of_year": np.arange(N_WEEKS) % 52,
            "unit": p.unit,
            "regular_price_usd": sim["regular"],
            "promo_price_usd": sim["promo"],
            "is_promo": sim["is_promo"],
            "promo_depth": sim["depth"],
            "price_usd": np.where(sim["is_promo"], sim["promo"], sim["regular"]),
            "baseline_price_usd": round(float(b["baseline_price_usd"]), 4),
            "baseline_source": b["baseline_source"],
            "n_real_obs": int(b["n_real_obs"]),
            "is_synthetic": True,
            "data_source": "synthetic",
            "seed": SEED,
        }))
    out = pd.concat(frames, ignore_index=True)[COLUMNS]
    out.to_csv(OUT_PATH, index=False)

    summary = out.groupby("product_id", sort=False).agg(
        baseline=("baseline_price_usd", "first"), source=("baseline_source", "first"),
        min_regular=("regular_price_usd", "min"), max_regular=("regular_price_usd", "max"),
        promo_weeks=("is_promo", "sum"))
    print(summary.to_string())
    print(f"\nwrote {OUT_PATH}: {len(out)} rows = {out['product_id'].nunique()} products x {N_WEEKS} weeks | "
          f"baselines from Open Prices: {(summary['source'] == 'open_prices_median').sum()} | "
          f"promo weeks: {out['is_promo'].mean():.1%} | seed {SEED}")


if __name__ == "__main__":
    main()
