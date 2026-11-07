"""Synthetic regional weekly prices for the demo. Run from the repo root:
    python -m src.generate_regional_prices

Input:  data/price_weekly_synthetic.csv (read only, not modified)
Output: data/price_weekly_regional_synthetic.csv - 5 regions x 36 products x 104 weeks = 18,720 rows

Each regional price is the base synthetic price times a fixed multiplier per (region, product):
    multiplier = REGION_INDEX[region] * exp(CATEGORY_ADJ[region][category] + noise)
    noise ~ Normal(0, PRODUCT_SIGMA), drawn once per (region, product) from a generator seeded with SEED.
Weekly fluctuations and promotion weeks/depths are inherited from the base file, so they stay the same in
every region; only the level changes. ALL regional adjustments are synthetic assumptions, not verified
local prices. The base file is treated as the reference level (index 1.00).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
BASE_PATH = Path("data/price_weekly_synthetic.csv")
PRODUCTS_PATH = Path("data/products.csv")
OUT_PATH = Path("data/price_weekly_regional_synthetic.csv")

# region_id: (display name, overall grocery price index vs. the base file). Assumed, loosely following the
# usual ordering of published cost-of-living grocery indices (SF and NYC highest, Southeast lowest).
REGIONS = {
    "durham_nc":        ("Durham, NC",        0.95),
    "san_francisco_ca": ("San Francisco, CA", 1.20),
    "atlanta_ga":       ("Atlanta, GA",       0.97),
    "new_york_ny":      ("New York, NY",      1.16),
    "chicago_il":       ("Chicago, IL",       1.03),
}
# Category adjustments in log points (assumed): e.g. California produce is relatively cheaper in SF,
# Midwest dairy relatively cheaper in Chicago, meat/fish relatively dearer in the large coastal cities.
CATEGORY_ADJ = {
    "san_francisco_ca": {"fruit": -0.04, "vegetables": -0.04, "protein": 0.03, "beverages": 0.02},
    "new_york_ny":      {"fruit": 0.03, "vegetables": 0.03, "protein": 0.03, "processed": 0.02},
    "chicago_il":       {"dairy": -0.03, "grains": -0.01, "protein": -0.01},
    "atlanta_ga":       {"protein": -0.02, "fruit": 0.01, "beverages": -0.02},
    "durham_nc":        {"protein": -0.02, "dairy": 0.01, "fats_oils": 0.02},
}
PRODUCT_SIGMA = 0.025          # product-level variation around the region/category level

COLUMNS = ["region_id", "region_name", "week", "product_id", "unit", "regular_price_usd", "promo_price_usd",
           "is_promo", "price_usd", "regional_multiplier", "source"]


def multipliers(products: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for region_id, (_, index) in REGIONS.items():                 # fixed order -> reproducible draws
        noise = rng.normal(0, PRODUCT_SIGMA, len(products))
        for p, eps in zip(products.itertuples(), noise):
            adj = CATEGORY_ADJ.get(region_id, {}).get(p.category, 0.0)
            rows.append({"region_id": region_id, "product_id": p.product_id,
                         "regional_multiplier": round(index * float(np.exp(adj + eps)), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    base = pd.read_csv(BASE_PATH)
    products = pd.read_csv(PRODUCTS_PATH)
    m = multipliers(products)
    frames = []
    for region_id, (name, _) in REGIONS.items():
        r = base.merge(m[m["region_id"] == region_id], on="product_id", how="left")
        mult = r["regional_multiplier"]
        regular = (r["regular_price_usd"] * mult).round(2)
        promo = (r["promo_price_usd"] * mult).round(2)             # NaN stays NaN outside promo weeks
        promo = promo.where(promo.isna() | (promo < regular), regular - 0.01)
        frames.append(pd.DataFrame({
            "region_id": region_id, "region_name": name, "week": r["week"], "product_id": r["product_id"],
            "unit": r["unit"], "regular_price_usd": regular, "promo_price_usd": promo,
            "is_promo": r["is_promo"].astype(bool), "price_usd": promo.where(r["is_promo"].astype(bool), regular),
            "regional_multiplier": mult, "source": "synthetic_regional",
        }))
    out = pd.concat(frames, ignore_index=True)[COLUMNS]
    out.to_csv(OUT_PATH, index=False)
    s = out.groupby("region_name", sort=False)["regional_multiplier"].agg(["mean", "min", "max"]).round(3)
    print(s.to_string())
    print(f"\nwrote {OUT_PATH}: {len(out)} rows = {out['region_id'].nunique()} regions x "
          f"{out['product_id'].nunique()} products x {out['week'].nunique()} weeks | seed {SEED}")


if __name__ == "__main__":
    main()
