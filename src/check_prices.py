"""Validates the price datasets. Run from the repo root after src.fetch_open_prices and
src.generate_synthetic_prices:
    python -m src.check_prices

Checks data/price_history.csv (real, Open Prices) and data/price_weekly_synthetic.csv (synthetic):
product coverage, units, missing values, duplicates, price ranges, promotional prices, week coverage and
that real and synthetic records are kept apart. Exits with status 1 if any check fails.
"""
from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

from src.fetch_open_prices import OUTPUT_COLUMNS as HISTORY_COLUMNS
from src.generate_synthetic_prices import COLUMNS as SYNTH_COLUMNS, N_WEEKS, SEED

failures: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {msg}")
    if not ok:
        failures.append(msg)


def check_history(h: pd.DataFrame, products: pd.DataFrame) -> None:
    print("data/price_history.csv (real)")
    check(list(h.columns) == HISTORY_COLUMNS, "columns match the documented schema")
    check(h["product_id"].isin(products["product_id"]).all(), "every product_id exists in products.csv")
    unit = h["product_id"].map(products.set_index("product_id")["unit"])
    check((h["unit"] == unit).all(), "unit equals the products.csv purchase unit")
    required = ["product_id", "unit", "date", "package_price_usd", "package_size", "package_unit",
                "units_per_package", "store_osm_id", "source", "source_price_id", "match_type"]
    na = h[required].isna().sum()
    check(na.sum() == 0, f"no missing values in required columns {'' if na.sum() == 0 else na[na > 0].to_dict()}")
    check((h["source"] == "open_prices").all(), "every record is labelled source = open_prices (no synthetic rows)")
    check(h["source_price_id"].is_unique, "no duplicate Open Prices ids")
    key = ["product_id", "date", "store_osm_id", "product_code", "category_tag", "package_price_usd",
           "package_regular_price_usd"]
    check(not h.duplicated(subset=key).any(), "no duplicate observations (same product, store, day, price)")
    d = pd.to_datetime(h["date"], errors="coerce")
    check(d.notna().all() and d.min() >= pd.Timestamp("2010-01-01") and d.max() <= pd.Timestamp(date.today()),
          f"dates valid and not in the future ({d.min().date()} .. {d.max().date()})")
    check(h["package_unit"].isin(["g", "ml", "piece"]).all(), "package_unit in {g, ml, piece}")
    check((h["package_size"] > 0).all() and (h["units_per_package"] > 0).all(), "package sizes positive")
    reg, promo = h["regular_price_usd"], h["promo_price_usd"]
    check((reg.dropna() > 0).all() and (promo.dropna() > 0).all(), "all prices positive")
    check((h["price_is_discounted"] == promo.notna()).all(), "promo price present exactly when discounted")
    both = reg.notna() & promo.notna()
    check((promo[both] < reg[both]).all(), "promo price below regular price")
    ref = reg.fillna(promo)
    cat = h["product_id"].map(products.set_index("product_id")["price_usd"])
    ratio = ref / cat
    check(ratio.between(0.1, 10).all(), f"prices within 0.1x..10x of products.csv "
                                        f"(range {ratio.min():.2f}x .. {ratio.max():.2f}x)")
    print(f"  info: {len(h)} rows | {h['product_id'].nunique()}/{len(products)} products | "
          f"{h['store_osm_id'].nunique()} stores | regular price missing (discounted, regular unknown): "
          f"{reg.isna().sum()} | discounted: {h['price_is_discounted'].sum()}")
    missing = sorted(set(products["product_id"]) - set(h["product_id"]))
    print(f"  info: products without real observations: {missing}")


def check_synthetic(s: pd.DataFrame, products: pd.DataFrame, demand_weeks: list[int]) -> None:
    print("data/price_weekly_synthetic.csv (synthetic)")
    check(list(s.columns) == SYNTH_COLUMNS, "columns match the documented schema")
    check(len(s) == len(products) * N_WEEKS, f"{len(s)} rows = {len(products)} products x {N_WEEKS} weeks")
    check(set(s["product_id"]) == set(products["product_id"]), "covers every product in products.csv")
    check(s["product_id"].drop_duplicates().tolist() == products["product_id"].tolist(),
          "products appear in products.csv row order")
    check(not s.duplicated(subset=["product_id", "week"]).any(), "one row per product and week")
    check(sorted(s["week"].unique()) == demand_weeks, "weeks equal the weeks of demand_weekly.csv")
    check((s["week_of_year"] == s["week"] % 52).all(), "week_of_year = week % 52")
    unit = s["product_id"].map(products.set_index("product_id")["unit"])
    check((s["unit"] == unit).all(), "unit equals the products.csv purchase unit")
    required = [c for c in SYNTH_COLUMNS if c not in ("promo_price_usd", "promo_depth")]
    na = s[required].isna().sum()
    check(na.sum() == 0, f"no missing values outside promo columns {'' if na.sum() == 0 else na[na > 0].to_dict()}")
    check(s["is_synthetic"].astype(bool).all() and (s["data_source"] == "synthetic").all(),
          "every row labelled is_synthetic = True, data_source = synthetic")
    check((s["seed"] == SEED).all(), f"seed recorded ({SEED})")
    check((s["regular_price_usd"] > 0).all(), "regular prices positive")
    promo = s["is_promo"].astype(bool)
    check((s["promo_price_usd"].notna() == promo).all() and (s["promo_depth"].notna() == promo).all(),
          "promo price and depth present exactly in promo weeks")
    check((s.loc[promo, "promo_price_usd"] < s.loc[promo, "regular_price_usd"]).all(), "promo price below regular")
    check(s.loc[promo, "promo_depth"].between(0.05, 0.5).all(), "promo depth between 5 % and 50 %")
    eff = np.where(promo, s["promo_price_usd"], s["regular_price_usd"])
    check(np.allclose(s["price_usd"], eff), "price_usd = promo price in promo weeks, else regular price")
    ratio = s["regular_price_usd"] / s["baseline_price_usd"]
    check(ratio.between(0.7, 1.4).all(), f"regular price within 0.7x..1.4x of baseline "
                                         f"({ratio.min():.2f}x .. {ratio.max():.2f}x)")
    lengths = []
    for _, g in s.sort_values(["product_id", "week"]).groupby("product_id"):
        flags = g["is_promo"].astype(bool).to_numpy()
        run = 0
        for f in list(flags) + [False]:
            if f:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
    check(set(lengths) <= {1, 2}, f"promotions last 1-2 weeks (runs: {pd.Series(lengths).value_counts().to_dict()})")
    print(f"  info: promo weeks {promo.mean():.1%} | baselines from Open Prices: "
          f"{s.drop_duplicates('product_id')['baseline_source'].eq('open_prices_median').sum()}/{len(products)}")


def check_regional(r: pd.DataFrame, base: pd.DataFrame, products: pd.DataFrame) -> None:
    from src.generate_regional_prices import COLUMNS as REGIONAL_COLUMNS, REGIONS
    print("data/price_weekly_regional_synthetic.csv (synthetic, regional)")
    check(list(r.columns) == REGIONAL_COLUMNS, "columns match the documented schema")
    n = len(REGIONS) * len(products) * N_WEEKS
    check(len(r) == n, f"{len(r)} rows = {len(REGIONS)} regions x {len(products)} products x {N_WEEKS} weeks")
    check(set(r["region_id"]) == set(REGIONS), "all regions present")
    check(not r.duplicated(subset=["region_id", "week", "product_id"]).any(), "no duplicate region/week/product rows")
    unit = r["product_id"].map(products.set_index("product_id")["unit"])
    check(unit.notna().all() and (r["unit"] == unit).all(), "products and units match products.csv")
    na = r.drop(columns="promo_price_usd").isna().sum().sum()
    check(na == 0, "no missing values outside promo_price_usd")
    check((r["source"] == "synthetic_regional").all(), "every row labelled source = synthetic_regional")
    check((r["regular_price_usd"] > 0).all() and (r["price_usd"] > 0).all(), "prices positive")
    promo = r["is_promo"].astype(bool)
    check((r["promo_price_usd"].notna() == promo).all(), "promo price present exactly in promo weeks")
    check((r.loc[promo, "promo_price_usd"] < r.loc[promo, "regular_price_usd"]).all(), "promo price below regular")
    eff = np.where(promo, r["promo_price_usd"], r["regular_price_usd"])
    check(np.allclose(r["price_usd"], eff), "price_usd = promo price in promo weeks, else regular price")
    check(r["regional_multiplier"].between(0.8, 1.35).all(),
          f"regional multipliers within 0.8..1.35 ({r['regional_multiplier'].min()}..{r['regional_multiplier'].max()})")
    j = r.merge(base[["product_id", "week", "regular_price_usd", "is_promo"]], on=["product_id", "week"],
                suffixes=("", "_base"))
    check((j["is_promo"].astype(bool) == j["is_promo_base"].astype(bool)).all(),
          "promotion weeks match the base synthetic file")
    expected = (j["regular_price_usd_base"] * j["regional_multiplier"]).round(2)
    check(np.allclose(j["regular_price_usd"], expected), "regular price = base price x regional multiplier")


def main() -> None:
    products = pd.read_csv("data/products.csv")
    history = pd.read_csv("data/price_history.csv")
    synthetic = pd.read_csv("data/price_weekly_synthetic.csv")
    demand_weeks = sorted(pd.read_csv("data/demand_weekly.csv", usecols=["week"])["week"].unique().tolist())
    check_history(history, products)
    check_synthetic(synthetic, products, demand_weeks)
    print("separation")
    check("is_synthetic" not in history.columns and (history["source"] == "open_prices").all(),
          "real dataset contains only Open Prices records (no synthetic rows or columns)")
    check((synthetic["data_source"] == "synthetic").all() and "source_price_id" not in synthetic.columns,
          "synthetic dataset contains only synthetic rows (no real Open Prices records)")
    regional_path = "data/price_weekly_regional_synthetic.csv"
    try:
        regional = pd.read_csv(regional_path)
    except FileNotFoundError:
        print(f"{regional_path} not found, skipped")
    else:
        check_regional(regional, synthetic, products)
    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        sys.exit(1)
    print("\nAll price checks passed.")


if __name__ == "__main__":
    main()
