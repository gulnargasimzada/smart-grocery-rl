"""Copies a REFERENCE price per product from the weekly price table into data/products.csv (column price_usd).

Why: the weekly price table is the source of truth for what is paid each week (RL, optimization).
products.csv keeps one reference price per product ("normal price") for things that need a single number:
household budget calibration, recipe cost, data checks.

Reference price = mean of regular_price_usd (price without promotions) over the training weeks.

Run from the repo root:
    python -m src.sync_reference_prices            # dry run: only shows old vs new prices
    python -m src.sync_reference_prices --apply    # writes data/products.csv
Afterwards: python -m src.households_gen, python -m src.data_gen, python -m src.check_data
Optional: --source <csv> (default data/price_weekly_synthetic.csv), --train-weeks 80
"""
import argparse

import pandas as pd

DEFAULT_SOURCE = "data/price_weekly_synthetic.csv"


def main(source: str, train_weeks: int, apply: bool) -> None:
    products = pd.read_csv("data/products.csv")
    prices = pd.read_csv(source)
    col = "regular_price_usd" if "regular_price_usd" in prices.columns else "price_usd"
    for need in ("week", "product_id", col):
        if need not in prices.columns:
            raise SystemExit(f"{source}: missing column '{need}' (columns: {list(prices.columns)})")
    if "region_id" in prices.columns and prices["region_id"].nunique() > 1:
        raise SystemExit(f"{source} has several regions; pass a single-region file")

    missing = sorted(set(products["product_id"]) - set(prices["product_id"]))
    if missing:
        raise SystemExit(f"{source}: no prices for {missing}")

    ref = (prices[prices["week"] < train_weeks].groupby("product_id")[col].mean().round(2))
    new = products["product_id"].map(ref)
    table = pd.DataFrame({"unit": products["unit"], "old": products["price_usd"], "new": new}).set_index(products["product_id"])
    table["ratio"] = (table["new"] / table["old"]).round(2)
    print(f"Reference price = mean {col} over weeks 0-{train_weeks - 1} of {source}\n")
    print(table.to_string())

    if not apply:
        print("\nDry run: nothing written. Add --apply to update data/products.csv")
        return
    products["price_usd"] = new.to_numpy()
    products.to_csv("data/products.csv", index=False)
    print("\nUpdated price_usd in data/products.csv. Now run: python -m src.households_gen, "
          "python -m src.data_gen, python -m src.check_data")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--train-weeks", type=int, default=80)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    main(a.source, a.train_weeks, a.apply)