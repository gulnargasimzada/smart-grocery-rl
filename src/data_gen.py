"""Creates the synthetic demand history. Run from the repo root:  python -m src.data_gen
Needs data/products.csv and data/households.csv (see src/households_gen.py).
Output: data/demand_weekly.csv (weekly totals per household and product, used by the forecast model)
Optional: python -m src.data_gen --daily also writes data/demand_daily.csv (large, not committed to git)."""
import argparse

import pandas as pd

from src.demand import simulate_demand

N_WEEKS = 104   # two years of history per household
SEED = 42


def main(write_daily: bool = False) -> None:
    products = pd.read_csv("data/products.csv")
    households = pd.read_csv("data/households.csv")

    frames = [simulate_demand(h, products, N_WEEKS, seed=SEED + i)
              for i, h in enumerate(households.to_dict("records"))]
    daily = pd.concat(frames, ignore_index=True)
    if write_daily:
        daily.to_csv("data/demand_daily.csv", index=False)

    weekly = daily.groupby(["household_id", "week", "guest_week", "away_week", "holiday_week", "product_id"],
                           as_index=False)["demand_units"].sum()
    weekly["demand_units"] = weekly["demand_units"].round(4)
    weekly.to_csv("data/demand_weekly.csv", index=False)
    print(f"daily rows: {len(daily)} | weekly rows: {len(weekly)} | "
          f"households: {weekly['household_id'].nunique()} | products: {weekly['product_id'].nunique()}")

    # Sanity check: does normal demand fit the budget and the protein target?
    p = products.set_index("product_id")
    w = weekly.copy()
    w["cost"] = w["demand_units"] * w["product_id"].map(p["price_usd"])
    w["protein"] = w["demand_units"] * w["product_id"].map(p["protein_g_per_unit"])
    tot = w.groupby(["household_id", "week"])[["cost", "protein"]].sum().reset_index()
    print("\nhousehold      budget | avg cost | weeks over budget | protein target | avg protein | weeks below target")
    for h in households.itertuples():
        t = tot[tot["household_id"] == h.household_id]
        print(f"{h.household_id:11s}  ${h.weekly_budget_usd:>4} | ${t['cost'].mean():>7.2f} | "
              f"{(t['cost'] > h.weekly_budget_usd).sum():>3}/{len(t)}            | "
              f"{h.weekly_protein_target_g:>6} g       | {t['protein'].mean():>7.0f} g   | "
              f"{(t['protein'] < h.weekly_protein_target_g).sum():>3}/{len(t)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", action="store_true", help="also write the large data/demand_daily.csv")
    main(ap.parse_args().daily)