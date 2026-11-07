"""Creates the synthetic history. Run from the repo root:  python -m src.data_gen
Outputs: data/demand_daily.csv (one row per household-day-product)
         data/demand_weekly.csv (weekly totals, convenient for the forecast model)"""
import pandas as pd

from src.demand import simulate_demand

N_WEEKS = 104   # two years of history per household
SEED = 42


def main():
    products = pd.read_csv("data/products.csv")
    households = pd.read_csv("data/households.csv")
    pids = products["product_id"].tolist()

    frames = [simulate_demand(h, pids, N_WEEKS, seed=SEED + i)
              for i, h in enumerate(households.to_dict("records"))]
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv("data/demand_daily.csv", index=False)

    weekly = daily.groupby(["household_id", "week", "guest_week", "product_id"],
                           as_index=False)["demand_units"].sum()
    weekly["demand_units"] = weekly["demand_units"].round(4)
    weekly.to_csv("data/demand_weekly.csv", index=False)
    print(f"daily rows: {len(daily)} | weekly rows: {len(weekly)}")

    # Sanity check: does normal demand fit the budget and the protein target?
    p = products.set_index("product_id")
    w = weekly.copy()
    w["cost"] = w["demand_units"] * w["product_id"].map(p["price_usd"])
    w["protein"] = w["demand_units"] * w["product_id"].map(p["protein_g_per_unit"])
    tot = w.groupby(["household_id", "week"])[["cost", "protein"]].sum().reset_index()
    for h in households.itertuples():
        t = tot[tot["household_id"] == h.household_id]
        print(f"{h.household_id}: avg weekly demand cost ${t['cost'].mean():.2f} (budget ${h.weekly_budget_usd}) | "
              f"weeks above budget {(t['cost'] > h.weekly_budget_usd).sum()}/{len(t)} | "
              f"avg protein {t['protein'].mean():.0f} g (target {h.weekly_protein_target_g})")


if __name__ == "__main__":
    main()