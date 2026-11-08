"""Step 1 diagnostic: cost of recorded weekly demand vs. household budget.

For each (household, region, week) the cost is sum_p demand_units * price_usd.
This is NOT RL spending or optimised shopping: it assumes the household buys
exactly what it consumed that week at that week's regional price (no stock,
no waste, no constraints).

Read-only on data/. Writes CSVs and a markdown report to results/step1/.

Run:  python -m src.budget_diagnostic
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "step1"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hh = pd.read_csv(DATA / "households.csv")
    products = pd.read_csv(DATA / "products.csv")
    demand = pd.read_csv(DATA / "demand_weekly.csv")
    prices = pd.read_csv(DATA / "price_weekly_regional_synthetic.csv")
    return hh, products, demand, prices


def weekly_costs(hh, demand, prices) -> pd.DataFrame:
    """Return one row per (household, region, week) with cost and budget."""
    merged = demand.merge(
        prices[["region_id", "week", "product_id", "price_usd", "is_promo"]],
        on=["week", "product_id"],
        how="left",
    )
    if merged["price_usd"].isna().any():
        raise ValueError("Some demand rows have no matching price")
    merged["cost"] = merged["demand_units"] * merged["price_usd"]
    cost = (
        merged.groupby(["household_id", "region_id", "week"], as_index=False)
        .agg(cost=("cost", "sum"), guest_week=("guest_week", "first"))
    )
    cost = cost.merge(
        hh[["household_id", "persona", "n_people", "weekly_budget_usd"]],
        on="household_id",
    )
    cost["ratio"] = cost["cost"] / cost["weekly_budget_usd"]
    cost["over_budget"] = cost["ratio"] > 1.0
    return cost


def summarise(cost: pd.DataFrame):
    def agg(g):
        return pd.Series(
            {
                "avg_cost": g["cost"].mean(),
                "budget": g["weekly_budget_usd"].mean(),
                "avg_ratio": g["ratio"].mean(),
                "pct_weeks_over": 100 * g["over_budget"].mean(),
                "pct_weeks_over_120": 100 * (g["ratio"] > 1.2).mean(),
                "pct_weeks_under_50": 100 * (g["ratio"] < 0.5).mean(),
                "p95_ratio": g["ratio"].quantile(0.95),
            }
        )

    by_region = cost.groupby("region_id").apply(agg, include_groups=False)
    by_hh = cost.groupby(["household_id", "persona", "n_people"]).apply(agg, include_groups=False)
    by_hh_region = cost.groupby(["household_id", "region_id"]).apply(agg, include_groups=False)
    by_persona = cost.groupby("persona").apply(agg, include_groups=False)
    return by_region, by_hh, by_hh_region, by_persona


def md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Render a DataFrame as a markdown table (no tabulate dependency)."""
    d = df.reset_index()
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in r]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    hh, products, demand, prices = load()
    OUT.mkdir(parents=True, exist_ok=True)

    cost = weekly_costs(hh, demand, prices)
    by_region, by_hh, by_hh_region, by_persona = summarise(cost)

    cost.to_csv(OUT / "weekly_cost_detail.csv", index=False)
    by_region.to_csv(OUT / "by_region.csv")
    by_hh.to_csv(OUT / "by_household.csv")
    by_hh_region.to_csv(OUT / "by_household_region.csv")

    overall = {
        "avg_cost": cost["cost"].mean(),
        "avg_budget": cost["weekly_budget_usd"].mean(),
        "avg_ratio": cost["ratio"].mean(),
        "pct_over": 100 * cost["over_budget"].mean(),
        "pct_under_50": 100 * (cost["ratio"] < 0.5).mean(),
    }
    worst_pairs = by_hh_region.sort_values("pct_weeks_over", ascending=False).head(10)
    worst_hh = by_hh.sort_values("avg_ratio", ascending=False).head(10)
    best_hh = by_hh.sort_values("avg_ratio").head(5)
    guest = cost.groupby("guest_week")["ratio"].mean()
    n_hh = cost["household_id"].nunique()
    n_reg = cost["region_id"].nunique()
    n_wk = cost["week"].nunique()
    n_prod = demand["product_id"].nunique()
    n_over_hh = int((by_hh["pct_weeks_over"] > 0).sum())

    lines = [
        "# Step 1: budget vs. price compatibility (diagnostic)",
        "",
        f"Scope: {n_hh} households x {n_reg} regions x {n_wk} weeks, {n_prod} products.",
        "Cost = sum(demand_units x regional weekly price_usd). This is the cost of the *recorded",
        "demand*, not RL spending or optimised shopping (no stock, waste or constraints).",
        "Reproduce: `python -m src.budget_diagnostic`.",
        "",
        "## Overall",
        f"- Average weekly cost: ${overall['avg_cost']:.2f} vs. average budget ${overall['avg_budget']:.2f}"
        f" (mean cost/budget = {overall['avg_ratio']:.2f})",
        f"- Household-region-weeks over budget: {overall['pct_over']:.1f}%",
        f"- Household-region-weeks using under 50% of budget: {overall['pct_under_50']:.1f}%",
        f"- Households with at least one over-budget week: {n_over_hh} of {n_hh}",
        f"- Mean cost/budget in guest weeks: {guest.get(1, float('nan')):.2f}"
        f" vs. normal weeks: {guest.get(0, float('nan')):.2f}",
        "",
        "## By region",
        md(by_region),
        "",
        "## By persona",
        md(by_persona),
        "",
        "## Households with the highest cost/budget (all regions pooled)",
        md(worst_hh),
        "",
        "## Households with the lowest cost/budget",
        md(best_hh),
        "",
        "## Worst household-region pairs (% weeks over budget)",
        md(worst_pairs),
        "",
        "## Full per-household table",
        md(by_hh.sort_values("avg_ratio", ascending=False)),
        "",
        "Column notes: `avg_ratio` = mean(weekly cost / budget); `pct_weeks_over_120` = weeks above 120% of budget;",
        "`pct_weeks_under_50` = weeks below 50% of budget; `p95_ratio` = 95th percentile of cost/budget.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
