"""Build synthetic regional weekly budgets for every household x region.

Output: data/household_region_budgets.csv (150 rows = 30 households x 5 regions).
Validation report: results/regional_budgets/validation.md (+ CSV).

Method
------
cost(h, r, w) = sum_p demand_units(h, w, p) * price_usd(r, w, p)   (diagnostic cost of
recorded demand, not RL spending).
Calibration uses NORMAL weeks only (guest_week = away_week = holiday_week = 0) in the
calibration window (weeks 0-77). Budget = P75 of those costs, rounded UP to a whole USD.
Validation uses the held-out weeks 78-103 (never used for calibration).
The original `weekly_budget_usd` from households.csv is kept as `original_budget_usd`
for the fixed-budget comparison scenario.

ALL BUDGETS ARE SYNTHETIC ASSUMPTIONS for the demo, not real household data.

Run:    python -m src.build_regional_budgets          # rebuild file + report
Check:  python -m src.build_regional_budgets --check  # verify committed file (exit 1 on failure)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_FILE = DATA / "household_region_budgets.csv"
REPORT_DIR = ROOT / "results" / "regional_budgets"

PERCENTILE = 75
CALIB_WEEKS = (0, 77)   # inclusive
VALID_WEEKS = (78, 103)  # inclusive
FLAGS = ["guest_week", "away_week", "holiday_week"]
COLUMNS = [
    "household_id", "region_id", "region_name", "original_budget_usd", "calibrated_budget_usd",
    "calibration_percentile", "calibration_weeks", "n_calibration_weeks",
    "calib_mean_cost_usd", "calib_median_cost_usd", "calib_p75_cost_usd",
    "budget_ratio_to_original", "budget_source",
    # Scenario columns (appended; the columns above are the original P75 results, unchanged)
    "normal_budget_usd", "challenging_budget_usd", "tight_budget_usd", "calib_p90_cost_usd",
]
NORMAL_PERCENTILE = 90  # "normal" scenario (default); "challenging" = P75; "tight" = original budget
SCENARIOS = {"normal": "normal_budget_usd", "challenging": "challenging_budget_usd", "tight": "tight_budget_usd"}


def weekly_costs() -> pd.DataFrame:
    """One row per (household, region, week) with cost and week-type flags."""
    demand = pd.read_csv(DATA / "demand_weekly.csv")
    prices = pd.read_csv(DATA / "price_weekly_regional_synthetic.csv")
    m = demand.merge(
        prices[["region_id", "region_name", "week", "product_id", "price_usd"]],
        on=["week", "product_id"], how="left",
    )
    if m["price_usd"].isna().any():
        raise ValueError("demand rows without a matching price")
    m["cost"] = m["demand_units"] * m["price_usd"]
    out = m.groupby(["household_id", "region_id", "region_name", "week"], as_index=False).agg(
        cost=("cost", "sum"), **{f: (f, "max") for f in FLAGS}
    )
    out["normal_week"] = out[FLAGS].sum(axis=1) == 0
    return out


def build_budgets(costs: pd.DataFrame | None = None) -> pd.DataFrame:
    costs = weekly_costs() if costs is None else costs
    hh = pd.read_csv(DATA / "households.csv")
    lo, hi = CALIB_WEEKS
    cal = costs[costs["week"].between(lo, hi) & costs["normal_week"]]
    g = cal.groupby(["household_id", "region_id", "region_name"])["cost"]
    stats = g.agg(
        n_calibration_weeks="count", calib_mean_cost_usd="mean", calib_median_cost_usd="median",
        calib_p75_cost_usd=lambda s: np.percentile(s, PERCENTILE),
        calib_p90_cost_usd=lambda s: np.percentile(s, NORMAL_PERCENTILE),
    ).reset_index()
    stats["calibrated_budget_usd"] = np.ceil(stats["calib_p75_cost_usd"]).astype(int)
    stats["normal_budget_usd"] = np.ceil(stats["calib_p90_cost_usd"]).astype(int)
    stats["challenging_budget_usd"] = stats["calibrated_budget_usd"]
    df = stats.merge(
        hh[["household_id", "weekly_budget_usd"]].rename(columns={"weekly_budget_usd": "original_budget_usd"}),
        on="household_id", how="left",
    )
    df["calibration_percentile"] = PERCENTILE
    df["calibration_weeks"] = f"{lo}-{hi} normal weeks only"
    df["budget_ratio_to_original"] = df["calibrated_budget_usd"] / df["original_budget_usd"]
    df["budget_source"] = "synthetic"
    df["tight_budget_usd"] = df["original_budget_usd"]
    for c in ["calib_mean_cost_usd", "calib_median_cost_usd", "calib_p75_cost_usd", "calib_p90_cost_usd",
              "budget_ratio_to_original"]:
        df[c] = df[c].round(4)
    df = df.sort_values(["household_id", "region_id"]).reset_index(drop=True)
    return df[COLUMNS]


def validate(costs: pd.DataFrame, budgets: pd.DataFrame) -> pd.DataFrame:
    """Over-budget rates on held-out weeks for calibrated and original budgets."""
    lo, hi = VALID_WEEKS
    v = costs[costs["week"].between(lo, hi)].merge(
        budgets[["household_id", "region_id", "original_budget_usd", "calibrated_budget_usd"]],
        on=["household_id", "region_id"],
    )
    v["over_cal"] = v["cost"] > v["calibrated_budget_usd"]
    v["over_orig"] = v["cost"] > v["original_budget_usd"]
    v["ratio_cal"] = v["cost"] / v["calibrated_budget_usd"]
    v["ratio_orig"] = v["cost"] / v["original_budget_usd"]
    return v


def summarise(v: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def f(g):
        n = g[g["normal_week"]]
        return pd.Series({
            "n_weeks": len(g),
            "calibrated_over_pct_all": 100 * g["over_cal"].mean(),
            "calibrated_over_pct_normal": 100 * n["over_cal"].mean(),
            "calibrated_mean_ratio": g["ratio_cal"].mean(),
            "original_over_pct_all": 100 * g["over_orig"].mean(),
            "original_over_pct_normal": 100 * n["over_orig"].mean(),
            "original_mean_ratio": g["ratio_orig"].mean(),
        })
    return v.groupby(by).apply(f, include_groups=False)


def scenario_table(costs: pd.DataFrame, budgets: pd.DataFrame, by_region: bool = True) -> pd.DataFrame:
    """Over-budget rate and mean cost/budget per scenario on held-out weeks 78-103."""
    lo, hi = VALID_WEEKS
    v = costs[costs["week"].between(lo, hi)].merge(
        budgets[["household_id", "region_id", *SCENARIOS.values()]], on=["household_id", "region_id"]
    )
    rows = []
    for region, g in [*v.groupby("region_id"), ("ALL", v)] if by_region else [("ALL", v)]:
        for name, col in SCENARIOS.items():
            ratio = g["cost"] / g[col]
            n = g["normal_week"]
            rows.append({
                "region_id": region, "scenario": name,
                "over_pct_all": 100 * (ratio > 1).mean(),
                "over_pct_normal": 100 * (ratio[n] > 1).mean(),
                "mean_ratio_all": ratio.mean(),
                "mean_ratio_normal": ratio[n].mean(),
                "mean_budget_usd": g[col].mean(),
            })
    return pd.DataFrame(rows).set_index(["region_id", "scenario"])


def write_scenario_report(costs: pd.DataFrame, budgets: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    t = scenario_table(costs, budgets)
    t.to_csv(REPORT_DIR / "scenarios_by_region.csv")
    lines = [
        "# Budget scenarios: validation on held-out weeks 78-103 (SYNTHETIC budgets)",
        "",
        "| Scenario | Budget column | Definition |",
        "|---|---|---|",
        f"| normal (default) | `normal_budget_usd` | P{NORMAL_PERCENTILE} of normal-week cost, weeks 0-77, rounded up |",
        f"| challenging | `challenging_budget_usd` (= `calibrated_budget_usd`) | P{PERCENTILE} of normal-week cost, weeks 0-77, rounded up |",
        "| tight | `tight_budget_usd` (= `original_budget_usd`) | original budget from households.csv |",
        "",
        "`all` = every held-out week; `normal` = held-out weeks without guest/away/holiday flag.",
        "Cost = recorded demand x regional price (diagnostic, not RL spending). Calibration never uses weeks 78-103.",
        "",
        md(t),
        "",
    ]
    (REPORT_DIR / "scenarios.md").write_text("\n".join(lines))
    print("\n".join(lines))


def md(df: pd.DataFrame) -> str:
    d = df.reset_index()
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        cells = []
        for c, x in zip(d.columns, r):
            cells.append(str(int(x)) if c == "n_weeks" else f"{x:.2f}" if isinstance(x, float) else str(x))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report(budgets: pd.DataFrame, v: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    by_region = summarise(v, ["region_id"])
    by_hh = summarise(v, ["household_id"])
    by_pair = summarise(v, ["household_id", "region_id"])
    by_region.to_csv(REPORT_DIR / "validation_by_region.csv")
    by_pair.to_csv(REPORT_DIR / "validation_by_household_region.csv")
    worst = by_pair.sort_values("calibrated_over_pct_normal", ascending=False).head(10)
    ratio = budgets.groupby("region_id")["budget_ratio_to_original"].agg(["min", "mean", "max"])
    lines = [
        "# Regional household budgets: calibration and validation (SYNTHETIC)",
        "",
        "All budgets are synthetic assumptions for the demo, not real household data.",
        f"Calibration: P{PERCENTILE} of weekly cost of recorded demand, weeks {CALIB_WEEKS[0]}-{CALIB_WEEKS[1]},"
        " normal weeks only (no guest, away or holiday flag), rounded up to whole USD.",
        f"Validation: held-out weeks {VALID_WEEKS[0]}-{VALID_WEEKS[1]} ({VALID_WEEKS[1]-VALID_WEEKS[0]+1} weeks)."
        " `all` includes guest/away/holiday weeks; `normal` excludes them.",
        "`calibrated_*` uses `calibrated_budget_usd`; `original_*` uses the fixed budget from households.csv.",
        "Reproduce: `python -m src.build_regional_budgets`; verify: `python -m src.build_regional_budgets --check`.",
        "",
        "## Over-budget rate by region (held-out weeks)",
        md(by_region),
        "",
        "## Calibrated budget / original budget by region",
        md(ratio),
        "",
        "## Over-budget rate by household (held-out weeks, regions pooled)",
        md(by_hh),
        "",
        "## Worst 10 household-region pairs (calibrated, normal weeks)",
        md(worst),
        "",
    ]
    (REPORT_DIR / "validation.md").write_text("\n".join(lines))
    print("\n".join(lines[:22]))


def check() -> int:
    """Reproducibility and integrity checks; returns number of failures."""
    fails: list[str] = []
    committed = pd.read_csv(OUT_FILE)
    costs = weekly_costs()
    fresh = build_budgets(costs)
    hh = pd.read_csv(DATA / "households.csv")
    regions = pd.read_csv(DATA / "price_weekly_regional_synthetic.csv")["region_id"].unique()

    def ok(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    ok(list(committed.columns) == COLUMNS, "columns match the schema")
    ok(len(committed) == len(hh) * len(regions) == 150, "150 rows = 30 households x 5 regions")
    ok(not committed.duplicated(["household_id", "region_id"]).any(), "no duplicate household-region keys")
    ok(set(committed["household_id"]) == set(hh["household_id"]), "all households present")
    ok((committed["budget_source"] == "synthetic").all(), "every row labelled synthetic")
    ok(committed.isna().sum().sum() == 0, "no missing values")
    ok((committed["calibrated_budget_usd"] > 0).all(), "budgets are positive")
    orig = committed.merge(hh[["household_id", "weekly_budget_usd"]], on="household_id")
    ok((orig["original_budget_usd"] == orig["weekly_budget_usd"]).all(), "original budget equals households.csv")
    ok(committed["n_calibration_weeks"].min() >= 50, "at least 50 normal calibration weeks per row")
    try:
        pd.testing.assert_frame_equal(committed.reset_index(drop=True), fresh, check_dtype=False, atol=1e-4)
        ok(True, "committed file equals a fresh rebuild")
    except AssertionError as e:
        ok(False, f"committed file differs from rebuild: {str(e)[:120]}")
    fresh2 = build_budgets(weekly_costs())
    ok(fresh.equals(fresh2), "two rebuilds are identical (deterministic)")
    ok((committed["challenging_budget_usd"] == committed["calibrated_budget_usd"]).all(),
       "challenging scenario equals the preserved P75 budget")
    ok((committed["tight_budget_usd"] == committed["original_budget_usd"]).all(),
       "tight scenario equals the original household budget")
    ok((committed["normal_budget_usd"] >= committed["challenging_budget_usd"]).all(),
       "normal (P90) budget >= challenging (P75) budget")
    ok((committed["normal_budget_usd"] == np.ceil(committed["calib_p90_cost_usd"] - 1e-4)).all(),
       "normal budget = ceil(P90 of calibration costs)")
    t = scenario_table(costs, committed, by_region=False).loc["ALL"]
    ok(t.loc["normal", "over_pct_normal"] < t.loc["challenging", "over_pct_normal"] < t.loc["tight", "over_pct_normal"],
       "over-budget rate ordering: normal < challenging < tight")
    lo, hi = CALIB_WEEKS
    cal_weeks = costs[costs["week"].between(lo, hi) & costs["normal_week"]]["week"]
    ok(cal_weeks.max() <= 77, "calibration uses weeks <= 77 only (no leakage from validation weeks)")
    return len(fails)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed file; do not write")
    args = ap.parse_args()
    if args.check:
        n = check()
        print(f"{n} check(s) failed" if n else "all checks passed")
        sys.exit(1 if n else 0)
    costs = weekly_costs()
    budgets = build_budgets(costs)
    budgets.to_csv(OUT_FILE, index=False)
    write_report(budgets, validate(costs, budgets))
    write_scenario_report(costs, budgets)
    print(f"\nwrote {OUT_FILE.relative_to(ROOT)} ({len(budgets)} rows)")


if __name__ == "__main__":
    main()
