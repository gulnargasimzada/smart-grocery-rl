"""Diagnostics for GroceryEnv v0: stockout investigation and baseline comparison.

Uses TRAINING and VALIDATION weeks only (episodes start in weeks 4-51 and 64). The test weeks 78-103 are not run here and
must not be used to tune the environment or the policies; the script refuses any episode that ends after week 77.

Identical conditions: every policy runs the same episodes (same household, region, budget scenario, storage scenario,
start week, recorded demand and prices). Results are simulated and deterministic (recorded-demand replay).

Run:  python -m src.diagnose_env          # writes results/env_diagnostics/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.env import HORIZON, GroceryEnv
from src.policies import AverageRule, BuyNothing, Fixed, FixedRounded, Policy

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "env_diagnostics"
TRAIN_STARTS = [4, 17, 30, 43, 51]     # windows end by week 63
VAL_STARTS = [64]                      # window 64-76
MAX_WEEK = 77                          # last week that diagnostics may touch (validation)
ABLATIONS = ["none", "budget", "stock_cap", "package", "shelf_life", "storage", "all"]


class NaiveRef(Policy):
    """Diagnostic only: order the trailing mean demand every week and let the environment round to the nearest package."""
    name = "naive_ref"

    def act(self, obs: dict) -> np.ndarray:
        return np.asarray(obs["ref_demand"], float)


def make_env(ablation: str = "none") -> GroceryEnv:
    """Counterfactual environment with one constraint relaxed (for attributing stockouts), or the real one."""
    env = GroceryEnv()
    d = env.d
    if ablation in ("budget", "all"):
        d.weekly_budget = 1e9
    if ablation in ("stock_cap", "all"):
        env.stock_cap_weeks = 1e6
    if ablation in ("package", "all"):
        d.step_size = np.full(d.n, 1e-3)
    if ablation in ("shelf_life", "all"):
        d.L_default = d.L_default * 1000.0
        d.L_freezer = d.L_freezer * 1000.0
    if ablation in ("storage", "all"):
        d.capacity = {k: 1e6 for k in d.capacity}
    return env


def make_policy(name: str, env: GroceryEnv) -> Policy:
    d = env.d
    train_mean = d.demand[4:64].mean(axis=0)               # training weeks only
    return {
        "buy_nothing": lambda: BuyNothing(d.n),
        "naive_ref": lambda: NaiveRef(),
        "fixed": lambda: Fixed(train_mean, d.step_size),
        "fixed_rounded": lambda: FixedRounded(train_mean),
        "average_rule": lambda: AverageRule(d.product_ids, d.step_size, 1.0),
        "average_rule_1.5": lambda: AverageRule(d.product_ids, d.step_size, 1.5),
        "average_rule_2.0": lambda: AverageRule(d.product_ids, d.step_size, 2.0),
    }[name]()


def run_episode(env: GroceryEnv, policy: Policy, start: int) -> dict:
    assert start + HORIZON - 1 <= MAX_WEEK, "diagnostics may only use training and validation weeks (<= 77)"
    n = env.d.n
    _, info = env.reset(options={"start_week": start})
    policy.reset()
    obs_dict, done = info["obs_dict"], False
    tot = dict(reward=0.0, spend=0.0, waste_cost=0.0, waste_units=0.0, stockout_units=0.0, stockout_value=0.0, stockout_days=0,
               protein=[], demand_value=0.0, demand_units=np.zeros(n), stockout_p=np.zeros(n), stockout_value_p=np.zeros(n),
               cut_rounding=np.zeros(n), cut_cap=np.zeros(n), cut_storage=np.zeros(n), cut_budget=np.zeros(n), requested=np.zeros(n))
    while not done:
        week = env.week
        _, r, done, _, info = env.step(policy.act(obs_dict))
        obs_dict = info["obs_dict"]
        price, dem = env.price_at(week), env.d.demand[week]
        tot["reward"] += r
        tot["spend"] += info["spend"]
        tot["waste_cost"] += info["waste_cost"]
        tot["waste_units"] += info["waste_units"]
        tot["stockout_units"] += info["stockout_units"]
        tot["stockout_value"] += info["stockout_value"]
        tot["stockout_days"] += info["stockout_days"]
        tot["protein"].append(info["protein_target_met"])
        tot["demand_value"] += float((dem * price).sum())
        tot["demand_units"] += dem
        tot["stockout_p"] += info["stockout_by_product"]
        tot["stockout_value_p"] += info["stockout_by_product"] * price
        a = info["adjust"]
        tot["requested"] += a["requested"]
        tot["cut_cap"] += a["cap_cut"]
        tot["cut_storage"] += a["storage_cut"]
        tot["cut_budget"] += a["budget_cut"]
        tot["cut_rounding"] += a["rounding_delta"]
    tot["net_cost"] = info["episode_net_cost"]
    tot["terminal_charge"] = info["terminal_charge"]
    tot["protein_coverage"] = float(np.mean(tot["protein"]))
    tot["fill_rate_value"] = 1.0 - tot["stockout_value"] / tot["demand_value"]
    return tot


def run_set(policy_name: str, ablation: str, starts: list[int]) -> list[dict]:
    out = []
    for s in starts:
        env = make_env(ablation)
        out.append(run_episode(env, make_policy(policy_name, env), s))
    return out


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    d = df.reset_index()
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        rows.append("| " + " | ".join(fmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |")
    return "\n".join(rows)


SUMMARY_COLS = ["reward", "spend", "waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days",
                "fill_rate_value", "protein_coverage", "net_cost", "terminal_charge"]


def summary(eps: list[dict]) -> pd.Series:
    return pd.Series({c: float(np.mean([e[c] for e in eps])) for c in SUMMARY_COLS})


def product_table(eps: list[dict], env: GroceryEnv) -> pd.DataFrame:
    s = lambda k: np.sum([e[k] for e in eps], axis=0)
    dem, so = s("demand_units"), s("stockout_p")
    df = pd.DataFrame({
        "demand_units": dem, "stockout_units": so, "fill_rate": 1 - so / np.maximum(dem, 1e-9),
        "stockout_value_usd": s("stockout_value_p"), "requested_units": s("requested"),
        "cut_by_rounding_units": s("cut_rounding"), "cut_by_stock_cap": s("cut_cap"),
        "cut_by_storage": s("cut_storage"), "cut_by_budget": s("cut_budget"),
    }, index=env.d.product_ids)
    df.index.name = "product"
    return df.sort_values("stockout_value_usd", ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    starts_all = TRAIN_STARTS + VAL_STARTS
    base_env = make_env()
    lines = [
        "# GroceryEnv v0 diagnostics (training and validation weeks only)",
        "",
        f"hh_student, durham_nc, normal budget (${base_env.d.weekly_budget:.0f}/week, synthetic), default storage, {HORIZON}-week episodes.",
        f"Training windows start in weeks {TRAIN_STARTS} (end by week 63); validation window starts in week {VAL_STARTS[0]} (weeks 64-76).",
        "No test weeks (78-103) are used. All policies run the same episodes (recorded-demand replay, deterministic). Simulated results.",
        "`reward` is the training signal; `spend` is actual cash spent; `net_cost` = consumed + waste + terminal charge.",
        "`fill_rate_value` = 1 - stockout value / demand value (price-weighted). `protein_coverage` = mean weekly protein ratio vs. target.",
        "",
    ]

    # ---- 1. baseline comparison
    pols = ["buy_nothing", "fixed", "fixed_rounded", "average_rule", "average_rule_1.5", "average_rule_2.0", "naive_ref"]
    results = {p: {"train": run_set(p, "none", TRAIN_STARTS), "val": run_set(p, "none", VAL_STARTS)} for p in pols}
    for split in ("train", "val"):
        tab = pd.DataFrame({p: summary(results[p][split]) for p in pols}).T
        tab.index.name = "policy"
        tab.to_csv(OUT / f"policies_{split}.csv")
        lines += [f"## Policy comparison, {split} windows (mean per 13-week episode)", md(tab), ""]
    lines += ["`fixed` = standing order of the mean weekly demand of weeks 4-63; `fixed_rounded` and `naive_ref` are diagnostic variants "
              "where the environment rounds fractional orders to the nearest package (small items are never bought). "
              "`average_rule_1.5/2.0` are cover sensitivity rows; the baseline definition stays at 1.0 week of cover.", ""]

    # ---- 2. stockouts by product
    for pol in ("naive_ref", "average_rule", "fixed"):
        eps = results[pol]["train"] + results[pol]["val"]
        pt = product_table(eps, base_env)
        pt.to_csv(OUT / f"stockouts_by_product_{pol}.csv")
        tot_dem = sum(e["demand_value"] for e in eps)
        tot_so = sum(e["stockout_value"] for e in eps)
        lines += [
            f"## Stockouts by product: {pol} (sum over the {len(eps)} train+validation episodes, top 12 by stockout value)",
            f"Total demand value ${tot_dem:.0f}, stockout value ${tot_so:.0f}, fill rate (value) {1 - tot_so / tot_dem:.1%}; "
            f"stockout units {sum(e['stockout_units'] for e in eps) / len(eps):.1f} per episode (mixed purchase units).",
            md(pt.head(12)), "",
            "`cut_by_*` = requested units removed by that step of the order pipeline (rounding is signed: negative = rounded down).", "",
        ]

    # ---- 3. cause attribution by relaxing one constraint at a time
    rows = []
    for pol in ("naive_ref", "fixed", "average_rule"):
        base = None
        for ab in ABLATIONS:
            eps = run_set(pol, ab, starts_all)
            so_val = float(np.mean([e["stockout_value"] for e in eps]))
            so_units = float(np.mean([e["stockout_units"] for e in eps]))
            base = so_val if ab == "none" else base
            rows.append({"policy": pol, "relaxed": ab, "stockout_units": so_units, "stockout_value_usd": so_val,
                         "fill_rate_value": float(np.mean([e["fill_rate_value"] for e in eps])),
                         "removed_vs_none_%": 100 * (1 - so_val / base) if base else 0.0})
    abl = pd.DataFrame(rows).set_index(["policy", "relaxed"])
    abl.to_csv(OUT / "stockout_attribution.csv")
    lines += [
        "## Cause attribution: stockouts when ONE constraint is relaxed (mean per episode, train+validation)",
        "`budget` = unlimited budget; `stock_cap` = no stock cap; `package` = 0.001 package step; `shelf_life` = 1000x shelf lives; "
        "`storage` = unlimited capacity (no thaw/space limits); `all` = everything relaxed. What remains under `all` is demand "
        "the policy did not order for (forecast/policy error), not an environment constraint.",
        md(abl), "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
