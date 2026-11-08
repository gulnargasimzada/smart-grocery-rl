"""Evaluate the optimization policy against the baselines with the shared evaluation framework (src/evaluate.py).

    python -m src.optimization_eval --tune                 # choose delta on the TRAINING windows (weeks 4-63) only
    python -m src.optimization_eval --split validation     # report on validation (weeks 64-76) with the tuned delta
    python -m src.optimization_eval --split train

Every policy runs the same EpisodeSpecs through the unchanged GroceryEnv and `src.evaluate.evaluate_policies`, so the metrics are
directly comparable with the baselines and with an RL agent (`RLPolicy`). The test split (weeks 78-103) is deliberately not offered;
`evaluate_policies("test", ...)` also refuses without `confirm_final_test=True`.
Outputs go to results/optimization/ (results/evaluation/ is left untouched).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.env import GroceryEnv
from src.evaluate import BASELINES, METRICS, EpisodeSpec, episode_specs, evaluate_policies, forecast_notes, md, spec_hash, validate_spec
from src.forecast_table import ForecastTable, ForecastUnavailable
from src.optimization import OptimizationPolicy

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "optimization"
DELTA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
TUNED_PATH = OUT / "tuned.json"
VARIANT = "no_events"


def load_table(path: str | Path = ROOT / "data" / "forecast_weekly.csv") -> ForecastTable:
    products = pd.read_csv(ROOT / "data" / "products.csv")["product_id"].tolist()
    return ForecastTable.from_csv(path, products)


def tune(table: ForecastTable, grid=DELTA_GRID) -> tuple[float, pd.DataFrame]:
    """Best delta by mean reward on the fixed TRAINING evaluation windows (ties -> smaller delta)."""
    pols = [OptimizationPolicy(delta=d, name=f"delta_{d:.1f}") for d in grid]
    _, summary, unavailable = evaluate_policies("train", pols, table=table, variant=VARIANT)
    if unavailable:
        raise ForecastUnavailable(f"cannot tune: {unavailable}")
    summary = summary.assign(delta=list(grid))
    best = float(summary.sort_values(["reward", "delta"], ascending=[False, True])["delta"].iloc[0])
    return best, summary


def pipeline_cuts(spec: EpisodeSpec, policy: OptimizationPolicy, table: ForecastTable) -> dict:
    """Replays one episode and sums what the environment pipeline removed (0 means the MILP order was already valid)."""
    validate_spec(spec)
    env = GroceryEnv(spec.household_id, spec.region_id, spec.budget_scenario, spec.storage_scenario, spec.horizon,
                     table.forecast_fn(spec.household_id, VARIANT))
    _, info = env.reset(options={"start_week": spec.start_week})
    policy.reset()
    tot = dict(rounding=0.0, stock_cap=0.0, storage=0.0, budget=0.0, max_budget_use=0.0)
    done = False
    while not done:
        _, _, _, done, info = env.step(policy.act(info["obs_dict"]))
        a = info["adjust"]
        tot["rounding"] += float(np.abs(a["rounding_delta"]).sum())
        tot["stock_cap"] += float(a["cap_cut"].sum())
        tot["storage"] += float(a["storage_cut"].sum())
        tot["budget"] += float(a["budget_cut"].sum())
        tot["max_budget_use"] = max(tot["max_budget_use"], info["spend"] / env.d.weekly_budget)
    return tot


def report(split: str, delta: float, table: ForecastTable) -> str:
    opt = OptimizationPolicy(delta=delta)
    episodes, summary, unavailable = evaluate_policies(split, list(BASELINES) + [opt], table=table, variant=VARIANT)
    specs = episode_specs(split)
    cuts = pd.DataFrame([{"start_week": s.start_week, **pipeline_cuts(s, OptimizationPolicy(delta=delta), table)} for s in specs])
    OUT.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(OUT / f"{split}_episodes.csv", index=False)
    summary.to_csv(OUT / f"{split}_summary.csv")
    cuts.to_csv(OUT / f"{split}_pipeline_cuts.csv", index=False)
    lines = [
        f"# Optimization vs. baselines: {split} split",
        "",
        f"hh_student, durham_nc, normal budget (synthetic), default storage; {len(specs)} episode(s) of 13 weeks starting in weeks "
        f"{[s.start_week for s in specs]} (spec hash `{spec_hash(specs)}`, same specs as `src.evaluate`). Forecasts: Ridge, `{VARIANT}`.",
        f"Optimization: weekly MILP, delta = {delta} (tuned on the training windows only). Deterministic paired replay of one recorded "
        "trajectory: no confidence intervals or significance claims. Simulated results. Test weeks 78-103 were not run.",
        "",
        "## Mean per 13-week episode",
        md(summary[METRICS]),
        "",
        "## Per episode",
        md(episodes.set_index(["policy", "start_week"])[["end_week"] + METRICS]),
        "",
        "## Environment pipeline cuts on the optimization orders (units removed; 0 = order already feasible)",
        md(cuts.set_index("start_week")),
        "",
    ]
    lines += [f"- `{k}` was NOT run: {v}" for k, v in unavailable.items()]
    lines += [f"- {n}" for n in forecast_notes(table, specs, VARIANT)]
    text = "\n".join(lines) + "\n"
    (OUT / f"report_{split}.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["train", "validation"])               # the test split is deliberately not offered
    ap.add_argument("--tune", action="store_true", help="choose delta on the training windows and save it")
    ap.add_argument("--forecast-csv", default=str(ROOT / "data" / "forecast_weekly.csv"))
    args = ap.parse_args()
    if not args.tune and not args.split:
        ap.error("give --tune and/or --split")
    table = load_table(args.forecast_csv)
    if args.tune:
        best, summary = tune(table)
        OUT.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUT / "tuning_train.csv")
        TUNED_PATH.write_text(json.dumps({"delta": best, "chosen_on": "train windows 4,17,30,43,51 (mean reward)",
                                          "grid": list(DELTA_GRID)}, indent=2) + "\n")
        print(summary[["delta"] + METRICS].round(3).to_string())
        print(f"\nchosen delta = {best} (saved to {TUNED_PATH.relative_to(ROOT)})")
    if args.split:
        delta = json.loads(TUNED_PATH.read_text())["delta"] if TUNED_PATH.exists() else OptimizationPolicy().delta
        print(report(args.split, delta, table))


if __name__ == "__main__":
    main()
