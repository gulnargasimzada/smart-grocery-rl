"""Verification of the Grocery Coach adapter on the VALIDATION episode (weeks 64-76), through the public JSON interface.

Each week the environment's inventory is exported as JSON lots, sent to `CoachAdapter.recommend`, the returned shopping list is executed in the
environment, and the result is compared with the pure models run by the unchanged evaluation framework:
    priority reduce_cost_waste  must equal  Optimization (src.optimization, delta 0.4)
    priority availability       must equal  PPO with forecast, checkpoint forecast_seed0
    priority balanced           runs the documented rule; reported for information (no reference model)
Also writes example responses (results/coach/examples/*.json). The test split (weeks 78-103) is not used.

Run:  python -m src.coach_check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.coach_api import PRIORITIES, CoachAdapter, lots_to_json
from src.evaluate import EpisodeSpec, run_episode
from src.optimization import OptimizationPolicy
from src.policies import RLPolicy

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "coach"
VAL_START = 64
EXAMPLE_WEEK_INDEX = 6               # decision 7 of the validation episode (week 70) for the example responses


def rolling_episode(adapter: CoachAdapter, priority: str, start: int = VAL_START, ppo_seed: int = 0, keep_responses: bool = False) -> dict:
    """Runs one 13-week validation episode where every weekly order comes from the adapter's JSON interface."""
    assert 64 <= start and start + 12 <= 77, "validation weeks only"
    env = adapter.env("hh_student", "durham_nc", "normal")
    _, info = env.reset(options={"start_week": start})
    tot = dict(reward=0.0, cash_spend=0.0, waste_cost=0.0, waste_units=0.0, stockout_units=0.0, stockout_value=0.0, demand_value=0.0)
    protein_ratio, done, responses, selected = 1.0, False, [], []
    while not done:
        week = env.week
        req = {"household_id": "hh_student", "region_id": "durham_nc", "week": week, "priority": priority, "weeks_remaining": 13 - env.t,
               "lots": lots_to_json(env), "last_week_protein_ratio": protein_ratio, "ppo_seed": ppo_seed}
        resp = adapter.recommend(req)
        assert resp["status"] == "ok", resp["errors"]
        order = np.zeros(env.d.n)
        for it in resp["shopping_list"]:
            order[env.d.product_ids.index(it["product_id"])] = it["quantity"]
        price, dem = env.price_at(week), env.d.demand[week]
        _, reward, terminated, truncated, info = env.step(order)
        done = terminated or truncated
        protein_ratio = info["protein_target_met"]
        tot["reward"] += reward
        tot["cash_spend"] += info["spend"]
        for k in ("waste_cost", "waste_units", "stockout_units", "stockout_value"):
            tot[k] += info[k]
        tot["demand_value"] += float((dem * price).sum())
        selected.append(resp["strategy"]["selected"])
        if keep_responses:
            responses.append(resp)
    tot["fill_rate_value"] = 1.0 - tot["stockout_value"] / tot["demand_value"]
    tot["net_cost"] = info["episode_net_cost"]
    return {"metrics": tot, "selected_per_week": selected, "responses": responses}


def reference(table, adapter: CoachAdapter) -> dict:
    spec = EpisodeSpec("validation", VAL_START)
    opt = run_episode(spec, OptimizationPolicy(delta=adapter._delta, name="optimization"), table, "no_events")
    ppo = run_episode(spec, adapter.ppo(0), table, "no_events")
    return {"optimization": opt, "ppo_forecast_seed0": ppo}


def main() -> None:
    adapter = CoachAdapter()
    table = adapter.table()
    ref = reference(table, adapter)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "examples").mkdir(exist_ok=True)
    rows = {}
    for pr in PRIORITIES:
        r = rolling_episode(adapter, pr, keep_responses=True)
        rows[f"adapter: {pr}"] = r["metrics"]
        rows[f"adapter: {pr}"]["weeks_using_optimization"] = float(sum(s == "optimization" for s in r["selected_per_week"]))
        (OUT / "examples" / f"{pr}_week70.json").write_text(json.dumps(r["responses"][EXAMPLE_WEEK_INDEX], indent=2) + "\n")
    for name, m in ref.items():
        rows[f"pure model: {name}"] = {k: m[k] for k in ("reward", "cash_spend", "waste_cost", "waste_units", "stockout_units", "stockout_value", "fill_rate_value")}
    df = pd.DataFrame(rows).T[["reward", "cash_spend", "waste_units", "waste_cost", "stockout_units", "stockout_value", "fill_rate_value", "weeks_using_optimization"]]
    df.to_csv(OUT / "validation_check.csv")
    cols = list(df.columns)
    lines = ["# Grocery Coach adapter: validation check (weeks 64-76, test weeks not used)", "",
             "Every weekly order below came from the adapter's JSON interface (inventory exported as lots, shopping list executed in the unchanged environment). "
             "`reduce_cost_waste` and `availability` must equal the pure Optimization and PPO-with-forecast (seed 0) runs; `balanced` applies the documented rule "
             "(see docs/GROCERY_COACH_INTEGRATION.md) and has no reference model. One recorded trajectory; no significance claims.", "",
             "| run | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, r in df.iterrows():
        lines.append(f"| {name} | " + " | ".join("" if pd.isna(v) else f"{v:.2f}" for v in r) + " |")
    (OUT / "validation_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
