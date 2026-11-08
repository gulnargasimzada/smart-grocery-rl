"""Export what each strategy actually did, week by week, for the app (results/ui/plans.json).

    python export_ui_plans.py

It replays the SAME frozen methods as `src/final_eval.py` (Fixed, Average, Forecast Rule, Optimization, PPO with and without the
Ridge forecast) on the held-out test episodes, and records for every week: the basket after the constraint pipeline, the
pipeline notes, prices, what was at home (each lot with its storage and days left), the forecast and what happened.
The replay is deterministic, so its totals match results/final/test exactly; nothing is trained, tuned or re-selected.
It refuses to run before the one-time test evaluation exists (results/final/test/), so it can't stand in for that run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import comparison as cmp
from src.env import GroceryEnv
from src.evaluate import episode_specs, make_policy
from src.rl_wrapper import OrderUpToWrapper

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "results" / "final" / "test"
OUT = ROOT / "results" / "ui" / "plans.json"
LABELS = {"fixed": "Fixed list", "average_rule": "Average rule", "forecast_rule": "Forecast rule (Ridge)",
          "optimization": "Optimization (MILP)", "ppo_forecast": "RL agent (PPO + forecast)", "ppo_no_forecast": "RL agent (PPO, no forecast)"}


def r(x, n=3):
    return round(float(x), n)


def best_seed(agent: str) -> int:
    """The seed with the best reward on the canonical VALIDATION episode (never chosen on test)."""
    v = pd.read_csv(ROOT / "results" / "comparison" / "validation_runs_canonical.csv")
    v = v[v["method"].str.startswith(f"ppo_{agent}_seed")]
    return int(v.sort_values("reward", ascending=False).iloc[0]["method"].rsplit("seed", 1)[1])


def replay(spec, policy, table):
    forecast_fn = table.forecast_fn(spec.household_id, cmp.VARIANT) if (table is not None and policy.uses_forecast) else None
    env = GroceryEnv(spec.household_id, spec.region_id, spec.budget_scenario, spec.storage_scenario, spec.horizon, forecast_fn)
    base, d = env, env.d
    if policy.normalized_action:
        env = OrderUpToWrapper(base)
    obs, info = env.reset(options={"start_week": spec.start_week})
    policy.reset()
    obs_dict, done, weeks = info["obs_dict"], False, {}
    while not done:
        week = base.week
        lots = [{"p": d.product_ids[l.p], "u": r(l.units), "days": r(base.remaining_days(l), 1), "where": base._location(l)}
                for l in base.lots if l.units > 1e-6]
        fc = obs_dict.get("forecast")
        action = policy.act_vec(obs) if policy.normalized_action else policy.act(obs_dict)
        obs, reward, done, _, info = env.step(action)
        order = {pid: r(u) for pid, u in zip(d.product_ids, np.asarray(info["order"])) if u > 1e-6}
        weeks[str(week)] = {
            "order": order, "notes": list(info.get("notes") or []), "home": lots,
            "price": {pid: r(p, 2) for pid, p in zip(d.product_ids, base.price_at(week))},
            "forecast": None if fc is None else {k: r(v) for k, v in fc.items()},
            "ref": {pid: r(v) for pid, v in zip(d.product_ids, obs_dict["ref_demand"])},
            "demand": {pid: r(v) for pid, v in zip(d.product_ids, d.demand[week]) if v > 1e-6},
            "spend": r(info["spend"], 2), "waste_cost": r(info["waste_cost"], 2), "waste_units": r(info["waste_units"]),
            "stockout_units": r(info["stockout_units"]), "stockout_value": r(info["stockout_value"], 2),
            "stockout_days": int(info["stockout_days"]), "protein_target_met": r(info["protein_target_met"]), "reward": r(reward, 4),
        }
        obs_dict = info["obs_dict"]
    return weeks, d


def main() -> None:
    if not (FINAL / "mean_over_episodes.csv").exists():
        sys.exit("results/final/test/ doesn't exist yet. Run the one-time held-out evaluation (src/final_eval.py) first.")
    table = cmp.load_table()
    specs = episode_specs("test")
    methods = {"fixed": "fixed", "average_rule": "average_rule", "forecast_rule": "forecast_rule",
               "optimization": cmp.build_methods.__globals__["OptimizationPolicy"](delta=cmp.delta_from_tuning(), name="optimization")}
    seeds = {}
    for agent in cmp.PPO_AGENTS:
        seeds[agent] = best_seed(agent)
        methods[f"ppo_{agent}"] = cmp.load_ppo(agent, seeds[agent])
    out_methods, d = {}, None
    for name, m in methods.items():
        weeks = {}
        for sp in specs:
            pol = make_policy(m, GroceryEnv(sp.household_id, sp.region_id, sp.budget_scenario, sp.storage_scenario, sp.horizon)) \
                if isinstance(m, str) else m
            w, d = replay(sp, pol, table if cmp.uses_forecast(pol) else None)
            weeks.update(w)
        out_methods[name] = {"label": LABELS[name], "weeks": weeks}
        if name.startswith("ppo_"):
            out_methods[name]["seed"] = seeds[name[4:]]
        print(f"{name:16s} {len(weeks)} weeks, spend ${sum(x['spend'] for x in weeks.values()):.2f}")
    final = pd.read_csv(FINAL / "mean_over_episodes.csv", index_col=0)
    payload = {
        "household": specs[0].household_id, "region": specs[0].region_id, "budget_scenario": specs[0].budget_scenario,
        "weekly_budget": r(d.weekly_budget, 2), "protein_target": r(d.protein_target, 1), "products": list(d.product_ids),
        "episodes": [{"start": s.start_week, "end": s.end_week} for s in specs], "forecast_variant": cmp.VARIANT,
        "methods": out_methods,
        "final_test": {k: {c: r(v, 4) for c, v in row.items()} for k, row in final.iterrows()},
        "source": "results/final/test (frozen held-out evaluation) + deterministic replay by export_ui_plans.py",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
