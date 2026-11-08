"""Export what each strategy actually did, week by week, for the app (results/ui/plans.json).

    python export_ui_plans.py

It replays the SAME frozen methods as `src/final_eval.py` (Fixed, Average, Forecast Rule, Optimization, PPO with and without the
Ridge forecast) on the held-out test episodes, and records for every week: the basket after the constraint pipeline, the
pipeline notes, prices, what was at home (each lot with its storage and days left), the forecast and what happened.
The replay is deterministic, so its totals match results/final/test exactly; nothing is trained, tuned or re-selected.
It refuses to run before the one-time test evaluation exists (results/final/test/), so it can't stand in for that run.

Seventh entry, `balanced_rule`: the backend's `balanced` priority (src/coach_api.py) replayed on the same test episodes. Each week it builds the
Optimization plan and the PPO-with-forecast plan for the current state and takes the Optimization plan unless its projected fill rate is more than
0.03 below PPO's. It is NOT part of the frozen six-method comparison (that comparison is untouched); it was added afterwards so the app's three
priorities match the backend's, and its rule and tolerance were fixed beforehand (docs/GROCERY_COACH_INTEGRATION.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import comparison as cmp
from src.env import GroceryEnv
from src.coach_api import STRATEGIES, CoachAdapter
from src.evaluate import episode_specs, make_policy
from src.policies import Policy
from src.rl_wrapper import OrderUpToWrapper

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "results" / "final" / "test"
OUT = ROOT / "results" / "ui" / "plans.json"
LABELS = {"fixed": "Fixed list", "average_rule": "Average rule", "forecast_rule": "Forecast rule (Ridge)",
          "optimization": "Optimization (MILP)", "ppo_forecast": "RL agent (PPO + forecast)", "ppo_no_forecast": "RL agent (PPO, no forecast)",
          "balanced_rule": "Balanced rule (Optimization or RL agent, chosen each week)"}
METRIC_COLS = ["reward", "cash_spend", "net_cost", "waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days",
               "fill_rate_value", "protein_coverage", "terminal_basis", "terminal_credit", "terminal_charge"]


class BalancedRulePolicy(Policy):
    """The backend's `balanced` priority as a policy: both plans for the current state, chosen by the documented projected-fill-rate rule."""
    name = "balanced_rule"
    uses_forecast = True

    def __init__(self, adapter: CoachAdapter, ppo_seed: int):
        self.adapter, self.ppo_seed, self.env, self.last = adapter, ppo_seed, None, None

    def bind(self, base_env) -> None:
        self.env = base_env

    def act(self, obs_dict):
        env, d = self.env, self.env.d
        p = {"household_id": d.household_id, "region_id": d.region_id, "budget_scenario": "normal", "weeks_remaining": 13 - env.t, "ppo_seed": self.ppo_seed}
        forecast = np.array([obs_dict["forecast"][pid] for pid in d.product_ids])
        plans = {s: self.adapter._plan(env, p, s, forecast) for s in STRATEGIES}
        strategy, why = self.adapter.select("balanced", plans, True)
        self.last = strategy
        return plans[strategy]["order"]


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
    if hasattr(policy, "bind"):
        policy.bind(base)
    obs_dict, done, weeks = info["obs_dict"], False, {}
    tot = dict.fromkeys(["reward", "cash_spend", "waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days", "demand_value"], 0.0)
    protein = []
    while not done:
        week = base.week
        lots = [{"p": d.product_ids[l.p], "u": r(l.units), "days": r(base.remaining_days(l), 1), "where": base._location(l)}
                for l in base.lots if l.units > 1e-6]
        fc = obs_dict.get("forecast")
        action = policy.act_vec(obs) if policy.normalized_action else policy.act(obs_dict)
        obs, reward, done, _, info = env.step(action)
        tot["reward"] += reward
        tot["cash_spend"] += info["spend"]
        for k in ("waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days"):
            tot[k] += info[k]
        tot["demand_value"] += float((d.demand[week] * base.price_at(week)).sum())
        protein.append(info["protein_target_met"])
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
        if getattr(policy, "last", None):
            weeks[str(week)]["strategy"] = policy.last          # which plan the balanced rule took this week
        obs_dict = info["obs_dict"]
    summary = {k: tot[k] for k in ("reward", "cash_spend", "waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days")}
    summary.update(net_cost=info["episode_net_cost"], fill_rate_value=1.0 - tot["stockout_value"] / tot["demand_value"],
                   protein_coverage=float(np.mean(protein)), terminal_basis=info["terminal_inventory_basis"],
                   terminal_credit=info["terminal_inventory_credit"], terminal_charge=info["terminal_charge"])
    return weeks, d, summary


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
    methods["balanced_rule"] = BalancedRulePolicy(CoachAdapter(table=table), seeds["forecast"])
    out_methods, d, summaries = {}, None, {}
    for name, m in methods.items():
        weeks, eps = {}, []
        for sp in specs:
            pol = make_policy(m, GroceryEnv(sp.household_id, sp.region_id, sp.budget_scenario, sp.storage_scenario, sp.horizon)) \
                if isinstance(m, str) else m
            w, d, summ = replay(sp, pol, table if cmp.uses_forecast(pol) else None)
            weeks.update(w)
            eps.append(summ)
        summaries[name] = {c: float(np.mean([e[c] for e in eps])) for c in METRIC_COLS}
        out_methods[name] = {"label": LABELS[name], "weeks": weeks}
        if name.startswith("ppo_"):
            out_methods[name]["seed"] = seeds[name[4:]]
        print(f"{name:16s} {len(weeks)} weeks, spend ${sum(x['spend'] for x in weeks.values()):.2f}")
    final = pd.read_csv(FINAL / "mean_over_episodes.csv", index_col=0)
    for name in ("fixed", "average_rule", "forecast_rule", "optimization"):       # the replay must reproduce the frozen results
        for c in METRIC_COLS:
            assert abs(summaries[name][c] - final.loc[name, c]) < 1e-6, (name, c)
    for name in ("ppo_forecast", "ppo_no_forecast"):
        seed = seeds[name[4:]]
        for c in METRIC_COLS:
            assert abs(summaries[name][c] - final.loc[f"{name}_seed{seed}", c]) < 1e-6, (name, c)
    final.loc["balanced_rule"] = pd.Series(summaries["balanced_rule"])           # supplementary row, not part of the frozen comparison
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
