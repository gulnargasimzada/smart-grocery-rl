"""Backend adapter for the Grocery Coach UI: weekly shopping recommendations from the frozen Optimization and PPO-with-forecast models.

One call = one weekly decision: `CoachAdapter().recommend(request) -> dict` (JSON-serializable, schema version 1.0). The adapter runs the EXISTING
models unchanged (nothing is retrained, tuned or modified) on the unchanged GroceryEnv, so every plan goes through the same order pipeline
(diet/allergen exclusions, whole packages, stock cap, storage capacity incl. freezer, weekly budget).

Strategies
----------
optimization   weekly MILP (src/optimization.py, delta 0.4 from results/optimization/tuned.json)
ppo_forecast   PPO with Ridge forecasts, one of the six frozen checkpoints (default seed 0; results/rl_pilot/selected/)

Priorities and the transparent selection rules
----------------------------------------------
reduce_cost_waste  -> always `optimization`.  Evidence (frozen held-out test, docs/FINAL_RESULTS.md): lowest waste (-75% vs Average Rule), lowest cash spend
                      among methods that serve demand, best reward.
availability       -> always `ppo_forecast`. Evidence: highest fill rate (0.89) and lowest stockout value on the held-out test; Optimization leaves ~79% more demand unserved.
balanced           -> both plans are built for this week's state and compared with a forecast-based availability estimate:
                          projected_fill = 1 - sum(price * max(0, forecast - (on_hand + order))) / sum(price * forecast)
                      Rule: use `optimization` if projected_fill(optimization) >= projected_fill(ppo_forecast) - BALANCED_FILL_TOLERANCE (0.03), otherwise `ppo_forecast`.
                      In words: take the cheaper, lower-waste plan unless it is projected to leave noticeably more of this week's forecast demand unmet than the PPO plan.
                      The 0.03 tolerance is a fixed a-priori choice; it was not tuned on any data.
The projected numbers use only this week's forecast, the inventory and the order; they ignore shelf life, daily timing and forecast error, so they are indicators, not predictions.

Scope and unsupported scenarios (flagged in the response, never silently ignored)
--------------------------------------------------------------------------------
- Data weeks 4..103 of the simulated dataset (the forecasts and prices exist for those weeks). `forecast_variant` must be `no_events`; storage scenario must be `default`.
- PPO checkpoints were trained for hh_student / durham_nc / normal budget only. For any other household, region or budget scenario the adapter falls back to
  `optimization` with a `ppo_unsupported_scenario` warning (status stays `ok`, `strategy.fallback` is true).
- The inventory must fit the household's storage capacity. Unknown products, negative quantities, frozen units of a product that cannot be frozen, and similar problems give `status: error`.
- The adapter never uses future demand or future prices: it reads the verified out-of-sample forecast of the decision week, the shelf prices up to that week and the trailing 4-week demand.

CLI:  python -m src.coach_api --example | --request request.json | --stdin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.env import LAST_WEEK, STOCK_CAP_WEEKS, CAP_TOL, GroceryEnv, Lot
from src.forecast_table import ForecastTable, ForecastUnavailable
from src.optimization import OptimizationPolicy
from src.rl_wrapper import action_to_order
from src.train_rl import policy_from_model

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "1.0"
PRIORITIES = ("reduce_cost_waste", "availability", "balanced")
STRATEGIES = ("optimization", "ppo_forecast")
PPO_SCENARIO = {"household_id": "hh_student", "region_id": "durham_nc", "budget_scenario": "normal"}
BUDGET_SCENARIOS = ("normal", "challenging", "tight")
BALANCED_FILL_TOLERANCE = 0.03
MIN_WEEK, MAX_WEEK = 4, LAST_WEEK
HORIZON = 13
SELECTED = ROOT / "results" / "rl_pilot" / "selected"
TUNED = ROOT / "results" / "optimization" / "tuned.json"
MANIFEST = ROOT / "results" / "final" / "freeze_manifest.json"
EXAMPLE_REQUEST = {"schema_version": SCHEMA_VERSION, "household_id": "hh_student", "region_id": "durham_nc", "week": 70,
                   "priority": "balanced", "weeks_remaining": 13,
                   "inventory": {"milk": 1.0, "rice": {"chilled": 0.5}, "chicken_breast": {"chilled": 0.4, "frozen": 1.0}}}

RULES = {
    "reduce_cost_waste": "Always the Optimization plan: it had the lowest simulated waste and cash spend among methods that serve demand, and the best reward, in the frozen validation and held-out test runs.",
    "availability": "Always the PPO-with-forecast plan: it had the highest fill rate and lowest stockout value in the frozen held-out test; Optimization leaves more demand unmet.",
    "balanced": f"Use the Optimization plan unless its forecast-based projected fill rate is more than {BALANCED_FILL_TOLERANCE:.2f} below the PPO plan's; then use PPO.",
}


class RequestError(ValueError):
    """Invalid request content (status 'error')."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class Unsupported(Exception):
    """Valid request that the models cannot serve (status 'unsupported')."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _num(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
        raise RequestError("invalid_number", f"{what} must be a finite number")
    if x < 0:
        raise RequestError("negative_quantity", f"{what} must not be negative")
    return float(x)


class CoachAdapter:
    def __init__(self, root: Path = ROOT, table: ForecastTable | None = None):
        self.root = Path(root)
        self.products = pd.read_csv(self.root / "data" / "products.csv")
        self.ids = list(self.products["product_id"])
        self._table = table
        self._envs: dict = {}
        self._opt: dict = {}
        self._ppo: dict = {}
        self._delta = float(json.loads((self.root / "results" / "optimization" / "tuned.json").read_text())["delta"])

    # ------------------------------------------------------------------ cached resources
    def table(self) -> ForecastTable:
        if self._table is None:
            self._table = ForecastTable.from_csv(self.root / "data" / "forecast_weekly.csv", self.ids)
        return self._table

    def env(self, hh: str, region: str, budget: str) -> GroceryEnv:
        key = (hh, region, budget)
        if key not in self._envs:
            fn = self.table().forecast_fn(hh, "no_events")
            self._envs[key] = GroceryEnv(hh, region, budget, "default", HORIZON, forecast_fn=fn)
        return self._envs[key]

    def optimization(self, hh: str, region: str, budget: str) -> OptimizationPolicy:
        key = (hh, region, budget)
        if key not in self._opt:
            self._opt[key] = OptimizationPolicy(hh, region, budget, delta=self._delta)
        return self._opt[key]

    def ppo(self, seed: int):
        if seed not in self._ppo:
            path = self.root / "results" / "rl_pilot" / "selected" / f"forecast_seed{seed}.zip"
            self._ppo[seed] = policy_from_model(PPO.load(path, device="cpu"), f"ppo_forecast_seed{seed}", True)
        return self._ppo[seed]

    # ------------------------------------------------------------------ request parsing
    def parse(self, req: dict) -> dict:
        if not isinstance(req, dict):
            raise RequestError("invalid_request", "the request must be a JSON object")
        if req.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise Unsupported("schema_version", f"only schema_version {SCHEMA_VERSION} is supported")
        p = {
            "household_id": req.get("household_id", PPO_SCENARIO["household_id"]),
            "region_id": req.get("region_id", PPO_SCENARIO["region_id"]),
            "budget_scenario": req.get("budget_scenario", PPO_SCENARIO["budget_scenario"]),
            "priority": req.get("priority", "balanced"),
            "week": req.get("week"),
            "weeks_remaining": req.get("weeks_remaining", HORIZON),
            "ppo_seed": req.get("ppo_seed", 0),
            "last_week_protein_ratio": req.get("last_week_protein_ratio", 1.0),
        }
        r = p["last_week_protein_ratio"]
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not np.isfinite(r) or r < 0:
            raise RequestError("invalid_number", "last_week_protein_ratio must be a non-negative number (last week's protein / weekly target; PPO observation input)")
        if p["priority"] not in PRIORITIES:
            raise RequestError("invalid_priority", f"priority must be one of {list(PRIORITIES)}")
        if isinstance(p["week"], bool) or not isinstance(p["week"], int):
            raise RequestError("invalid_week", "week must be an integer (simulated data week)")
        if not MIN_WEEK <= p["week"] <= MAX_WEEK:
            raise Unsupported("week_out_of_range", f"week must be in {MIN_WEEK}..{MAX_WEEK}: forecasts need 4 weeks of history and data ends at week {MAX_WEEK}")
        wr = p["weeks_remaining"]
        if isinstance(wr, bool) or not isinstance(wr, int) or not 1 <= wr <= HORIZON:
            raise RequestError("invalid_weeks_remaining", f"weeks_remaining must be an integer in 1..{HORIZON}")
        if p["week"] - (HORIZON - wr) < MIN_WEEK:
            raise Unsupported("week_out_of_range", "week - (13 - weeks_remaining) must be at least 4")
        if p["ppo_seed"] not in (0, 1, 2):
            raise RequestError("invalid_ppo_seed", "ppo_seed must be 0, 1 or 2 (the three frozen checkpoints)")
        if req.get("forecast_variant", "no_events") != "no_events":
            raise Unsupported("forecast_variant", "only the no_events Ridge forecasts are supported (guest/away information is not part of this interface)")
        if req.get("storage_scenario", "default") != "default":
            raise Unsupported("storage_scenario", "only the default storage scenario is supported")
        if p["budget_scenario"] not in BUDGET_SCENARIOS:
            raise Unsupported("budget_scenario", f"budget_scenario must be one of {list(BUDGET_SCENARIOS)}")
        if "inventory" in req and "lots" in req:
            raise RequestError("invalid_inventory", "give either `inventory` or `lots`, not both")
        p["inventory"], p["lots"] = req.get("inventory"), req.get("lots")
        return p

    def build_state(self, p: dict) -> GroceryEnv:
        """Env positioned at the decision week with the requested inventory (or the standard initial inventory if none is given)."""
        hh, region, budget, week = p["household_id"], p["region_id"], p["budget_scenario"], p["week"]
        try:
            env = self.env(hh, region, budget)
        except (KeyError, ValueError, IndexError, FileNotFoundError) as e:
            raise Unsupported("scenario_not_available", f"no data for household {hh!r}, region {region!r}, budget {budget!r}: {e}") from e
        t0 = HORIZON - p["weeks_remaining"]
        env.reset(options={"start_week": MIN_WEEK})
        env.horizon = HORIZON
        if p["inventory"] is not None or p["lots"] is not None:
            env.lots = self._lots(env, p)
        else:                                                       # standard starting inventory, built for the decision week
            env.start_week, env.t, env.lots = week, 0, []
            env._init_inventory()
        env.start_week, env.t = week - t0, t0                       # the decision week; `weeks left` feature follows weeks_remaining
        env.last_protein_ratio = float(p["last_week_protein_ratio"])
        for s, v in env.volumes().items():
            if v > env.d.capacity[s] + CAP_TOL:
                raise Unsupported("inventory_exceeds_capacity", f"the inventory needs {v:.1f} L of {s} space but the household has {env.d.capacity[s]:.0f} L")
        return env

    def _lots(self, env: GroceryEnv, p: dict) -> list[Lot]:
        d, price0 = env.d, env.price_at(p["week"] - 1)
        lots: list[Lot] = []
        if p["lots"] is not None:
            if not isinstance(p["lots"], list):
                raise RequestError("invalid_inventory", "`lots` must be a list")
            for i, x in enumerate(p["lots"]):
                pid = x.get("product_id") if isinstance(x, dict) else None
                if pid not in self.ids:
                    raise RequestError("unknown_product", f"lots[{i}]: unknown product_id {pid!r}")
                k = self.ids.index(pid)
                storage = x.get("storage", "default")
                if storage not in ("default", "freezer"):
                    raise RequestError("invalid_inventory", f"lots[{i}]: storage must be 'default' or 'freezer'")
                frozen = storage == "freezer" or bool(d.sold_frozen[k])
                if frozen and not (d.freezable[k] or d.sold_frozen[k]):
                    raise RequestError("not_freezable", f"lots[{i}]: {pid} cannot be stored in the freezer")
                units = _num(x.get("units"), f"lots[{i}].units")
                if units > 0:
                    lots.append(Lot(k, units, float(x.get("unit_cost", price0[k])), frozen=frozen, thawed=bool(x.get("thawed", False)),
                                    chilled_days=int(_num(x.get("chilled_days", 0), "chilled_days")), frozen_days=int(_num(x.get("frozen_days", 0), "frozen_days")),
                                    post_thaw_days=int(_num(x.get("post_thaw_days", 0), "post_thaw_days"))))
            return lots
        if not isinstance(p["inventory"], dict):
            raise RequestError("invalid_inventory", "`inventory` must be an object {product_id: units | {chilled, frozen, chilled_days}}")
        for pid, v in p["inventory"].items():
            if pid not in self.ids:
                raise RequestError("unknown_product", f"inventory: unknown product_id {pid!r}")
            k = self.ids.index(pid)
            if isinstance(v, dict):
                chilled, frozen, days = _num(v.get("chilled", 0), f"{pid}.chilled"), _num(v.get("frozen", 0), f"{pid}.frozen"), int(_num(v.get("chilled_days", 0), "chilled_days"))
            else:
                chilled, frozen, days = _num(v, pid), 0.0, 0
            if d.sold_frozen[k]:                                    # frozen vegetables / pizza are always in the freezer
                chilled, frozen = 0.0, chilled + frozen
            if frozen > 0 and not (d.freezable[k] or d.sold_frozen[k]):
                raise RequestError("not_freezable", f"{pid} cannot be stored in the freezer")
            if chilled > 0:
                lots.append(Lot(k, chilled, float(price0[k]), chilled_days=days))
            if frozen > 0:
                lots.append(Lot(k, frozen, float(price0[k]), frozen=True))
        return lots

    # ------------------------------------------------------------------ planning
    def _plan(self, env: GroceryEnv, p: dict, strategy: str, forecast: np.ndarray) -> dict:
        d, week = env.d, env.week
        ref, step = env.ref(week), d.step_size
        have = env.on_hand_units()
        if strategy == "optimization":
            pol = self.optimization(p["household_id"], p["region_id"], p["budget_scenario"])
            pol.reset()
            pol.t = HORIZON - p["weeks_remaining"]                  # the policy treats the last week of the horizon specially
            proposed = pol.act(env._obs_dict())
        else:
            pol = self.ppo(p["ppo_seed"])
            proposed = action_to_order(pol.act_vec(env._obs()), ref, have, step)
        n_steps, notes = env._adjust_order(proposed, week)
        order = n_steps * step
        price = env.price_at(week)
        chilled, frozen, overflow = env._place(order, ref)
        supply = have + order
        short = np.maximum(forecast - supply, 0.0)
        fc_value = float((price * forecast).sum())
        return {
            "strategy": strategy, "order": order, "proposed": proposed, "packages": n_steps, "notes": notes, "chilled": chilled, "frozen": frozen,
            "overflow": overflow, "cost": float((order * price).sum()),
            "projected_fill_rate": 1.0 - float((price * short).sum()) / fc_value if fc_value > 0 else 1.0,
            "projected_shortfall_units": float(short.sum()), "projected_leftover_units": float(np.maximum(supply - forecast, 0.0).sum()),
            "adjusted_by_pipeline": bool(not np.allclose(order, proposed)),
        }

    def select(self, priority: str, plans: dict, ppo_ok: bool) -> tuple[str, dict]:
        """Returns (strategy, explanation). `plans` maps strategy -> plan (ppo_forecast only if the scenario is supported)."""
        info = {"priority": priority, "rule": RULES[priority], "fallback": False, "fallback_reason": None}
        if priority == "reduce_cost_waste":
            return "optimization", info
        if not ppo_ok:
            info.update(fallback=True, fallback_reason="ppo_unsupported_scenario")
            return "optimization", info
        if priority == "availability":
            return "ppo_forecast", info
        o, q = plans["optimization"]["projected_fill_rate"], plans["ppo_forecast"]["projected_fill_rate"]
        chosen = "optimization" if o >= q - BALANCED_FILL_TOLERANCE else "ppo_forecast"
        info["comparison"] = {"optimization_projected_fill": round(o, 4), "ppo_projected_fill": round(q, 4), "tolerance": BALANCED_FILL_TOLERANCE,
                              "decision": f"optimization fill {o:.3f} {'>=' if chosen == 'optimization' else '<'} ppo fill {q:.3f} - {BALANCED_FILL_TOLERANCE:.2f}"}
        return chosen, info

    def recommend(self, req: dict) -> dict:
        try:
            return self._recommend(req)
        except RequestError as e:
            return self._failure("error", e.code, e.message, req)
        except Unsupported as e:
            return self._failure("unsupported", e.code, e.message, req)
        except ForecastUnavailable as e:
            return self._failure("unsupported", "forecast_unavailable", str(e), req)

    @staticmethod
    def _failure(status: str, code: str, message: str, req) -> dict:
        return {"schema_version": SCHEMA_VERSION, "status": status, "errors": [{"code": code, "message": message}], "warnings": [],
                "request": req if isinstance(req, dict) else None}

    def _recommend(self, req: dict) -> dict:
        p = self.parse(req)
        env = self.build_state(p)
        d, week = env.d, env.week
        forecast = self.table().get(p["household_id"], week, "no_events")
        method = self.table().method(p["household_id"], week, "no_events")
        ppo_ok = all(p[k] == v for k, v in PPO_SCENARIO.items())
        warnings = []
        if not ppo_ok and p["priority"] != "reduce_cost_waste":
            warnings.append({"code": "ppo_unsupported_scenario",
                             "message": "The PPO checkpoints were trained for hh_student / durham_nc / normal budget only; this scenario uses the Optimization plan instead."})
        if method == "ma4_fallback":
            warnings.append({"code": "forecast_moving_average", "message": "Weeks 4-11 use a 4-week moving-average forecast, not a Ridge prediction."})
        if p["inventory"] is None and p["lots"] is None:
            warnings.append({"code": "default_inventory", "message": "No inventory was given: the standard starting inventory (3 days of average demand) is assumed."})
        if p["weeks_remaining"] != HORIZON and ppo_ok:
            warnings.append({"code": "partial_horizon", "message": "PPO was trained on 13-week plans; the 'weeks left' input is set from weeks_remaining."})
        need = ["optimization"] + (["ppo_forecast"] if ppo_ok and p["priority"] != "reduce_cost_waste" else [])
        plans = {s: self._plan(env, p, s, forecast) for s in need}
        strategy, why = self.select(p["priority"], plans, ppo_ok)
        plan = plans[strategy]
        if plan["overflow"]:                                          # cannot happen after the pipeline; guard anyway
            raise Unsupported("storage_overflow", f"the plan does not fit the storage: {plan['overflow']}")
        return self._response(env, p, plan, plans, why, forecast, method, warnings)

    # ------------------------------------------------------------------ response
    def _model_info(self, strategy: str, p: dict) -> dict:
        frozen_hash = None
        try:
            m = json.loads((self.root / "results" / "final" / "freeze_manifest.json").read_text())
            import hashlib
            frozen_hash = hashlib.sha256(json.dumps({k: v for k, v in m.items() if k != "libraries"}, sort_keys=True).encode()).hexdigest()[:12]
        except (FileNotFoundError, KeyError, ValueError):
            pass
        if strategy == "optimization":
            return {"name": "Traditional Optimization (weekly MILP)", "delta": self._delta, "frozen_manifest": frozen_hash}
        return {"name": "PPO with Ridge forecasts", "checkpoint": f"results/rl_pilot/selected/forecast_seed{p['ppo_seed']}.zip", "seed": p["ppo_seed"],
                "frozen_manifest": frozen_hash}

    def _response(self, env, p, plan, plans, why, forecast, method, warnings) -> dict:
        d, week = env.d, env.week
        pr = self.products.set_index("product_id")
        price, avg4 = env.price_at(week), d.price[max(week - 4, 0):week].mean(axis=0)
        have, ref = env.on_hand_units(), env.ref(week)
        order = plan["order"]
        chilled_inv, frozen_inv = env._units(True), env._units(False)
        min_left = {}
        for lot in env.lots:
            min_left[lot.p] = min(min_left.get(lot.p, 1e9), env.remaining_days(lot))
        vols, caps = env.volumes(), d.capacity
        r2 = lambda x: round(float(x), 4)
        items = []
        for i, pid in enumerate(self.ids):
            if order[i] > 0:
                items.append({"product_id": pid, "name": str(pr.loc[pid, "name"]), "category": str(pr.loc[pid, "category"]), "unit": str(pr.loc[pid, "unit"]),
                              "quantity": r2(order[i]), "packages": int(plan["packages"][i]), "package_size": r2(d.step_size[i]), "unit_price": r2(price[i]),
                              "line_cost": r2(order[i] * price[i]), "default_storage": str(d.default_storage[i]),
                              "chilled_units": r2(plan["chilled"][i]), "freezer_units": r2(plan["frozen"][i])})
        cost = plan["cost"]
        budget = d.weekly_budget
        excluded = [pid for i, pid in enumerate(self.ids) if not d.allowed[i]]
        n_steps = plan["packages"]
        stock_cap = np.maximum(STOCK_CAP_WEEKS * ref, d.step_size + ref)
        constraints = {
            "within_budget": cost <= budget + 1e-6,
            "whole_packages": bool(np.allclose(order / d.step_size, np.round(order / d.step_size))),
            "diet_allergen_ok": bool(np.all(order[~d.allowed] == 0)),
            "stock_cap_ok": bool(np.all((order == 0) | (have + order <= stock_cap + 1e-6))),
            "storage_ok": not plan["overflow"],
            "adjusted_by_environment_pipeline": plan["adjusted_by_pipeline"], "pipeline_notes": plan["notes"],
        }
        cand = {s: {"cost": r2(pl["cost"]), "n_items": int((pl["order"] > 0).sum()), "projected_fill_rate": r2(pl["projected_fill_rate"]),
                    "projected_shortfall_units": r2(pl["projected_shortfall_units"]), "projected_leftover_units": r2(pl["projected_leftover_units"])}
                for s, pl in plans.items()}
        return {
            "schema_version": SCHEMA_VERSION, "status": "ok", "errors": [], "warnings": warnings,
            "request": {"household_id": p["household_id"], "region_id": p["region_id"], "budget_scenario": p["budget_scenario"], "week": week,
                        "priority": p["priority"], "weeks_remaining": p["weeks_remaining"], "forecast_variant": "no_events", "storage_scenario": "default"},
            "strategy": {"priority": p["priority"], "selected": plan["strategy"], "label": "Optimization (MILP)" if plan["strategy"] == "optimization" else "PPO with forecast",
                         "rule": why["rule"], "fallback": why["fallback"], "fallback_reason": why["fallback_reason"], "comparison": why.get("comparison"),
                         "candidates": cand, "model": self._model_info(plan["strategy"], p)},
            "shopping_list": items,
            "totals": {"total_cost": r2(cost), "weekly_budget": r2(budget), "budget_used_pct": r2(100 * cost / budget), "n_items": len(items),
                       "n_packages": int(n_steps.sum()), "currency": "USD", "projected_fill_rate": r2(plan["projected_fill_rate"]),
                       "projected_shortfall_units": r2(plan["projected_shortfall_units"]), "projected_leftover_units": r2(plan["projected_leftover_units"])},
            "inventory": {
                "items": [{"product_id": pid, "name": str(pr.loc[pid, "name"]), "unit": str(pr.loc[pid, "unit"]), "on_hand_units": r2(have[i]),
                           "chilled_units": r2(chilled_inv[i]), "frozen_units": r2(frozen_inv[i]),
                           "days_left_min": None if i not in min_left else int(min_left[i]),
                           "after_purchase_units": r2(have[i] + order[i])} for i, pid in enumerate(self.ids) if have[i] > 0 or order[i] > 0],
                "storage": {s: {"used_l": r2(vols[s]), "capacity_l": r2(caps[s])} for s in ("pantry", "fridge", "freezer")}},
            "forecast": {"variant": "no_events", "week": week, "method": "ridge" if method == "ridge" else "ma4_fallback", "units": "purchase units for the week",
                         "items": [{"product_id": pid, "forecast_units": r2(forecast[i]), "reference_units": r2(ref[i])} for i, pid in enumerate(self.ids)
                                   if forecast[i] > 0 or ref[i] > 0]},
            "prices": {"week": week, "region_id": p["region_id"], "currency": "USD",
                       "items": [{"product_id": pid, "unit_price": r2(price[i]), "avg_4_week_price": r2(avg4[i]),
                                  "vs_avg_pct": r2(100 * (price[i] / avg4[i] - 1)) if avg4[i] > 0 else 0.0} for i, pid in enumerate(self.ids)]},
            "constraints": constraints, "excluded_products": excluded,
            "notes": ["Simulated data: all demand, prices, budgets and storage values are synthetic.",
                      "Projected fill / shortfall / leftover use this week's forecast only and ignore shelf life and timing: they are indicators, not predictions."],
        }


def lots_to_json(env: GroceryEnv) -> list[dict]:
    """Full-fidelity inventory export of an environment (used by tests and by clients that track lots)."""
    return [{"product_id": env.d.product_ids[l.p], "units": l.units, "storage": "freezer" if l.frozen else "default", "unit_cost": l.unit_cost,
             "thawed": l.thawed, "chilled_days": l.chilled_days, "frozen_days": l.frozen_days, "post_thaw_days": l.post_thaw_days} for l in env.lots]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--example", action="store_true", help="run the example request")
    g.add_argument("--request", help="path to a JSON request file")
    g.add_argument("--stdin", action="store_true", help="read the JSON request from stdin")
    args = ap.parse_args()
    req = EXAMPLE_REQUEST if args.example else json.loads(Path(args.request).read_text()) if args.request else json.load(sys.stdin)
    print(json.dumps(CoachAdapter().recommend(req), indent=2))


if __name__ == "__main__":
    main()
