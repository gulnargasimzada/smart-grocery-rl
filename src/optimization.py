"""Traditional optimization policy for GroceryEnv: a weekly mixed-integer program (rolling horizon).

Every week the policy solves one small MILP (scipy.optimize.milp / HiGHS) that decides how many PACKAGES of each product to buy.
It acts in the same space as every baseline: `act(obs_dict) -> order quantities in purchase units, shape (36,)`, and is evaluated
by the unchanged `src.evaluate` runner, so it sees exactly the same episodes, prices, budget, storage, diet rules and metrics as
the baselines and an RL agent (`RLPolicy`). The environment still applies its own pipeline; the MILP already satisfies those
constraints, so normally nothing is cut.

Information used at decision time (never the recorded demand of the current or future weeks):
- `obs["forecast"]`  Ridge forecast of this week's demand (verified out-of-sample, `src.forecast_table`); required, like ForecastRule
- `obs["inventory"]` chilled and frozen units on hand;  `obs["ref_demand"]` trailing 4-week mean demand (known history)
- static household / region data from `GroceryData`: shelf prices of the CURRENT week and the 4 weeks before it, weekly budget,
  package steps, diet/allergen mask, shelf lives, storage volumes and capacities, protein target

Model for week t (all money in USD; it mirrors the environment's reward multiplied by the weekly budget):

    min  sum_p c_p x_p + sum_j w_j [ sum_p ( k c_p s_pj + c_p z_pj - v_p y_pj ) + lam u_j ]

    x_p = step_p * n_p,  n_p integer, 0 <= n_p <= nmax_p        whole packages; nmax_p = environment stock cap; 0 if excluded by diet
    s_pj >= D_pj - I_p - x_p                                     shortage in demand scenario j (D_pj = servable forecast demand)
    z_pj >= I_p + x_p - D_pj + s_pj - keep_p                     waste: leftover that expires before it can be eaten
    y_pj <= I_p + x_p - D_pj + s_pj,  y_pj <= ycap_p              useful carry-over to next week, valued at v_p
    u_j  >= target - sum_p protein_p (D_pj - s_pj)                protein shortfall (grams)
    sum_p c_p x_p <= budget                                       weekly budget
    sum_{p in storage} vol_p x_p <= free volume                   pantry, fridge (non-freezable items), freezer (frozen + freezable)

- Demand scenarios: D_pj = f_p * m_j * min(1, L_p / 7) with m = (1 - delta, 1, 1 + delta), weights (0.25, 0.5, 0.25); f = Ridge
  forecast, L = usable shelf life (freezer life for freezable / frozen products), so a 5-day product can only cover 5/7 of the week.
- k = STOCKOUT_MULT (2, as in the reward); lam = PROTEIN_WEIGHT * budget / protein target (the reward's protein term in USD).
- Carry-over value v_p = (1 - HOLDING) * mean price of the last 4 weeks: buying ahead pays only when today's price is below the recent
  level (a promotion). ycap = keep_p = the leftover that is still eaten before it expires (forecast rate x remaining shelf life), so a
  package that lasts several weeks is not treated as a loss; the environment stock cap (nmax) limits stockpiling. In the last week v_p follows the
  environment's terminal credit (min(1, remaining days / 14), at most 2 weeks of reference demand).

Tuned parameter: `delta` (demand uncertainty), chosen on the TRAINING windows only (`python -m src.optimization_eval --tune`).
The test weeks 78-103 are not used for tuning or evaluation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from src.env import DAYS, PROTEIN_WEIGHT, STOCK_CAP_WEEKS, STOCKOUT_MULT, GroceryData
from src.forecast_table import ForecastUnavailable
from src.policies import Policy, on_hand

EPS = 1e-9
SCENARIO_WEIGHTS = (0.25, 0.5, 0.25)
DEFAULT_DELTA = 0.4          # tuned on the training windows (results/optimization/tuned.json, docs/OPTIMIZATION.md)
HOLDING = 0.02               # carry-over is worth 2 % less than the recent price (ties -> buy later)
TERMINAL_CREDIT_DAYS = 14.0  # same rule as the environment's terminal valuation
TERMINAL_BUFFER_WEEKS = 2.0
HORIZON = 13


class OptimizationPolicy(Policy):
    name = "optimization"
    uses_forecast = True

    def __init__(self, household_id: str = "hh_student", region_id: str = "durham_nc", budget_scenario: str = "normal",
                 storage_scenario: str = "default", horizon: int = HORIZON, delta: float = DEFAULT_DELTA,
                 name: str = "optimization"):
        d = GroceryData(household_id, region_id, budget_scenario, storage_scenario)
        self.name = name
        self.delta = float(delta)
        self.horizon = horizon
        self.ids = d.product_ids
        self.n = d.n
        # static data only; the recorded demand (d.demand) is deliberately NOT copied
        self._prices = d.price.copy()                       # weekly shelf prices; only weeks <= the current week are read
        self.step = d.step_size.copy()
        self.allowed = d.allowed.copy()
        self.budget = d.weekly_budget
        self.protein = d.protein.copy()
        self.protein_target = d.protein_target
        self.volume = d.volume.copy()
        self.capacity = dict(d.capacity)
        self.default_storage = d.default_storage.copy()
        self.to_freezer = d.sold_frozen | d.freezable       # purchases that may end up in the freezer
        self.life = np.where(self.to_freezer, d.L_freezer, d.L_default).astype(float)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.last_solution: dict | None = None

    # ------------------------------------------------------------------ decision-time information
    def price_now(self, week: int) -> np.ndarray:
        return self._prices[week]

    def price_recent(self, week: int) -> np.ndarray:
        return self._prices[max(week - 4, 0):week].mean(axis=0) if week > 0 else self._prices[0]

    def act(self, obs: dict) -> np.ndarray:
        fc = obs.get("forecast")
        if fc is None:
            raise ForecastUnavailable("optimization needs the Ridge forecast, but the environment has no verified forecast")
        week = int(obs["week"])
        f = np.maximum(np.array([fc[p] for p in self.ids], float), 0.0)
        inv = on_hand(obs, self.ids)
        chilled = np.array([obs["inventory"][p]["chilled"] for p in self.ids])
        frozen = np.array([obs["inventory"][p]["frozen"] for p in self.ids])
        ref = np.asarray(obs["ref_demand"], float)
        last = self.t >= self.horizon - 1
        n_pk = self.solve(week, f, inv, chilled, frozen, ref, last)
        self.t += 1
        return n_pk * self.step

    # ------------------------------------------------------------------ MILP
    def solve(self, week: int, f, inv, chilled, frozen, ref, last: bool) -> np.ndarray:
        n, J = self.n, len(SCENARIO_WEIGHTS)
        c = self.price_now(week)
        step = self.step
        mult = (1 - self.delta, 1.0, 1 + self.delta)
        servable = np.minimum(1.0, self.life / DAYS)
        D = np.stack([f * m * servable for m in mult], axis=1)                       # (n, J)
        keep = f / DAYS * np.maximum(self.life - DAYS, 0.0)                          # leftover still eaten before it expires
        if last:
            credit = np.clip(np.maximum(self.life - DAYS, 0.0) / TERMINAL_CREDIT_DAYS, 0.0, 1.0)
            value = c * credit
            ycap = np.minimum(keep, TERMINAL_BUFFER_WEEKS * ref)
        else:
            value = (1 - HOLDING) * self.price_recent(week)
            ycap = keep

        cap_units = np.maximum(STOCK_CAP_WEEKS * ref, step + ref)                    # environment stock cap
        nmax = np.maximum(np.floor((cap_units - inv) / step + EPS), 0.0)
        nmax[~self.allowed] = 0.0

        # variable layout: n (n) | s (n*J) | z (n*J) | y (n*J) | u (J)
        i_s = lambda p, j: n + p * J + j
        i_z = lambda p, j: n + n * J + p * J + j
        i_y = lambda p, j: n + 2 * n * J + p * J + j
        i_u = lambda j: n + 3 * n * J + j
        nv = n + 3 * n * J + J
        lam = PROTEIN_WEIGHT * self.budget / max(self.protein_target, EPS)
        obj = np.zeros(nv)
        obj[:n] = c * step
        for j, w in enumerate(SCENARIO_WEIGHTS):
            for p in range(n):
                obj[i_s(p, j)] = w * STOCKOUT_MULT * c[p]
                obj[i_z(p, j)] = w * c[p]
                obj[i_y(p, j)] = -w * value[p]
            obj[i_u(j)] = w * lam

        rows = 3 * n * J + J + 1 + 3
        A = lil_matrix((rows, nv))
        lo, hi = np.full(rows, -np.inf), np.zeros(rows)
        r = 0
        for p in range(n):
            for j in range(J):
                # -x - s <= I - D
                A[r, p] = -step[p]; A[r, i_s(p, j)] = -1.0; hi[r] = inv[p] - D[p, j]; r += 1
                # x + s - z <= keep - I + D
                A[r, p] = step[p]; A[r, i_s(p, j)] = 1.0; A[r, i_z(p, j)] = -1.0; hi[r] = keep[p] - inv[p] + D[p, j]; r += 1
                # y - x - s <= I - D
                A[r, i_y(p, j)] = 1.0; A[r, p] = -step[p]; A[r, i_s(p, j)] = -1.0; hi[r] = inv[p] - D[p, j]; r += 1
        for j in range(J):
            # sum protein * s - u <= sum protein * D - target
            for p in range(n):
                A[r, i_s(p, j)] = self.protein[p]
            A[r, i_u(j)] = -1.0
            hi[r] = float(self.protein @ D[:, j]) - self.protein_target
            r += 1
        A[r, :n] = c * step; hi[r] = self.budget; r += 1                            # budget
        vol = self.volume * step
        for s in ("pantry", "fridge"):                                                # chilled storages: non-freezable purchases
            mask = (self.default_storage == s) & ~self.to_freezer
            used = float((chilled * self.volume)[self.default_storage == s].sum())
            A[r, :n] = np.where(mask, vol, 0.0); hi[r] = max(self.capacity[s] - used, 0.0); r += 1
        used_f = float((frozen * self.volume).sum())                                   # freezer: frozen + freezable purchases
        A[r, :n] = np.where(self.to_freezer, vol, 0.0); hi[r] = max(self.capacity["freezer"] - used_f, 0.0); r += 1

        ub = np.full(nv, np.inf)
        ub[:n] = nmax
        for p in range(n):
            for j in range(J):
                ub[i_y(p, j)] = ycap[p]
        integrality = np.zeros(nv)
        integrality[:n] = 1
        res = milp(obj, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integrality,
                   bounds=Bounds(np.zeros(nv), ub), options={"disp": False})
        if res.status != 0 or res.x is None:
            raise RuntimeError(f"optimization MILP failed in week {week}: {res.message}")
        n_pk = np.round(res.x[:n])
        self.last_solution = {"week": week, "objective": float(res.fun), "packages": n_pk,
                              "expected_shortage": res.x[n:n + n * J].reshape(n, J) @ np.array(SCENARIO_WEIGHTS),
                              "cost": float((n_pk * step * c).sum())}
        return n_pk
