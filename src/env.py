"""GroceryEnv v0: weekly grocery shopping with lots, three storages and a hard-constraint order pipeline.

First incremental version (see docs/RL_TECHNICAL_DESIGN.md). Defaults: hh_student, durham_nc, normal budget,
default storage, 13-week episodes, recorded demand replay (deterministic, no demand randomness).

What is implemented: inventory lots, storage transitions (freeze on delivery, thaw just in time), weekly purchases
with package steps, daily consumption (first-expiring-first), order constraints (diet/allergen, stock cap, storage
capacity, budget), simulated disposal, reward and terminal inventory accounting.
Not implemented yet: RL action wrapper, baseline policies, forecaster integration (use `forecast_fn`), the `limited`
storage scenario, other budget scenarios beyond the column lookup, evaluation specs.

ALL BUDGETS, STORAGE RULES AND CAPACITIES ARE SYNTHETIC ASSUMPTIONS. Waste is *simulated disposal*, not food-safety
expiry (see the design doc).

Only reads data/*.csv and data_rl/*.csv; nothing is written.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.diets import allowed_mask      # shared diet and allergen rules (owned by the data team)

ROOT = Path(__file__).resolve().parent.parent

# ---- Initial assumptions to be tested (docs/RL_TECHNICAL_DESIGN.md) -------------------------------------------
HORIZON = 13                 # weekly steps per episode
DAYS = 7
THAW_LEAD_DAYS = 2.0         # keep this many days of expected demand thawed
INIT_COVER_DAYS = 3.0        # initial inventory: days of trailing 4-week mean demand
STOCKOUT_MULT = 2.0          # unserved demand costs this multiple of its price
PROTEIN_WEIGHT = 0.5
STOCK_CAP_WEEKS = 3.0        # on-hand + order <= max(this many weeks of reference demand, one package + one week)
CREDIT_DAYS = 14.0           # terminal credit = min(1, remaining_days / CREDIT_DAYS)
CREDIT_BUFFER_WEEKS = 2.0    # terminal credit only for units up to this many weeks of reference demand
AT_RISK_DAYS = 3             # observation: chilled units with this many days left or fewer
TRAIN_START_RANGE = (4, 51)  # training episodes start here (inclusive)
LAST_WEEK = 103
LOOSE_STEP = 0.5             # package step for loose items sold by weight
LOOSE_PRODUCTS = {"bananas", "apples", "carrots", "oranges", "onions"}
EPS = 1e-9
CAP_TOL = 1e-6

BUDGET_COLUMNS = {"normal": "normal_budget_usd", "challenging": "challenging_budget_usd", "tight": "tight_budget_usd"}
STORAGES = ("pantry", "fridge", "freezer")


class GroceryData:
    """Static arrays for one household and region (product order = products.csv)."""

    def __init__(self, household_id: str, region_id: str, budget_scenario: str, storage_scenario: str, root: Path = ROOT):
        if storage_scenario != "default":
            raise NotImplementedError("only the 'default' storage scenario exists so far")
        if budget_scenario not in BUDGET_COLUMNS:
            raise ValueError(f"unknown budget scenario {budget_scenario!r}")
        data, data_rl = root / "data", root / "data_rl"
        products = pd.read_csv(data / "products.csv")
        rules = pd.read_csv(data_rl / "storage_rules.csv").set_index("product_id").loc[products["product_id"]]
        hh = pd.read_csv(data / "households.csv").set_index("household_id").loc[household_id]
        cap = pd.read_csv(data_rl / "household_storage.csv").set_index("household_id").loc[household_id]
        bud = pd.read_csv(data / "household_region_budgets.csv")
        bud = bud[(bud["household_id"] == household_id) & (bud["region_id"] == region_id)]
        if len(bud) != 1:
            raise ValueError(f"no budget row for {household_id} / {region_id}")
        bud = bud.iloc[0]

        self.household_id, self.region_id = household_id, region_id
        self.product_ids = list(products["product_id"])
        n = self.n = len(self.product_ids)
        self.price_usd_nominal = products["price_usd"].to_numpy(float)
        self.kcal = products["kcal_per_unit"].to_numpy(float)
        self.protein = products["protein_g_per_unit"].to_numpy(float)
        self.L_default = products["shelf_life_days"].to_numpy(float)
        self.L_freezer = rules["shelf_life_freezer_days"].astype(float).to_numpy()
        self.default_storage = rules["default_storage"].to_numpy()
        self.volume = rules["volume_l_per_unit"].to_numpy(float)
        self.sold_frozen = self.default_storage == "freezer"
        self.freezable = rules["freezing_supported"].to_numpy(bool) & ~self.sold_frozen
        self.safety_kind = (rules["shelf_life_kind"] == "safety_conservative").to_numpy()
        self.step_size = np.array([LOOSE_STEP if pid in LOOSE_PRODUCTS else 1.0 for pid in self.product_ids])
        self.capacity = {"fridge": float(cap["fridge_l"]), "freezer": float(cap["freezer_l"]), "pantry": float(cap["pantry_l"])}

        # diet and allergen exclusions -> products that may never be bought
        self.allowed = allowed_mask(hh.to_dict(), products)

        self.weekly_budget = float(bud[BUDGET_COLUMNS[budget_scenario]])
        self.budget_scale = float(bud["calib_median_cost_usd"])
        self.protein_target = float(hh["weekly_protein_target_g"])

        demand = pd.read_csv(data / "demand_weekly.csv")
        demand = demand[demand["household_id"] == household_id]
        wide = demand.pivot(index="week", columns="product_id", values="demand_units")[self.product_ids].sort_index()
        self.demand = wide.to_numpy(float)
        self.holiday = demand.groupby("week")["holiday_week"].max().sort_index().to_numpy(int)
        prices = pd.read_csv(data / "price_weekly_regional_synthetic.csv")
        prices = prices[prices["region_id"] == region_id]
        pw = prices.pivot(index="week", columns="product_id", values="price_usd")[self.product_ids].sort_index()
        self.price = pw.to_numpy(float)
        assert self.demand.shape == self.price.shape == (LAST_WEEK + 1, n)


@dataclass
class Lot:
    """One purchase of one product. The three day counters are never reset by a move."""
    p: int
    units: float
    unit_cost: float
    frozen: bool = False
    thawed: bool = False
    chilled_days: int = 0
    frozen_days: int = 0
    post_thaw_days: int = 0


class GroceryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, household_id: str = "hh_student", region_id: str = "durham_nc", budget_scenario: str = "normal",
                 storage_scenario: str = "default", horizon: int = HORIZON, forecast_fn=None, root: Path = ROOT,
                 start_range: tuple[int, int] = TRAIN_START_RANGE):
        super().__init__()
        self.d = GroceryData(household_id, region_id, budget_scenario, storage_scenario, root)
        self.horizon = horizon
        self.forecast_fn = forecast_fn
        self.start_range = start_range               # inclusive range from which reset() samples a start week
        self.stock_cap_weeks = STOCK_CAP_WEEKS       # attribute so diagnostics can relax it
        n = self.d.n
        self.action_space = spaces.Box(0.0, 1e6, shape=(n,), dtype=np.float32)   # order quantities in purchase units
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(5 * n + 10,), dtype=np.float32)
        self.lots: list[Lot] = []
        self.start_week = self.t = 0
        self.last_protein_ratio = 1.0

    # ------------------------------------------------------------------ helpers
    @property
    def week(self) -> int:
        return self.start_week + self.t

    def ref(self, week: int) -> np.ndarray:
        """Reference demand: mean of the 4 weeks before `week` (recorded demand), in purchase units per week."""
        return self.d.demand[week - 4:week].mean(axis=0)

    def price_at(self, week: int) -> np.ndarray:
        return self.d.price[min(week, LAST_WEEK)]

    def _location(self, lot: Lot) -> str:
        return "freezer" if lot.frozen else str(self.d.default_storage[lot.p])

    def volumes(self) -> dict[str, float]:
        v = dict.fromkeys(STORAGES, 0.0)
        for lot in self.lots:
            v[self._location(lot)] += lot.units * self.d.volume[lot.p]
        return v

    def _units(self, chilled: bool | None = None) -> np.ndarray:
        """Units per product. chilled=True: not frozen; False: frozen; None: all."""
        u = np.zeros(self.d.n)
        for lot in self.lots:
            if chilled is None or lot.frozen != chilled:
                u[lot.p] += lot.units
        return u

    def on_hand_units(self) -> np.ndarray:
        """Units on hand per product (chilled + frozen)."""
        return self._units(None)

    def _consumable(self, lot: Lot) -> bool:
        return (not lot.frozen) or bool(self.d.sold_frozen[lot.p])

    def remaining_days(self, lot: Lot) -> float:
        """Days until the lot would be disposed in its current storage state."""
        d = self.d
        if lot.frozen:
            return d.L_freezer[lot.p] - lot.frozen_days
        return d.L_default[lot.p] - lot.chilled_days - lot.post_thaw_days

    def _track_volumes(self) -> None:
        for s, v in self.volumes().items():
            self.max_volume_seen[s] = max(self.max_volume_seen[s], v)

    def basis_remaining(self) -> float:
        return float(sum(lot.units * lot.unit_cost for lot in self.lots))

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        lo, hi = self.start_range
        start = options.get("start_week")
        if start is None:
            start = int(self.np_random.integers(lo, hi + 1))
        self.horizon = int(options.get("horizon", self.horizon))
        if start < 4 or start + self.horizon - 1 > LAST_WEEK:
            raise ValueError(f"episode weeks {start}..{start + self.horizon - 1} outside 4..{LAST_WEEK}")
        self.start_week, self.t = int(start), 0
        self.lots = []
        self.max_volume_seen = dict.fromkeys(STORAGES, 0.0)
        self.last_protein_ratio = 1.0
        # episode ledger (units per product, and cost basis in USD)
        n = self.d.n
        self.ledger = {k: np.zeros(n) for k in ("initial", "purchased", "consumed", "wasted")}
        self.cash = dict(spend=0.0, consumed_cost=0.0, waste_cost=0.0, stockout_value=0.0, terminal_charge=0.0, initial_basis=0.0)
        self._init_inventory()
        self._track_volumes()
        return self._obs(), {"obs_dict": self._obs_dict(), "start_week": self.start_week}

    def _init_inventory(self) -> None:
        d, ref = self.d, self.ref(self.start_week)
        price0 = self.price_at(self.start_week - 1)
        for p in range(d.n):
            u = ref[p] / DAYS * INIT_COVER_DAYS
            if u > EPS:
                self.lots.append(Lot(p, float(u), float(price0[p]), frozen=bool(d.sold_frozen[p])))
        # same proportional cut for everyone if a storage would overflow
        for s, v in self.volumes().items():
            if v > self.d.capacity[s] + CAP_TOL:
                f = self.d.capacity[s] / v
                for lot in self.lots:
                    if self._location(lot) == s:
                        lot.units *= f
        self.ledger["initial"] = self._units(None).copy()
        self.cash["initial_basis"] = self.basis_remaining()

    # ------------------------------------------------------------------ order pipeline
    def _place(self, units: np.ndarray, ref: np.ndarray):
        """Split an order into chilled (default storage) and frozen (freezer) parts and check capacity.
        Returns (chilled, frozen, overflowing storages)."""
        d = self.d
        n = d.n
        existing = self._units(chilled=True)
        chilled, frozen = np.zeros(n), np.zeros(n)
        for p in range(n):
            u = units[p]
            if u <= EPS:
                continue
            if d.sold_frozen[p]:
                frozen[p] = u
            elif d.freezable[p]:
                need = max(0.0, ref[p] / DAYS * d.L_default[p] - existing[p])
                chilled[p] = min(u, need)
                frozen[p] = u - chilled[p]
            else:
                chilled[p] = u
        base = self.volumes()

        def added(s: str) -> float:
            if s == "freezer":
                return float((frozen * d.volume).sum())
            return float((chilled * d.volume)[d.default_storage == s].sum())

        # relieve a full pantry/fridge by sending more of its freezable products straight to the freezer
        for s in ("pantry", "fridge"):
            over = base[s] + added(s) - d.capacity[s]
            if over <= CAP_TOL:
                continue
            movers = [p for p in range(n) if d.default_storage[p] == s and d.freezable[p] and chilled[p] > EPS]
            movers.sort(key=lambda p: (-chilled[p] * d.volume[p], p))
            for p in movers:
                free_f = d.capacity["freezer"] - (base["freezer"] + added("freezer"))
                if free_f <= CAP_TOL or over <= CAP_TOL:
                    break
                move_vol = min(over, chilled[p] * d.volume[p], free_f)
                du = move_vol / d.volume[p]
                chilled[p] -= du
                frozen[p] += du
                over -= move_vol
        overflow = [s for s in STORAGES if base[s] + added(s) > d.capacity[s] + CAP_TOL]
        return chilled, frozen, overflow

    def _adjust_order(self, raw: np.ndarray, week: int):
        """Raw order (units) -> package counts after diet, rounding, stock cap, storage and budget. Returns (n_steps, notes)."""
        d = self.d
        notes: list[str] = []
        ref, price, on_hand = self.ref(week), self.price_at(week), self._units(None)
        q = np.where(np.isfinite(raw), np.maximum(raw, 0.0), 0.0)
        blocked = (~d.allowed) & (q > EPS)
        if blocked.any():
            notes.append("removed excluded products: " + ", ".join(d.product_ids[p] for p in np.where(blocked)[0]))
        q[~d.allowed] = 0.0
        n_steps = np.floor(q / d.step_size + 0.5)                       # 1. round to packages (nearest, ties up)
        log = {"requested": q.copy(), "rounding_delta": n_steps * d.step_size - q}
        cap_units = np.maximum(self.stock_cap_weeks * ref, d.step_size + ref)   # 2. stock cap (whole steps only)
        max_steps = np.maximum(np.floor((cap_units - on_hand) / d.step_size + EPS), 0.0)
        capped = n_steps > max_steps
        if capped.any():
            notes.append("stock cap applied to: " + ", ".join(d.product_ids[p] for p in np.where(capped)[0]))
        log["cap_cut"] = (n_steps - np.minimum(n_steps, max_steps)) * d.step_size
        n_steps = np.minimum(n_steps, max_steps)
        log["storage_cut"] = np.zeros(d.n)
        log["budget_cut"] = np.zeros(d.n)

        def key(p: int):    # cut first: most overstocked (stock after the order, in weeks of demand), then costlier, then index
            return ((on_hand[p] + n_steps[p] * d.step_size[p]) / max(ref[p], 1e-6), d.step_size[p] * price[p], -p)

        removed_storage = removed_budget = 0
        while True:                                                      # 3. capacity-aware placement, then reduction
            chilled, frozen, overflow = self._place(n_steps * d.step_size, ref)
            if not overflow:
                break
            cands = [p for p in range(d.n) if n_steps[p] > 0 and (
                (d.default_storage[p] in overflow and chilled[p] > EPS and not d.sold_frozen[p])
                or ("freezer" in overflow and frozen[p] > EPS))]
            cands = cands or [p for p in range(d.n) if n_steps[p] > 0]
            if not cands:
                break
            pick = max(cands, key=key)
            n_steps[pick] -= 1
            log["storage_cut"][pick] += d.step_size[pick]
            removed_storage += 1
        if removed_storage:
            notes.append(f"removed {removed_storage} package step(s) to fit storage")
        while float((n_steps * d.step_size * price).sum()) > d.weekly_budget + EPS:   # 4. budget
            cands = [p for p in range(d.n) if n_steps[p] > 0]
            pick = max(cands, key=key)
            n_steps[pick] -= 1
            log["budget_cut"][pick] += d.step_size[pick]
            removed_budget += 1
        if removed_budget:
            notes.append(f"removed {removed_budget} package step(s) to fit the budget of ${d.weekly_budget:.0f}")
        self.last_adjust = log
        return n_steps, notes

    # ------------------------------------------------------------------ daily dynamics
    def _thaw(self, demand_day: np.ndarray, ref: np.ndarray) -> None:
        """Just-in-time thaw into free default-storage volume only. Thawed lots are never refrozen."""
        d = self.d
        for p in np.where(d.freezable)[0]:
            frozen_lots = [l for l in self.lots if l.p == p and l.frozen]
            if not frozen_lots:
                continue
            desired = max(THAW_LEAD_DAYS * ref[p] / DAYS, demand_day[p])
            chilled_now = sum(l.units for l in self.lots if l.p == p and not l.frozen)
            need = desired - chilled_now
            if need <= EPS:
                continue
            free = d.capacity[d.default_storage[p]] - self.volumes()[d.default_storage[p]]
            amount = min(need, sum(l.units for l in frozen_lots), max(free, 0.0) / d.volume[p])
            for lot in sorted(frozen_lots, key=lambda l: (-l.chilled_days, l.frozen_days)):
                if amount <= EPS:
                    break
                take = min(lot.units, amount)
                if take < lot.units - EPS:                               # split the lot
                    rest = copy.copy(lot)
                    rest.units = lot.units - take
                    lot.units = take
                    self.lots.append(rest)
                lot.frozen, lot.thawed = False, True
                amount -= take

    def _serve(self, demand_day: np.ndarray, price: np.ndarray, acc: dict) -> None:
        d = self.d
        for p in np.where(demand_day > EPS)[0]:
            left = demand_day[p]
            lots = sorted((l for l in self.lots if l.p == p and self._consumable(l)), key=lambda l: (self.remaining_days(l), l.unit_cost))
            for lot in lots:
                if left <= EPS:
                    break
                take = min(lot.units, left)
                lot.units -= take
                left -= take
                acc["consumed"][p] += take
                acc["consumed_cost"] += take * lot.unit_cost
            if left > EPS:
                acc["stockout_p"][p] += left
                acc["stockout_units"] += left
                acc["stockout_value"] += left * price[p]
                acc["stockout_days"] += 1
        self.lots = [l for l in self.lots if l.units > EPS]

    def _age_and_dispose(self, acc: dict) -> None:
        d = self.d
        keep = []
        for lot in self.lots:
            if lot.frozen:
                lot.frozen_days += 1
                dead = lot.frozen_days >= d.L_freezer[lot.p]
                safety = False                                           # frozen limit is a quality estimate
            else:
                if lot.thawed:
                    lot.post_thaw_days += 1
                else:
                    lot.chilled_days += 1
                dead = lot.chilled_days + lot.post_thaw_days >= d.L_default[lot.p]
                safety = bool(d.safety_kind[lot.p])
            if dead:
                acc["wasted"][lot.p] += lot.units
                acc["waste_cost"] += lot.units * lot.unit_cost
                acc["waste_units_safety_cutoff" if safety else "waste_units_quality"] += lot.units
            else:
                keep.append(lot)
        self.lots = keep

    # ------------------------------------------------------------------ terminal accounting
    def terminal_valuation(self, week: int) -> tuple[float, float, float]:
        """(basis, credit, charge) of the leftover inventory. credit = value kept as useful carry-over:
        per product, units up to CREDIT_BUFFER_WEEKS x reference demand (highest-credit lots first), each lot weighted by
        min(1, remaining_days / CREDIT_DAYS). charge = basis - credit."""
        ref = self.ref(week)
        basis = credit = 0.0
        for p in range(self.d.n):
            lots = [l for l in self.lots if l.p == p]
            basis += sum(l.units * l.unit_cost for l in lots)
            buffer = CREDIT_BUFFER_WEEKS * ref[p]
            for lot in sorted(lots, key=lambda l: -self.remaining_days(l)):
                w = float(np.clip(self.remaining_days(lot) / CREDIT_DAYS, 0.0, 1.0))
                u = min(lot.units, buffer)
                credit += u * lot.unit_cost * w
                buffer -= u
        return basis, credit, basis - credit

    # ------------------------------------------------------------------ step
    def step(self, action):
        d = self.d
        week = self.week
        order = np.asarray(action, dtype=float).reshape(-1)
        if order.shape != (d.n,):
            raise ValueError(f"action must have shape ({d.n},)")
        ref, price = self.ref(week), self.price_at(week)
        n_steps, notes = self._adjust_order(order, week)
        final_units = n_steps * d.step_size
        spend = float((final_units * price).sum())
        chilled, frozen, overflow = self._place(final_units, ref)
        assert not overflow and spend <= d.weekly_budget + 1e-6, "pipeline produced an invalid order"
        for p in range(d.n):
            if chilled[p] > EPS:
                self.lots.append(Lot(p, float(chilled[p]), float(price[p])))
            if frozen[p] > EPS:                                          # direct to freezer, all counters 0, not 'thawed'
                self.lots.append(Lot(p, float(frozen[p]), float(price[p]), frozen=True))
        self.ledger["purchased"] += final_units
        self._track_volumes()

        acc = dict(consumed=np.zeros(d.n), consumed_cost=0.0, waste_cost=0.0, stockout_value=0.0, stockout_units=0.0,
                   stockout_days=0, stockout_p=np.zeros(d.n), wasted=np.zeros(d.n), waste_units_quality=0.0, waste_units_safety_cutoff=0.0)
        demand_week = d.demand[week]
        demand_day = demand_week / DAYS
        for _ in range(DAYS):
            self._thaw(demand_day, ref)
            self._track_volumes()
            self._serve(demand_day, price, acc)
            self._age_and_dispose(acc)
        self.ledger["consumed"] += acc["consumed"]
        self.ledger["wasted"] += acc["wasted"]

        protein_g = float((acc["consumed"] * d.protein).sum())
        kcal = float((acc["consumed"] * d.kcal).sum())
        met = protein_g / d.protein_target
        shortfall = max(0.0, 1.0 - met)
        self.last_protein_ratio = met
        self.cash["spend"] += spend
        self.cash["consumed_cost"] += acc["consumed_cost"]
        self.cash["waste_cost"] += acc["waste_cost"]
        self.cash["stockout_value"] += acc["stockout_value"]

        self.t += 1
        terminated = self.t >= self.horizon       # finite-horizon task: terminal charge defined, value after the last step is 0
        info = {
            "spend": spend, "waste_units": float(acc["wasted"].sum()), "waste_cost": acc["waste_cost"],
            "stockout_units": acc["stockout_units"], "stockout_days": acc["stockout_days"], "kcal": kcal,
            "protein_g": protein_g, "protein_target_met": met, "notes": notes, "order": final_units,
            "waste_units_quality": acc["waste_units_quality"], "waste_units_safety_cutoff": acc["waste_units_safety_cutoff"],
            "consumed_cost": acc["consumed_cost"], "stockout_value": acc["stockout_value"],
            "episode_cash_spend": self.cash["spend"], "adjust": self.last_adjust,
            "stockout_by_product": acc["stockout_p"], "consumed_by_product": acc["consumed"],
        }
        reward = -(acc["consumed_cost"] + acc["waste_cost"] + STOCKOUT_MULT * acc["stockout_value"]) / d.weekly_budget \
                 - PROTEIN_WEIGHT * shortfall
        if terminated:
            basis, credit, charge = self.terminal_valuation(self.week)
            self.cash["terminal_charge"] = charge
            reward -= charge / d.weekly_budget
            info.update(terminal_inventory_basis=basis, terminal_inventory_credit=credit, terminal_charge=charge,
                        episode_net_cost=self.cash["consumed_cost"] + self.cash["waste_cost"] + charge)
        info["obs_dict"] = self._obs_dict()
        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ observation
    def _forecast(self, week: int) -> np.ndarray | None:
        # No forecast once the episode is over: the terminal observation must not request a week of the next episode or split.
        if self.forecast_fn is None or self.t >= self.horizon:
            return None
        return np.asarray(self.forecast_fn(week), dtype=float)

    def _obs_dict(self) -> dict:
        d, week = self.d, min(self.week, LAST_WEEK)
        fc = self._forecast(week)
        chilled, frozen = self._units(True), self._units(False)
        return {
            "inventory": {pid: {"chilled": float(chilled[i]), "frozen": float(frozen[i])} for i, pid in enumerate(d.product_ids)},
            "week": week, "forecast": None if fc is None else dict(zip(d.product_ids, fc.tolist())),
            "budget_left": d.weekly_budget,
            "ref_demand": self.ref(week).tolist(),       # trailing 4-week mean demand (known history)
        }

    def _obs(self) -> np.ndarray:
        d = self.d
        week = min(self.week, LAST_WEEK)
        ref = np.maximum(self.ref(week), 1e-6)
        price = self.price_at(week)
        price_ref = np.maximum(d.price[max(week - 4, 0):week].mean(axis=0), 1e-6)
        on_hand = self._units(None)
        frozen = self._units(False)
        at_risk = np.zeros(d.n)
        for lot in self.lots:
            if not lot.frozen and self.remaining_days(lot) <= AT_RISK_DAYS:
                at_risk[lot.p] += lot.units
        fc = self._forecast(week)
        fc_block = np.zeros(d.n) if fc is None else fc / ref
        vols = self.volumes()
        free = [1.0 - vols[s] / d.capacity[s] for s in STORAGES]
        angle = 2 * np.pi * (week % 52) / 52
        glob = [np.sin(angle), np.cos(angle), (self.horizon - self.t) / self.horizon, d.weekly_budget / d.budget_scale,
                *free, self.last_protein_ratio, float(d.holiday[week]), float(fc is not None)]
        obs = np.concatenate([np.clip(on_hand / ref, 0, 4), np.clip(at_risk / ref, 0, 4), np.clip(frozen / ref, 0, 4),
                              np.clip(price / price_ref, 0, 4), np.clip(fc_block, 0, 4), np.clip(glob, -5, 5)])
        return obs.astype(np.float32)
