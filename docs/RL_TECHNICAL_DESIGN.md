# RL technical design (MVP)

Status: design only. `GroceryEnv` is not implemented yet. This document refines `docs/RL_MODELLING_RULES.md`
and builds on `data_rl/storage_rules.csv` and `data_rl/household_storage.csv` (see `docs/STORAGE_DATA.md`).
All storage numbers and budgets are synthetic assumptions. Results are simulated, not real savings.

MVP principles: one order per week, deterministic storage handling, hard constraints enforced by code (not by reward),
plain Python/NumPy dynamics.

---

## 1. Storage-age model

### Lot state
Inventory is a list of **lots** (one purchase of one product). Each lot carries:

| Field | Meaning |
|---|---|
| `product_id`, `units` | What and how much (fractional units allowed) |
| `unit_cost` | Price paid per unit (cost basis, used by the reward) |
| `frozen` | True while the lot is in the freezer |
| `chilled_days` | Days spent in the default storage (pantry or fridge) **before** freezing |
| `frozen_days` | Days spent in the freezer |
| `post_thaw_days` | Days spent in the default storage **after** thawing |
| `thawed` | True once the lot has been frozen and thawed |

The three day counters are tracked separately and are **never reset** by a move. `total age = chilled_days + frozen_days + post_thaw_days`
is kept for reporting only.

### Ageing and disposal (per simulated day)
- Lot in default storage and never frozen: `chilled_days += 1`.
- Lot in the freezer: `frozen_days += 1`. The chilled clock is **paused**, not reset.
- Thawed lot (back in default storage): `post_thaw_days += 1`. The chilled clock **resumes** where it stopped.

A lot is disposed (simulated waste) when either condition holds:
- `chilled_days + post_thaw_days >= L_default`, where `L_default` is the shelf life at default storage (`shelf_life_days` in `products.csv`);
- `frozen_days >= L_freezer` (`shelf_life_freezer_days` in `storage_rules.csv`).

Example: chicken (`L_default = 3`, `L_freezer = 180`) bought, 1 day in the fridge, frozen for 30 days, thawed: it has 2 chilled days left
(`3 - 1 - 0`). Freezing did not reset freshness and did not extend it beyond the days it paused.

This replaces the earlier single "fraction of life used" and the 0.25 thaw surcharge. No new constants are needed: the rule uses only
the shelf lives already in the data. It is optimistic for real thawed food (it is a simulation rule, see section 5), so the only extra
restriction is **no refreezing** (`thawed` lots never return to the freezer).

### Moves
- Allowed moves: default storage -> freezer (for `freezing_supported` products that are not sold frozen) and freezer -> default storage.
  There are no pantry <-> fridge moves in the MVP. Products sold frozen (`frozen_vegetables`, `frozen_pizza`) stay in the freezer.
- A move needs free volume in the target storage; otherwise it is skipped.
- A lot is only frozen if it still has at least 1 chilled day left.

### Capacity and consumption
- Volume of a storage = sum over its lots of `units x volume_l_per_unit` (litres), at most `fridge_l`, `freezer_l` or `pantry_l`.
- Weekly demand is split evenly over 7 days. Each day demand is served from non-frozen lots of the product, **first-expiring-first**
  (smallest remaining chilled life). Frozen lots are never consumed directly. Unserved demand is a stockout.
- Daily order of events: serve demand -> age lots by one day -> dispose lots that reached their limit.

## 2. Deterministic freezing policy (MVP)

The RL action contains no storage decisions. A fixed rule runs identically for RL and every baseline:
1. **Freeze surplus after delivery.** For each freezing-capable product, `need = expected_daily_demand x remaining_chilled_days`.
   Units above `need` (and at least one chilled day left) move to the freezer if there is room.
   Expected daily demand = trailing 4-week mean / 7 (the forecast is not used here, so the rule is the same with or without a forecaster).
2. **Thaw just in time.** Each day, if the chilled stock of the product covers less than `THAW_LEAD_DAYS = 2` days of expected demand,
   thaw enough units to cover them, oldest `chilled_days` first.
3. Nothing else moves.

`THAW_LEAD_DAYS` is the only constant of this rule (an assumption). Later extension, not in the MVP: freeze/thaw decisions in the action space,
compared against this rule as a baseline.

## 3. Storage scenarios

| Scenario | Capacities | Role |
|---|---|---|
| `default` | `data_rl/household_storage.csv` (unchanged; at least 4x the P95 weekly demand volume, so it rarely binds) | Headline results |
| `limited` | Separate file `data_rl/household_storage_limited.csv`, to be created in a later step | Stress test |

Plan for `limited`: capacity per storage = a multiple (about 1.0-1.5x, to be calibrated) of the P95 weekly demand volume for that default storage,
calibrated on weeks 4-63 only and checked so a household can still store one normal week. Environment parameter:
`storage_scenario = "default" | "limited"`. Headline grid: region x budget scenario (normal, challenging, tight) on `default` storage;
`limited` runs only on the `normal` budget.

## 4. Package sizes and order adjustment

Quantities are in purchase units. Each product has a **package step** (defined in the environment code, not in `products.csv`):
`1.0` for packaged and counted items; `0.5` for loose items sold by weight (bananas, apples, carrots, oranges, onions).
Final orders must always be an integer number of steps.

Pipeline inside `step()`, in this order:
1. **Filter:** zero the products excluded by the household's `allergen_exclusions` and by its diet (vegetarian, vegan, pescatarian rules
   by product category). Own filter, `apply_constraints` (Person 3) is not modified; if it is used, see step 5.
2. **Round to packages:** `n_p = round(q_p / step_p)` (nearest, ties up). Tiny requests become 0.
3. **Stock cap:** reduce `n_p` so that on-hand + order is at most 3 weeks of reference demand (but at least one step allowed).
4. **Storage:** compute the volume added to each default storage. While a storage would overflow, remove **one step** from the product with the
   highest coverage ratio (on-hand + ordered, divided by weekly reference demand), i.e. the most overstocked item.
5. **Budget:** while `cost = sum(n_p x step_p x price_p) > budget`, remove one step from the product with the highest coverage ratio
   (ties: higher cost first).
6. **Notes:** every change is recorded in `info["notes"]`.

Rule that keeps package sizes intact: rounding to packages happens **first**; after that, storage and budget adjustments only **remove whole
steps** (and any external scaling is followed by flooring to a multiple of the step, never rounding up). Cost and volume only decrease when
steps are removed, so the final order satisfies every constraint and is a multiple of the step. A unit test must assert this
(`order % step == 0`, `cost <= budget`, `volume <= capacity`, no excluded product) on random orders.

## 5. Simulated disposal vs. actual food-safety expiration

This project does **not** model real food-safety expiry. It models *simulated disposal*:

| Term | Meaning | Source |
|---|---|---|
| Quality disposal | A lot with `shelf_life_kind = quality` reaches its planning shelf life and counts as waste | Synthetic estimate |
| Safety-conservative cutoff | A lot with `shelf_life_kind = safety_conservative` reaches its cutoff and counts as waste. Never extended | Synthetic, deliberately cautious, not verified |
| Real food-safety expiration | Use-by dates, storage guidance, spoilage and illness risk in the real world | **Not modelled, not claimed** |

- `info` keeps `waste_units` and `waste_cost` as in `INTERFACES.md`, and adds `waste_units_quality` and `waste_units_safety_cutoff` (they sum to `waste_units`).
- Wording: "simulated waste" or "simulated disposal". Never "expired" or "unsafe", and never "safe to eat". The pause-and-resume rule for thawed
  food is a simulation simplification, not food-safety guidance.

## 6. RL formulation

### Episode and time
- One episode = one household, one region, one budget scenario, one storage scenario, **13 weekly steps**.
- One step = one shopping decision at the start of a week, then 7 simulated days.
- Start week: training samples from weeks 4-51 (so the episode ends by week 63); validation uses windows inside weeks 64-77 (start 64, 13 weeks);
  test uses two non-overlapping windows, 78-90 and 91-103. Weeks 0-3 are never used (no forecast history).
- `terminated` is always False; `truncated` is True after the 13th step (time limit).

### Observation (flat `float32` vector, about 200 values)
Per product (36 each), all scaled by the product's weekly reference demand `ref` (trailing 4-week mean demand):
- `on_hand / ref` (chilled + frozen, clipped to [0, 4])
- `at_risk / ref`: chilled units with 3 or fewer days left
- `frozen / ref`
- `price / trailing_4w_mean_price` (promo and price-level signal)
- `forecast / ref` (zeros when no forecaster; a separate flag says so)

Global values: week of year as sin/cos; weeks left in the episode / 13; weekly `budget / budget_scale` (`budget_scale` = household's `calib_median_cost_usd`);
free-volume fraction of pantry, fridge, freezer; last week's protein ratio; holiday flag; `forecast_available` flag.
Guest and away flags are **not** given (unknown in advance, consistent with the no-events forecaster).
`INTERFACES.md` asks for a dict with `inventory`, `week`, `forecast`, `budget_left`; `GroceryEnv` exposes that dict as `info["obs_dict"]` and the flat
vector as the observation for training. `budget_left` is the weekly budget (it does not carry over between weeks).

### Action
- Contract with `INTERFACES.md`: `env.step(order)` takes **order quantities** in purchase units, shape `(36,)`, values >= 0. Baselines act in this space.
- RL agent: a thin wrapper maps a normalized action `a in [0, 1]^36` to order-up-to targets
  `target_p = 2 x a_p x ref_p`, and `order_p = max(0, target_p - on_hand_p)`. This keeps the learning problem well scaled.
- Everything then goes through the pipeline in section 4.

### Step behaviour
1. Read the action, build the raw order, run the pipeline (section 4).
2. Pay the order; add lots to the default storage with `unit_cost = price`.
3. Apply the freezing rule (section 2).
4. For each of 7 days: thaw rule, serve demand, age, dispose.
5. Compute metrics (`INTERFACES.md` section 5 keys plus `notes`, `waste_units_quality`, `waste_units_safety_cutoff`), reward and next observation.

### Reward (per step)
```
reward = - ( consumed_cost + waste_cost + 2.0 x stockout_value ) / budget
         - 0.5 x protein_shortfall
```
- `consumed_cost`: cost basis (`unit_cost`) of the units eaten this week. `waste_cost`: cost basis of disposed units.
  `stockout_value`: unserved demand valued at the current price. `protein_shortfall = max(0, 1 - protein_target_met)`.
- Charging cost when units are **eaten or wasted**, not when bought, avoids an end-of-episode effect (the agent is not punished for stock it
  still holds, and a cheap promo purchase is rewarded when eaten later).
- **Why "buy nothing" cannot win:** serving a unit costs its price; failing to serve it costs 2.0 times its price plus the protein shortfall.
  A policy that buys nothing pays the stockout multiplier on all demand; buying what is eaten pays 1.0 times. Waste costs the price once, so
  over-buying is also worse than buying exactly what is needed.
- The budget, allergens, diet, capacity and package sizes are **hard constraints** in the pipeline, not reward terms.
- **Acceptance test before training:** on validation episodes, the `buy_nothing` policy must score clearly below `fixed`, `average_rule` and
  `forecast_rule`; if not, adjust the multipliers before any training.
- Reported metrics stay the `INTERFACES.md` ones (weekly spend, waste, stockout days, protein target met); the reward is only a training signal.

### Reset
`reset(seed, options={"spec": EvalSpec})` builds the episode from a spec (below). Without a spec it samples a training spec from the training weeks.
Reset sets the week, loads that region's prices and the household's budget and capacities, and creates the initial inventory (section 7).

## 7. Identical starting conditions for all policies

An **`EvalSpec`** fully defines an episode: `household_id, region_id, budget_scenario, storage_scenario, start_week, horizon (13), demand_seed`.
- `demand_seed = None` (MVP) replays the recorded `demand_weekly.csv`, so demand, prices, budgets and capacities are identical for every policy.
  Independent evaluation seeds (resampled demand) are a later option; see section 8.
- **Initial inventory** is a deterministic function of the spec: for each product, `INIT_COVER_DAYS = 3` days of the trailing 4-week mean
  demand (weeks before `start_week`), in default storage with `chilled_days = 0`, `unit_cost = price at start_week - 1`. Capacity permitting; otherwise
  the same proportional cut for everyone. No randomness.
- All policies (`buy_nothing`, `fixed`, `average_rule`, `forecast_rule`, `rl`) receive the same list of specs and the same pipeline, rewards and metrics.
  RL acts deterministically (no exploration noise) during evaluation.
- The evaluation set is fixed in a file written once (list of specs plus their hash), so results can be reproduced and compared paired
  (per-episode differences with confidence intervals).
- Training uses random training specs and training seeds only. The validation set selects checkpoints. The test set is run once at the end.

## 8. ML/RL split and calibration limitations

Checked in `src/forecast.py` on `origin/main` (commit `01bbee9`). We do not modify that file.

| Item | ML forecast | RL design |
|---|---|---|
| Weeks | train 4-63, validation 64-77, test 78-103 | train 4-63, validation 64-77, test 78-103 |
| Final scoring models | `models/eval/` trained on weeks 4-77 | n/a |
| Default models | `models/` trained on **all** weeks, including 78-103 | must not be used on test weeks |

Issues to coordinate with the ML teammate:
1. **Leakage:** `predict_next_week` defaults to the all-weeks model. RL evaluation must use `for_evaluation=True`.
2. **In-sample forecasts:** forecasts for weeks 4-77 come from models trained on those weeks (`forecast_weekly.csv` flags only weeks >= 78 as
   `out_of_sample`). An agent trained on them would over-trust the forecast. Request: an honest forecast file for weeks 4-103 with a
   `forecast_source` column (4-63 out-of-fold or expanding-window; 64-77 from the model trained on 4-63; 78-103 from `models/eval/`).
3. **Independent seeds:** `demand_weekly.csv` is one recorded realisation. If evaluation resamples demand, forecasts must be recomputed from the
   simulated history. To be decided with the ML teammate. Until then the MVP runs with `forecaster=None` and replayed demand.

### Budget calibration limitation (validation weeks)
The budget files (`data/household_region_budgets.csv`) were calibrated on normal weeks 0-77. That window includes the ML/RL **validation** weeks 64-77
(and the training weeks 4-63). Consequences:
- The **test weeks 78-103 are not used** in calibration, so test results are unaffected.
- Budgets are fixed environment constants, not learned parameters, but validation-set budget feasibility is slightly optimistic, because those
  weeks helped set the budget level. Validation scores must not be read as an unbiased estimate of budget pressure.
- The budget files stay unchanged. Optional later sensitivity check: recalibrate on weeks 0-63 only and compare test results.

## Open decisions
- Scope of the first agent: one shared policy for all households, or per-household policies (suggest starting with a few demo households, e.g. `hh_student`, one region).
- Constants: `THAW_LEAD_DAYS = 2`, `INIT_COVER_DAYS = 3`, stockout multiplier 2.0, protein weight 0.5, stock-cap multiple 3 weeks.
- Calibration target for the `limited` storage scenario.
- Whether to merge `origin/main` into `RL` (the `RL` branch is behind `main`).
