# RL technical design: storage, disposal, scenarios and evaluation split

Status: design only. `GroceryEnv` is not implemented yet. This document refines `docs/RL_MODELLING_RULES.md`
and builds on `data_rl/storage_rules.csv` and `data_rl/household_storage.csv` (see `docs/STORAGE_DATA.md`).
All storage numbers are synthetic assumptions.

## 1. Product movement between pantry, fridge and freezer

### Inventory representation
Inventory is a list of **lots**. A lot is one purchase of one product in one storage place:

| Field | Meaning |
|---|---|
| `product_id`, `units` | What and how much (fractional units allowed) |
| `storage` | `pantry`, `fridge` or `freezer` |
| `used_life` | Fraction of shelf life already used, in `[0, 1)`. **This is the lot's age.** |
| `thawed` | True if the lot has been frozen and thawed |

Age is stored as **fraction of life used**, not as days, so it survives a change of storage.

### Ageing rule
Each simulated day a lot adds `1 / shelf_life_<storage>_days` to `used_life`, using the shelf life of the storage it is in
(from `storage_rules.csv`). A lot reaches the end of its life when `used_life >= 1`.

### Moving a lot (no age reset)
Moving a lot never sets `used_life` back to 0. Only the ageing *rate* changes with the new storage:
- Pantry <-> fridge (where both are allowed): `used_life` is kept. Remaining days = `(1 - used_life) x shelf_life_new_storage`.
- Fridge/pantry -> freezer: `used_life` is kept. In the freezer it grows slowly (`1 / shelf_life_freezer_days` per day).
  Freezing pauses most of the ageing, it does not rewind it. Example: chicken frozen on day 2 of 3 (`used_life = 0.67`) has
  only 33% of its frozen life left, not a full reset.
- Freezer -> fridge (thaw): `used_life` is kept, then a **thaw surcharge** of `THAW_SURCHARGE = 0.25` is added. If the sum reaches 1, the lot is
  disposed at thawing. The lot is marked `thawed`. The surcharge is a modelling assumption for "thawed food keeps worse",
  not a verified figure; it is a tunable constant.
- **No refreezing** in the MVP: a `thawed` lot may not return to the freezer.
- Products sold frozen (`frozen_vegetables`, `frozen_pizza`) arrive in the freezer and never move.
- Moves are only allowed to storage types listed in `allowed_storage`, and only if the target has free volume.

### Capacity
Volume of a storage = sum over its lots of `units x volume_l_per_unit` (litres). It must stay <= the household capacity
(`fridge_l`, `freezer_l`, `pantry_l`). A purchase that does not fit its default storage is first tried in another allowed storage, otherwise the
overflow is **not bought** (the order is cut before payment, with a note). This happens inside the optimizer/constraint step,
before the budget check, so the cost is always the cost of what is stored.

### Consumption
Weekly demand is split evenly over 7 days (MVP). Each day, demand is served from lots of the product that are in
the **fridge and pantry first** (freezer lots are only used after being thawed). Within a product, lots with the highest
`used_life` are used first. Unserved demand is a stockout.

## 2. Deterministic freezing policy (MVP)

The RL action stays as in `INTERFACES.md`: one order quantity per product. **Storage decisions are not part of the action in the MVP.**
A fixed rule handles them, identically for RL and for every baseline, so differences between policies come from buying decisions only.

Rule, applied to products with `freezing_supported = True` that are not sold frozen:
1. **Freeze surplus on arrival.** After a delivery, estimate the units that will be used before the lot's default-storage life ends:
   `need = expected_daily_demand x remaining_days_in_default_storage` (expected demand = the forecast if available, otherwise the mean of the last 4 weeks).
   Units above `need` are moved to the freezer, if the lot has `used_life <= 0.5` and the freezer has room.
2. **Thaw just in time.** Each day, if the fridge stock of the product covers less than `THAW_LEAD_DAYS = 2` days of expected demand,
   thaw enough units from the freezer to cover them (oldest `used_life` first).
3. Everything else stays in its default storage. No other moves.

Constants (`THAW_SURCHARGE`, `THAW_LEAD_DAYS`, the `0.5` freeze limit) are assumptions to be set in one place in the environment code.

Later extension (not in the MVP): add a discrete freeze/thaw decision per product to the action space and compare it with this rule as a baseline.

## 3. Storage scenarios

| Scenario | Capacities | Role |
|---|---|---|
| `default` | `data_rl/household_storage.csv` (unchanged). Capacity is at least 4x the P95 weekly demand volume, so it rarely binds | Headline results |
| `limited` | A separate file, `data_rl/household_storage_limited.csv` (to be created in a later step, not now) | Stress test |

Plan for `limited`: capacity per storage = a multiple of the household's P95 weekly demand volume for that default storage
(target around 1.0-1.5x, to be calibrated), built by the same reproducible script with a scenario flag and checked for feasibility
(a household must still be able to store one normal week). Weeks 0-63 only are used for calibration, to avoid leaking validation and test weeks.

Environment parameter: `storage_scenario = "default" | "limited"`. Results are reported per scenario. The evaluation grid is
region x budget scenario (normal, challenging, tight) x storage scenario. The headline grid uses `default` storage; `limited` is run on
the `normal` budget as an extra stress test, to keep the number of runs manageable.

## 4. Simulated disposal vs. actual food-safety expiration

This project does **not** model real food-safety expiry. It models *simulated disposal*:

| Term | Meaning | Source |
|---|---|---|
| Quality disposal | A lot with `shelf_life_kind = quality` reaches the end of its planning shelf life and is counted as waste | Synthetic quality estimate |
| Safety-conservative cutoff | A lot with `shelf_life_kind = safety_conservative` reaches its cutoff and is counted as waste. The cutoff is never extended | Synthetic, deliberately cautious assumption, not verified |
| Real food-safety expiration | Use-by dates, storage guidance, spoilage and illness risk in the real world | **Not modelled, not claimed** |

Rules for code, reports and the app:
- The `info` dict keeps `waste_units` and `waste_cost` as in `INTERFACES.md`, and adds `waste_units_quality` and `waste_units_safety_cutoff`
  (they sum to `waste_units`).
- Wording: "simulated waste" or "simulated disposal". Never "expired" or "unsafe", and never say a product is safe to eat.
- Results are simulated, not real savings and not food-safety advice.

## 5. ML/RL train-test split mismatch (to coordinate with the ML teammate)

Checked in `src/forecast.py` on `origin/main` (commit `01bbee9`). Files are not modified by us.

| Item | ML forecast (`forecast.py`) | RL plan so far (`RL_MODELLING_RULES.md`) |
|---|---|---|
| Target weeks | train 4-63, validation 64-77, test 78-103 | train 0-77, test 78-103, no validation set |
| Final scoring models | `models/eval/` trained on weeks 4-77 | n/a |
| Default models | `models/` trained on **all** weeks, including 78-103 | n/a |

Issues:
1. **Leakage risk with the default models.** `predict_next_week` loads the all-weeks model unless `for_evaluation=True`. Any RL evaluation on
   weeks 78-103 must use `for_evaluation=True` (the `models/eval/` models). The default models must not be used for test weeks.
2. **In-sample forecasts during RL training.** The forecasts the RL agent would see in weeks 4-77 come from models trained on those same weeks
   (and `forecast_weekly.csv` marks only weeks >= 78 as `out_of_sample`). Those forecasts are more accurate than they will be on the test weeks,
   so an agent trained on them would over-trust the forecast and look worse at test time.
3. **No RL validation set.** The ML team uses weeks 64-77 for model selection. RL needs its own selection set too, otherwise policy and
   hyperparameter choices are tuned on the test weeks.
4. **Budgets were calibrated on weeks 0-77**, which includes the ML validation weeks 64-77. Budgets are fixed environment constants, so this is
   a minor point, but it is stated here for honesty. We keep the budgets unchanged.
5. **Independent seeds.** `demand_weekly.csv` is one recorded realisation. If the environment resamples demand with new seeds via
   `simulate_demand`, forecasts for those samples need to be computed from the simulated history (`predict_next_week`), not read from
   `forecast_weekly.csv`. To be decided with the ML teammate.

Proposed RL split (aligned with ML):
- RL **training**: target weeks 4-63 (weeks 0-3 have no forecast history).
- RL **validation / model selection**: weeks 64-77.
- RL **test**: weeks 78-103, run once at the end, with independent evaluation seeds.

Requests for the ML teammate (nothing is changed on our side without agreement):
1. An honest forecast file for every week 4-103, with a `forecast_source` column:
   - weeks 4-63: out-of-fold or expanding-window forecasts (each week predicted by a model that did not train on it),
   - weeks 64-77: from the model trained on weeks 4-63 only,
   - weeks 78-103: from the `models/eval/` models.
2. Confirm that RL evaluation calls `predict_next_week(..., for_evaluation=True)`.
3. Agree on how forecasts are produced when the environment resamples demand with independent seeds (item 5 above).

Until then, the RL baselines and agent can run without a forecaster (`forecaster=None` in `GroceryEnv`) to avoid training on leaked or in-sample forecasts.

## Open decisions
- Final values of `THAW_SURCHARGE`, `THAW_LEAD_DAYS` and the freeze limit.
- Calibration target for the `limited` storage scenario.
- Whether to merge `origin/main` into `RL` to pick up the teammate's forecast work (the `RL` branch is currently behind `main`).
