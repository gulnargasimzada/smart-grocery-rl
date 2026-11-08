# RL action wrapper, evaluation framework and forecast interface

Implementation notes for the pre-training stage. No agent has been trained. All budgets, storage rules and demand are synthetic;
results are simulated.

## 1. Action wrapper (`src/rl_wrapper.py`)
`OrderUpToWrapper(GroceryEnv)` exposes `Box(0, 1, (36,), float32)`; the observation is the unchanged flat `float32` vector (190 values).

```
target_p  = 2.0 x a_p x ref_p                ref_p = trailing 4-week mean demand (known history); a_p clipped to [0, 1]
deficit_p = max(0, target_p - on_hand_p)     on_hand = chilled + frozen units
order_p   = ceil(deficit_p / step_p) x step_p
```
- `a = 0` never buys, `a = 0.5` orders up to one week of average demand (**identical to the Average Rule baseline**, tested over a full episode),
  `a = 1` orders up to two weeks. The proposal is already a whole number of packages (rounded up when there is a deficit).
- The proposal then goes through the unchanged `GroceryEnv` pipeline: diet/allergen exclusions (shared `src/diets.py`), package rounding,
  stock cap, capacity-aware storage placement including the freezer, weekly budget. `info["proposed_order"]` is the order before the pipeline,
  `info["order"]` after it. The pipeline only removes whole packages.
- Reward, observation and environment constants are unchanged (tested: the wrapped episode equals the raw env fed with the proposed orders).
- NaN or wrongly shaped actions raise `ValueError`; values outside [0, 1] are clipped. Conversion is deterministic.
- Stable-Baselines3 is not installed or used. The Box spaces follow the Gymnasium API (`check_env` passes).

## 2. Evaluation framework (`src/evaluate.py`)
| Split | Target weeks | Episodes (13 weeks each) |
|---|---|---|
| train | 4-63 | sampler starts 4..51 (`training_starts()`); fixed evaluation set starts 4, 17, 30, 43, 51 (the last two windows overlap) |
| validation | 64-77 | complete windows only: start 64 (weeks 64-76). Start 65 is also complete but overlaps it (`validation_starts(overlapping=True)`) |
| test | 78-103 | two fixed episodes: 78-90 and 91-103. **Reserved for the final evaluation.** |

- `validate_spec` rejects any episode that is not fully inside its split. `EpisodeSpec` = split, household, region, budget scenario,
  storage scenario, start week, horizon (MVP: `hh_student`, `durham_nc`, `normal`, `default`). The spec lists and their hashes are in
  `results/evaluation/specs.json`.
- Every policy runs the same specs: identical recorded demand, prices, budget, capacities and initial inventory. **Deterministic, paired replay of
  one recorded trajectory.** Episodes and policy differences are not independent samples: no confidence intervals, no significance claims.
- Policies: `buy_nothing` (diagnostic floor), `fixed`, `average_rule`, `forecast_rule`, and an `RLPolicy(predict)` interface
  (`predict(obs_vector) -> action in [0, 1]^36`, evaluated on the wrapper). `forecast_rule` runs only with a verified forecast table; otherwise it is
  listed as "not run" with the reason, and the other policies still run.
- Metrics per policy and episode: cumulative reward, cash spending, net cost, waste cost and units, stockout units and value, stockout days,
  price-weighted fill rate, protein coverage, terminal inventory basis, credit and charge.
- Commands: `python -m src.evaluate --split validation` or `--split train`. The command line does not offer `test`, and
  `evaluate_policies("test", ...)` raises unless `confirm_final_test=True`. Test weeks were not run, and were not used for debugging or decisions.
- Baseline results are preserved: the evaluation module reproduces `src.diagnose_env` exactly (regression tests), e.g. validation window 64:
  fixed -16.08, average_rule -15.32, buy_nothing -29.76.

## 3. Forecast interface (`src/forecast_table.py`)
Reads a forecast file produced by the ML side; never trains models, never touches `src/forecast.py`, never uses actual demand.
- Variants: `no_events` (guest/away unknown; default) and `with_events` (the household knows guest/away weeks in advance).
- A forecast is used only if it is verifiably out-of-sample: the row has `trained_through_week < week`, or `out_of_sample == 1`
  (if both columns exist, both must hold). A file with neither column is rejected as a whole. `forecast_source == "in_sample"` rows are rejected.
- Missing, non-finite or negative forecasts raise `ForecastUnavailable` naming the weeks; nothing is imputed. Actual-demand columns
  (`actual_units`, `demand_units`) are never loaded.
- Accepted columns: `household_id, week, product_id`, forecast values `forecast_no_events` and/or `forecast_with_events` (legacy names
  `forecast_units_no_events` / `forecast_units` are mapped), plus the validity columns above.
- Forecast-dependent code (`forecast_rule`, an agent trained with the forecast block) fails clearly when forecasts are unavailable. Non-forecast
  baselines never read the forecast file. The tests use synthetic mock forecasts only.

### State of `main` (checked, not merged; commit `1938065`)
- The final forecaster is Ridge (`models/ridge_with_events.pkl`, `models/ridge_no_events.pkl`), trained on target weeks 4-77. The API is
  `predict_next_week(history, household_id, week, guest=None, away=None)`.
- `data/forecast_weekly.csv` (git-ignored, generated by `python -m src.forecast --export`) holds Ridge predictions with an `out_of_sample` flag that is
  1 only for weeks >= 78. Weeks 4-77 are **in-sample**, so this file is valid for the test weeks only and is rejected by the table for training and
  validation weeks. It also contains `actual_units`, which we ignore.
- **Needed from the ML teammate** for RL training and validation: walk-forward, out-of-sample Ridge forecasts for target weeks 4-77, one row per
  household, week and product, with `trained_through_week` (the last target week used to fit the model that produced the row; it must be < `week`),
  and values for both variants. For weeks 78-103 the saved models (trained on 4-77) are already honest.

## 4. Limitations
- One household and region, one recorded demand trajectory, one validation window and two test windows: results describe this simulated
  setting only. Seeds in later training control training randomness, not demand.
- The $75 Normal budget binds for the Average Rule; waste is simulated disposal, not food-safety expiry.
- The train evaluation windows overlap (51-63 and 43-55), and the agent trains on weeks 4-63, so train-split numbers are not generalisation evidence.
- No forecast-based results exist yet; `forecast_rule` and forecast features in the observation are untested on real out-of-sample forecasts.
- `limited` storage, other budget scenarios and other households are not evaluated yet.
