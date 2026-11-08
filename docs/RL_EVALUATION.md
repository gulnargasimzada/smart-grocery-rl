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
Reads the forecast file produced by the ML side; never trains models, never touches `src/forecast.py`, never uses actual demand.
- Variants: `no_events` (guest/away unknown; **the default** for the table, the environment and the evaluation) and `with_events` (only if the household
  knows guest/away weeks in advance). Value columns of the export: `forecast_units_no_events` (no events) and `forecast_units` (with events).
- **Validity** (checked per row): Ridge rows need `model_trained_through_week < week` (alias `trained_through_week`). When a cutoff column exists it
  decides; the legacy `out_of_sample` column (1 only from week 78) is ignored, because it is 0 for valid walk-forward predictions before week 78.
  `out_of_sample` is used only by old files with no cutoff column. A ridge row without a cutoff, a cutoff >= week, or `forecast_source == "in_sample"`
  is rejected. A file with neither column is rejected as a whole.
- **`forecast_method == "ma4_fallback"`** (weeks 4-11, too little history to fit Ridge): historical 4-week moving averages from earlier weeks only.
  They are accepted, identified (`method()`, `fallback_weeks()`), listed in the evaluation report, and can be refused with `allow_fallback=False`.
- Missing, non-finite or negative values raise `ForecastUnavailable` naming the weeks; nothing is imputed. `actual_units` / `demand_units` are
  never loaded.
- **Environment connection:** `GroceryEnv(forecast_fn=table.forecast_fn(household_id, "no_events"))` puts `forecast / ref` (36 values, products.csv
  order, clipped to [0, 4]) into the observation, with `forecast_available = 1`. The environment only requests the forecast of the episode's own weeks:
  the terminal observation (after the last step) has no forecast block and `forecast_available = 0`, so a training window never asks for week 64 and a
  validation window never asks for week 77 or the next episode. With stable-baselines3 time-limit bootstrapping, the critic sees that terminal
  observation without forecast; keep that in mind when reading value-loss curves.
- Forecast-dependent code (`forecast_rule`, an agent trained with the forecast block) fails clearly when forecasts are unavailable. Non-forecast baselines
  never read the forecast file. `python -m src.evaluate` reads `data/forecast_weekly.csv` by default and prints a BLOCKER line if it is unusable.

### State of `main` (checked, not merged; commit `0be89d0`)
- Final forecaster: Ridge (`models/ridge_*.pkl`, trained through week 77). `python -m src.forecast --export` writes the git-ignored
  `data/forecast_weekly.csv` with honest forecasts for all weeks: weeks 4-11 `ma4_fallback`, weeks 12-63 walk-forward (a new model per week, trained
  through week t-1), validation 64-77 from a model trained through week 63, test 78-103 through week 77. Columns: `household_id, week, product_id,
  actual_units, forecast_units, forecast_units_no_events, split, forecast_method, model_trained_through_week, out_of_sample`.
- **Status on this machine:** the exported file is NOT present (git-ignored, and `main` is not merged). Tests built from a synthetic frame with exactly these
  columns pass; the checks on the real file (`tests/test_real_forecasts.py`) are skipped and report this blocker. Run the export on the ML side (or copy
  the file to `data/forecast_weekly.csv`) and re-run `python -m pytest tests -q -rs` and `python -m src.evaluate --split validation`.

## 4. Limitations
- One household and region, one recorded demand trajectory, one validation window and two test windows: results describe this simulated
  setting only. Seeds in later training control training randomness, not demand.
- The $75 Normal budget binds for the Average Rule; waste is simulated disposal, not food-safety expiry.
- The train evaluation windows overlap (51-63 and 43-55), and the agent trains on weeks 4-63, so train-split numbers are not generalisation evidence.
- No forecast-based results on the real forecasts exist yet (the exported file is missing locally); `forecast_rule` and the forecast features are verified on synthetic frames of the exact export format only.
- `limited` storage, other budget scenarios and other households are not evaluated yet.
