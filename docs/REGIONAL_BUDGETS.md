# Regional household budgets (SYNTHETIC)

`data/household_region_budgets.csv` has one weekly budget per household and region
(30 households x 5 regions = 150 rows). **All budgets are synthetic assumptions for the demo, not real household data.**
`households.csv` is unchanged; this file is a separate, additive dataset. Demographics, diets, allergies and demand are not touched.

## Columns
| Column | Meaning |
|---|---|
| `household_id`, `region_id`, `region_name` | Key (same ids as `households.csv` and the regional price file) |
| `original_budget_usd` | `weekly_budget_usd` from `households.csv`, kept for the fixed-budget comparison scenario |
| `calibrated_budget_usd` | Regional budget: P75 of weekly cost of recorded demand, rounded up to whole USD |
| `calibration_percentile`, `calibration_weeks`, `n_calibration_weeks` | How it was calibrated (P75, weeks 0-77, normal weeks only, count used) |
| `calib_mean_cost_usd`, `calib_median_cost_usd`, `calib_p75_cost_usd` | Cost statistics on the calibration weeks |
| `budget_ratio_to_original` | `calibrated_budget_usd / original_budget_usd` |
| `budget_source` | Always `synthetic` |

## Scenarios (columns appended; the P75 columns above are preserved unchanged)
| Scenario | Column | Definition |
|---|---|---|
| normal (default) | `normal_budget_usd` | P90 of normal-week cost, weeks 0-77, rounded up |
| challenging | `challenging_budget_usd` | P75 (same as `calibrated_budget_usd`) |
| tight | `tight_budget_usd` | original household budget (same as `original_budget_usd`) |

`calib_p90_cost_usd` holds the P90 cost used for the normal scenario. Validation per scenario and region:
`results/regional_budgets/scenarios.md`.

## Method
- Weekly cost = sum over products of `demand_units` x regional `price_usd` (cost of recorded demand, not RL spending).
- Calibration: weeks 0-77, excluding any week with `guest_week`, `away_week` or `holiday_week` set.
- Validation: held-out weeks 78-103, never used for calibration. Reported with and without special weeks.
- Two scenarios are available for later steps: `original_budget_usd` (fixed) and `calibrated_budget_usd` (regional).

## Reproduce and verify
```bash
python -m src.build_regional_budgets          # rebuilds the CSV and results/regional_budgets/
python -m src.build_regional_budgets --check  # integrity + reproducibility checks, exit 1 on failure
```
Results and the validation tables are in `results/regional_budgets/validation.md`.
