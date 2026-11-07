# Regional household budgets: P75 (challenging) vs. original (tight) validation (SYNTHETIC)

All budgets are synthetic assumptions for the demo, not real household data.
Calibration: P75 of weekly cost of recorded demand, weeks 0-77, normal weeks only (no guest, away or holiday flag), rounded up to whole USD.
Validation: held-out weeks 78-103 (26 weeks). `all` includes guest/away/holiday weeks; `normal` excludes them.
`calibrated_*` uses `calibrated_budget_usd`; `original_*` uses the fixed budget from households.csv.
The normal (P90) scenario is reported in scenarios.md.
Reproduce: `python -m src.build_regional_budgets`; verify: `python -m src.build_regional_budgets --check`.

## Over-budget rate by region (held-out weeks)
| region_id | n_weeks | calibrated_over_pct_all | calibrated_over_pct_normal | calibrated_mean_ratio | original_over_pct_all | original_over_pct_normal | original_mean_ratio |
|---|---|---|---|---|---|---|---|
| atlanta_ga | 780 | 34.62 | 26.79 | 0.95 | 80.26 | 80.22 | 1.17 |
| chicago_il | 780 | 34.74 | 26.95 | 0.95 | 88.85 | 90.34 | 1.25 |
| durham_nc | 780 | 34.36 | 26.79 | 0.95 | 78.97 | 78.66 | 1.16 |
| new_york_ny | 780 | 35.13 | 27.41 | 0.95 | 96.15 | 99.22 | 1.44 |
| san_francisco_ca | 780 | 34.74 | 26.95 | 0.95 | 96.15 | 99.22 | 1.46 |

## Calibrated budget / original budget by region
| region_id | min | mean | max |
|---|---|---|---|
| atlanta_ga | 1.04 | 1.23 | 1.52 |
| chicago_il | 1.11 | 1.31 | 1.63 |
| durham_nc | 1.04 | 1.22 | 1.50 |
| new_york_ny | 1.27 | 1.51 | 1.86 |
| san_francisco_ca | 1.29 | 1.53 | 1.90 |

## Over-budget rate by household (held-out weeks, regions pooled)
| household_id | n_weeks | calibrated_over_pct_all | calibrated_over_pct_normal | calibrated_mean_ratio | original_over_pct_all | original_over_pct_normal | original_mean_ratio |
|---|---|---|---|---|---|---|---|
| hh_04 | 130 | 37.69 | 37.14 | 0.93 | 90.77 | 98.10 | 1.35 |
| hh_05 | 130 | 49.23 | 51.43 | 0.92 | 86.92 | 98.10 | 1.41 |
| hh_06 | 130 | 45.38 | 40.00 | 1.00 | 96.15 | 100.00 | 1.47 |
| hh_07 | 130 | 49.23 | 42.61 | 1.01 | 100.00 | 100.00 | 1.51 |
| hh_08 | 130 | 42.31 | 33.33 | 0.99 | 89.23 | 91.43 | 1.33 |
| hh_09 | 130 | 34.62 | 25.00 | 0.93 | 71.54 | 73.00 | 1.14 |
| hh_10 | 130 | 23.08 | 9.52 | 0.96 | 80.00 | 75.24 | 1.17 |
| hh_11 | 130 | 46.15 | 42.86 | 0.98 | 92.31 | 100.00 | 1.65 |
| hh_12 | 130 | 42.31 | 37.50 | 1.02 | 95.38 | 95.00 | 1.28 |
| hh_13 | 130 | 50.00 | 36.84 | 0.96 | 89.23 | 90.53 | 1.38 |
| hh_14 | 130 | 38.46 | 23.81 | 1.01 | 95.38 | 94.29 | 1.38 |
| hh_15 | 130 | 15.38 | 9.09 | 0.89 | 80.00 | 76.36 | 1.18 |
| hh_16 | 130 | 30.77 | 19.05 | 0.95 | 89.23 | 86.67 | 1.29 |
| hh_17 | 130 | 23.08 | 17.39 | 0.92 | 90.00 | 90.43 | 1.21 |
| hh_18 | 130 | 38.46 | 31.82 | 0.93 | 85.38 | 87.27 | 1.21 |
| hh_19 | 130 | 33.08 | 26.36 | 0.97 | 88.46 | 86.36 | 1.19 |
| hh_20 | 130 | 26.15 | 12.73 | 0.97 | 86.92 | 84.55 | 1.25 |
| hh_21 | 130 | 30.77 | 26.09 | 0.95 | 84.62 | 82.61 | 1.26 |
| hh_22 | 130 | 34.62 | 15.79 | 0.94 | 90.00 | 91.58 | 1.32 |
| hh_23 | 130 | 26.92 | 22.73 | 0.91 | 80.77 | 81.82 | 1.16 |
| hh_24 | 130 | 23.08 | 18.18 | 0.85 | 77.69 | 82.73 | 1.16 |
| hh_25 | 130 | 31.54 | 19.09 | 0.99 | 94.62 | 93.64 | 1.35 |
| hh_26 | 130 | 38.46 | 33.33 | 0.98 | 88.46 | 90.48 | 1.23 |
| hh_27 | 130 | 30.00 | 19.05 | 0.94 | 93.85 | 97.14 | 1.35 |
| hh_28 | 130 | 19.23 | 13.64 | 0.90 | 92.31 | 100.00 | 1.28 |
| hh_29 | 130 | 23.08 | 14.29 | 0.89 | 91.54 | 89.52 | 1.34 |
| hh_30 | 130 | 32.31 | 25.71 | 0.95 | 66.92 | 65.71 | 1.09 |
| hh_couple | 130 | 33.85 | 27.62 | 0.94 | 86.15 | 87.62 | 1.24 |
| hh_family | 130 | 46.15 | 38.10 | 1.02 | 96.15 | 95.24 | 1.38 |
| hh_student | 130 | 46.15 | 40.00 | 0.97 | 92.31 | 100.00 | 1.34 |

## Worst 10 household-region pairs (calibrated, normal weeks)
| household_id | region_id | n_weeks | calibrated_over_pct_all | calibrated_over_pct_normal | calibrated_mean_ratio | original_over_pct_all | original_over_pct_normal | original_mean_ratio |
|---|---|---|---|---|---|---|---|---|
| hh_05 | atlanta_ga | 26 | 50.00 | 52.38 | 0.92 | 84.62 | 95.24 | 1.27 |
| hh_05 | durham_nc | 26 | 50.00 | 52.38 | 0.93 | 84.62 | 95.24 | 1.26 |
| hh_05 | new_york_ny | 26 | 50.00 | 52.38 | 0.93 | 88.46 | 100.00 | 1.56 |
| hh_05 | san_francisco_ca | 26 | 50.00 | 52.38 | 0.93 | 88.46 | 100.00 | 1.58 |
| hh_05 | chicago_il | 26 | 46.15 | 47.62 | 0.92 | 88.46 | 100.00 | 1.35 |
| hh_07 | atlanta_ga | 26 | 50.00 | 43.48 | 1.01 | 100.00 | 100.00 | 1.37 |
| hh_07 | new_york_ny | 26 | 50.00 | 43.48 | 1.01 | 100.00 | 100.00 | 1.68 |
| hh_07 | durham_nc | 26 | 50.00 | 43.48 | 1.01 | 100.00 | 100.00 | 1.35 |
| hh_07 | chicago_il | 26 | 50.00 | 43.48 | 1.01 | 100.00 | 100.00 | 1.45 |
| hh_11 | atlanta_ga | 26 | 46.15 | 42.86 | 0.97 | 92.31 | 100.00 | 1.48 |
