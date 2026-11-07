# Budget scenarios: validation on held-out weeks 78-103 (SYNTHETIC budgets)

| Scenario | Budget column | Definition |
|---|---|---|
| normal (default) | `normal_budget_usd` | P90 of normal-week cost, weeks 0-77, rounded up |
| challenging | `challenging_budget_usd` (= `calibrated_budget_usd`) | P75 of normal-week cost, weeks 0-77, rounded up |
| tight | `tight_budget_usd` (= `original_budget_usd`) | original budget from households.csv |

`all` = every held-out week; `normal` = held-out weeks without guest/away/holiday flag.
Cost = recorded demand x regional price (diagnostic, not RL spending). Calibration never uses weeks 78-103.

| region_id | scenario | over_pct_all | over_pct_normal | mean_ratio_all | mean_ratio_normal | mean_budget_usd |
|---|---|---|---|---|---|---|
| atlanta_ga | normal | 18.97 | 10.90 | 0.88 | 0.87 | 167.43 |
| atlanta_ga | challenging | 34.62 | 26.79 | 0.95 | 0.94 | 155.27 |
| atlanta_ga | tight | 80.26 | 80.22 | 1.17 | 1.15 | 126.33 |
| chicago_il | normal | 19.23 | 11.06 | 0.88 | 0.87 | 178.30 |
| chicago_il | challenging | 34.74 | 26.95 | 0.95 | 0.94 | 165.13 |
| chicago_il | tight | 88.85 | 90.34 | 1.25 | 1.23 | 126.33 |
| durham_nc | normal | 19.10 | 11.06 | 0.88 | 0.87 | 165.70 |
| durham_nc | challenging | 34.36 | 26.79 | 0.95 | 0.94 | 153.77 |
| durham_nc | tight | 78.97 | 78.66 | 1.16 | 1.14 | 126.33 |
| new_york_ny | normal | 19.49 | 11.53 | 0.88 | 0.87 | 205.27 |
| new_york_ny | challenging | 35.13 | 27.41 | 0.95 | 0.94 | 190.27 |
| new_york_ny | tight | 96.15 | 99.22 | 1.44 | 1.42 | 126.33 |
| san_francisco_ca | normal | 18.97 | 10.90 | 0.88 | 0.87 | 208.23 |
| san_francisco_ca | challenging | 34.74 | 26.95 | 0.95 | 0.94 | 192.80 |
| san_francisco_ca | tight | 96.15 | 99.22 | 1.46 | 1.44 | 126.33 |
| ALL | normal | 19.15 | 11.09 | 0.88 | 0.87 | 184.99 |
| ALL | challenging | 34.72 | 26.98 | 0.95 | 0.94 | 171.45 |
| ALL | tight | 88.08 | 89.53 | 1.30 | 1.28 | 126.33 |
