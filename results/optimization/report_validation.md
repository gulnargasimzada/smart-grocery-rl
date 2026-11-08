# Optimization vs. baselines: validation split

hh_student, durham_nc, normal budget (synthetic), default storage; 1 episode(s) of 13 weeks starting in weeks [64] (spec hash `66807c1116e1`, same specs as `src.evaluate`). Forecasts: Ridge, `no_events`.
Optimization: weekly MILP, delta = 0.4 (tuned on the training windows only). Deterministic paired replay of one recorded trajectory: no confidence intervals or significance claims. Simulated results. Test weeks 78-103 were not run.

## Mean per 13-week episode
| policy | reward | cash_spend | net_cost | waste_cost | waste_units | stockout_units | stockout_value | stockout_days | fill_rate_value | protein_coverage | terminal_basis | terminal_credit | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_nothing | -29.76 | 0.00 | 28.56 | 0.21 | 0.05 | 227.80 | 870.08 | 3173.00 | 0.03 | 0.05 | 0.00 | 0.00 | 0.00 |
| fixed | -16.08 | 732.57 | 730.20 | 46.52 | 17.56 | 61.42 | 230.61 | 1292.00 | 0.74 | 1.12 | 48.12 | 30.93 | 17.18 |
| average_rule | -15.32 | 926.15 | 928.99 | 133.56 | 53.56 | 23.28 | 110.05 | 796.00 | 0.88 | 1.44 | 35.50 | 25.73 | 9.77 |
| forecast_rule | -15.53 | 908.77 | 917.99 | 138.59 | 53.89 | 24.43 | 123.22 | 835.00 | 0.86 | 1.43 | 25.43 | 19.35 | 6.08 |
| optimization | -14.12 | 797.69 | 782.92 | 18.14 | 6.78 | 42.68 | 138.11 | 730.00 | 0.85 | 1.44 | 52.73 | 43.34 | 9.39 |

## Per episode
| policy | start_week | end_week | reward | cash_spend | net_cost | waste_cost | waste_units | stockout_units | stockout_value | stockout_days | fill_rate_value | protein_coverage | terminal_basis | terminal_credit | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_nothing | 64 | 76 | -29.76 | 0.00 | 28.56 | 0.21 | 0.05 | 227.80 | 870.08 | 3173.00 | 0.03 | 0.05 | 0.00 | 0.00 | 0.00 |
| fixed | 64 | 76 | -16.08 | 732.57 | 730.20 | 46.52 | 17.56 | 61.42 | 230.61 | 1292.00 | 0.74 | 1.12 | 48.12 | 30.93 | 17.18 |
| average_rule | 64 | 76 | -15.32 | 926.15 | 928.99 | 133.56 | 53.56 | 23.28 | 110.05 | 796.00 | 0.88 | 1.44 | 35.50 | 25.73 | 9.77 |
| forecast_rule | 64 | 76 | -15.53 | 908.77 | 917.99 | 138.59 | 53.89 | 24.43 | 123.22 | 835.00 | 0.86 | 1.43 | 25.43 | 19.35 | 6.08 |
| optimization | 64 | 76 | -14.12 | 797.69 | 782.92 | 18.14 | 6.78 | 42.68 | 138.11 | 730.00 | 0.85 | 1.44 | 52.73 | 43.34 | 9.39 |

## Environment pipeline cuts on the optimization orders (units removed; 0 = order already feasible)
| start_week | rounding | stock_cap | storage | budget | max_budget_use |
|---|---|---|---|---|---|
| 64.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

- forecast weeks 64-76: all Ridge predictions (walk-forward / trained before the week)
