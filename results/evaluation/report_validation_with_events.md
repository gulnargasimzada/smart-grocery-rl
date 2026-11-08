# Policy evaluation: validation split

hh_student, durham_nc, normal budget (synthetic), default storage; 1 episode(s) of 13 weeks, starting in weeks [64] (spec hash `66807c1116e1`).
Deterministic, paired replay of the recorded demand and prices: every policy sees the same episodes. This is one recorded trajectory, so the episodes are not independent samples; no confidence intervals or significance claims are made. Simulated results.
`reward` is the training signal; `cash_spend` is money paid; `net_cost` = consumed + waste + terminal charge; `fill_rate_value` is price-weighted; terminal columns value the leftover inventory (basis, credit, charge).

## Mean per 13-week episode
| policy | reward | cash_spend | net_cost | waste_cost | waste_units | stockout_units | stockout_value | stockout_days | fill_rate_value | protein_coverage | terminal_basis | terminal_credit | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_nothing | -29.76 | 0.00 | 28.56 | 0.21 | 0.05 | 227.80 | 870.08 | 3173.00 | 0.03 | 0.05 | 0.00 | 0.00 | 0.00 |
| fixed | -16.08 | 732.57 | 730.20 | 46.52 | 17.56 | 61.42 | 230.61 | 1292.00 | 0.74 | 1.12 | 48.12 | 30.93 | 17.18 |
| average_rule | -15.32 | 926.15 | 928.99 | 133.56 | 53.56 | 23.28 | 110.05 | 796.00 | 0.88 | 1.44 | 35.50 | 25.73 | 9.77 |
| forecast_rule | -15.55 | 907.78 | 915.37 | 138.08 | 54.79 | 24.53 | 125.50 | 844.00 | 0.86 | 1.43 | 27.22 | 20.98 | 6.24 |

## Per episode
| policy | start_week | end_week | reward | cash_spend | net_cost | waste_cost | waste_units | stockout_units | stockout_value | stockout_days | fill_rate_value | protein_coverage | terminal_basis | terminal_credit | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_nothing | 64 | 76 | -29.76 | 0.00 | 28.56 | 0.21 | 0.05 | 227.80 | 870.08 | 3173.00 | 0.03 | 0.05 | 0.00 | 0.00 | 0.00 |
| fixed | 64 | 76 | -16.08 | 732.57 | 730.20 | 46.52 | 17.56 | 61.42 | 230.61 | 1292.00 | 0.74 | 1.12 | 48.12 | 30.93 | 17.18 |
| average_rule | 64 | 76 | -15.32 | 926.15 | 928.99 | 133.56 | 53.56 | 23.28 | 110.05 | 796.00 | 0.88 | 1.44 | 35.50 | 25.73 | 9.77 |
| forecast_rule | 64 | 76 | -15.55 | 907.78 | 915.37 | 138.08 | 54.79 | 24.53 | 125.50 | 844.00 | 0.86 | 1.43 | 27.22 | 20.98 | 6.24 |

- forecast weeks 64-76: all Ridge predictions (walk-forward / trained before the week) (with_events)
