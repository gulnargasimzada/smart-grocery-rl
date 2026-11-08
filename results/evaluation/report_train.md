# Policy evaluation: train split

hh_student, durham_nc, normal budget (synthetic), default storage; 5 episode(s) of 13 weeks, starting in weeks [4, 17, 30, 43, 51] (spec hash `fa084da3f013`).
Deterministic, paired replay of the recorded demand and prices: every policy sees the same episodes. This is one recorded trajectory, so the episodes are not independent samples; no confidence intervals or significance claims are made. Simulated results.
`reward` is the training signal; `cash_spend` is money paid; `net_cost` = consumed + waste + terminal charge; `fill_rate_value` is price-weighted; terminal columns value the leftover inventory (basis, credit, charge).

## Mean per 13-week episode
| policy | reward | cash_spend | net_cost | waste_cost | waste_units | stockout_units | stockout_value | stockout_days | fill_rate_value | protein_coverage | terminal_basis | terminal_credit | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_nothing | -28.12 | 0.00 | 25.62 | 0.23 | 0.04 | 213.34 | 808.60 | 3199.40 | 0.03 | 0.04 | 0.00 | 0.00 | 0.00 |
| fixed | -15.41 | 698.76 | 693.27 | 57.80 | 22.49 | 53.77 | 214.33 | 1248.80 | 0.74 | 1.04 | 48.37 | 31.12 | 17.25 |
| average_rule | -14.82 | 906.10 | 897.85 | 146.18 | 51.32 | 23.03 | 101.55 | 705.00 | 0.88 | 1.28 | 51.92 | 33.87 | 18.06 |
| forecast_rule | -14.77 | 910.82 | 901.75 | 147.35 | 51.19 | 22.23 | 97.73 | 684.00 | 0.88 | 1.28 | 52.09 | 34.70 | 17.39 |

## Per episode
| policy | start_week | end_week | reward | cash_spend | net_cost | waste_cost | waste_units | stockout_units | stockout_value | stockout_days | fill_rate_value | protein_coverage | terminal_basis | terminal_credit | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_nothing | 4 | 16 | -26.63 | 0.00 | 26.45 | 0.38 | 0.05 | 201.40 | 753.14 | 3177.00 | 0.03 | 0.05 | 0.00 | 0.00 | 0.00 |
| buy_nothing | 17 | 29 | -26.76 | 0.00 | 26.28 | 0.00 | 0.00 | 199.94 | 757.69 | 3200.00 | 0.03 | 0.05 | 0.00 | 0.00 | 0.00 |
| buy_nothing | 30 | 42 | -29.75 | 0.00 | 24.32 | 0.00 | 0.00 | 230.70 | 869.96 | 3212.00 | 0.03 | 0.04 | 0.00 | 0.00 | 0.00 |
| buy_nothing | 43 | 55 | -28.32 | 0.00 | 26.33 | 0.77 | 0.14 | 211.58 | 815.99 | 3174.00 | 0.03 | 0.04 | 0.00 | 0.00 | 0.00 |
| buy_nothing | 51 | 63 | -29.13 | 0.00 | 24.73 | 0.00 | 0.00 | 223.09 | 846.24 | 3234.00 | 0.03 | 0.04 | 0.00 | 0.00 | 0.00 |
| fixed | 4 | 16 | -14.07 | 696.00 | 682.65 | 61.07 | 24.13 | 42.65 | 172.45 | 1097.00 | 0.78 | 1.07 | 57.12 | 39.80 | 17.32 |
| fixed | 17 | 29 | -14.40 | 688.99 | 684.92 | 63.80 | 24.48 | 46.38 | 183.86 | 1211.00 | 0.77 | 1.00 | 51.18 | 30.34 | 20.84 |
| fixed | 30 | 42 | -16.10 | 703.54 | 697.81 | 37.50 | 15.93 | 63.05 | 246.53 | 1346.00 | 0.72 | 1.07 | 45.16 | 30.05 | 15.10 |
| fixed | 43 | 55 | -15.70 | 718.91 | 709.62 | 60.43 | 23.47 | 51.91 | 210.77 | 1249.00 | 0.75 | 1.04 | 52.82 | 35.63 | 17.19 |
| fixed | 51 | 63 | -16.79 | 686.39 | 691.35 | 66.21 | 24.45 | 64.83 | 258.01 | 1341.00 | 0.70 | 1.03 | 35.55 | 19.78 | 15.77 |
| average_rule | 4 | 16 | -13.67 | 886.52 | 873.50 | 155.23 | 55.64 | 18.04 | 75.79 | 665.00 | 0.90 | 1.27 | 55.65 | 39.47 | 16.18 |
| average_rule | 17 | 29 | -13.60 | 904.19 | 892.36 | 146.48 | 48.04 | 16.98 | 63.97 | 497.00 | 0.92 | 1.27 | 62.79 | 38.11 | 24.68 |
| average_rule | 30 | 42 | -15.84 | 929.42 | 922.83 | 136.42 | 51.34 | 28.37 | 132.50 | 937.00 | 0.85 | 1.29 | 52.25 | 30.92 | 21.33 |
| average_rule | 43 | 55 | -14.98 | 915.58 | 901.80 | 145.35 | 49.60 | 22.76 | 102.50 | 627.00 | 0.88 | 1.27 | 56.04 | 40.10 | 15.94 |
| average_rule | 51 | 63 | -16.03 | 894.77 | 898.78 | 147.40 | 52.00 | 28.99 | 132.98 | 799.00 | 0.85 | 1.33 | 32.88 | 20.73 | 12.15 |
| forecast_rule | 4 | 16 | -13.64 | 886.87 | 873.28 | 152.46 | 54.26 | 17.70 | 74.78 | 645.00 | 0.90 | 1.27 | 57.74 | 40.04 | 17.70 |
| forecast_rule | 17 | 29 | -13.65 | 899.17 | 888.59 | 147.53 | 49.16 | 17.09 | 67.71 | 528.00 | 0.91 | 1.26 | 61.92 | 36.86 | 25.06 |
| forecast_rule | 30 | 42 | -15.64 | 932.72 | 921.92 | 132.53 | 49.05 | 28.20 | 125.35 | 896.00 | 0.86 | 1.29 | 51.67 | 35.12 | 16.56 |
| forecast_rule | 43 | 55 | -14.92 | 925.22 | 913.53 | 150.74 | 50.64 | 20.87 | 95.01 | 609.00 | 0.89 | 1.27 | 53.32 | 38.02 | 15.30 |
| forecast_rule | 51 | 63 | -15.98 | 910.14 | 911.41 | 153.46 | 52.83 | 27.28 | 125.79 | 742.00 | 0.86 | 1.33 | 35.79 | 23.47 | 12.32 |

- forecast weeks 4-16: weeks [4, 5, 6, 7, 8, 9, 10, 11] use the historical ma4_fallback (4-week moving average), not Ridge; the rest are Ridge (no_events)
- forecast weeks 17-29: all Ridge predictions (walk-forward / trained before the week) (no_events)
- forecast weeks 30-42: all Ridge predictions (walk-forward / trained before the week) (no_events)
- forecast weeks 43-55: all Ridge predictions (walk-forward / trained before the week) (no_events)
- forecast weeks 51-63: all Ridge predictions (walk-forward / trained before the week) (no_events)
