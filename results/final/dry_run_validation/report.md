# DRY RUN of the final pipeline on the VALIDATION episode (not a result)

Pipeline check only, run on validation week 64-76 with the frozen state (manifest hash `cbb36102bebe`). PPO validation numbers are optimistic because the checkpoints were selected on these weeks.
Episodes: [(64, 76)] (spec hash `66807c1116e1`). hh_student, durham_nc, normal budget (synthetic), default storage, stockout multiplier 2.0. Forecasts: Ridge `no_events`, identical for every forecast-dependent method. PPO checkpoints and the Optimization delta are frozen (see `results/final/freeze_manifest.json`). Deterministic replay of one recorded trajectory: no confidence intervals or significance claims. Metrics are reported separately; **no overall winner is declared**.

## Mean over the episodes
| index | cash_spend | net_cost | waste_units | waste_cost | stockout_units | stockout_value | fill_rate_value | reward | protein_coverage | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 732.57 | 730.20 | 17.56 | 46.52 | 61.42 | 230.61 | 0.74 | -16.08 | 1.12 | 17.18 |
| average_rule | 926.15 | 928.99 | 53.56 | 133.56 | 23.28 | 110.05 | 0.88 | -15.32 | 1.44 | 9.77 |
| forecast_rule | 908.77 | 917.99 | 53.89 | 138.59 | 24.43 | 123.22 | 0.86 | -15.53 | 1.43 | 6.08 |
| optimization | 797.69 | 782.92 | 6.78 | 18.14 | 42.68 | 138.11 | 0.85 | -14.12 | 1.44 | 9.39 |
| ppo_forecast_mean | 918.08 | 913.35 | 33.76 | 101.19 | 24.60 | 93.79 | 0.90 | -14.68 | 1.48 | 10.97 |
| ppo_forecast_seed0 | 906.24 | 899.69 | 28.38 | 80.62 | 24.73 | 91.49 | 0.90 | -14.44 | 1.49 | 16.18 |
| ppo_forecast_seed1 | 919.84 | 923.05 | 39.25 | 119.36 | 24.24 | 97.69 | 0.89 | -14.91 | 1.47 | 5.57 |
| ppo_forecast_seed2 | 928.15 | 917.30 | 33.64 | 103.59 | 24.84 | 92.19 | 0.90 | -14.69 | 1.49 | 11.16 |
| ppo_no_forecast_mean | 917.94 | 911.08 | 32.89 | 97.50 | 24.90 | 94.66 | 0.89 | -14.67 | 1.49 | 12.98 |
| ppo_no_forecast_seed0 | 913.07 | 901.35 | 26.86 | 74.73 | 23.47 | 83.03 | 0.91 | -14.23 | 1.52 | 15.59 |
| ppo_no_forecast_seed1 | 929.50 | 928.13 | 39.63 | 119.89 | 25.22 | 99.59 | 0.89 | -15.03 | 1.47 | 12.31 |
| ppo_no_forecast_seed2 | 911.24 | 903.75 | 32.19 | 97.87 | 26.01 | 101.36 | 0.89 | -14.75 | 1.47 | 11.06 |

## Rank of the six methods on each metric (best to worst)
| index | #1 | #2 | #3 | #4 | #5 | #6 |
|---|---|---|---|---|---|---|
| cash_spend (lower better) | Fixed Rule 732.57 | Optimization 797.69 | Forecast Rule 908.77 | PPO without forecast 917.94 | PPO with forecast 918.08 | Average Rule 926.15 |
| net_cost (lower better) | Fixed Rule 730.20 | Optimization 782.92 | PPO without forecast 911.08 | PPO with forecast 913.35 | Forecast Rule 917.99 | Average Rule 928.99 |
| waste_units (lower better) | Optimization 6.78 | Fixed Rule 17.56 | PPO without forecast 32.89 | PPO with forecast 33.76 | Average Rule 53.56 | Forecast Rule 53.89 |
| waste_cost (lower better) | Optimization 18.14 | Fixed Rule 46.52 | PPO without forecast 97.50 | PPO with forecast 101.19 | Average Rule 133.56 | Forecast Rule 138.59 |
| stockout_units (lower better) | Average Rule 23.28 | Forecast Rule 24.43 | PPO with forecast 24.60 | PPO without forecast 24.90 | Optimization 42.68 | Fixed Rule 61.42 |
| stockout_value (lower better) | PPO with forecast 93.79 | PPO without forecast 94.66 | Average Rule 110.05 | Forecast Rule 123.22 | Optimization 138.11 | Fixed Rule 230.61 |
| fill_rate_value (higher better) | PPO with forecast 0.90 | PPO without forecast 0.89 | Average Rule 0.88 | Forecast Rule 0.86 | Optimization 0.85 | Fixed Rule 0.74 |
| reward (higher better) | Optimization -14.12 | PPO without forecast -14.67 | PPO with forecast -14.68 | Average Rule -15.32 | Forecast Rule -15.53 | Fixed Rule -16.08 |
| protein_coverage (higher better) | PPO without forecast 1.49 | PPO with forecast 1.48 | Optimization 1.44 | Average Rule 1.44 | Forecast Rule 1.43 | Fixed Rule 1.12 |
| terminal_charge (lower better) | Forecast Rule 6.08 | Optimization 9.39 | Average Rule 9.77 | PPO with forecast 10.97 | PPO without forecast 12.98 | Fixed Rule 17.18 |

## Change relative to the Average Rule (%)
Positive is better for reward, fill rate, protein coverage; negative is better for spend, net cost, waste, stockouts, terminal charge.
| index | cash_spend | net_cost | waste_units | waste_cost | stockout_units | stockout_value | fill_rate_value | reward | protein_coverage | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | -20.9 | -21.4 | -67.2 | -65.2 | +163.8 | +109.5 | -15.3 | -4.9 | -22.5 | +76.0 |
| forecast_rule | -1.9 | -1.2 | +0.6 | +3.8 | +4.9 | +12.0 | -1.7 | -1.3 | -0.5 | -37.7 |
| optimization | -13.9 | -15.7 | -87.3 | -86.4 | +83.3 | +25.5 | -3.6 | +7.8 | +0.4 | -3.9 |
| ppo_forecast_mean | -0.9 | -1.7 | -37.0 | -24.2 | +5.7 | -14.8 | +2.1 | +4.2 | +3.2 | +12.3 |
| ppo_forecast_seed0 | -2.1 | -3.2 | -47.0 | -39.6 | +6.2 | -16.9 | +2.4 | +5.8 | +3.9 | +65.7 |
| ppo_forecast_seed1 | -0.7 | -0.6 | -26.7 | -10.6 | +4.1 | -11.2 | +1.6 | +2.7 | +2.4 | -43.0 |
| ppo_forecast_seed2 | +0.2 | -1.3 | -37.2 | -22.4 | +6.7 | -16.2 | +2.3 | +4.1 | +3.4 | +14.3 |
| ppo_no_forecast_mean | -0.9 | -1.9 | -38.6 | -27.0 | +6.9 | -14.0 | +2.0 | +4.2 | +3.4 | +32.9 |
| ppo_no_forecast_seed0 | -1.4 | -3.0 | -49.9 | -44.0 | +0.8 | -24.6 | +3.4 | +7.1 | +5.8 | +59.6 |
| ppo_no_forecast_seed1 | +0.4 | -0.1 | -26.0 | -10.2 | +8.3 | -9.5 | +1.3 | +1.9 | +2.2 | +26.0 |
| ppo_no_forecast_seed2 | -1.6 | -2.7 | -39.9 | -26.7 | +11.7 | -7.9 | +1.1 | +3.7 | +2.4 | +13.2 |

## Episode weeks 64-76
| index | cash_spend | net_cost | waste_units | waste_cost | stockout_units | stockout_value | fill_rate_value | reward | protein_coverage | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 732.57 | 730.20 | 17.56 | 46.52 | 61.42 | 230.61 | 0.74 | -16.08 | 1.12 | 17.18 |
| average_rule | 926.15 | 928.99 | 53.56 | 133.56 | 23.28 | 110.05 | 0.88 | -15.32 | 1.44 | 9.77 |
| forecast_rule | 908.77 | 917.99 | 53.89 | 138.59 | 24.43 | 123.22 | 0.86 | -15.53 | 1.43 | 6.08 |
| optimization | 797.69 | 782.92 | 6.78 | 18.14 | 42.68 | 138.11 | 0.85 | -14.12 | 1.44 | 9.39 |
| ppo_forecast_mean | 918.08 | 913.35 | 33.76 | 101.19 | 24.60 | 93.79 | 0.90 | -14.68 | 1.48 | 10.97 |
| ppo_forecast_seed0 | 906.24 | 899.69 | 28.38 | 80.62 | 24.73 | 91.49 | 0.90 | -14.44 | 1.49 | 16.18 |
| ppo_forecast_seed1 | 919.84 | 923.05 | 39.25 | 119.36 | 24.24 | 97.69 | 0.89 | -14.91 | 1.47 | 5.57 |
| ppo_forecast_seed2 | 928.15 | 917.30 | 33.64 | 103.59 | 24.84 | 92.19 | 0.90 | -14.69 | 1.49 | 11.16 |
| ppo_no_forecast_mean | 917.94 | 911.08 | 32.89 | 97.50 | 24.90 | 94.66 | 0.89 | -14.67 | 1.49 | 12.98 |
| ppo_no_forecast_seed0 | 913.07 | 901.35 | 26.86 | 74.73 | 23.47 | 83.03 | 0.91 | -14.23 | 1.52 | 15.59 |
| ppo_no_forecast_seed1 | 929.50 | 928.13 | 39.63 | 119.89 | 25.22 | 99.59 | 0.89 | -15.03 | 1.47 | 12.31 |
| ppo_no_forecast_seed2 | 911.24 | 903.75 | 32.19 | 97.87 | 26.01 | 101.36 | 0.89 | -14.75 | 1.47 | 11.06 |

## PPO seeds
Individual seeds appear in the tables above (`ppo_*_seed0/1/2`); the `_mean` rows are the mean over the three seeds.
