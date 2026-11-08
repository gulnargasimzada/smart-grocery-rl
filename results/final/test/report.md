# Held-out TEST evaluation

One-time held-out run with the frozen state (manifest hash `cbb36102bebe`, base commit `2f5a644`).
Episodes: [(78, 90), (91, 103)] (spec hash `94911ee3f75b`). hh_student, durham_nc, normal budget (synthetic), default storage, stockout multiplier 2.0. Forecasts: Ridge `no_events`, identical for every forecast-dependent method. PPO checkpoints and the Optimization delta are frozen (see `results/final/freeze_manifest.json`). Deterministic replay of one recorded trajectory: no confidence intervals or significance claims. Metrics are reported separately; **no overall winner is declared**.

## Mean over the episodes
| index | cash_spend | net_cost | waste_units | waste_cost | stockout_units | stockout_value | fill_rate_value | reward | protein_coverage | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 731.73 | 716.72 | 23.75 | 58.84 | 59.09 | 231.93 | 0.73 | -16.27 | 1.04 | 24.59 |
| average_rule | 912.61 | 900.11 | 50.01 | 124.14 | 23.27 | 109.68 | 0.87 | -15.19 | 1.32 | 19.88 |
| forecast_rule | 927.31 | 913.99 | 50.33 | 128.10 | 22.65 | 99.91 | 0.88 | -15.12 | 1.30 | 22.66 |
| optimization | 796.57 | 778.56 | 12.34 | 31.23 | 41.56 | 128.63 | 0.85 | -14.09 | 1.32 | 14.93 |
| ppo_forecast_mean | 911.39 | 895.05 | 36.69 | 97.19 | 23.59 | 94.57 | 0.89 | -14.71 | 1.36 | 25.89 |
| ppo_forecast_seed0 | 902.42 | 882.96 | 33.10 | 79.50 | 22.79 | 87.93 | 0.90 | -14.37 | 1.37 | 25.97 |
| ppo_forecast_seed1 | 899.94 | 885.24 | 38.01 | 101.11 | 24.70 | 104.42 | 0.88 | -14.84 | 1.34 | 22.08 |
| ppo_forecast_seed2 | 931.80 | 916.95 | 38.95 | 110.97 | 23.29 | 91.37 | 0.89 | -14.92 | 1.36 | 29.61 |
| ppo_no_forecast_mean | 904.91 | 892.24 | 37.10 | 101.96 | 25.21 | 103.72 | 0.88 | -14.91 | 1.34 | 26.46 |
| ppo_no_forecast_seed0 | 898.12 | 884.60 | 34.38 | 93.51 | 25.64 | 103.29 | 0.88 | -14.80 | 1.35 | 27.12 |
| ppo_no_forecast_seed1 | 914.31 | 901.23 | 38.03 | 104.40 | 24.29 | 100.64 | 0.88 | -14.95 | 1.34 | 29.61 |
| ppo_no_forecast_seed2 | 902.29 | 890.88 | 38.88 | 107.97 | 25.71 | 107.23 | 0.88 | -14.99 | 1.34 | 22.66 |

## Rank of the six methods on each metric (best to worst)
| index | #1 | #2 | #3 | #4 | #5 | #6 |
|---|---|---|---|---|---|---|
| cash_spend (lower better) | Fixed Rule 731.73 | Optimization 796.57 | PPO without forecast 904.91 | PPO with forecast 911.39 | Average Rule 912.61 | Forecast Rule 927.31 |
| net_cost (lower better) | Fixed Rule 716.72 | Optimization 778.56 | PPO without forecast 892.24 | PPO with forecast 895.05 | Average Rule 900.11 | Forecast Rule 913.99 |
| waste_units (lower better) | Optimization 12.34 | Fixed Rule 23.75 | PPO with forecast 36.69 | PPO without forecast 37.10 | Average Rule 50.01 | Forecast Rule 50.33 |
| waste_cost (lower better) | Optimization 31.23 | Fixed Rule 58.84 | PPO with forecast 97.19 | PPO without forecast 101.96 | Average Rule 124.14 | Forecast Rule 128.10 |
| stockout_units (lower better) | Forecast Rule 22.65 | Average Rule 23.27 | PPO with forecast 23.59 | PPO without forecast 25.21 | Optimization 41.56 | Fixed Rule 59.09 |
| stockout_value (lower better) | PPO with forecast 94.57 | Forecast Rule 99.91 | PPO without forecast 103.72 | Average Rule 109.68 | Optimization 128.63 | Fixed Rule 231.93 |
| fill_rate_value (higher better) | PPO with forecast 0.89 | Forecast Rule 0.88 | PPO without forecast 0.88 | Average Rule 0.87 | Optimization 0.85 | Fixed Rule 0.73 |
| reward (higher better) | Optimization -14.09 | PPO with forecast -14.71 | PPO without forecast -14.91 | Forecast Rule -15.12 | Average Rule -15.19 | Fixed Rule -16.27 |
| protein_coverage (higher better) | PPO with forecast 1.36 | PPO without forecast 1.34 | Optimization 1.32 | Average Rule 1.32 | Forecast Rule 1.30 | Fixed Rule 1.04 |
| terminal_charge (lower better) | Optimization 14.93 | Average Rule 19.88 | Forecast Rule 22.66 | Fixed Rule 24.59 | PPO with forecast 25.89 | PPO without forecast 26.46 |

## Change relative to the Average Rule (%)
Positive is better for reward, fill rate, protein coverage; negative is better for spend, net cost, waste, stockouts, terminal charge.
| index | cash_spend | net_cost | waste_units | waste_cost | stockout_units | stockout_value | fill_rate_value | reward | protein_coverage | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | -19.8 | -20.4 | -52.5 | -52.6 | +154.0 | +111.5 | -16.1 | -7.1 | -20.6 | +23.7 |
| forecast_rule | +1.6 | +1.5 | +0.6 | +3.2 | -2.7 | -8.9 | +1.3 | +0.5 | -1.0 | +14.0 |
| optimization | -12.7 | -13.5 | -75.3 | -74.8 | +78.6 | +17.3 | -2.4 | +7.3 | +0.6 | -24.9 |
| ppo_forecast_mean | -0.1 | -0.6 | -26.6 | -21.7 | +1.4 | -13.8 | +2.0 | +3.2 | +3.1 | +30.2 |
| ppo_forecast_seed0 | -1.1 | -1.9 | -33.8 | -36.0 | -2.1 | -19.8 | +2.9 | +5.4 | +4.1 | +30.7 |
| ppo_forecast_seed1 | -1.4 | -1.7 | -24.0 | -18.6 | +6.1 | -4.8 | +0.8 | +2.3 | +1.8 | +11.0 |
| ppo_forecast_seed2 | +2.1 | +1.9 | -22.1 | -10.6 | +0.1 | -16.7 | +2.5 | +1.8 | +3.5 | +48.9 |
| ppo_no_forecast_mean | -0.8 | -0.9 | -25.8 | -17.9 | +8.4 | -5.4 | +0.9 | +1.8 | +2.2 | +33.1 |
| ppo_no_forecast_seed0 | -1.6 | -1.7 | -31.3 | -24.7 | +10.2 | -5.8 | +0.9 | +2.5 | +2.7 | +36.4 |
| ppo_no_forecast_seed1 | +0.2 | +0.1 | -24.0 | -15.9 | +4.4 | -8.2 | +1.3 | +1.6 | +2.1 | +48.9 |
| ppo_no_forecast_seed2 | -1.1 | -1.0 | -22.3 | -13.0 | +10.5 | -2.2 | +0.4 | +1.3 | +1.7 | +14.0 |

## Episode weeks 78-90
| index | cash_spend | net_cost | waste_units | waste_cost | stockout_units | stockout_value | fill_rate_value | reward | protein_coverage | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 745.25 | 720.65 | 23.30 | 60.93 | 55.23 | 216.91 | 0.74 | -15.86 | 1.04 | 37.01 |
| average_rule | 911.76 | 887.79 | 55.95 | 139.42 | 22.38 | 112.34 | 0.87 | -15.07 | 1.30 | 19.89 |
| forecast_rule | 928.48 | 903.21 | 56.78 | 144.69 | 22.50 | 101.49 | 0.88 | -14.98 | 1.27 | 22.74 |
| optimization | 807.56 | 779.67 | 14.26 | 33.68 | 32.83 | 110.03 | 0.87 | -13.58 | 1.32 | 22.48 |
| ppo_forecast_mean | 921.50 | 890.97 | 36.04 | 99.83 | 21.29 | 85.28 | 0.90 | -14.38 | 1.35 | 35.97 |
| ppo_forecast_seed0 | 905.13 | 874.13 | 32.44 | 80.39 | 21.61 | 83.04 | 0.90 | -14.09 | 1.36 | 36.89 |
| ppo_forecast_seed1 | 918.87 | 887.81 | 38.20 | 106.69 | 20.94 | 90.41 | 0.89 | -14.47 | 1.33 | 33.71 |
| ppo_forecast_seed2 | 940.50 | 910.96 | 37.49 | 112.40 | 21.30 | 82.39 | 0.90 | -14.57 | 1.37 | 37.32 |
| ppo_no_forecast_mean | 920.47 | 894.75 | 37.15 | 108.50 | 22.15 | 89.51 | 0.89 | -14.54 | 1.35 | 36.13 |
| ppo_no_forecast_seed0 | 907.27 | 878.21 | 32.23 | 87.16 | 22.06 | 84.95 | 0.90 | -14.19 | 1.37 | 37.01 |
| ppo_no_forecast_seed1 | 933.94 | 909.19 | 39.14 | 116.86 | 21.31 | 88.08 | 0.90 | -14.69 | 1.34 | 40.35 |
| ppo_no_forecast_seed2 | 920.19 | 896.87 | 40.09 | 121.47 | 23.09 | 95.49 | 0.89 | -14.73 | 1.33 | 31.02 |

## Episode weeks 91-103
| index | cash_spend | net_cost | waste_units | waste_cost | stockout_units | stockout_value | fill_rate_value | reward | protein_coverage | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 718.21 | 712.80 | 24.19 | 56.76 | 62.95 | 246.95 | 0.72 | -16.68 | 1.05 | 12.17 |
| average_rule | 913.46 | 912.43 | 44.07 | 108.86 | 24.15 | 107.01 | 0.88 | -15.31 | 1.33 | 19.88 |
| forecast_rule | 926.14 | 924.77 | 43.87 | 111.50 | 22.79 | 98.33 | 0.89 | -15.25 | 1.33 | 22.59 |
| optimization | 785.59 | 777.44 | 10.42 | 28.78 | 50.29 | 147.23 | 0.83 | -14.60 | 1.33 | 7.39 |
| ppo_forecast_mean | 901.27 | 899.13 | 37.33 | 94.55 | 25.90 | 103.87 | 0.88 | -15.04 | 1.36 | 15.80 |
| ppo_forecast_seed0 | 899.71 | 891.78 | 33.77 | 78.60 | 23.97 | 92.83 | 0.90 | -14.66 | 1.38 | 15.06 |
| ppo_forecast_seed1 | 881.00 | 882.66 | 37.82 | 95.53 | 28.45 | 118.42 | 0.87 | -15.20 | 1.35 | 10.45 |
| ppo_forecast_seed2 | 923.11 | 922.94 | 40.40 | 109.54 | 25.28 | 100.35 | 0.89 | -15.27 | 1.36 | 21.89 |
| ppo_no_forecast_mean | 889.35 | 889.72 | 37.04 | 95.43 | 28.27 | 117.94 | 0.87 | -15.29 | 1.34 | 16.80 |
| ppo_no_forecast_seed0 | 888.96 | 891.00 | 36.53 | 99.86 | 29.21 | 121.64 | 0.86 | -15.41 | 1.33 | 17.23 |
| ppo_no_forecast_seed1 | 894.67 | 893.27 | 36.92 | 91.94 | 27.27 | 113.20 | 0.87 | -15.21 | 1.34 | 18.88 |
| ppo_no_forecast_seed2 | 884.40 | 884.89 | 37.67 | 94.48 | 28.32 | 118.97 | 0.87 | -15.25 | 1.34 | 14.29 |

## PPO seeds
Individual seeds appear in the tables above (`ppo_*_seed0/1/2`); the `_mean` rows are the mean over the three seeds.
