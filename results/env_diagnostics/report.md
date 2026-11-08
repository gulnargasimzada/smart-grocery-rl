# GroceryEnv v0 diagnostics (training and validation weeks only)

hh_student, durham_nc, normal budget ($75/week, synthetic), default storage, 13-week episodes.
Training windows start in weeks [4, 17, 30, 43, 51] (end by week 63); validation window starts in week 64 (weeks 64-76).
No test weeks (78-103) are used. All policies run the same episodes (recorded-demand replay, deterministic). Simulated results.
`reward` is the training signal; `spend` is actual cash spent; `net_cost` = consumed + waste + terminal charge.
`fill_rate_value` = 1 - stockout value / demand value (price-weighted). `protein_coverage` = mean weekly protein ratio vs. target.

## Policy comparison, train windows (mean per 13-week episode)
| policy | reward | spend | waste_cost | waste_units | stockout_units | stockout_value | stockout_days | fill_rate_value | protein_coverage | net_cost | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_nothing | -28.12 | 0.00 | 0.23 | 0.04 | 213.34 | 808.60 | 3199.40 | 0.03 | 0.04 | 25.62 | 0.00 |
| fixed | -15.41 | 698.76 | 57.80 | 22.49 | 53.77 | 214.33 | 1248.80 | 0.74 | 1.04 | 693.27 | 17.25 |
| fixed_rounded | -18.20 | 525.55 | 53.44 | 23.90 | 80.52 | 352.88 | 2016.20 | 0.58 | 0.76 | 537.12 | 2.94 |
| average_rule | -14.82 | 906.10 | 146.18 | 51.32 | 23.03 | 101.55 | 705.00 | 0.88 | 1.28 | 897.85 | 18.06 |
| average_rule_1.5 | -15.94 | 927.81 | 200.23 | 82.36 | 27.07 | 134.29 | 949.40 | 0.84 | 1.20 | 912.66 | 11.66 |
| average_rule_2.0 | -16.93 | 928.60 | 236.83 | 98.48 | 32.45 | 169.01 | 1131.00 | 0.80 | 1.13 | 910.87 | 9.26 |
| naive_ref | -17.95 | 539.40 | 54.85 | 24.10 | 76.23 | 342.01 | 1973.20 | 0.59 | 0.78 | 549.49 | 3.17 |

## Policy comparison, val windows (mean per 13-week episode)
| policy | reward | spend | waste_cost | waste_units | stockout_units | stockout_value | stockout_days | fill_rate_value | protein_coverage | net_cost | terminal_charge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_nothing | -29.76 | 0.00 | 0.21 | 0.05 | 227.80 | 870.08 | 3173.00 | 0.03 | 0.05 | 28.56 | 0.00 |
| fixed | -16.08 | 732.57 | 46.52 | 17.56 | 61.42 | 230.61 | 1292.00 | 0.74 | 1.12 | 730.20 | 17.18 |
| fixed_rounded | -19.10 | 552.48 | 50.56 | 20.81 | 90.16 | 381.25 | 2034.00 | 0.58 | 0.80 | 567.96 | 1.49 |
| average_rule | -15.32 | 926.15 | 133.56 | 53.56 | 23.28 | 110.05 | 796.00 | 0.88 | 1.44 | 928.99 | 9.77 |
| average_rule_1.5 | -16.72 | 936.55 | 190.96 | 82.35 | 30.38 | 159.04 | 1091.00 | 0.82 | 1.35 | 933.70 | 5.54 |
| average_rule_2.0 | -17.94 | 927.58 | 221.32 | 100.04 | 39.21 | 214.37 | 1319.00 | 0.76 | 1.22 | 911.52 | 7.88 |
| naive_ref | -18.62 | 572.36 | 51.54 | 22.14 | 79.66 | 367.49 | 1954.00 | 0.59 | 0.85 | 583.21 | 2.38 |

`fixed` = standing order of the mean weekly demand of weeks 4-63; `fixed_rounded` and `naive_ref` are diagnostic variants where the environment rounds fractional orders to the nearest package (small items are never bought). `average_rule_1.5/2.0` are cover sensitivity rows; the baseline definition stays at 1.0 week of cover.

## Stockouts by product: naive_ref (sum over the 6 train+validation episodes, top 12 by stockout value)
Total demand value $5070, stockout value $2078, fill rate (value) 59.0%; stockout units 76.8 per episode (mixed purchase units).
| product | demand_units | stockout_units | fill_rate | stockout_value_usd | requested_units | cut_by_rounding_units | cut_by_stock_cap | cut_by_storage | cut_by_budget |
|---|---|---|---|---|---|---|---|---|---|
| breakfast_cereal | 32.85 | 28.90 | 0.12 | 195.80 | 32.58 | -28.58 | 1.00 | 0.00 | 0.00 |
| rice | 21.67 | 21.01 | 0.03 | 185.92 | 21.65 | -21.65 | 0.00 | 0.00 | 0.00 |
| salmon_fillet | 19.41 | 18.88 | 0.03 | 159.85 | 19.38 | -19.38 | 0.00 | 0.00 | 0.00 |
| olive_oil | 11.81 | 11.44 | 0.03 | 158.76 | 11.86 | -11.86 | 0.00 | 0.00 | 0.00 |
| eggs | 18.12 | 17.50 | 0.03 | 143.80 | 18.13 | -18.13 | 0.00 | 0.00 | 0.00 |
| ground_beef | 16.63 | 16.12 | 0.03 | 137.35 | 16.58 | -16.58 | 0.00 | 0.00 | 0.00 |
| cheddar_cheese | 32.21 | 23.63 | 0.27 | 110.75 | 32.39 | -17.39 | 6.00 | 0.00 | 0.00 |
| oats | 21.24 | 20.63 | 0.03 | 90.36 | 20.96 | -20.96 | 0.00 | 0.00 | 0.00 |
| avocados | 16.80 | 16.30 | 0.03 | 88.02 | 16.55 | -16.55 | 0.00 | 0.00 | 0.00 |
| bread | 13.87 | 13.42 | 0.03 | 80.15 | 13.84 | -13.84 | 0.00 | 0.00 | 0.00 |
| tofu | 17.37 | 16.77 | 0.03 | 53.00 | 17.50 | -17.50 | 0.00 | 0.00 | 0.00 |
| pasta | 19.67 | 19.02 | 0.03 | 51.49 | 19.47 | -19.47 | 0.00 | 0.00 | 0.00 |

`cut_by_*` = requested units removed by that step of the order pipeline (rounding is signed: negative = rounded down).

## Stockouts by product: average_rule (sum over the 6 train+validation episodes, top 12 by stockout value)
Total demand value $5070, stockout value $618, fill rate (value) 87.8%; stockout units 23.1 per episode (mixed purchase units).
| product | demand_units | stockout_units | fill_rate | stockout_value_usd | requested_units | cut_by_rounding_units | cut_by_stock_cap | cut_by_storage | cut_by_budget |
|---|---|---|---|---|---|---|---|---|---|
| olive_oil | 11.81 | 7.34 | 0.38 | 100.80 | 58.00 | 0.00 | 0.00 | 0.00 | 52.00 |
| avocados | 16.80 | 9.84 | 0.41 | 53.91 | 78.00 | 0.00 | 0.00 | 0.00 | 27.00 |
| spinach | 67.92 | 19.57 | 0.71 | 47.97 | 90.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| butter | 7.46 | 6.07 | 0.19 | 38.74 | 68.00 | 0.00 | 0.00 | 0.00 | 66.00 |
| frozen_vegetables | 7.86 | 6.05 | 0.23 | 36.87 | 68.00 | 0.00 | 0.00 | 0.00 | 65.00 |
| bread | 13.87 | 5.72 | 0.59 | 34.56 | 46.00 | 0.00 | 0.00 | 0.00 | 36.00 |
| ground_beef | 16.63 | 3.94 | 0.76 | 33.79 | 38.00 | 0.00 | 0.00 | 0.00 | 23.00 |
| salmon_fillet | 19.41 | 3.00 | 0.85 | 26.32 | 34.00 | 0.00 | 0.00 | 0.00 | 16.00 |
| eggs | 18.12 | 3.09 | 0.83 | 25.66 | 39.00 | 0.00 | 0.00 | 0.00 | 17.00 |
| potatoes | 11.28 | 6.63 | 0.41 | 23.90 | 56.00 | 0.00 | 0.00 | 0.00 | 47.00 |
| rice | 21.67 | 2.30 | 0.89 | 20.71 | 29.00 | 0.00 | 0.00 | 0.00 | 7.00 |
| orange_juice | 50.11 | 7.03 | 0.86 | 19.55 | 59.00 | 0.00 | 0.00 | 0.00 | 0.00 |

`cut_by_*` = requested units removed by that step of the order pipeline (rounding is signed: negative = rounded down).

## Stockouts by product: fixed (sum over the 6 train+validation episodes, top 12 by stockout value)
Total demand value $5070, stockout value $1302, fill rate (value) 74.3%; stockout units 55.0 per episode (mixed purchase units).
| product | demand_units | stockout_units | fill_rate | stockout_value_usd | requested_units | cut_by_rounding_units | cut_by_stock_cap | cut_by_storage | cut_by_budget |
|---|---|---|---|---|---|---|---|---|---|
| avocados | 16.80 | 15.36 | 0.09 | 83.08 | 12.00 | 0.00 | 0.00 | 0.00 | 4.00 |
| eggs | 18.12 | 9.83 | 0.46 | 80.59 | 18.00 | 0.00 | 0.00 | 0.00 | 3.00 |
| ground_beef | 16.63 | 9.21 | 0.45 | 77.55 | 12.00 | 0.00 | 0.00 | 0.00 | 4.00 |
| olive_oil | 11.81 | 5.47 | 0.54 | 75.78 | 6.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| frozen_pizza | 69.53 | 6.84 | 0.90 | 63.60 | 66.00 | 0.00 | 3.00 | 0.00 | 0.00 |
| spinach | 67.92 | 25.91 | 0.62 | 63.57 | 66.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| rice | 21.67 | 7.13 | 0.67 | 62.77 | 18.00 | 0.00 | 3.00 | 0.00 | 0.00 |
| salmon_fillet | 19.41 | 7.13 | 0.63 | 60.31 | 18.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| chicken_breast | 87.45 | 12.28 | 0.86 | 51.73 | 78.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| bell_peppers | 96.97 | 17.58 | 0.82 | 48.10 | 90.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| breakfast_cereal | 32.85 | 6.36 | 0.81 | 43.33 | 30.00 | 0.00 | 3.00 | 0.00 | 0.00 |
| broccoli | 38.14 | 19.55 | 0.49 | 39.43 | 36.00 | 0.00 | 0.00 | 0.00 | 0.00 |

`cut_by_*` = requested units removed by that step of the order pipeline (rounding is signed: negative = rounded down).

## Cause attribution: stockouts when ONE constraint is relaxed (mean per episode, train+validation)
`budget` = unlimited budget; `stock_cap` = no stock cap; `package` = 0.001 package step; `shelf_life` = 1000x shelf lives; `storage` = unlimited capacity (no thaw/space limits); `all` = everything relaxed. What remains under `all` is demand the policy did not order for (forecast/policy error), not an environment constraint.
| policy | relaxed | stockout_units | stockout_value_usd | fill_rate_value | removed_vs_none_% |
|---|---|---|---|---|---|
| naive_ref | none | 76.80 | 346.26 | 0.59 | 0.00 |
| naive_ref | budget | 76.80 | 346.26 | 0.59 | 0.00 |
| naive_ref | stock_cap | 75.19 | 340.06 | 0.60 | 1.79 |
| naive_ref | package | 15.61 | 47.51 | 0.94 | 86.28 |
| naive_ref | shelf_life | 68.40 | 327.55 | 0.61 | 5.40 |
| naive_ref | storage | 76.80 | 346.26 | 0.59 | 0.00 |
| naive_ref | all | 5.30 | 20.96 | 0.98 | 93.95 |
| fixed | none | 55.04 | 217.04 | 0.74 | 0.00 |
| fixed | budget | 51.46 | 200.51 | 0.76 | 7.61 |
| fixed | stock_cap | 53.43 | 209.51 | 0.75 | 3.47 |
| fixed | package | 19.56 | 62.16 | 0.93 | 71.36 |
| fixed | shelf_life | 39.96 | 178.92 | 0.79 | 17.56 |
| fixed | storage | 55.04 | 217.04 | 0.74 | 0.00 |
| fixed | all | 10.47 | 37.52 | 0.96 | 82.71 |
| average_rule | none | 23.07 | 102.97 | 0.88 | 0.00 |
| average_rule | budget | 11.98 | 34.47 | 0.96 | 66.53 |
| average_rule | stock_cap | 23.07 | 102.97 | 0.88 | 0.00 |
| average_rule | package | 30.57 | 111.34 | 0.87 | -8.13 |
| average_rule | shelf_life | 10.14 | 50.01 | 0.94 | 51.44 |
| average_rule | storage | 23.07 | 102.97 | 0.88 | 0.00 |
| average_rule | all | 25.91 | 99.00 | 0.88 | 3.86 |
