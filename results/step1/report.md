# Step 1: budget vs. price compatibility (diagnostic)

Scope: 30 households x 5 regions x 104 weeks, 36 products.
Cost = sum(demand_units x regional weekly price_usd). This is the cost of the *recorded
demand*, not RL spending or optimised shopping (no stock, waste or constraints).
Reproduce: `python -m src.budget_diagnostic`.

## Overall
- Average weekly cost: $159.20 vs. average budget $126.33 (mean cost/budget = 1.26)
- Household-region-weeks over budget: 86.3%
- Household-region-weeks using under 50% of budget: 2.8%
- Households with at least one over-budget week: 30 of 30
- Mean cost/budget in guest weeks: 1.67 vs. normal weeks: 1.24

## By region
| region_id | avg_cost | budget | avg_ratio | pct_weeks_over | pct_weeks_over_120 | pct_weeks_under_50 | p95_ratio |
|---|---|---|---|---|---|---|---|
| atlanta_ga | 144.10 | 126.33 | 1.14 | 77.82 | 37.34 | 3.46 | 1.55 |
| chicago_il | 153.35 | 126.33 | 1.22 | 86.54 | 51.73 | 3.17 | 1.66 |
| durham_nc | 142.64 | 126.33 | 1.13 | 76.22 | 34.74 | 3.49 | 1.54 |
| new_york_ny | 176.76 | 126.33 | 1.40 | 95.35 | 81.86 | 2.12 | 1.90 |
| san_francisco_ca | 179.15 | 126.33 | 1.42 | 95.58 | 83.49 | 1.96 | 1.94 |

## By persona
| persona | avg_cost | budget | avg_ratio | pct_weeks_over | pct_weeks_over_120 | pct_weeks_under_50 | p95_ratio |
|---|---|---|---|---|---|---|---|
| Athlete | 176.91 | 150.00 | 1.17 | 81.86 | 46.86 | 4.36 | 1.56 |
| Family with children | 279.44 | 222.50 | 1.26 | 86.96 | 56.63 | 1.89 | 1.73 |
| Senior couple | 150.40 | 120.00 | 1.26 | 87.88 | 56.54 | 2.56 | 1.74 |
| Shared student house | 261.41 | 195.00 | 1.34 | 93.08 | 72.60 | 1.54 | 1.79 |
| Single professional | 47.63 | 45.00 | 1.06 | 60.38 | 23.27 | 3.65 | 1.42 |
| Student | 66.10 | 48.00 | 1.38 | 93.62 | 74.27 | 2.62 | 1.92 |
| Vegan household | 82.63 | 70.00 | 1.18 | 82.40 | 48.46 | 4.33 | 1.59 |
| Vegetarian household | 101.60 | 80.00 | 1.25 | 86.09 | 55.19 | 2.63 | 1.74 |
| Working couple | 149.28 | 118.00 | 1.27 | 84.08 | 56.54 | 3.35 | 1.85 |

## Households with the highest cost/budget (all regions pooled)
| household_id | persona | n_people | avg_cost | budget | avg_ratio | pct_weeks_over | pct_weeks_over_120 | pct_weeks_under_50 | p95_ratio |
|---|---|---|---|---|---|---|---|---|---|
| hh_11 | Working couple | 2 | 174.41 | 110.00 | 1.59 | 95.77 | 92.12 | 1.15 | 2.16 |
| hh_06 | Student | 1 | 71.73 | 50.00 | 1.43 | 94.42 | 81.92 | 1.73 | 2.03 |
| hh_07 | Student | 1 | 63.82 | 45.00 | 1.42 | 96.15 | 81.35 | 1.92 | 1.89 |
| hh_05 | Student | 1 | 62.76 | 45.00 | 1.39 | 93.27 | 75.58 | 3.08 | 1.93 |
| hh_29 | Shared student house | 4 | 273.22 | 200.00 | 1.37 | 93.08 | 75.00 | 1.15 | 1.83 |
| hh_27 | Senior couple | 2 | 147.50 | 110.00 | 1.34 | 94.81 | 67.31 | 1.35 | 1.85 |
| hh_04 | Student | 1 | 60.09 | 45.00 | 1.34 | 91.35 | 68.85 | 4.42 | 1.90 |
| hh_22 | Vegetarian household | 2 | 146.13 | 110.00 | 1.33 | 92.50 | 70.19 | 2.50 | 1.78 |
| hh_14 | Family with children | 4 | 298.14 | 225.00 | 1.33 | 94.04 | 63.46 | 0.96 | 1.82 |
| hh_13 | Family with children | 4 | 276.57 | 210.00 | 1.32 | 87.12 | 66.54 | 0.96 | 1.77 |

## Households with the lowest cost/budget
| household_id | persona | n_people | avg_cost | budget | avg_ratio | pct_weeks_over | pct_weeks_over_120 | pct_weeks_under_50 | p95_ratio |
|---|---|---|---|---|---|---|---|---|---|
| hh_30 | Single professional | 1 | 47.63 | 45.00 | 1.06 | 60.38 | 23.27 | 3.65 | 1.42 |
| hh_09 | Working couple | 2 | 152.13 | 135.00 | 1.13 | 72.88 | 37.69 | 5.38 | 1.54 |
| hh_19 | Athlete | 1 | 107.09 | 95.00 | 1.13 | 75.58 | 37.31 | 3.46 | 1.49 |
| hh_10 | Working couple | 2 | 131.63 | 115.00 | 1.14 | 74.62 | 38.85 | 4.81 | 1.60 |
| hh_26 | Senior couple | 2 | 151.00 | 130.00 | 1.16 | 79.81 | 41.73 | 3.85 | 1.56 |

## Worst household-region pairs (% weeks over budget)
| household_id | region_id | avg_cost | budget | avg_ratio | pct_weeks_over | pct_weeks_over_120 | pct_weeks_under_50 | p95_ratio |
|---|---|---|---|---|---|---|---|---|
| hh_14 | new_york_ny | 331.32 | 225.00 | 1.47 | 99.04 | 91.35 | 0.96 | 1.93 |
| hh_14 | san_francisco_ca | 335.18 | 225.00 | 1.49 | 99.04 | 93.27 | 0.96 | 1.96 |
| hh_16 | new_york_ny | 262.52 | 190.00 | 1.38 | 98.08 | 83.65 | 0.00 | 1.81 |
| hh_27 | new_york_ny | 163.72 | 110.00 | 1.49 | 98.08 | 95.19 | 0.96 | 1.99 |
| hh_27 | san_francisco_ca | 166.30 | 110.00 | 1.51 | 98.08 | 95.19 | 0.00 | 2.03 |
| hh_15 | san_francisco_ca | 342.92 | 250.00 | 1.37 | 98.08 | 82.69 | 0.96 | 1.69 |
| hh_20 | san_francisco_ca | 59.99 | 45.00 | 1.33 | 98.08 | 68.27 | 1.92 | 1.84 |
| hh_16 | san_francisco_ca | 265.57 | 190.00 | 1.40 | 98.08 | 84.62 | 0.00 | 1.83 |
| hh_15 | new_york_ny | 338.40 | 250.00 | 1.35 | 98.08 | 79.81 | 0.96 | 1.67 |
| hh_28 | new_york_ny | 276.90 | 190.00 | 1.46 | 97.12 | 91.35 | 0.96 | 1.85 |

## Full per-household table
| household_id | persona | n_people | avg_cost | budget | avg_ratio | pct_weeks_over | pct_weeks_over_120 | pct_weeks_under_50 | p95_ratio |
|---|---|---|---|---|---|---|---|---|---|
| hh_11 | Working couple | 2 | 174.41 | 110.00 | 1.59 | 95.77 | 92.12 | 1.15 | 2.16 |
| hh_06 | Student | 1 | 71.73 | 50.00 | 1.43 | 94.42 | 81.92 | 1.73 | 2.03 |
| hh_07 | Student | 1 | 63.82 | 45.00 | 1.42 | 96.15 | 81.35 | 1.92 | 1.89 |
| hh_05 | Student | 1 | 62.76 | 45.00 | 1.39 | 93.27 | 75.58 | 3.08 | 1.93 |
| hh_29 | Shared student house | 4 | 273.22 | 200.00 | 1.37 | 93.08 | 75.00 | 1.15 | 1.83 |
| hh_27 | Senior couple | 2 | 147.50 | 110.00 | 1.34 | 94.81 | 67.31 | 1.35 | 1.85 |
| hh_04 | Student | 1 | 60.09 | 45.00 | 1.34 | 91.35 | 68.85 | 4.42 | 1.90 |
| hh_22 | Vegetarian household | 2 | 146.13 | 110.00 | 1.33 | 92.50 | 70.19 | 2.50 | 1.78 |
| hh_14 | Family with children | 4 | 298.14 | 225.00 | 1.33 | 94.04 | 63.46 | 0.96 | 1.82 |
| hh_13 | Family with children | 4 | 276.57 | 210.00 | 1.32 | 87.12 | 66.54 | 0.96 | 1.77 |
| hh_28 | Shared student house | 4 | 249.60 | 190.00 | 1.31 | 93.08 | 70.19 | 1.92 | 1.71 |
| hh_student | Student | 1 | 72.11 | 55.00 | 1.31 | 92.88 | 63.65 | 1.92 | 1.82 |
| hh_08 | Working couple | 2 | 139.98 | 110.00 | 1.27 | 90.00 | 59.42 | 2.50 | 1.75 |
| hh_25 | Senior couple | 2 | 152.71 | 120.00 | 1.27 | 89.04 | 60.58 | 2.50 | 1.75 |
| hh_16 | Family with children | 4 | 236.82 | 190.00 | 1.25 | 88.27 | 55.96 | 1.15 | 1.66 |
| hh_21 | Vegetarian household | 2 | 105.19 | 85.00 | 1.24 | 84.81 | 52.69 | 3.46 | 1.73 |
| hh_couple | Working couple | 2 | 148.25 | 120.00 | 1.24 | 87.12 | 54.62 | 2.88 | 1.65 |
| hh_family | Family with children | 4 | 269.19 | 220.00 | 1.22 | 83.27 | 55.19 | 5.77 | 1.74 |
| hh_15 | Family with children | 4 | 304.29 | 250.00 | 1.22 | 84.23 | 51.54 | 0.96 | 1.64 |
| hh_12 | Family with children | 5 | 291.62 | 240.00 | 1.22 | 84.81 | 47.12 | 1.54 | 1.59 |
| hh_17 | Athlete | 2 | 210.29 | 175.00 | 1.20 | 87.50 | 51.35 | 3.65 | 1.59 |
| hh_20 | Vegetarian household | 1 | 53.48 | 45.00 | 1.19 | 80.96 | 42.69 | 1.92 | 1.63 |
| hh_23 | Vegan household | 1 | 59.37 | 50.00 | 1.19 | 84.04 | 46.92 | 2.31 | 1.57 |
| hh_18 | Athlete | 2 | 213.36 | 180.00 | 1.19 | 82.50 | 51.92 | 5.96 | 1.56 |
| hh_24 | Vegan household | 2 | 105.89 | 90.00 | 1.18 | 80.77 | 50.00 | 6.35 | 1.59 |
| hh_26 | Senior couple | 2 | 151.00 | 130.00 | 1.16 | 79.81 | 41.73 | 3.85 | 1.56 |
| hh_10 | Working couple | 2 | 131.63 | 115.00 | 1.14 | 74.62 | 38.85 | 4.81 | 1.60 |
| hh_19 | Athlete | 1 | 107.09 | 95.00 | 1.13 | 75.58 | 37.31 | 3.46 | 1.49 |
| hh_09 | Working couple | 2 | 152.13 | 135.00 | 1.13 | 72.88 | 37.69 | 5.38 | 1.54 |
| hh_30 | Single professional | 1 | 47.63 | 45.00 | 1.06 | 60.38 | 23.27 | 3.65 | 1.42 |

Column notes: `avg_ratio` = mean(weekly cost / budget); `pct_weeks_over_120` = weeks above 120% of budget;
`pct_weeks_under_50` = weeks below 50% of budget; `p95_ratio` = 95th percentile of cost/budget.

## Assessment and recommendations (no data was changed)

- Budgets are too low for the recorded demand. Even the cheapest regions (Durham, Atlanta) exceed budget in about 76-78% of weeks, and the median week costs 125% of budget. No household has an average ratio below 1.06.
- Regional spread is large: New York and San Francisco cost about 40% of budget above 1.0 on average (ratio 1.40-1.42) and exceed budget in about 95% of weeks, versus 1.13-1.14 in Durham and Atlanta.
- Worst households: hh_11 (pescatarian couple, 1.59), the students hh_06, hh_07, hh_05, hh_04 (1.34-1.43), and hh_29, hh_27. Least affected: hh_30, hh_09, hh_19, hh_10.
- Guest weeks cost 1.67x budget versus 1.24x normally, so they are a built-in budget breaker.
- Interpretation: this is a diagnostic ceiling. An RL agent buys under constraints and can use promos, so it can lower cost, but if the unconstrained cost is already 25% above budget, the budget constraint binds every week and the agent mostly learns to under-buy (stockouts, missed protein), not to optimise.

Recommended adjustments, in order of preference (to be decided by the data owner):
1. Raise `weekly_budget_usd` so that the cost of demand sits near 85-95% of budget in a mid-price region (for example scale by about 1.25-1.3, or set per household budget = ~1.1 x mean cost in Durham/Atlanta). This leaves room for guest-week spikes and for the agent to show savings.
2. Make budgets region-aware (or scale regional price multipliers down for NY/SF), otherwise one region is infeasible for most households. Alternatively, train and report per region.
3. Treat the guest weeks (about 5% of weeks) as an explicit stress test, or allow a guest-week budget bump.
4. If budgets must stay as they are, lower `demand_scale` / raise `eats_out_share` instead. This is less preferred because it changes the forecasting data.
