# Grocery Coach adapter: validation check (weeks 64-76, test weeks not used)

Every weekly order below came from the adapter's JSON interface (inventory exported as lots, shopping list executed in the unchanged environment). `reduce_cost_waste` and `availability` must equal the pure Optimization and PPO-with-forecast (seed 0) runs; `balanced` applies the documented rule (see docs/GROCERY_COACH_INTEGRATION.md) and has no reference model. One recorded trajectory; no significance claims.

| run | reward | cash_spend | waste_units | waste_cost | stockout_units | stockout_value | fill_rate_value | weeks_using_optimization |
|---|---|---|---|---|---|---|---|---|
| adapter: reduce_cost_waste | -14.12 | 797.69 | 6.78 | 18.14 | 42.68 | 138.11 | 0.85 | 13.00 |
| adapter: availability | -14.44 | 906.24 | 28.38 | 80.62 | 24.73 | 91.49 | 0.90 | 0.00 |
| adapter: balanced | -14.25 | 908.93 | 27.32 | 80.39 | 24.97 | 81.91 | 0.91 | 2.00 |
| pure model: optimization | -14.12 | 797.69 | 6.78 | 18.14 | 42.68 | 138.11 | 0.85 |  |
| pure model: ppo_forecast_seed0 | -14.44 | 906.24 | 28.38 | 80.62 | 24.73 | 91.49 | 0.90 |  |
