# Step 1: assessment and recommendations (hand-written; no data was changed)


- Budgets are too low for the recorded demand. Even the cheapest regions (Durham, Atlanta) exceed budget in about 76-78% of weeks, and the median week costs 125% of budget. No household has an average ratio below 1.06.
- Regional spread is large: New York and San Francisco cost about 40% more than budget on average (ratio 1.40-1.42) and exceed budget in about 95% of weeks, versus 1.13-1.14 in Durham and Atlanta.
- Worst households: hh_11 (pescatarian couple, 1.59), the students hh_06, hh_07, hh_05, hh_04 (1.34-1.43), and hh_29, hh_27. Least affected: hh_30, hh_09, hh_19, hh_10.
- Guest weeks cost 1.67x budget versus 1.24x normally, so they are a built-in budget breaker.
- Interpretation: this is a diagnostic ceiling. An RL agent buys under constraints and can use promos, so it can lower cost, but if the unconstrained cost is already 25% above budget, the budget constraint binds every week and the agent mostly learns to under-buy (stockouts, missed protein), not to optimise.

Recommended adjustments, in order of preference (to be decided by the data owner):
1. Raise `weekly_budget_usd` so that the cost of demand sits near 85-95% of budget in a mid-price region (for example scale by about 1.25-1.3, or set per household budget = ~1.1 x mean cost in Durham/Atlanta). This leaves room for guest-week spikes and for the agent to show savings.
2. Make budgets region-aware (or scale regional price multipliers down for NY/SF), otherwise one region is infeasible for most households. Alternatively, train and report per region.
3. Treat the guest weeks (about 5% of weeks) as an explicit stress test, or allow a guest-week budget bump.
4. If budgets must stay as they are, lower `demand_scale` / raise `eats_out_share` instead. This is less preferred because it changes the forecasting data.
