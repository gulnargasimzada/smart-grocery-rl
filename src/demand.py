"""Shared demand generator = the ONE source of truth for household consumption.
- src/data_gen.py uses it to create the synthetic history (data/*.csv) for the ML forecast
- the RL environment (src/env.py) will call it to create every simulated episode
So the forecast model and the RL agent live in the same simulated world.

Units are PURCHASE UNITS from data/products.csv (e.g. milk 2.0 = 2 litres).
All numbers below are PLACEHOLDERS to be checked later by the business teammate."""
import numpy as np
import pandas as pd

# Weekly units ONE person consumes at demand_scale = 1.0
BASE_WEEKLY_UNITS = {
    "milk": 1.5, "eggs": 0.5, "chicken_breast": 1.0, "rice": 0.3, "bread": 1.0, "bananas": 0.8,
    "apples": 0.5, "tomatoes": 1.0, "spinach": 0.8, "lentils": 0.3, "yogurt": 1.0, "oats": 0.3,
}
# Seasonal swing over a 52-week cycle (share of normal demand; negative = opposite season)
SEASONAL_AMPLITUDE = {"bananas": 0.15, "apples": -0.10, "yogurt": 0.10, "tomatoes": 0.10, "lentils": -0.15}
GUEST_WEEK_PROB = 0.06   # chance a week has guests (visible to the forecast model)
GUEST_WEEK_MULT = 1.30   # demand multiplier in a guest week

COLUMNS = ["household_id", "week", "day_of_week", "is_weekend", "guest_week", "product_id", "demand_units"]


def simulate_demand(household: dict, product_ids: list, n_weeks: int, seed: int = 0) -> pd.DataFrame:
    """Daily demand for one household. Returns a long table (COLUMNS). day_of_week: 0=Mon ... 6=Sun."""
    rng = np.random.default_rng(seed)
    n = len(product_ids)
    base = np.array([BASE_WEEKLY_UNITS[p] for p in product_ids])
    amp = np.array([SEASONAL_AMPLITUDE.get(p, 0.0) for p in product_ids])
    scale = float(household["demand_scale"])
    sigma = float(household["demand_noise"])
    boost = float(household["weekend_boost"])

    raw = np.array([1, 1, 1, 1, 1, boost, boost], dtype=float)
    day_factor = raw / raw.mean()  # weekly average stays equal to the weekly demand

    rows = []
    for w in range(n_weeks):
        guest = rng.random() < GUEST_WEEK_PROB
        season = 1 + amp * np.sin(2 * np.pi * w / 52)
        house_shock = rng.lognormal(-0.5 * 0.12 ** 2, 0.12)                       # busy / quiet week
        prod_shock = rng.lognormal(-0.5 * (0.6 * sigma) ** 2, 0.6 * sigma, n)     # product-specific level
        weekly = base * scale * season * prod_shock * house_shock * (GUEST_WEEK_MULT if guest else 1.0)
        for d in range(7):
            daily_noise = rng.lognormal(-0.5 * sigma ** 2, sigma, n)
            demand = weekly / 7 * day_factor[d] * daily_noise
            for i, pid in enumerate(product_ids):
                rows.append((household["household_id"], w, d, int(d >= 5), int(guest), pid, round(float(demand[i]), 4)))
    return pd.DataFrame(rows, columns=COLUMNS)