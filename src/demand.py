"""Shared demand generator (data v2) = the ONE source of truth for household consumption.
- src/data_gen.py uses it to create the synthetic history (data/*.csv) for the ML forecast
- src/households_gen.py uses it to calibrate budgets
- the RL environment (src/env.py) calls it to create every simulated episode
So the forecast model and the RL agent live in the same simulated world.

How demand is built (nutrition based, so diets and allergens work automatically):
  1. A household needs kcal: adult-equivalents * kcal per adult per day * 7, and only the part NOT eaten out
     (1 - eats_out_share) comes from groceries.
  2. Grocery kcal are split across product categories (CATEGORY_KCAL_SHARE), then across the products of a
     category (WITHIN_CATEGORY_WEIGHT * household taste). Products excluded by diet or allergen get zero weight;
     their kcal flow to the remaining products (a category with no allowed product hands its share to the others).
  3. kcal -> purchase units using kcal_per_unit from data/products.csv.
  4. Realism on top: household taste that drifts slowly, seasonality, weekend pattern, guest weeks,
     away weeks (travel), holiday weeks, busy/quiet weeks and daily noise.

Units are PURCHASE UNITS from data/products.csv (e.g. milk 2.0 = 2 litres).
All numbers below are PLACEHOLDERS to be checked later by the business teammate."""
import numpy as np
import pandas as pd

from src.diets import allowed_mask

# Share of grocery kcal per category (sums to 1)
CATEGORY_KCAL_SHARE = {
    "grains": 0.30, "protein": 0.16, "dairy": 0.12, "vegetables": 0.08, "fruit": 0.07,
    "legumes": 0.06, "fats_oils": 0.06, "beverages": 0.06, "processed": 0.09,
}
# Share of a category's kcal per product (relative weights, renormalised among allowed products)
WITHIN_CATEGORY_WEIGHT = {
    "rice": 0.22, "bread": 0.22, "oats": 0.10, "pasta": 0.20, "tortillas": 0.10, "breakfast_cereal": 0.16,
    "eggs": 0.20, "chicken_breast": 0.22, "tofu": 0.08, "salmon_fillet": 0.08, "ground_beef": 0.14,
    "canned_tuna": 0.06, "peanut_butter": 0.22,
    "milk": 0.35, "yogurt": 0.20, "cheddar_cheese": 0.30, "butter": 0.15,
    "tomatoes": 0.15, "spinach": 0.06, "carrots": 0.12, "potatoes": 0.30, "broccoli": 0.08,
    "onions": 0.10, "bell_peppers": 0.07, "frozen_vegetables": 0.12,
    "bananas": 0.35, "apples": 0.25, "oranges": 0.20, "avocados": 0.20,
    "lentils": 0.40, "chickpeas_canned": 0.30, "black_beans_canned": 0.30,
    "olive_oil": 1.0,
    "orange_juice": 0.60, "plant_milk": 0.40,
    "frozen_pizza": 1.0,
}
PLANT_MILK_BOOST = 4.0   # households with no allowed dairy drink more plant milk

# Seasonal swing over a 52-week cycle (week 0 = early January; positive peaks around week 13, negative around week 39)
SEASONAL_AMPLITUDE = {
    "bananas": 0.15, "apples": -0.10, "yogurt": 0.10, "tomatoes": 0.10, "lentils": -0.15,
    "oranges": -0.20, "orange_juice": -0.10, "potatoes": -0.15, "carrots": -0.10, "broccoli": -0.10, "avocados": 0.10,
}
GUEST_WEEK_MULT = 1.30        # demand multiplier in a guest week (probability is per household: guest_prob)
AWAY_WEEK_PROB = 0.04         # household travels: very little cooking at home
AWAY_WEEK_MULT = 0.35
HOLIDAY_WEEKS = {46, 51}      # week of year: Thanksgiving, Christmas
HOLIDAY_MULT = 1.20
HOLIDAY_PRODUCT_BOOST = {"butter": 1.4, "potatoes": 1.3, "cheddar_cheese": 1.2}
TASTE_SIGMA = 0.5             # how different household tastes are (per product, fixed)
DRIFT_PHI, DRIFT_SIGMA = 0.95, 0.05   # slow weekly drift of taste (AR(1))
CHILD_KCAL_FACTOR = 0.65      # a child eats about 65% of an adult

DEFAULT_SEED = 42            # data_gen uses DEFAULT_SEED + household index as the seed of each household


def draw_taste(seed: int, n_products: int) -> np.ndarray:
    """Fixed taste of a household (per product). Same draw as the first random numbers of simulate_demand."""
    return np.random.default_rng(seed).lognormal(-0.5 * TASTE_SIGMA ** 2, TASTE_SIGMA, n_products)


COLUMNS = ["household_id", "week", "day_of_week", "is_weekend", "guest_week", "away_week", "holiday_week",
           "product_id", "demand_units"]


def adult_equivalents(household: dict) -> float:
    return float(household["n_adults"]) + CHILD_KCAL_FACTOR * float(household["n_children"])


def base_weekly_units(household: dict, products: pd.DataFrame, taste: np.ndarray | None = None) -> np.ndarray:
    """Normal-week demand in purchase units per product (noise free). taste=None means average taste."""
    n = len(products)
    allowed = allowed_mask(household, products)
    w = np.array([WITHIN_CATEGORY_WEIGHT[p] for p in products["product_id"]], dtype=float) * allowed
    if taste is not None:
        w = w * taste
    cats = products["category"].to_numpy()
    if not allowed[cats == "dairy"].any():
        w = w * np.where(products["product_id"].to_numpy() == "plant_milk", PLANT_MILK_BOOST, 1.0)
    active = [c for c in CATEGORY_KCAL_SHARE if w[cats == c].sum() > 0]
    total_share = sum(CATEGORY_KCAL_SHARE[c] for c in active)
    share = np.zeros(n)
    for c in active:
        idx = cats == c
        share[idx] = CATEGORY_KCAL_SHARE[c] / total_share * w[idx] / w[idx].sum()
    grocery_kcal_week = (adult_equivalents(household) * float(household["kcal_per_adult_day"]) * 7
                         * (1.0 - float(household["eats_out_share"])))
    return grocery_kcal_week * share / products["kcal_per_unit"].to_numpy()


def expected_weekly_cost(household: dict, products: pd.DataFrame, taste: np.ndarray | None = None) -> float:
    """Cost of a normal week (no noise, no events) for the given taste (used to calibrate budgets)."""
    return float((base_weekly_units(household, products, taste) * products["price_usd"].to_numpy()).sum())


def simulate_demand(household: dict, products: pd.DataFrame, n_weeks: int, seed: int = 0) -> pd.DataFrame:
    """Daily demand for one household. Returns a long table (COLUMNS). day_of_week: 0=Mon ... 6=Sun."""
    rng = np.random.default_rng(seed)
    pids = products["product_id"].tolist()
    n = len(pids)
    amp = np.array([SEASONAL_AMPLITUDE.get(p, 0.0) for p in pids])
    holiday_boost = np.array([HOLIDAY_PRODUCT_BOOST.get(p, 1.0) for p in pids])
    sigma = float(household["demand_noise"])
    boost = float(household["weekend_boost"])
    guest_prob = float(household["guest_prob"])

    raw = np.array([1, 1, 1, 1, 1, boost, boost], dtype=float)
    day_factor = raw / raw.mean()                       # weekly average stays equal to the weekly demand

    taste_base = rng.lognormal(-0.5 * TASTE_SIGMA ** 2, TASTE_SIGMA, n)   # fixed household taste (= draw_taste)
    drift = np.zeros(n)

    demand = np.zeros((n_weeks, 7, n))
    guest = np.zeros(n_weeks, dtype=int)
    away = np.zeros(n_weeks, dtype=int)
    holiday = np.zeros(n_weeks, dtype=int)
    for w in range(n_weeks):
        drift = DRIFT_PHI * drift + DRIFT_SIGMA * rng.standard_normal(n)
        is_away = rng.random() < AWAY_WEEK_PROB
        is_guest = (not is_away) and rng.random() < guest_prob
        is_holiday = (w % 52) in HOLIDAY_WEEKS
        away[w], guest[w], holiday[w] = int(is_away), int(is_guest), int(is_holiday)

        weekly = base_weekly_units(household, products, taste_base * np.exp(drift))
        weekly = weekly * (1 + amp * np.sin(2 * np.pi * w / 52))
        weekly = weekly * rng.lognormal(-0.5 * 0.12 ** 2, 0.12)                        # busy / quiet week
        weekly = weekly * rng.lognormal(-0.5 * (0.6 * sigma) ** 2, 0.6 * sigma, n)     # product specific level
        if is_guest:
            weekly = weekly * GUEST_WEEK_MULT
        if is_away:
            weekly = weekly * AWAY_WEEK_MULT
        if is_holiday:
            weekly = weekly * HOLIDAY_MULT * holiday_boost
        for d in range(7):
            demand[w, d] = weekly / 7 * day_factor[d] * rng.lognormal(-0.5 * sigma ** 2, sigma, n)

    per_week = 7 * n
    return pd.DataFrame({
        "household_id": household["household_id"],
        "week": np.repeat(np.arange(n_weeks), per_week),
        "day_of_week": np.tile(np.repeat(np.arange(7), n), n_weeks),
        "is_weekend": np.tile(np.repeat((np.arange(7) >= 5).astype(int), n), n_weeks),
        "guest_week": np.repeat(guest, per_week),
        "away_week": np.repeat(away, per_week),
        "holiday_week": np.repeat(holiday, per_week),
        "product_id": np.tile(np.array(pids), n_weeks * 7),
        "demand_units": demand.ravel().round(4),
    })[COLUMNS]