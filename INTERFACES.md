# Interface Contract

Every teammate codes against this file. If you need to change it, tell the team first.
All code, comments, column names, docs and commit messages are in **English**.

## 1. Conventions

- **Product IDs** are lowercase with underscores (`chicken_breast`). They are defined in `data/products.csv` and never renamed.
- **Product order** everywhere (arrays, vectors) is the row order of `data/products.csv`.
- **Quantities** are always in *purchase units* (see the `unit` column in `products.csv`), e.g. `milk = 2.0` means 2 litres.
- **Weeks** are indexed from 0. **Days**: 0 = Monday ... 6 = Sunday.
- **Money** is in USD per week. **Nutrition** values are per purchase unit.
- Staples (salt, pepper, oil, spices) are assumed to be in every household and are not tracked.
- All numbers in the CSV files are **placeholders** until the business teammate verifies them from official sources.

## 2. Data files (`data/`)

| File | Owner | Columns |
|---|---|---|
| `products.csv` | Person 1 | `product_id, name, category, unit, price_usd, shelf_life_days, kcal_per_unit, protein_g_per_unit, allergens` |
| `households.csv` | Person 1 | `household_id, persona, n_people, weekly_budget_usd, weekly_protein_target_g, demand_scale, weekend_boost, demand_noise, allergen_exclusions, description` |
| `recipes.csv` | Person 1 | `recipe_id, name, meal_type, servings, prep_time_min, ingredients, tags` (`ingredients` = `product_id:quantity;product_id:quantity`) |
| `demand_daily.csv` | generated | `household_id, week, day_of_week, is_weekend, guest_week, product_id, demand_units` |
| `demand_weekly.csv` | generated | `household_id, week, guest_week, product_id, demand_units` |

Regenerate the demand files with `python -m src.data_gen`. Validate the CSVs with `python -m src.check_data`.

## 3. Shared demand generator

`src/demand.py` is the single source of truth for consumption.

```python
simulate_demand(household: dict, product_ids: list[str], n_weeks: int, seed: int = 0) -> pd.DataFrame
```

The forecast training data and the RL environment must both use it, so they live in the same simulated world.

## 4. Forecast (`src/forecast.py`, Person 1)

```python
def moving_average_forecast(history_weekly: pd.DataFrame, household_id: str, week: int, window: int = 4) -> dict[str, float]:
    """Baseline: mean demand of the last `window` weeks per product."""

def fit_forecaster(weekly: pd.DataFrame) -> object:
    """Train the ML model (e.g. XGBoost) on demand_weekly.csv."""

def predict_next_week(model, history_weekly: pd.DataFrame, household_id: str, week: int, guest_week: int = 0) -> dict[str, float]:
    """Return {product_id: predicted demand in purchase units} for `week`."""
```

Report the forecast MAE and always compare it with `moving_average_forecast`.

## 5. Environment (`src/env.py`, Person 2)

Gymnasium-style:

```python
env = GroceryEnv(household_id: str, seed: int = 0, forecaster=None)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

- **action**: `np.ndarray` of shape `(n_products,)`, order quantities in purchase units (>= 0).
- **Inside `step()`** the environment calls `apply_constraints` (section 6) before executing the order, so the agent always learns under the real constraints.
- **obs**: dict with `inventory` (per product, units by age), `week`, `forecast` (per product, if a forecaster is given), `budget_left`.
- **info** must contain these keys:

| Key | Meaning |
|---|---|
| `spend` | money spent this week (USD) |
| `waste_units` | units that expired this week |
| `waste_cost` | `waste_units` valued at price (USD) |
| `stockout_units` | demand that could not be served |
| `stockout_days` | product-days with an empty shelf |
| `kcal`, `protein_g` | nutrition actually consumed this week |
| `protein_target_met` | consumed protein / household weekly target (1.0 = met) |

## 6. Constraints (`src/constraints.py`, Person 3)

```python
def apply_constraints(order: np.ndarray, inventory: dict, household: dict, products: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (valid_order, notes). The order is scaled down / cut so that:
    - total cost <= weekly_budget_usd
    - nothing is bought from the household's `allergen_exclusions`
    - stock per product does not exceed a sensible maximum
    `notes` lists in plain English what was changed and why."""
```

## 7. Policies (`src/policies.py`, Person 2)

Every policy exposes `act(obs) -> np.ndarray` (same shape as the environment action):

| Policy | Idea |
|---|---|
| `fixed` | buy the same quantities every week |
| `average_rule` | buy the recent average consumption |
| `forecast_rule` | buy the ML forecast minus current stock (**strongest baseline**) |
| `rl` | trained agent |

## 8. Meals (`src/meals.py`, Person 4)

```python
def suggest_meals(inventory: dict, household: dict, recipes: pd.DataFrame, products: pd.DataFrame, n: int = 3) -> list[dict]:
    """Return a list of {recipe_id, name, reason, kcal_per_serving, protein_g_per_serving, cost_per_serving}."""
```

- The LLM only **chooses and explains** recipes that exist in `recipes.csv` and use available items (prefer `use_it_first` recipes and items close to expiry).
- **Calories, protein and cost are computed in code** from `products.csv`, never by the LLM.
- Meals are suggestions only; they do not change the simulated demand in v1.

## 9. Evaluation output (`results/results.json`, Person 2)

```json
{
  "household": "hh_student",
  "n_seeds": 10,
  "policies": {
    "fixed":         {"weekly_cost": 0.0, "waste_units": 0.0, "stockout_days": 0.0, "protein_target_met": 0.0, "weekly": []},
    "average_rule":  {},
    "forecast_rule": {},
    "rl":            {}
  }
}
```

Each metric is the **mean over seeds**, with a `_std` companion (for example `weekly_cost_std`). Results are always presented as *simulated* results, not real savings.

## 10. App (`app.py`, Person 4)

Reads only `data/*.csv`, `results/results.json` and the functions above. Tabs: **Results**, **Inventory and shopping list**, **Meals**, **Barcode lookup** (optional).

## 11. File ownership

| Person | Files |
|---|---|
| 1 | `data/*.csv`, `src/demand.py`, `src/data_gen.py`, `src/forecast.py` |
| 2 | `src/env.py`, `src/policies.py`, `src/train_rl.py`, `src/evaluate.py` |
| 3 | `src/constraints.py` |
| 4 | `src/meals.py`, `app.py`, `README.md` |

Edit only your own files. Run `git pull` before each work session and push small commits often.