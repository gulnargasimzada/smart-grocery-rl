# Grocery Coach integration contract (backend adapter v1.0)

For Supriya's frontend (`ui-grocery-coach`). Nothing on that branch was touched. The adapter runs the **frozen** Optimization (MILP, delta 0.4) and PPO-with-Ridge-forecast models
(no retraining, no changes) on the unchanged `GroceryEnv`, so every plan already respects the household's diet/allergens, whole packages, stock cap, storage capacity and weekly budget.
All data is simulated (demand, prices, budgets, storage); "week" is a week of the simulated dataset.

## Two ways to connect

**A. HTTP (decoupled, recommended).** Start the small local server in a terminal from the repo root of branch `grocery-coach-integration` (needs `data/forecast_weekly.csv`, see "Setup"):
```
python -m src.coach_server --port 8000
```
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{"status": "ok", "schema_version": "1.0"}` |
| GET | `/contract` | priorities, selection rules, supported PPO scenario, example request |
| POST | `/recommend` | body = request JSON, answer = response JSON |
```
curl -s -X POST http://127.0.0.1:8000/recommend -H 'Content-Type: application/json' -d @docs/coach_request.example.json
```
HTTP 200 is returned for `ok`, `unsupported` and `error` (read the `status` field); HTTP 400 only for a body that is not JSON. The server answers one request at a time (about 0.2-0.5 s each) and sends
`Access-Control-Allow-Origin: *` for local development only: do not expose it publicly. From the Streamlit app call it with `requests`/`urllib`, or fetch it from the browser page.

**B. Direct Python call** (only if the UI code runs in the same environment as this branch's `src/`):
```python
from src.coach_api import CoachAdapter
adapter = CoachAdapter()                 # create once; loads the models and forecasts lazily
response = adapter.recommend(request)    # dict, JSON-serializable
```

## Request
```json
{"schema_version": "1.0", "household_id": "hh_student", "region_id": "durham_nc", "week": 70, "priority": "balanced",
 "weeks_remaining": 13, "inventory": {"milk": 1.0, "rice": {"chilled": 0.5}, "chicken_breast": {"chilled": 0.4, "frozen": 1.0}}}
```
| Field | Required | Meaning |
|---|---|---|
| `week` | yes | integer decision week of the dataset, 4..103 |
| `priority` | no (`balanced`) | `reduce_cost_waste`, `availability` or `balanced` |
| `household_id`, `region_id` | no (`hh_student`, `durham_nc`) | any household in `data/households.csv`; regions `durham_nc`, `atlanta_ga`, `chicago_il`, `new_york_ny`, `san_francisco_ca` |
| `budget_scenario` | no (`normal`) | `normal`, `challenging` or `tight` |
| `inventory` | no | `{product_id: units}` or `{product_id: {"chilled": u, "frozen": u, "chilled_days": n}}`, units in the product's purchase unit (see `products.csv`). Omitted = standard starting inventory (3 days of average demand), reported as the warning `default_inventory` |
| `lots` | no | full-fidelity alternative to `inventory` (per lot `product_id, units, storage (default/freezer), chilled_days, frozen_days, post_thaw_days, thawed, unit_cost`); not together with `inventory` |
| `weeks_remaining` | no (13) | 1..13, weeks left in the 13-week plan (first week of a plan = 13). `1` makes Optimization plan for the final week |
| `ppo_seed` | no (0) | which of the three frozen PPO checkpoints (seed 0 is the default by convention, not because it scored best) |
| `last_week_protein_ratio` | no (1.0) | last week's protein / weekly target; only an input of the PPO observation |
| `forecast_variant`, `storage_scenario` | no | only `no_events` and `default` are supported; anything else returns `unsupported` |

## Response (`schema_version` "1.0")
Always has `schema_version`, `status` (`ok` | `unsupported` | `error`), `errors` (list of `{code, message}`) and `warnings` (list of `{code, message}`). Only `ok` responses contain the fields below.
Full example: `docs/coach_response.example.json`.

| Field | Content |
|---|---|
| `request` | echo of the resolved request (household, region, budget scenario, week, priority, weeks_remaining) |
| `strategy` | `selected` (`optimization` or `ppo_forecast`), `label`, `priority`, `rule` (plain-language selection rule), `fallback` + `fallback_reason`, `comparison` (balanced only), `candidates` (cost, items, projected fill / shortfall / leftover of each plan), `model` (name, delta or checkpoint + seed, `frozen_manifest` hash) |
| `shopping_list` | items to buy (quantity > 0): `product_id, name, category, unit, quantity` (purchase units), `packages`, `package_size`, `unit_price`, `line_cost`, `default_storage`, `chilled_units`, `freezer_units` |
| `totals` | `total_cost`, `weekly_budget`, `budget_used_pct`, `n_items`, `n_packages`, `projected_fill_rate`, `projected_shortfall_units`, `projected_leftover_units` |
| `inventory` | `items` (on hand, chilled/frozen split, `days_left_min`, `after_purchase_units`) and `storage` (litres used / capacity for pantry, fridge, freezer) |
| `forecast` | Ridge `no_events` forecast of this week per product (`forecast_units`) and the 4-week reference (`reference_units`); `method` is `ridge` or `ma4_fallback` (weeks 4-11, flagged by a warning) |
| `prices` | this week's price per product, the 4-week average and `vs_avg_pct` (negative = cheaper than usual) |
| `constraints` | booleans `within_budget, whole_packages, diet_allergen_ok, stock_cap_ok, storage_ok`, and whether the environment pipeline had to trim the model's order (`adjusted_by_environment_pipeline`, `pipeline_notes`) |
| `excluded_products` | products the household may not buy (diet / allergens) |

## Priorities and the transparent selection rules
| `priority` | Strategy | Why (evidence = frozen validation and held-out test runs, `docs/FINAL_RESULTS.md`) |
|---|---|---|
| `reduce_cost_waste` | always Optimization | lowest waste (-75% vs the Average Rule) and lowest spend among methods that serve demand; best reward; but about 79% more unmet demand |
| `availability` | always PPO with forecast | highest fill rate (0.89) and lowest stockout value; waste only moderately reduced, spend like the Average Rule |
| `balanced` | per week: Optimization unless its projected fill rate is more than **0.03** below PPO's | both plans are built for the same state; `projected_fill = 1 - sum(price * max(0, forecast - (on_hand + order))) / sum(price * forecast)`. The 0.03 is a fixed a-priori choice, not tuned; on the validation episode it picked PPO in 11 of 13 weeks |

The projected numbers use this week's forecast only and ignore shelf life and timing: they are indicators for the UI, not predictions. Display `strategy.rule` and `strategy.comparison.decision` to explain the choice.

## Unsupported scenarios and errors (never silently ignored)
- PPO exists only for `hh_student` / `durham_nc` / `normal`. For other households, regions or budget scenarios `availability` and `balanced` **fall back to Optimization**: `status: "ok"`, `strategy.fallback: true`, warning `ppo_unsupported_scenario`.
- `status: "unsupported"` (valid request that cannot be served): `week_out_of_range`, `forecast_variant`, `storage_scenario`, `budget_scenario`, `schema_version`, `scenario_not_available` (unknown household or region), `inventory_exceeds_capacity`, `forecast_unavailable` (no verified Ridge forecast for the week).
- `status: "error"` (bad content): `invalid_priority`, `invalid_week`, `invalid_weeks_remaining`, `invalid_ppo_seed`, `invalid_number`, `negative_quantity`, `unknown_product`, `not_freezable`, `invalid_inventory`, `invalid_request`.
- Warnings on `ok`: `default_inventory`, `forecast_moving_average` (weeks 4-11), `partial_horizon`, `ppo_unsupported_scenario`.

## Setup (once)
1. Check out `grocery-coach-integration` (it contains the models, checkpoints and frozen results; the UI branch has older placeholder `src/` files, so integrate over HTTP or merge this branch).
2. `pip install -r requirements.txt -r requirements_rl.txt -r requirements_optimization.txt`.
3. `data/forecast_weekly.csv` is git-ignored: produce it with `python -m src.forecast --export` from `main` (ML side), or copy it from the RL machine. Without it every request returns `forecast_unavailable`.
4. Tests: `python -m pytest tests -q` (the adapter tests are in `tests/test_coach_api.py`).

## Verification
`python -m src.coach_check` runs the three priorities through the JSON interface over the validation episode (weeks 64-76) and compares them with the pure models:
`reduce_cost_waste` equals Optimization and `availability` equals PPO seed 0 exactly (reward -14.12 and -14.44); `balanced` gives -14.25. Output: `results/coach/validation_check.md`,
example responses in `results/coach/examples/`.
