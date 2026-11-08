# How the app connects to the team's models

The app (`app.py` → `ui_payload.py` → `ui/coach.html`) reads the files below. Each is optional: when one is missing the
app falls back to a clearly labelled in-browser preview.

| File | Made by | What the app shows |
|---|---|---|
| `data/forecast_weekly.csv` | `python -m src.forecast --export` (Ridge, walk-forward) | Step 1 of every plan ("Ridge forecast") and the "How good is the forecast?" card on Results. |
| `results/ui/plans.json` | `python export_ui_plans.py` | The three shopping priorities on the Plan page, the Inventory page and the Results page for the evaluated household. |
| `results/final/test/` | `python -m src.final_eval` (one-time held-out run) | Read by `export_ui_plans.py`; the "All six methods" table is copied from it. |
| `results/results.json` | `INTERFACES.md` §9 format | Results page for other households, if anyone writes it. |

Week split comes from `src/forecast.py`: train 4-63, validation 64-77, test 78-103. The app only shows test weeks.

## Static demo vs. live Python inference
The deployed site (Netlify) is a **static demo**. `build_static.py` embeds the data and `results/ui/plans.json` into one HTML page; nothing on that page calls the Python models.
Each shopping priority shows a **precomputed replay** of the team's frozen models on the held-out test weeks (78-103) for the evaluated household (`hh_student`, Durham, normal budget).
Other households use an in-browser preview (an order-up-to rule), clearly labelled as such. The only live calls are the AI features (`netlify/functions/explain.mjs`) and Open Food Facts barcode lookups.

A real, tested Python API exists in this repository: `python -m src.coach_server` (adapter `src/coach_api.py`, contract in [GROCERY_COACH_INTEGRATION.md](GROCERY_COACH_INTEGRATION.md)).
It is **not deployed** and the page does not call it: the PyTorch/SciPy stack plus the frozen checkpoints do not fit a static Netlify build. To connect it, host `coach_server` on a service that can run Python
(container or small VM, behind HTTPS), have the page `POST` the household, week, priority and current inventory to `/recommend`, and render `shopping_list`, `strategy` and `constraints` from the response.
Until then the demo must keep saying "precomputed".

## The three shopping priorities
The app's options use the same strategies as the backend's priorities (`src/coach_api.py`); a test (`tests/test_ui_integration.py`) keeps the two in sync.
| Option in the app | Backend priority | Method shown (precomputed) |
|---|---|---|
| Save Money & Reduce Waste | `reduce_cost_waste` | Optimization (MILP) |
| Balanced | `balanced` | Balanced rule: each week the Optimization plan, unless its projected fill rate is more than 0.03 below the RL agent's, then the RL agent (PPO + Ridge forecast, the seed with the best **validation** reward) |
| Keep My Groceries Stocked | `availability` | RL agent (PPO + Ridge forecast) |
To change the mapping, edit `PRIO` near the top of the "Shopping priority" section in `ui/coach.html`.
The `balanced_rule` replay is a seventh entry that `export_ui_plans.py` adds **after** the frozen six-method comparison: the rule and its 0.03 tolerance were fixed beforehand, nothing was tuned on it, and
it is not part of `results/final/test/`.

## Meal suggestions and the shopping list
Meals are ranked from the recipes the household's diet and allergens allow (`hh.ok`, from `src/diets.py`), preferring food that must be used soon. Ingredients that are missing at home are added **on top of the strategy's
recommended purchases**, but only through `fitExtras()` in `ui/coach.html`: quantities are rounded up to whole packages (0.5 for loose produce, like the models), and an item is left off, with a note, if it would exceed the
weekly budget or the household's storage capacity (`data_rl/`, litres). Storage is checked conservatively in the product's default storage. Manual additions go through the same function.

## `export_ui_plans.py`
Replays the same six frozen methods as `src/final_eval.py` (plus the balanced-rule replay described above) on the held-out test episodes and records, for every week:
the basket after the constraint pipeline, the pipeline notes, the week's prices, every lot at home (storage and days left),
the Ridge forecast, and the simulated outcome (spend, waste, running out, protein). The replay is deterministic, so its
totals match `results/final/test/runs.csv` (checked to the cent). Nothing is trained, tuned or re-selected, and the
script refuses to run before `results/final/test/` exists, so it can't stand in for the one-time test run.

Needs the same packages as the RL code: `gymnasium`, `stable-baselines3`, `torch`, `scipy`.

## Scope
The team's evaluation covers one household (`hh_student`), Durham NC prices and the normal budget ($75/week). Other
households in the app use an in-browser preview (an order-up-to rule tuned on training weeks 4-63, with a heavier or
lighter penalty on running out for each priority). All numbers are simulated, never real savings.

## Tracking without typing (Inventory page)

People don't log what they eat. The app tracks what's bought, estimates use from the forecast, and asks only about exceptions:

- **Receipt photo / order email** → the AI (`netlify/functions/explain.mjs`, `receipt` / `email` modes) matches each line to the catalogue in catalogue units. Receipt amounts replace the planned amounts for those items. Pasted emails are also parsed on-device when no AI is connected.
- **Fridge photo** → the AI (`fridge` mode) estimates what's on the shelves and corrects drifted estimates, after a review step.
- **"Did you finish the …?"** → one question at a time, only when a little of a perishable is left and the estimate is unsure.
- **Weekly check-in** → swipe cards (used / tossed / still have it), +15 points. **"Tossed it"** sits next to the +10 rescue bubbles.
- **Feedback** → anything tossed in the last 4 weeks is bought 20% less per week tossed (at most 50% less); the shopping list shows it as "Your check-ins" in the item's trace.

Reports are stored in the browser (localStorage) for the demo. Without the AI connection, receipt and fridge photos offer a clearly labelled sample.

## AI provider

`netlify/functions/explain.mjs` uses Google Gemini when `GEMINI_API_KEY` is set (free tier from aistudio.google.com/apikey; model `GEMINI_MODEL`, default `gemini-3.6-flash`), otherwise Claude with `ANTHROPIC_API_KEY`. The prompts and answers are the same either way.
