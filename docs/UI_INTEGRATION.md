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

## The three shopping priorities
| Option in the app | Method (frozen, from `results/final/test`) |
|---|---|
| Save Money & Reduce Waste | Optimization (MILP) |
| Balanced | PPO with Ridge forecast (seed with the best **validation** reward) |
| Keep My Groceries Stocked | Forecast Rule (Ridge) |

To change the mapping, edit `PRIO` near the top of the "Shopping priority" section in `ui/coach.html`.

## `export_ui_plans.py`
Replays the same six frozen methods as `src/final_eval.py` on the held-out test episodes and records, for every week:
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
