# Documentation index

Start with the [project README](../README.md). Everything here describes the simulated setting (`hh_student`, Durham NC, $75/week) unless stated otherwise.

## App and integration
| Document | What it covers |
|---|---|
| [UI_INTEGRATION.md](UI_INTEGRATION.md) | How the web app gets its data, static demo vs. live Python, shopping priorities, meal-extra rules, AI provider |
| [GROCERY_COACH_INTEGRATION.md](GROCERY_COACH_INTEGRATION.md) | JSON contract of the Python recommendation API (request, response, priorities, unsupported scenarios), with `coach_request.example.json` / `coach_response.example.json` |

## Models
| Document | What it covers |
|---|---|
| [RL_TECHNICAL_DESIGN.md](RL_TECHNICAL_DESIGN.md) | Environment design: inventory lots, storage, freezing, constraints, reward, episode and split |
| [RL_MODELLING_RULES.md](RL_MODELLING_RULES.md) | Rules the environment and evaluation follow (packages, budgets, no leakage) |
| [RL_EVALUATION.md](RL_EVALUATION.md) | Action wrapper, evaluation framework, forecast interface |
| [RL_PPO_PILOT.md](RL_PPO_PILOT.md) | PPO pilot: setup, learning curves, validation results |
| [OPTIMIZATION.md](OPTIMIZATION.md) | The weekly MILP policy |

## Evaluation
| Document | What it covers |
|---|---|
| [COMPARISON.md](COMPARISON.md) | Unified validation comparison of baselines, Optimization and PPO |
| [FINAL_EVALUATION.md](FINAL_EVALUATION.md) | The frozen state and procedure of the one-time held-out test |
| [FINAL_RESULTS.md](FINAL_RESULTS.md) | Held-out test results (weeks 78-103) and trade-offs |

## Data
| Document | What it covers |
|---|---|
| [PRICE_DATA.md](PRICE_DATA.md), [INTERFACES_PRICES_PROPOSAL.md](INTERFACES_PRICES_PROPOSAL.md) | Real and synthetic price data and their sources |
| [REGIONAL_BUDGETS.md](REGIONAL_BUDGETS.md) | Synthetic budgets per household and region (normal / challenging / tight) |
| [STORAGE_DATA.md](STORAGE_DATA.md) | Synthetic storage rules and household capacities |
| [../INTERFACES.md](../INTERFACES.md) | The original hackathon planning contract (partly superseded, see its status note) |

## Reproducing results
`python -m pytest tests -q` runs all tests. `python -m src.final_eval --verify` checks that code, data, checkpoints and settings still match the frozen manifest
(`results/final/freeze_manifest.json`). `python -m src.comparison`, `python -m src.coach_check` and `python export_ui_plans.py` regenerate the validation comparison, the API check and the app's precomputed plans.
