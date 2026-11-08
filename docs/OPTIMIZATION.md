# Traditional optimization policy

A non-learning benchmark for the RL agent: every week it solves a small **mixed-integer program** that decides how many whole packages of each product to buy.
It uses the Ridge demand forecast, current inventory, this week's prices, the weekly budget, package sizes, diet and allergen rules, the stock cap and storage capacities.

- Code: `src/optimization.py` (`OptimizationPolicy`), `src/optimization_eval.py` (tuning and reports), `tests/test_optimization.py`.
- Solver: `scipy.optimize.milp` (HiGHS), see `requirements_optimization.txt`.
- All budgets, storage rules and demand are synthetic. **Results are simulated.**

## Compatibility with GroceryEnv and the RL evaluation

- **Same interface as the baselines.** `OptimizationPolicy` is a `src.policies.Policy`: `act(info["obs_dict"]) -> order quantities in purchase units, shape (36,)`, in `products.csv` order, with `reset()` at the start of each episode.
- **Same runner, same specs, same metrics.** It is evaluated by the unchanged `src.evaluate.evaluate_policies` / `run_episode`:
  - the same `EpisodeSpec`s (household, region, budget scenario, storage scenario, start week, 13-week horizon);
  - the same recorded demand, prices, initial inventory and environment pipeline;
  - the same metrics (reward, cash spend, net cost, waste, stockouts, fill rate, protein coverage, terminal accounting).

  A trained agent wrapped in `RLPolicy(predict)` goes through the same function, so the comparison is paired and fair:

  ```python
  from src.evaluate import evaluate_policies
  from src.optimization import OptimizationPolicy
  from src.policies import RLPolicy
  episodes, summary, _ = evaluate_policies("validation", [OptimizationPolicy(), RLPolicy(predict, uses_forecast=True)], table=forecast_table)
  ```
- **Hard constraints are still enforced by the environment.** The MILP already satisfies diet, package steps, stock cap, storage and budget.
  On every training and validation episode the pipeline removed **nothing** from its orders (`results/optimization/*_pipeline_cuts.csv`; asserted in the tests).
- **Not modified:** `src/env.py`, `src/rl_wrapper.py`, `src/policies.py`, `src/evaluate.py`, `src/forecast_table.py`, the ML code, the training code and all datasets.
  Outputs go to `results/optimization/`; `results/evaluation/` is untouched.

## Information used at decision time

| Used | Not used |
|---|---|
| `obs["forecast"]`: Ridge forecast of this week (verified out of sample by `ForecastTable`, `no_events` variant) | recorded demand of the current or any future week (the policy does not even keep a copy) |
| `obs["inventory"]` (chilled and frozen units), `obs["ref_demand"]` (trailing 4-week mean) | future prices (only the current week and the 4 weeks before it are read; tested) |
| static data from `GroceryData`: this week's shelf prices, weekly budget, package steps, diet mask, shelf lives, volumes, capacities, protein target | guest or away flags, test weeks |

Like `forecast_rule`, the policy needs a forecast and raises `ForecastUnavailable` without one; there is no silent fallback.
Weeks 4–11 of the forecast file are the forecaster's own `ma4_fallback` (too little history for Ridge), which the report lists.

## Model (one MILP per week)

All money is in USD. The objective is the environment's reward multiplied by the weekly budget:

```
min  Σp c_p x_p + Σj w_j [ Σp ( 2 c_p s_pj + c_p z_pj − v_p y_pj ) + λ u_j ]

x_p = step_p · n_p,  n_p ∈ {0, 1, …, nmax_p}      whole packages; nmax_p = environment stock cap (0 if diet-excluded)
s_pj ≥ D_pj − I_p − x_p                            shortage in demand scenario j
z_pj ≥ I_p + x_p − D_pj + s_pj − keep_p            waste: leftover that expires before it is eaten
y_pj ≤ I_p + x_p − D_pj + s_pj,  y_pj ≤ keep_p      useful leftover, carried to later weeks
u_j  ≥ target − Σp protein_p (D_pj − s_pj)          protein shortfall (g)
Σp c_p x_p ≤ budget                                weekly budget
Σ vol_p x_p ≤ free volume                          pantry and fridge (non-freezable items); freezer (frozen + freezable items)
```

| Symbol | Meaning |
|---|---|
| `c_p` | this week's shelf price |
| `I_p` | units on hand |
| `D_pj` | servable demand in scenario j: `f_p · m_j · min(1, L_p / 7)`, with `f_p` the Ridge forecast and `L_p` the usable shelf life (freezer life for freezable and frozen products), so a 5-day product covers only 5/7 of a week bought on day 1 |
| `m_j`, `w_j` | demand scenarios `(1 − δ, 1, 1 + δ)` with weights `(0.25, 0.5, 0.25)`: forecast uncertainty |
| `keep_p` | leftover that is still eaten before it expires: `f_p / 7 · max(L_p − 7, 0)` |
| `v_p` | value of a carried unit: `0.98 ×` the mean price of the last 4 weeks. Buying ahead pays only when today's price is below the recent level (a promotion); the 2 % holding cost breaks ties. In the episode's last week it is the environment's terminal credit (`min(1, remaining days / 14)`, at most 2 weeks of reference demand) |
| `2`, `λ` | the reward's stockout multiplier and protein weight (`λ = 0.5 · budget / protein target`) |

Each solve is a few hundred variables and takes milliseconds. The storage rows are a conservative approximation of the environment's placement rule: freezable purchases are counted in the freezer.

### Tuning (training weeks only)

`δ` is chosen from {0, 0.1, …, 0.5} by mean reward on the five fixed **training** windows (starts 4, 17, 30, 43, 51; all inside weeks 4–63).
The result was `δ = 0.4` (`results/optimization/tuned.json`, `tuning_train.csv`). The grid is nearly flat (−13.44 to −13.40), so the choice hardly matters.
Validation was run once, afterwards. **The test weeks 78–103 were not used for tuning, evaluation or debugging:**
- `src.optimization_eval` does not offer `--split test`;
- `evaluate_policies("test", …)` refuses without `confirm_final_test=True`;
- the forecast file was regenerated without printing the forecaster's test-week scores.

## Results (simulated, `hh_student`, `durham_nc`, normal budget, default storage, Ridge `no_events`)

Mean per 13-week episode. The baseline numbers are identical to `results/evaluation/` (same runner).

| Split | Policy | Reward | Cash spend | Waste cost | Stockout value | Fill rate (value) | Protein coverage | Terminal charge |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation 64–76 | fixed | −16.08 | 732.57 | 46.52 | 230.61 | 0.74 | 1.12 | 17.18 |
| | average_rule | −15.32 | 926.15 | 133.56 | 110.05 | 0.88 | 1.44 | 9.77 |
| | forecast_rule | −15.53 | 908.77 | 138.59 | 123.22 | 0.86 | 1.43 | 6.08 |
| | **optimization** | **−14.12** | 797.69 | 18.14 | 138.11 | 0.85 | 1.44 | 9.39 |
| train (5 windows) | average_rule | −14.82 | 906.10 | 146.18 | 101.55 | 0.88 | 1.28 | 18.06 |
| | forecast_rule | −14.77 | 910.82 | 147.35 | 97.73 | 0.88 | 1.28 | 17.39 |
| | **optimization** | **−13.40** | 741.35 | 25.61 | 131.68 | 0.84 | 1.28 | 9.31 |

How to read this:
- The optimization policy trades a slightly lower fill rate (0.85 vs 0.88) for much less waste (cost $18 vs about $134), with the same protein coverage. It has the best reward on both splits.
- These numbers come from **one recorded demand trajectory**: one validation window, and five overlapping training windows that `δ` was tuned on. They are paired replays, not independent samples, so there are no confidence intervals or significance claims.
- Full tables: `results/optimization/report_validation.md`, `report_train.md`.

## Reproduce

```bash
pip install -r requirements.txt -r requirements_rl.txt -r requirements_optimization.txt
# data/forecast_weekly.csv (git-ignored) comes from the ML side's `python -m src.forecast --export` on main (see docs/RL_EVALUATION.md);
# the file used here has SHA-256 44f62eb8...48fa7d, identical to the one in RL_EVALUATION.md
python -m src.optimization_eval --tune                # training windows only
python -m src.optimization_eval --split validation
python -m src.optimization_eval --split train
python -m pytest tests -q                              # 106 tests; test_optimization.py uses mock forecasts
```

## Limitations

- **Myopic, one-week model.** Only the current week's forecast is known (later forecasts would need models trained after today).
  Future needs enter only through the value of carried-over stock, and future promotions are not anticipated.
- **Approximations:**
  - lot ages are not in `obs_dict`, so on-hand stock is treated as fresh;
  - consumption within a week is modelled in aggregate, not day by day;
  - the storage rows approximate the environment's freezer placement.

  The environment simulates the real dynamics, and these approximations are why the policy still has some stockouts and waste.
- **Uncertainty:** a symmetric three-point scenario set, not a fitted error distribution.
- **Scope:** one household, region, budget and storage scenario, as in the RL MVP. Other scenarios run through the same constructor arguments but were not evaluated.
