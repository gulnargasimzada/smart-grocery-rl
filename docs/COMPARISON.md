# Unified validation comparison: baselines, Optimization (MILP) and PPO

Branch `comparison` = `origin/RL` (`2eeef71`) + the Optimization commit `02b6685` (cherry-picked as `ce9f67d`) + one adaptation commit (episode loops in
`src/optimization_eval.py` and `tests/test_optimization.py` now use `terminated or truncated`, because `RL` ends episodes with `terminated=True`).
The optimization policy, reward, environment, constraints and datasets are unchanged. `main`, `RL` and `optimization` are untouched.

Run: `python -m src.comparison` (needs the git-ignored `data/forecast_weekly.csv`). Outputs: `results/comparison/report.md` (full tables and the % change vs. the
Average Rule), `comparison_table.csv`, `vs_average_rule_pct.csv`, `comparison.png`. Tests: `tests/test_comparison.py` (122 tests pass in the whole suite, 0 skipped).

## Setup
hh_student, durham_nc, normal budget (synthetic), default storage, validation episode = weeks 64-76. All methods run the same `EpisodeSpec` through the unchanged
environment (same prices, recorded demand, budget, initial inventory and metrics). Forecast Rule, Optimization and the PPO-with-forecast agents all received the
same verified Ridge `no_events` forecasts (recorded and checked: identical values, only the episode's own weeks requested). PPO uses the six selected checkpoints in
`results/rl_pilot/selected/` (not retrained) and is reported per seed and as the mean of three seeds. Optimization uses delta = 0.4, tuned on the training windows only.
Deterministic replay of ONE recorded trajectory; the PPO checkpoints were selected on validation windows 64 and 65, so their validation numbers are optimistic. No confidence
intervals or significance claims. Test weeks 78-103 were not run or inspected.

## Results (weeks 64-76; PPO = mean of 3 seeds)
| Method | Reward | Cash spend | Net cost | Waste units / cost | Stockout units / value | Fill rate | Protein | Terminal charge |
|---|---|---|---|---|---|---|---|---|
| Fixed | -16.08 | 732.57 | 730.20 | 17.6 / 46.5 | 61.4 / 230.6 | 0.74 | 1.12 | 17.18 |
| Average Rule | -15.32 | 926.15 | 928.99 | 53.6 / 133.6 | 23.3 / 110.1 | 0.88 | 1.44 | 9.77 |
| Forecast Rule | -15.53 | 908.77 | 917.99 | 53.9 / 138.6 | 24.4 / 123.2 | 0.86 | 1.43 | 6.08 |
| **Optimization** | **-14.12** | 797.69 | 782.92 | **6.8 / 18.1** | 42.7 / 138.1 | 0.85 | 1.44 | 9.39 |
| PPO with forecast | -14.68 | 918.08 | 913.35 | 33.8 / 101.2 | 24.6 / 93.8 | 0.90 | 1.48 | 10.97 |
| PPO without forecast | -14.67 | 917.94 | 911.08 | 32.9 / 97.5 | 24.9 / 94.7 | 0.89 | 1.49 | 12.98 |

PPO per seed (reward): with forecast -14.44 / -14.91 / -14.69, without forecast -14.23 / -15.03 / -14.75 (all tables in the report).

Change versus the Average Rule: Optimization reward +7.8%, cash spend -13.9%, waste cost -86%, stockout units +83%, stockout value +26%, fill rate -3.6%.
PPO (both agents) reward +4.2%, cash spend -0.9%, waste cost -24% / -27%, stockout units +6% / +7%, stockout value -15% / -14%, fill rate +2%, terminal charge +12% / +33%.
Forecast Rule reward -1.3%. Fixed reward -4.9% with 2.6x the stockouts.

## Trade-offs
- **Optimization** wins on reward, spending and waste by a wide margin (waste 6.8 units instead of 53.6, $128 less cash spent), but it leaves 83% more demand unserved
  than the Average Rule (42.7 vs 23.3 units; fill rate 0.85 vs 0.88). Its reward is highest because the reward prices a stockout at 2x the item price, which the model
  weighs against the waste it avoids and the money it saves.
- **PPO** keeps the Average Rule's service level (fill rate 0.89-0.90, stockout value 14-15% lower) and cuts waste by about a third, but spends the same cash and saves
  less than Optimization. Its gain over the Average Rule is smaller (+4.2% vs +7.8% reward).
- If unmet demand matters more than the reward weights say (the 2x stockout multiplier is a modelling choice), PPO is the better-balanced policy; if cost and waste matter
  most, Optimization is. That is a judgement about the objective, not something these numbers settle.
- PPO with and without forecasts are indistinguishable; Forecast Rule does not beat the Average Rule. The Ridge forecast helps only Optimization (which requires it) in this setting,
  and this comparison does not isolate that effect (Optimization without forecasts was not run).

## Integration checks
- Baselines reproduce `results/evaluation/validation_summary.csv`; PPO checkpoints reproduce `results/rl_pilot/checkpoint_results.csv` (selected rows) to 1e-6; the Optimization validation report
  regenerates byte-identically on the combined branch.
- Optimization orders need no environment cuts (rounding, stock cap, storage and budget all 0 units); maximum weekly budget use 100%.
- Optimization is unchanged when all future shelf prices are replaced by absurd values, and it does not hold the recorded demand; the other methods only use `obs_dict`/observations.

## Readiness for the final held-out test
Ready from an integration standpoint (one command and one fixed set of checkpoints). Still to decide before running weeks 78-90 and 91-103 once: whether to train PPO longer (curves were still
rising), which PPO checkpoints and which Optimization setting count as final, and whether the objective weights (especially the stockout multiplier) are acceptable, since the ranking of Optimization
versus PPO depends on them. The Optimization delta must stay the one tuned on training windows.
