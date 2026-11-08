# PPO pilot (validation results only)

First PPO experiment on the MVP setting: hh_student, durham_nc, normal budget, default storage, 13-week episodes. Everything is simulated and one
recorded demand trajectory is replayed. Test weeks 78-103 were not run. Code: `src/train_rl.py` (training), `src/report_rl.py` (selection, report, plot).
Full tables, per-seed numbers and the learning-curve plot: `results/rl_pilot/` (`report.md`, `learning_curves.png`, `learning_curves_seed*.csv`,
`checkpoint_results.csv`, selected checkpoints in `results/rl_pilot/selected/`; all other checkpoints stay local in untracked `models_rl/`).

## Termination fix (before training)
The episode used to end with `truncated=True, terminated=False`. Stable-Baselines3 then bootstraps the value from the last observation, but this task is
a finite horizon with a terminal inventory charge inside the final reward, and the observation contains "weeks left", so the value after step 13 is
exactly 0. `GroceryEnv` now returns `terminated=True` at the 13th step and never truncates (reward and inventory dynamics unchanged; tests updated, and
a test checks that SB3 reports `TimeLimit.truncated = False`). The terminal observation also carries no forecast, which would have been a second
reason to avoid bootstrapping from it.

## Setup (identical for both agents)
- Agents: `forecast` (verified `no_events` Ridge forecasts in the observation) and `no_forecast` (forecast block zeros, `forecast_available = 0`).
- Training episodes start in weeks 12-51 (end by week 63), so the forecast agent only sees Ridge predictions, never the MA4 fallback.
- PPO (stable-baselines3 2.9.2, torch 2.13.0, CPU, 1 thread), MLP 64x64 for policy and value, 4 envs x 512 steps per update, batch 256, 10 epochs, lr 3e-4,
  gamma 1.0, GAE 0.95, clip 0.2, no entropy bonus, `log_std_init = -1`, reward normalisation by `VecNormalize` (learning signal only; the environment reward is unchanged).
- Budget: 300,000 timesteps (301,056 after rounding to whole updates) per run, seeds 0, 1, 2, same seeds for both agents. Training time about 10 minutes per run (three seeds ran in parallel).
- Action: the agent acts in [-1, 1]^36, mapped to the existing normalized wrapper action `a = (x + 1) / 2`. Zero maps to a = 0.5, which is the Average Rule, so the
  **initial policy is close to the Average Rule** (step-0 validation reward -15.31 vs -15.32). Improvement is therefore measured against a good starting point, not against random.
- Checkpoints every ~20k steps and at step 0; selection by the mean validation reward over the two complete windows starting in weeks 64 and 65. They overlap, so selection
  rests on essentially one trajectory and the selected numbers are optimistic. Final-checkpoint numbers are shown for comparison.

## Results, canonical validation episode (weeks 64-76), mean over 3 seeds
| Policy | Reward | Cash spend | Waste cost | Stockout units | Fill rate (value) | Protein coverage |
|---|---|---|---|---|---|---|
| Fixed | -16.08 | 732.57 | 46.52 | 61.4 | 0.74 | 1.12 |
| Average Rule | -15.32 | 926.15 | 133.56 | 23.3 | 0.88 | 1.44 |
| Forecast Rule | -15.53 | 908.77 | 138.59 | 24.4 | 0.86 | 1.43 |
| PPO forecast, selected | -14.68 | 918.08 | 101.19 | 24.6 | 0.90 | 1.48 |
| PPO forecast, final | -14.79 | 912.29 | 101.89 | 25.8 | 0.89 | 1.47 |
| PPO no_forecast, selected | -14.67 | 917.94 | 97.50 | 24.9 | 0.89 | 1.49 |
| PPO no_forecast, final | -14.76 | 908.14 | 95.03 | 26.2 | 0.89 | 1.48 |

Per seed (selected, reward): forecast -14.44 / -14.91 / -14.69, no_forecast -14.23 / -15.03 / -14.75. Fixed -16.08 is far below all of them.

## What this does and does not show
- PPO beats the Average Rule in all 3 seeds for both agents on this validation episode, by about 0.6 reward (4%) on average (0.3 to 1.1 by seed). The gain comes mainly from
  lower simulated waste (about 134 to 100 USD) with similar stockouts and slightly higher fill rate and protein coverage; cash spend is about the same.
- **The forecast does not help in this pilot:** forecast and no-forecast agents are indistinguishable (-14.68 vs -14.67, differences within the seed spread), consistent with Forecast Rule
  not beating the Average Rule.
- Seed spread is large (best -14.23, worst -15.03 for the selected checkpoints; seed 1 of the ablation ends at -15.28, barely level with the Average Rule). The training curves
  are still rising at 300k steps, so this is a pilot, not a converged result.
- One recorded validation trajectory, checkpoints selected on it: no confidence intervals, no significance claims, and no statement about the test weeks or other households.

## Stability
No NaN or infinite values in any run; validation reward moves non-monotonically (the training curve improves steadily, validation is noisier). No reward weights, constraints
or environment constants were changed. Hyperparameters were not tuned.

## Before the test split
1. Decide whether to continue training (curves still rising), e.g. 1M steps, more seeds, or an entropy bonus; any such change is judged on validation only.
2. Fix the final agent, checkpoint rule and seeds, then run the reserved test episodes (78-90 and 91-103) once, with all baselines.
