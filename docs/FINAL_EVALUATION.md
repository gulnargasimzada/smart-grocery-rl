# Final held-out evaluation: frozen state and procedure

Status: **prepared and verified, NOT executed.** The held-out test episodes (weeks 78-90 and 91-103) have not been run or inspected. The run waits for explicit approval.

## What is frozen (`results/final/freeze_manifest.json`, manifest hash `cbb36102bebe`, base commit `2f5a644`)
- **Methods:** Fixed Rule (standing order of the mean weekly demand of training weeks 4-63), Average Rule (1 week of cover), Forecast Rule (Ridge `no_events`),
  Traditional Optimization (weekly MILP, **delta = 0.4**, tuned on training windows only) and PPO with / without Ridge forecasts.
- **PPO:** the six checkpoints in `results/rl_pilot/selected/` (SHA-256 recorded), selected on validation windows 64 and 65. No further training or tuning.
- **Environment and reward:** unchanged; stockout multiplier **2.0**, protein weight 0.5, stock cap 3 weeks, terminal credit 14 days / 2 weeks of buffer.
- **Evaluation settings:** hh_student, durham_nc, normal budget (synthetic), default storage, 13-week episodes, deterministic paired replay of the recorded demand and prices; test episodes
  (78-90, 91-103), spec hash recorded; Ridge `no_events` forecasts (walk-forward/trained through week 77 for the test weeks), identical for every forecast-dependent method.
- **Hashes:** code files, data files (including the git-ignored `data/forecast_weekly.csv`), checkpoints, the validation reference table; library versions are recorded
  (Python 3.12.11, numpy 2.2.6, torch 2.13.0, stable-baselines3 2.9.2, gymnasium 1.4.0, scipy 1.17.1). Any change to a hashed item makes the run refuse.
- **Metrics (reported separately):** cash spend, net cost, waste units and cost, stockout units and value, fill rate, reward, protein coverage, terminal charge. Rankings are given
  per metric; **no overall winner is declared and reward is not used as the only criterion.** PPO seeds 0, 1, 2 are reported individually and as their mean.

## Commands
```
python -m src.final_eval --verify          # frozen state still matches the manifest; prints the approval phrase
python -m src.final_eval --dry-run         # same pipeline on the VALIDATION episode -> results/final/dry_run_validation/ (not a result)
python -m src.final_eval --run-test --approve-test <phrase>    # the one-time held-out run -> results/final/test/
```
The held-out run additionally refuses if the manifest does not verify, if the phrase is wrong, or if `results/final/test/` already exists (it is run once and never overwritten).
The phrase is a safeguard against accidents, not a security control: it must only be used after the project owner has explicitly approved the run.
If a hashed file must change before the run, re-freeze with `--freeze` and review the new manifest.

## Verification done
- `tests/test_final_eval.py`: manifest verifies and records all settings; tampering with a checkpoint or a constant is detected; the held-out run is refused without approval, never evaluates anything
  in the refused cases and runs only once; the approved code path is exercised with a stub that asks for split `test` exactly once; the test split appears in one place only; forecast metadata exists for weeks 78-103 (Ridge, trained earlier;
  metadata only, no outcomes); the dry run covers all methods and seeds, reports every metric and declares no winner.
- Dry run on validation weeks 64-76 reproduces the unified comparison exactly (`results/final/dry_run_validation/`).

## What the test results will and will not show
Two episodes of one recorded trajectory for one household and region: no confidence intervals, no significance statements, no generalisation to other households. Optimization showed the best reward
and the lowest waste on validation at the cost of more unmet demand; PPO kept service close to the Average Rule. The test run reports whether that pattern repeats; it will not be used to retune anything.
