# Held-out test results (weeks 78-90 and 91-103), run once with the frozen state

Run on the frozen state (manifest hash `cbb36102bebe`, code at `950ddc1`): no retraining, retuning or code/data change. Full tables, per-episode results and
individual PPO seeds: `results/final/test/` (`report.md`, `runs.csv`, `mean_over_episodes.csv`, `episode_78.csv`, `episode_91.csv`, `rankings_per_metric.csv`,
`vs_average_rule_pct.csv`). hh_student, durham_nc, normal budget (synthetic), default storage, stockout multiplier 2. Deterministic replay of one recorded trajectory
(two episodes): no confidence intervals or significance claims. Nothing was changed after seeing these numbers.

## Mean of the two test episodes (PPO = mean of 3 seeds)
| Method | Reward | Cash spend | Net cost | Waste units / cost | Stockout units / value | Fill rate | Protein | Terminal charge |
|---|---|---|---|---|---|---|---|---|
| Fixed Rule | -16.27 | 731.73 | 716.72 | 23.8 / 58.8 | 59.1 / 231.9 | 0.73 | 1.04 | 24.59 |
| Average Rule | -15.19 | 912.61 | 900.11 | 50.0 / 124.1 | 23.3 / 109.7 | 0.87 | 1.32 | 19.88 |
| Forecast Rule (Ridge) | -15.12 | 927.31 | 913.99 | 50.3 / 128.1 | 22.7 / 99.9 | 0.88 | 1.30 | 22.66 |
| Optimization (MILP) | **-14.09** | 796.57 | 778.56 | **12.3 / 31.2** | 41.6 / 128.6 | 0.85 | 1.32 | **14.93** |
| PPO with forecast | -14.71 | 911.39 | 895.05 | 36.7 / 97.2 | 23.6 / **94.6** | **0.89** | **1.36** | 25.89 |
| PPO without forecast | -14.91 | 904.91 | 892.24 | 37.1 / 102.0 | 25.2 / 103.7 | 0.88 | 1.34 | 26.46 |

PPO seeds (reward): with forecast -14.37 / -14.84 / -14.92, without forecast -14.80 / -14.95 / -14.99.

## Best method by criterion
- **Reward:** Optimization (-14.09, +7.3% vs the Average Rule), best in both test episodes (-13.58 and -14.60). PPO with forecast is second (-14.71, +3.2%), PPO without forecast third (-14.91, +1.8%).
  Forecast Rule (+0.5%) is practically level with the Average Rule.
- **Cost:** the Fixed Rule spends least ($732) but only because it under-buys (fill rate 0.73). Among methods that serve demand, Optimization spends least: $797, 12.7% below the Average Rule (net cost -13.5%).
  PPO and Forecast Rule spend about the same as the Average Rule ($905-927).
- **Food waste (simulated):** Optimization by far: 12.3 units / $31 (-75% vs the Average Rule). PPO cuts waste by about a quarter (36.7-37.1 units); Forecast Rule does not reduce it.
- **Meeting grocery needs:** PPO with forecast has the highest fill rate (0.89) and lowest stockout value ($94.6, -13.8% vs the Average Rule); Forecast Rule has the fewest stockout units (22.7), with the Average Rule and PPO within about one unit.
  Optimization is the weakest among the non-trivial methods: 41.6 stockout units (+79%), stockout value +17%, fill rate 0.85, falling to 0.83 in weeks 91-103 (50 stockout units). Fixed is far worse (59 units, 0.73).
  Protein coverage is 1.30-1.36 for every method except Fixed (1.04).
- **Terminal inventory charge:** lowest for Optimization (14.9); PPO leaves the most stock at the end (25.9-26.5, about 30% above the Average Rule), which costs it reward.

## Trade-offs
- **Optimization vs PPO:** Optimization saves money and avoids waste (and wins on reward) by buying leaner, at the price of about 75% more unmet demand than the Average Rule or PPO. PPO keeps service at or slightly above the
  Average Rule's level and cuts waste moderately, but does not save cash. Which is "better" depends on how much unmet demand is worth compared with cost and waste; the reward's stockout multiplier of 2 favours Optimization here, and these results
  do not decide that choice. No single method is best on every criterion.
- **Forecast value:** PPO with forecast beats PPO without on reward (-14.71 vs -14.91) and stockout value, but the seed ranges overlap (-14.37 to -14.92 vs -14.80 to -14.99), so this is suggestive only. Forecast Rule barely improves on the Average Rule.
- **Consistency with validation:** the validation pattern held on unseen weeks: Optimization best reward/cost/waste with weakest service; PPO about 3% above the Average Rule with similar service. Absolute reward values moved little
  (Optimization -14.12 to -14.09, PPO with forecast -14.68 to -14.71), so there was no sign of the PPO checkpoint selection overfitting the validation window.
- **Limits:** one household and region, one recorded demand trajectory, two overlapping-in-nature episodes; fixed budget and reward weights; PPO only a 300k-step pilot with three seeds. These results describe this simulated setting only.
