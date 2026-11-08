"""Figure for the README: held-out test results (weeks 78-103) from the frozen evaluation files.

Reads only results/final/test/mean_over_episodes.csv (written once by `python -m src.final_eval --run-test`); recomputes nothing.
Run:  python -m src.plot_final_results      ->  docs/images/test_results.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
t = pd.read_csv(ROOT / "results" / "final" / "test" / "mean_over_episodes.csv", index_col=0)
main = {"fixed": "Fixed list", "average_rule": "Average rule", "forecast_rule": "Forecast rule", "optimization": "Optimization (MILP)",
        "ppo_forecast_mean": "PPO + forecast", "ppo_no_forecast_mean": "PPO, no forecast"}
colors = {"fixed": "#8a8f98", "average_rule": "#2e8b57", "forecast_rule": "#d9534f", "optimization": "#7b3fa0",
          "ppo_forecast_mean": "#1f77b4", "ppo_no_forecast_mean": "#ff7f0e"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
names = list(main)
LO = -16.8                                        # bars start at the left edge of the axis (rewards are negative)
ax[0].barh(range(len(names)), [t.loc[n, "reward"] - LO for n in names], left=LO, color=[colors[n] for n in names])
ax[0].set_yticks(range(len(names)), [main[n] for n in names])
ax[0].invert_yaxis()
ax[0].set(xlim=(LO, -13.4), title="Reward per 13-week episode\n(longer bar = higher = better)")
for i, n in enumerate(names):
    ax[0].text(t.loc[n, "reward"] + 0.04, i, f"{t.loc[n, 'reward']:.2f}", va="center", fontsize=8)
ax[1].barh(range(len(names)), [t.loc[n, "cash_spend"] / 13 for n in names], color=[colors[n] for n in names])
ax[1].set_yticks(range(len(names)), [])
ax[1].invert_yaxis()
ax[1].set(xlim=(0, 82), title="Cash spend per week, USD\n(lower is cheaper, not necessarily better)")
for i, n in enumerate(names):
    ax[1].text(t.loc[n, "cash_spend"] / 13 + 1, i, f"{t.loc[n, 'cash_spend'] / 13:.0f}", va="center", fontsize=8)
for n in names:
    ax[2].scatter(t.loc[n, "waste_units"], t.loc[n, "stockout_units"], s=110, color=colors[n], label=main[n], zorder=3)
for a in ("forecast", "no_forecast"):
    for s in range(3):
        k = f"ppo_{a}_seed{s}"
        ax[2].scatter(t.loc[k, "waste_units"], t.loc[k, "stockout_units"], s=28, marker="x", color=colors[f"ppo_{a}_mean"], alpha=0.7)
ax[2].set(title="Trade-off: waste vs. unmet demand\n(lower left is better; x = individual PPO seeds)", xlabel="food waste, units per episode",
          ylabel="unmet demand (stockouts), units per episode")
ax[2].legend(fontsize=7, loc="upper right")
ax[2].grid(alpha=0.25)
fig.suptitle("Held-out test, weeks 78-103, hh_student / Durham / normal budget (simulated, mean of 2 episodes)", fontsize=10)
fig.tight_layout()
out = ROOT / "docs" / "images" / "test_results.png"
fig.savefig(out, dpi=140)
print("wrote", out)
