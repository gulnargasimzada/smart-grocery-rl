"""Unified VALIDATION comparison: baselines, traditional optimization (MILP) and the six selected PPO checkpoints.

Every method runs the same EpisodeSpec through the unchanged GroceryEnv and `src.evaluate.run_episode`: same household (hh_student), region
(durham_nc), normal budget, default storage, initial inventory, recorded prices and demand trajectory, 13-week validation episode (weeks 64-76).
Forecast-dependent methods (Forecast Rule, Optimization, PPO with forecasts) read the SAME verified no_events Ridge forecasts from one
`ForecastTable`; the check below records what each method actually received. No policy sees future demand or future prices.

The reserved test split (weeks 78-103) is not offered here, and this module never builds a test EpisodeSpec. No agent is trained: the PPO
checkpoints are loaded from results/rl_pilot/selected/.

Run:  python -m src.comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.evaluate import EpisodeSpec, make_policy, run_episode, validate_spec
from src.forecast_table import ForecastTable
from src.optimization import OptimizationPolicy
from src.optimization_eval import TUNED_PATH, pipeline_cuts
from src.train_rl import policy_from_model

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "comparison"
SELECTED = ROOT / "results" / "rl_pilot" / "selected"
SEEDS = (0, 1, 2)
CANONICAL = 64                      # validation episode, weeks 64-76
SECOND = 65                         # second complete validation window (overlaps the first; supplementary view only)
VARIANT = "no_events"
METRICS = ["reward", "cash_spend", "net_cost", "waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days",
           "fill_rate_value", "protein_coverage", "terminal_basis", "terminal_credit", "terminal_charge"]
HIGHER_IS_BETTER = {"reward", "fill_rate_value", "protein_coverage", "terminal_credit"}
TABLE_COLS = ["reward", "cash_spend", "net_cost", "waste_units", "waste_cost", "stockout_units", "stockout_value", "fill_rate_value",
              "protein_coverage", "terminal_charge"]
PPO_AGENTS = ("forecast", "no_forecast")


class RecordingTable:
    """Wraps a ForecastTable and records every forecast the environment hands to a method (week -> vector)."""

    def __init__(self, table: ForecastTable):
        self.table, self.log = table, {}

    def forecast_fn(self, household_id: str, variant: str):
        base = self.table.forecast_fn(household_id, variant)

        def fn(week):
            v = base(week)
            self.log[int(week)] = v.copy()
            return v
        return fn

    def __getattr__(self, name):                 # require(), get(), ... pass through
        return getattr(self.table, name)


def load_table() -> ForecastTable:
    products = pd.read_csv(ROOT / "data" / "products.csv")["product_id"].tolist()
    return ForecastTable.from_csv(ROOT / "data" / "forecast_weekly.csv", products)


def delta_from_tuning() -> float:
    return float(json.loads(TUNED_PATH.read_text())["delta"])


def load_ppo(agent: str, seed: int):
    model = PPO.load(SELECTED / f"{agent}_seed{seed}.zip", device="cpu")
    return policy_from_model(model, f"ppo_{agent}_seed{seed}", agent == "forecast")


def build_methods() -> dict:
    """name -> policy object or baseline name. Optimization uses the delta tuned on the TRAINING windows (results/optimization/tuned.json)."""
    methods: dict = {"fixed": "fixed", "average_rule": "average_rule", "forecast_rule": "forecast_rule",
                     "optimization": OptimizationPolicy(delta=delta_from_tuning(), name="optimization")}
    for agent in PPO_AGENTS:
        for seed in SEEDS:
            methods[f"ppo_{agent}_seed{seed}"] = load_ppo(agent, seed)
    return methods


def uses_forecast(m) -> bool:
    return m == "forecast_rule" or (not isinstance(m, str) and m.uses_forecast)


def run_all(starts=(CANONICAL,)) -> tuple[pd.DataFrame, dict]:
    """One row per (method, start week). Also returns the forecasts each forecast-dependent method received."""
    table = load_table()
    rows, received = [], {}
    for name, method in build_methods().items():
        rec = RecordingTable(table)
        for s in starts:
            spec = EpisodeSpec("validation", s)
            validate_spec(spec)
            r = run_episode(spec, method, rec if uses_forecast(method) else None, VARIANT)
            r["method"] = name
            rows.append(r)
        if uses_forecast(method):
            received[name] = rec.log
    df = pd.DataFrame(rows)
    return df, received


def forecast_consistency(received: dict, table: ForecastTable, weeks: range) -> dict:
    """Every forecast-dependent method must have received exactly the table's no_events forecasts of the episode weeks, and nothing else."""
    out = {}
    for name, log in received.items():
        out[name] = {"weeks_requested": sorted(log) == list(weeks),
                     "values_equal_table": all(np.array_equal(log[w], table.get("hh_student", w, VARIANT)) for w in log)}
    return out


def aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    """Per-method table: baselines/optimization as they are, PPO per seed and the mean (+ min/max) over seeds."""
    base = per_run.set_index("method")[METRICS]
    rows = {m: base.loc[m] for m in ("fixed", "average_rule", "forecast_rule", "optimization")}
    for agent in PPO_AGENTS:
        g = base.loc[[f"ppo_{agent}_seed{s}" for s in SEEDS]]
        rows[f"ppo_{agent}_mean"] = g.mean()
        for s in SEEDS:
            rows[f"ppo_{agent}_seed{s}"] = base.loc[f"ppo_{agent}_seed{s}"]
    return pd.DataFrame(rows).T


def vs_average_rule(table: pd.DataFrame) -> pd.DataFrame:
    """Percent change relative to the Average Rule: (x - AR) / |AR| * 100. Sign: for reward, fill rate, protein coverage and terminal credit a
    positive value is better; for every other column (costs, waste, stockouts, spend) a negative value is better."""
    ar = table.loc["average_rule"]
    return (table - ar) / ar.abs().replace(0, np.nan) * 100.0


def md(df: pd.DataFrame, fmt="{:.2f}") -> str:
    d = df.reset_index()
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        rows.append("| " + " | ".join(fmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |")
    return "\n".join(rows)


def tradeoff_text(t: pd.DataFrame) -> list[str]:
    o, ar = t.loc["optimization"], t.loc["average_rule"]
    pf, pn = t.loc["ppo_forecast_mean"], t.loc["ppo_no_forecast_mean"]
    pp = (pf + pn) / 2
    lines = [
        f"- **Optimization** has the best reward ({o['reward']:.2f} vs Average Rule {ar['reward']:.2f}, PPO means {pf['reward']:.2f} / {pn['reward']:.2f}). "
        f"It cuts simulated waste to {o['waste_units']:.1f} units (${o['waste_cost']:.0f}) from {ar['waste_units']:.1f} units (${ar['waste_cost']:.0f}), "
        f"and spends ${o['cash_spend']:.0f} against ${ar['cash_spend']:.0f}.",
        f"- The price is unmet demand: {o['stockout_units']:.1f} stockout units (${o['stockout_value']:.0f}) and fill rate {o['fill_rate_value']:.2f}, against "
        f"{ar['stockout_units']:.1f} units (${ar['stockout_value']:.0f}) and {ar['fill_rate_value']:.2f} for the Average Rule.",
        f"- **PPO** (mean of 3 seeds) keeps the Average Rule's service level (stockouts {pf['stockout_units']:.1f} / {pn['stockout_units']:.1f} units, fill rate "
        f"{pf['fill_rate_value']:.2f} / {pn['fill_rate_value']:.2f}) and reduces waste to {pf['waste_units']:.1f} / {pn['waste_units']:.1f} units "
        f"(${pf['waste_cost']:.0f} / ${pn['waste_cost']:.0f}), less than Optimization does.",
        f"- Optimization versus the PPO means: waste {o['waste_units']:.1f} vs {pp['waste_units']:.1f} units (PPO average of both agents), stockouts {o['stockout_units']:.1f} vs "
        f"{pp['stockout_units']:.1f} units, fill rate {o['fill_rate_value']:.2f} vs {pp['fill_rate_value']:.2f}, cash spend ${o['cash_spend']:.0f} vs ${pp['cash_spend']:.0f}. "
        "Optimization trades more unmet demand for much less waste and lower spending; PPO keeps service closer to the Average Rule.",
        f"- Protein coverage is {o['protein_coverage']:.2f} for Optimization against {ar['protein_coverage']:.2f} (Average Rule) and {pp['protein_coverage']:.2f} (PPO); "
        "values above 1 mean the recorded demand already exceeds the 340 g target.",
    ]
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table_src = load_table()
    per_run, received = run_all(starts=(CANONICAL,))
    per_run.to_csv(OUT / "validation_runs_canonical.csv", index=False)
    t = aggregate(per_run)
    pct = vs_average_rule(t)
    t.to_csv(OUT / "comparison_table.csv")
    pct.to_csv(OUT / "vs_average_rule_pct.csv")
    consistency = forecast_consistency(received, table_src, range(CANONICAL, CANONICAL + 13))

    second, _ = run_all(starts=(CANONICAL, SECOND))
    t2 = aggregate(second.groupby("method", sort=False)[METRICS].mean().reset_index())
    t2.to_csv(OUT / "comparison_table_two_windows.csv")

    delta = delta_from_tuning()
    cuts = pipeline_cuts(EpisodeSpec("validation", CANONICAL), OptimizationPolicy(delta=delta), table_src)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    names = ["fixed", "average_rule", "forecast_rule", "optimization", "ppo_forecast_mean", "ppo_no_forecast_mean"]
    cols = ["gray", "green", "red", "purple", "tab:blue", "tab:orange"]
    ax[0].bar(range(6), [t.loc[n, "reward"] for n in names], color=cols)
    ax[0].set_xticks(range(6), [n.replace("_mean", "") for n in names], rotation=30, ha="right")
    ax[0].set(title="Validation reward (weeks 64-76; higher is better)", ylim=(-16.5, -13.5))
    for n, c in zip(names, cols):
        ax[1].scatter(t.loc[n, "waste_units"], t.loc[n, "stockout_units"], s=90, color=c, label=n.replace("_mean", ""))
    for agent, c in (("forecast", "tab:blue"), ("no_forecast", "tab:orange")):
        for s in SEEDS:
            r = t.loc[f"ppo_{agent}_seed{s}"]
            ax[1].scatter(r["waste_units"], r["stockout_units"], s=25, color=c, alpha=0.5, marker="x")
    ax[1].set(title="Waste vs. unmet demand (x = individual PPO seeds)", xlabel="waste (units)", ylabel="stockout (units)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=130)

    lines = [
        "# Unified validation comparison (validation split only; test weeks 78-103 not run)",
        "",
        f"hh_student, durham_nc, normal budget (synthetic), default storage; validation episode weeks {CANONICAL}-{CANONICAL + 12}. Every method runs the same "
        "`EpisodeSpec` through the unchanged environment: same prices, recorded demand, budget, initial inventory and metrics. Forecasts: the same verified "
        "Ridge `no_events` table for every forecast-dependent method. PPO checkpoints are the six selected ones from `results/rl_pilot/selected/` (loaded, "
        "not retrained; selected on validation windows 64 and 65, so PPO validation numbers are optimistic). Optimization: weekly MILP, delta "
        f"{delta} tuned on the training windows only. Deterministic replay of ONE recorded trajectory: no confidence intervals or significance claims.",
        "",
        "## Results (canonical validation episode)",
        md(t[TABLE_COLS]),
        "",
        "Remaining metrics: stockout days, terminal inventory basis and credit are in `comparison_table.csv`.",
        "",
        "## Change relative to the Average Rule (%)",
        "(x - AR) / |AR| x 100. Positive is better for reward, fill rate and protein coverage; negative is better for spend, net cost, waste, stockouts and terminal charge.",
        md(pct[TABLE_COLS].drop(index="average_rule"), "{:+.1f}"),
        "",
        "## Trade-offs: Optimization vs. PPO",
        *tradeoff_text(t),
        "",
        "![comparison](comparison.png)",
        "",
        f"## Supplementary: mean of the two complete validation windows ({CANONICAL} and {SECOND}; they overlap by 12 weeks)",
        md(t2[TABLE_COLS]),
        "",
        "## Integration checks",
        f"- Forecast received by each forecast-dependent method equals the table's no_events forecasts for weeks {CANONICAL}-{CANONICAL + 12}, and only those weeks "
        f"(no other week requested): {json.dumps(consistency)}",
        f"- Optimization orders on this episode vs. the environment pipeline (units removed; 0 = already feasible): "
        f"rounding {cuts['rounding']:.1f}, stock cap {cuts['stock_cap']:.1f}, storage {cuts['storage']:.1f}, budget {cuts['budget']:.1f}; "
        f"maximum weekly budget use {cuts['max_budget_use']:.3f}.",
        "- Baselines and PPO checkpoints reproduce the earlier results (`tests/test_comparison.py`): `results/evaluation/report_validation.md` and `results/rl_pilot/checkpoint_results.csv`.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
