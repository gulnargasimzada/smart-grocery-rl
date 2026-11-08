"""Integration tests for the unified validation comparison (src/comparison.py).

Tests that need the real exported forecasts (git-ignored data/forecast_weekly.csv) are skipped, with the reason, when the file is missing.
Only validation weeks (64-77) are used; nothing here touches the reserved test split.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import comparison as cmp
from src.env import GroceryEnv
from src.evaluate import EpisodeSpec, run_episode
from src.optimization import OptimizationPolicy
from src.policies import AverageRule

ROOT = Path(__file__).resolve().parent.parent
needs_forecasts = pytest.mark.skipif(not (ROOT / "data" / "forecast_weekly.csv").exists(),
                                     reason="BLOCKER: real data/forecast_weekly.csv missing (git-ignored; export it on the ML side)")


# ---------------------------------------------------------------- pure functions (no forecast file needed)
def fake_runs() -> pd.DataFrame:
    rows = []
    for m, base in [("fixed", 2.0), ("average_rule", 1.0), ("forecast_rule", 1.0), ("optimization", 0.5)] + \
                   [(f"ppo_{a}_seed{s}", 1.0 + s) for a in cmp.PPO_AGENTS for s in cmp.SEEDS]:
        rows.append({"method": m, **{k: base * (1 if k not in cmp.HIGHER_IS_BETTER else 1) for k in cmp.METRICS}})
    return pd.DataFrame(rows)


def test_aggregate_means_ppo_seeds_and_keeps_individual_seeds():
    t = cmp.aggregate(fake_runs())
    assert t.loc["ppo_forecast_mean", "reward"] == pytest.approx(np.mean([1.0, 2.0, 3.0]))
    assert [f"ppo_forecast_seed{s}" in t.index for s in cmp.SEEDS] == [True] * 3
    assert t.loc["ppo_no_forecast_seed2", "waste_cost"] == 3.0


def test_percent_change_versus_average_rule():
    t = cmp.aggregate(fake_runs())
    pct = cmp.vs_average_rule(t)
    assert (pct.loc["average_rule"] == 0).all()
    assert pct.loc["fixed", "waste_cost"] == pytest.approx(100.0)          # twice the Average Rule
    assert pct.loc["optimization", "waste_cost"] == pytest.approx(-50.0)
    neg = pd.DataFrame({"reward": [-15.0, -14.0]}, index=["average_rule", "x"])
    assert cmp.vs_average_rule(neg).loc["x", "reward"] == pytest.approx(100 / 15)      # -14 is 6.7% better than -15 (|AR| in the denominator)


def test_module_cannot_build_a_test_episode():
    src = inspect.getsource(cmp)
    assert "confirm_final_test" not in src and '"test"' not in src.replace('weeks 78-103', '')
    assert cmp.CANONICAL == 64 and cmp.SECOND == 65 and cmp.SECOND + 12 <= 77


# ---------------------------------------------------------------- optimization: constraints and no future information
def test_optimization_ignores_future_prices():
    """Poisoning every shelf price after the current week must not change any order."""
    from tests.mock_forecasts import mock_frame
    from src.forecast_table import ForecastTable
    env = GroceryEnv()
    table = ForecastTable(mock_frame(weeks=range(4, 78)), env.d.product_ids)
    env = GroceryEnv(forecast_fn=table.forecast_fn("hh_student", "no_events"))
    clean, poisoned = OptimizationPolicy(), OptimizationPolicy()
    _, info = env.reset(options={"start_week": 64})
    done = False
    while not done:
        week = env.week
        poisoned._prices = clean._prices.copy()
        poisoned._prices[week + 1:] = 1e6                       # absurd future prices
        a, b = clean.act(info["obs_dict"]), poisoned.act(info["obs_dict"])
        assert np.array_equal(a, b)
        _, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated


def test_optimization_does_not_hold_the_recorded_demand():
    pol = OptimizationPolicy()
    assert not any(isinstance(v, np.ndarray) and v.shape == (104, 36) and np.array_equal(v, GroceryEnv().d.demand) for v in vars(pol).values())


# ---------------------------------------------------------------- real forecasts and checkpoints
@needs_forecasts
def test_forecast_dependent_methods_receive_identical_forecasts_for_the_episode_weeks_only():
    per_run, received = cmp.run_all()
    table = cmp.load_table()
    names = {"forecast_rule", "optimization", *[f"ppo_forecast_seed{s}" for s in cmp.SEEDS]}
    assert set(received) == names                                  # ablation agents and non-forecast baselines never touch the table
    for name, ok in cmp.forecast_consistency(received, table, range(64, 77)).items():
        assert ok == {"weeks_requested": True, "values_equal_table": True}, name
    ref = received["forecast_rule"]
    for name in names:
        assert all(np.array_equal(ref[w], received[name][w]) for w in ref)


@needs_forecasts
def test_baselines_reproduce_the_earlier_evaluation():
    per_run, _ = cmp.run_all()
    old = pd.read_csv(ROOT / "results" / "evaluation" / "validation_summary.csv").set_index("policy")
    new = per_run.set_index("method")
    for m in ("fixed", "average_rule", "forecast_rule"):
        for col in cmp.METRICS:
            assert new.loc[m, col] == pytest.approx(old.loc[m, col], rel=1e-9, abs=1e-9), (m, col)


@needs_forecasts
def test_ppo_checkpoints_load_and_reproduce_the_pilot_validation_results():
    per_run, _ = cmp.run_all()
    old = pd.read_csv(ROOT / "results" / "rl_pilot" / "checkpoint_results.csv")
    old = old[old["checkpoint"] == "selected"]
    assert len(old) == 6
    new = per_run.set_index("method")
    for _, r in old.iterrows():
        row = new.loc[f"ppo_{r['agent']}_seed{int(r['seed'])}"]
        for col in ("reward", "cash_spend", "waste_cost", "waste_units", "stockout_units", "stockout_value", "fill_rate_value", "protein_coverage"):
            assert row[col] == pytest.approx(r[col], rel=1e-6, abs=1e-6), (r["agent"], r["seed"], col)


@needs_forecasts
def test_optimization_orders_are_already_feasible_for_the_environment_on_real_forecasts():
    from src.optimization_eval import pipeline_cuts
    cuts = pipeline_cuts(EpisodeSpec("validation", 64), OptimizationPolicy(delta=cmp.delta_from_tuning()), cmp.load_table())
    assert cuts["rounding"] == cuts["stock_cap"] == cuts["storage"] == cuts["budget"] == 0
    assert cuts["max_budget_use"] <= 1.0 + 1e-9


@needs_forecasts
def test_all_methods_use_the_same_episode_conditions():
    per_run, _ = cmp.run_all()
    assert set(per_run["start_week"]) == {64} and set(per_run["end_week"]) == {76} and set(per_run["split"]) == {"validation"}
    assert len(per_run) == 4 + 6
    # same recorded demand and prices behind every fill rate
    dv = per_run["stockout_value"] / (1.0 - per_run["fill_rate_value"])
    assert dv.max() == pytest.approx(dv.min())
