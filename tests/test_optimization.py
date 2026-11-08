"""Tests for the traditional optimization policy (src/optimization.py) and its evaluation (src/optimization_eval.py).

Uses synthetic mock forecasts (tests/mock_forecasts.py), so the git-ignored data/forecast_weekly.csv is not needed.
Only training and validation weeks are used; nothing here runs the test weeks 78-103.

Run:  python -m pytest tests/test_optimization.py -q
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from src import optimization_eval as oe
from src.env import STOCK_CAP_WEEKS, GroceryEnv
from src.evaluate import METRICS, EpisodeSpec, episode_specs, evaluate_policies, run_episode
from src.forecast_table import ForecastTable, ForecastUnavailable
from src.optimization import OptimizationPolicy
from tests.mock_forecasts import mock_frame


@pytest.fixture(scope="module")
def table():
    env = GroceryEnv()
    return ForecastTable(mock_frame(weeks=range(4, 78)), env.d.product_ids)


def run_checked(table, start=64, household_id="hh_student", budget_scenario="normal", policy=None):
    """Runs one episode and checks every order against the environment's constraints before the pipeline sees it."""
    env = GroceryEnv(household_id, "durham_nc", budget_scenario, "default", 13, table.forecast_fn(household_id, "no_events"))
    pol = policy or OptimizationPolicy(household_id, "durham_nc", budget_scenario)
    _, info = env.reset(options={"start_week": start})
    pol.reset()
    d, orders, done = env.d, [], False
    while not done:
        obs = info["obs_dict"]
        week = env.week
        order = pol.act(obs)
        have = env.on_hand_units()
        ref = env.ref(week)
        steps = order / d.step_size
        assert np.all(order >= 0) and np.allclose(steps, np.round(steps))              # whole packages
        assert float((order * env.price_at(week)).sum()) <= d.weekly_budget + 1e-6      # budget
        assert np.all(order[~d.allowed] == 0)                                            # diet / allergens
        assert np.all(have + order <= np.maximum(STOCK_CAP_WEEKS * ref, d.step_size + ref) + 1e-6)   # stock cap
        _, _, terminated, truncated, info = env.step(order)
        done = terminated or truncated
        a = info["adjust"]
        assert np.allclose(info["order"], order)                                         # the pipeline changed nothing
        assert a["cap_cut"].sum() == a["storage_cut"].sum() == a["budget_cut"].sum() == 0
        orders.append(order)
    return np.array(orders), info


def test_orders_satisfy_all_constraints_and_pass_the_pipeline_unchanged(table):
    orders, info = run_checked(table)
    assert orders.sum() > 0 and "terminal_charge" in info


def test_tight_budget_and_allergen_household(table):
    run_checked(table, budget_scenario="tight")
    t07 = ForecastTable(mock_frame("hh_07", weeks=range(4, 78)), GroceryEnv("hh_07").d.product_ids)
    orders, _ = run_checked(t07, household_id="hh_07")                                   # peanut allergy
    pb = GroceryEnv("hh_07").d.product_ids.index("peanut_butter")
    assert np.all(orders[:, pb] == 0)


def test_needs_a_forecast():
    env = GroceryEnv()
    _, info = env.reset(options={"start_week": 30})
    with pytest.raises(ForecastUnavailable):
        OptimizationPolicy().act(info["obs_dict"])


def test_uses_no_recorded_demand_and_no_future_prices(table):
    pol = OptimizationPolicy()
    assert not hasattr(pol, "demand") and all(not isinstance(v, type(GroceryEnv().d)) for v in vars(pol).values())
    env = GroceryEnv(forecast_fn=table.forecast_fn("hh_student", "no_events"))
    _, info = env.reset(options={"start_week": 30})
    obs = info["obs_dict"]
    a = pol.act(obs)
    pol.reset()
    pol._prices[31:] *= 3.0                                     # future prices must not matter
    assert np.array_equal(a, pol.act(obs))


def test_buys_ahead_on_a_promotion_but_not_at_a_high_price(table):
    env = GroceryEnv(forecast_fn=table.forecast_fn("hh_student", "no_events"))
    _, info = env.reset(options={"start_week": 30})
    obs = info["obs_dict"]
    p = env.d.product_ids.index("canned_tuna")                  # long shelf life, stock cap leaves room for 2 cans
    base = OptimizationPolicy()
    normal = base.act(obs)[p]
    promo = OptimizationPolicy()
    promo._prices[30, p] *= 0.6
    dear = OptimizationPolicy()
    dear._prices[30, p] *= 1.5
    assert promo.act(obs)[p] >= normal >= dear.act(obs)[p] and promo.act(obs)[p] > dear.act(obs)[p]


def test_storage_capacity_is_respected():
    pol = OptimizationPolicy()
    pol.capacity = {"pantry": 175.0, "fridge": 0.5, "freezer": 40.0}
    n = pol.n
    f = np.full(n, 2.0)
    order = pol.solve(30, f, np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, 2.0), last=False) * pol.step
    fridge_only = (pol.default_storage == "fridge") & ~pol.to_freezer
    assert float((order * pol.volume)[fridge_only].sum()) <= 0.5 + 1e-9


def test_deterministic_and_compatible_with_the_shared_evaluation(table):
    episodes, summary, unavailable = evaluate_policies("validation", ["average_rule", OptimizationPolicy()], table=table)
    assert not unavailable and list(summary.index) == ["average_rule", "optimization"]
    assert set(METRICS) <= set(episodes.columns)
    again = run_episode(EpisodeSpec("validation", 64), OptimizationPolicy(), table)
    row = episodes[episodes["policy"] == "optimization"].iloc[0]
    assert all(np.isclose(row[m], again[m]) for m in METRICS)


def test_test_weeks_are_not_offered(monkeypatch):
    with pytest.raises(PermissionError):
        evaluate_policies("test", [OptimizationPolicy()])
    monkeypatch.setattr(sys, "argv", ["optimization_eval", "--split", "test"])
    with pytest.raises(SystemExit):
        oe.main()
    assert max(s.end_week for s in episode_specs("train")) == 63    # tuning windows end before validation
