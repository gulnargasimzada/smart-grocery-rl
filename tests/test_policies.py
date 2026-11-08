"""Tests for the baseline policies (Fixed, FixedRounded, AverageRule, BuyNothing).

Run:  python -m pytest tests/test_policies.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.diagnose_env import make_policy, run_episode
from src.env import GroceryEnv


def env_and_obs(start=30):
    env = GroceryEnv()
    _, info = env.reset(options={"start_week": start})
    return env, info["obs_dict"]


def test_fixed_standing_order_matches_plan_on_average_and_is_package_sized():
    env, obs = env_and_obs()
    pol = make_policy("fixed", env)
    plan = env.d.demand[4:64].mean(axis=0)
    orders = np.array([pol.act(obs) for _ in range(52)])
    steps = orders / env.d.step_size
    assert np.allclose(steps, np.round(steps))                       # whole packages
    assert np.allclose(orders.sum(axis=0), 52 * plan, atol=env.d.step_size.max())   # same long-run quantity as the plan
    pol.reset()
    assert np.array_equal(pol.act(obs), orders[0])                    # reset restores the schedule


def test_fixed_ignores_the_inventory():
    env, obs = env_and_obs()
    a, b = make_policy("fixed", env), make_policy("fixed", env)
    obs2 = {**obs, "inventory": {k: {"chilled": 99.0, "frozen": 0.0} for k in obs["inventory"]}}
    assert all(np.array_equal(a.act(obs), b.act(obs2)) for _ in range(5))


def test_average_rule_orders_up_to_recent_average_in_whole_packages():
    env, obs = env_and_obs()
    pol = make_policy("average_rule", env)
    ref = np.array(obs["ref_demand"])
    have = np.array([obs["inventory"][p]["chilled"] + obs["inventory"][p]["frozen"] for p in env.d.product_ids])
    order = pol.act(obs)
    step = env.d.step_size
    assert np.allclose(order / step, np.round(order / step))
    deficit = ref - have
    assert np.all(order[deficit <= 1e-9] == 0)                         # no order without a deficit
    assert np.all(order[deficit > 1e-9] >= deficit[deficit > 1e-9] - 1e-9)   # rounded up to cover the deficit
    assert np.all(order[deficit > 1e-9] < deficit[deficit > 1e-9] + step[deficit > 1e-9])


def test_buy_nothing_and_identical_conditions_for_all_policies():
    results = {}
    for name in ("buy_nothing", "fixed", "average_rule"):
        env = GroceryEnv()
        results[name] = run_episode(env, make_policy(name, env), 30)
    # same episode for everyone: same demand and same prices -> same total demand value
    assert len({round(r["demand_value"], 9) for r in results.values()}) == 1
    assert results["buy_nothing"]["spend"] == 0.0
    assert results["buy_nothing"]["reward"] < results["fixed"]["reward"]
    assert results["buy_nothing"]["reward"] < results["average_rule"]["reward"]


def test_diagnostics_refuse_test_weeks():
    env = GroceryEnv()
    with pytest.raises(AssertionError):
        run_episode(env, make_policy("fixed", env), 78)
