"""Tests for the RL action wrapper (normalized action -> order quantities through the unchanged env pipeline)."""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.env import CAP_TOL, STORAGES, GroceryEnv
from src.policies import AverageRule
from src.rl_wrapper import ACTION_MAX_COVER_WEEKS, OrderUpToWrapper, action_to_order


def wrapped(**kw):
    return OrderUpToWrapper(GroceryEnv(**kw))


def test_action_space_is_normalized_float32_box_and_gymnasium_compatible():
    env = wrapped()
    assert env.action_space.shape == (36,) and env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == 0.0) and np.all(env.action_space.high == 1.0)
    assert env.observation_space.shape == (190,) and env.observation_space.dtype == np.float32
    check_env(env, skip_render_check=True)


def test_conversion_is_deterministic_clipped_and_scaled():
    ref = np.array([1.0, 0.27, 2.0])
    have = np.array([0.5, 0.0, 5.0])
    step = np.array([1.0, 1.0, 0.5])
    a = np.array([0.5, 0.5, 1.0])
    o1, o2 = action_to_order(a, ref, have, step), action_to_order(a, ref, have, step)
    assert np.array_equal(o1, o2)
    # product 0: target 1.0 - 0.5 = 0.5 deficit -> one package; 1: 0.27 -> one package; 2: target 4.0 < stock 5 -> nothing
    assert np.array_equal(o1, [1.0, 1.0, 0.0])
    assert np.array_equal(action_to_order(np.array([2.0, -1.0, 7.0]), ref, have, step), action_to_order(np.array([1.0, 0.0, 1.0]), ref, have, step))
    assert np.array_equal(action_to_order(np.zeros(3), ref, have, step), np.zeros(3))
    assert ACTION_MAX_COVER_WEEKS == 2.0


def test_shape_and_nan_actions_are_rejected():
    env = wrapped()
    env.reset(options={"start_week": 30})
    with pytest.raises(ValueError):
        env.step(np.zeros(35))
    with pytest.raises(ValueError):
        env.step(np.full(36, np.nan))


def test_half_action_reproduces_the_average_rule_baseline_exactly():
    env = wrapped()
    raw = GroceryEnv()
    _, info = env.reset(options={"start_week": 64})
    _, raw_info = raw.reset(options={"start_week": 64})
    pol = AverageRule(raw.d.product_ids, raw.d.step_size, 1.0)
    done, r_w, r_raw = False, 0.0, 0.0
    while not done:
        order = pol.act(raw_info["obs_dict"])
        _, r2, _, _, raw_info = raw.step(order)
        _, r1, done, _, info = env.step(np.full(36, 0.5))
        assert np.allclose(info["proposed_order"], order)
        assert np.allclose(info["order"], raw_info["order"])
        r_w, r_raw = r_w + r1, r_raw + r2
    assert r_w == pytest.approx(r_raw)


def test_more_cover_orders_at_least_as_much_and_zero_orders_nothing():
    env = wrapped()
    env.reset(options={"start_week": 30})
    orders = []
    for a in (0.0, 0.25, 0.5, 1.0):
        e = copy_env(env)
        _, _, _, _, info = e.step(np.full(36, a))
        orders.append(info["proposed_order"])
    assert orders[0].sum() == 0
    for lo, hi in zip(orders, orders[1:]):
        assert np.all(hi >= lo - 1e-12)


def copy_env(env):
    import copy
    return copy.deepcopy(env)


def run_checked(env, actions_fn, start, base_check=True):
    """Runs an episode and checks the hard constraints every step. Returns the infos."""
    base = env.unwrapped
    env.reset(options={"start_week": start})
    infos, done = [], False
    while not done:
        week = base.week
        on_hand_before = base.on_hand_units().copy()
        ref, step = base.ref(week), base.d.step_size
        _, _, done, _, info = env.step(actions_fn(base))
        infos.append(info)
        prop, order = info["proposed_order"], info["order"]
        assert np.allclose(prop / step, np.round(prop / step)), "proposed order must be whole packages"
        assert np.allclose(order / step, np.round(order / step)), "final order must be whole packages"
        assert np.all(order <= prop + 1e-9), "the pipeline only removes packages"
        assert info["spend"] <= base.d.weekly_budget + 1e-9
        assert np.all(order[~base.d.allowed] == 0)
        cap_units = np.maximum(3.0 * ref, step + ref)
        assert np.all((order == 0) | (on_hand_before + order <= cap_units + 1e-9)), "stock cap violated"
        for s, v in base.volumes().items():
            assert v <= base.d.capacity[s] + CAP_TOL
    for s in STORAGES:
        assert base.max_volume_seen[s] <= base.d.capacity[s] + CAP_TOL
    return infos


@pytest.mark.parametrize("kind", ["ones", "uniform", "bernoulli", "half"])
def test_orders_stay_feasible_under_extreme_and_random_actions(kind):
    env = wrapped()
    rng = np.random.default_rng(3)
    fn = {"ones": lambda b: np.ones(36), "uniform": lambda b: rng.uniform(0, 1, 36),
          "bernoulli": lambda b: rng.integers(0, 2, 36).astype(float), "half": lambda b: np.full(36, 0.5)}[kind]
    infos = run_checked(env, fn, 51)
    assert len(infos) == 13
    if kind == "ones":
        assert any("budget" in n or "storage" in n or "cap" in n for i in infos for n in i["notes"]), "constraints should bind for a=1"


def test_diet_exclusions_hold_for_a_vegan_household():
    env = wrapped(household_id="hh_23")
    base = env.unwrapped
    assert not base.d.allowed.all()
    infos = run_checked(env, lambda b: np.ones(36), 30)
    assert all(np.all(i["order"][~base.d.allowed] == 0) for i in infos)


def test_pipeline_removes_an_excluded_product_that_the_action_requests():
    env = wrapped()
    base = env.unwrapped
    chicken = base.d.product_ids.index("chicken_breast")
    base.d.allowed[chicken] = False                          # the household may not buy chicken (demand reference is > 0)
    env.reset(options={"start_week": 30})
    _, _, _, _, info = env.step(np.ones(36))
    assert info["proposed_order"][chicken] > 0 and info["order"][chicken] == 0
    assert any("excluded" in n for n in info["notes"])


def test_wrapper_does_not_change_reward_or_dynamics():
    env, raw = wrapped(), GroceryEnv()
    rng = np.random.default_rng(5)
    env.reset(options={"start_week": 40})
    raw.reset(options={"start_week": 40})
    done = False
    while not done:
        _, r1, done, _, i1 = env.step(rng.uniform(0, 1, 36))
        _, r2, _, _, i2 = raw.step(i1["proposed_order"])
        assert r1 == pytest.approx(r2) and i1["spend"] == pytest.approx(i2["spend"])
    assert i1["episode_net_cost"] == pytest.approx(i2["episode_net_cost"])


def test_same_actions_give_identical_wrapped_episodes():
    out = []
    for _ in range(2):
        env = wrapped()
        rng = np.random.default_rng(9)
        env.reset(options={"start_week": 20})
        done, total = False, 0.0
        while not done:
            _, r, done, _, info = env.step(rng.uniform(0, 1, 36).astype(np.float32))
            total += r
        out.append((total, info["episode_cash_spend"]))
    assert out[0] == out[1]
