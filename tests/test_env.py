"""Deterministic unit tests for GroceryEnv v0 (hh_student, durham_nc, normal budget, default storage).

Run:  python -m pytest tests/test_env.py -q
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.env import (CAP_TOL, CREDIT_BUFFER_WEEKS, DAYS, HORIZON, STORAGES, GroceryEnv, Lot)


STOCKOUT_FLOOR = 5.0                                               # buying nothing over 13 weeks must be clearly negative


def make_env(**kw) -> GroceryEnv:
    return GroceryEnv(**kw)


def pid(env: GroceryEnv, name: str) -> int:
    return env.d.product_ids.index(name)


def run_episode(env: GroceryEnv, start: int, policy, collect=False):
    env.reset(options={"start_week": start})
    infos, rewards, done = [], [], False
    while not done:
        _, r, term, done, info = env.step(policy(env))
        rewards.append(r)
        infos.append(info)
    return (rewards, infos) if collect else sum(rewards)


def random_policy(seed: int, scale: float):
    rng = np.random.default_rng(seed)
    return lambda env: rng.uniform(0, scale, env.d.n)


def assert_conserved(env: GroceryEnv) -> None:
    """units: initial + purchased = consumed + wasted + remaining (per product); cost basis likewise (USD)."""
    L = env.ledger
    remaining = env._units(None)
    assert np.allclose(L["initial"] + L["purchased"], L["consumed"] + L["wasted"] + remaining, atol=1e-7)
    c = env.cash
    lhs = c["initial_basis"] + c["spend"]
    rhs = c["consumed_cost"] + c["waste_cost"] + env.basis_remaining()
    assert lhs == pytest.approx(rhs, abs=1e-6)


# ---------------------------------------------------------------- basics
def test_gymnasium_api_and_spaces():
    env = make_env()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(options={"start_week": 10})
    assert obs.shape == env.observation_space.shape == (190,)
    assert env.observation_space.contains(obs)
    assert set(info["obs_dict"]) == {"inventory", "week", "forecast", "budget_left", "ref_demand"}
    assert env.d.weekly_budget == 75  # normal budget of hh_student / durham_nc (synthetic)


def test_reset_is_deterministic_and_initial_inventory_is_fresh():
    a, b = make_env(), make_env()
    oa, _ = a.reset(options={"start_week": 20})
    ob, _ = b.reset(options={"start_week": 20})
    assert np.array_equal(oa, ob)
    ref = a.ref(20)
    for lot in a.lots:
        assert (lot.chilled_days, lot.frozen_days, lot.post_thaw_days, lot.thawed) == (0, 0, 0, False)
    got = a._units(None)
    assert np.allclose(got, ref / DAYS * 3.0)           # 3 days of trailing demand fits in default storage
    assert all(v <= a.d.capacity[s] + CAP_TOL for s, v in a.volumes().items())


def test_same_actions_give_identical_results():
    env = make_env()
    r1, i1 = run_episode(env, 30, random_policy(1, 3.0), collect=True)
    r2, i2 = run_episode(env, 30, random_policy(1, 3.0), collect=True)
    assert r1 == r2
    assert [x["spend"] for x in i1] == [x["spend"] for x in i2]


def test_episode_length_and_truncation():
    env = make_env()
    env.reset(options={"start_week": 91})
    flags = [env.step(np.zeros(env.d.n))[3] for _ in range(HORIZON)]
    assert flags == [False] * (HORIZON - 1) + [True]
    with pytest.raises(ValueError):
        env.reset(options={"start_week": 92})            # would run past week 103
    with pytest.raises(ValueError):
        env.reset(options={"start_week": 3})             # no forecast history


# ---------------------------------------------------------------- conservation
@pytest.mark.parametrize("start,seed,scale", [(4, 0, 2.0), (30, 1, 10.0), (51, 2, 60.0), (64, 3, 0.7), (78, 4, 5.0), (91, 5, 100.0)])
def test_inventory_and_cost_conservation_every_step(start, seed, scale):
    env = make_env()
    policy = random_policy(seed, scale)
    env.reset(options={"start_week": start})
    done = False
    while not done:
        _, _, _, done, info = env.step(policy(env))
        assert_conserved(env)
        assert info["waste_units"] == pytest.approx(info["waste_units_quality"] + info["waste_units_safety_cutoff"])
        assert all(l.units > 0 for l in env.lots)


def test_net_cost_reconciles_with_terminal_accounting():
    env = make_env()
    _, infos = run_episode(env, 40, random_policy(7, 4.0), collect=True)
    last = infos[-1]
    c = env.cash
    assert last["episode_net_cost"] == pytest.approx(c["consumed_cost"] + c["waste_cost"] + last["terminal_charge"])
    assert last["terminal_charge"] == pytest.approx(last["terminal_inventory_basis"] - last["terminal_inventory_credit"])
    assert last["episode_cash_spend"] == pytest.approx(sum(i["spend"] for i in infos))


# ---------------------------------------------------------------- constraints
@pytest.mark.parametrize("scale", [1.0, 20.0, 1000.0])
def test_order_pipeline_respects_budget_packages_and_storage(scale):
    env = make_env()
    policy = random_policy(11, scale)
    env.reset(options={"start_week": 25})
    done = False
    while not done:
        _, _, _, done, info = env.step(policy(env))
        assert info["spend"] <= env.d.weekly_budget + 1e-9
        steps = info["order"] / env.d.step_size
        assert np.allclose(steps, np.round(steps)), "order must be a whole number of package steps"
        for s, v in env.volumes().items():
            assert v <= env.d.capacity[s] + CAP_TOL
    for s in STORAGES:
        assert env.max_volume_seen[s] <= env.d.capacity[s] + CAP_TOL, f"{s} exceeded capacity during the week"


def test_excluded_products_are_never_bought():
    env = make_env()
    env.reset(options={"start_week": 25})
    env.d.allowed[pid(env, "chicken_breast")] = False
    order = np.zeros(env.d.n)
    order[pid(env, "chicken_breast")] = 3
    order[pid(env, "rice")] = 1
    n_steps, notes = env._adjust_order(order, 25)
    assert n_steps[pid(env, "chicken_breast")] == 0 and n_steps[pid(env, "rice")] == 1
    assert any("excluded" in n for n in notes)


def test_stock_cap_and_rounding():
    env = make_env()
    env.reset(options={"start_week": 25})
    p = pid(env, "bananas")                              # loose item, step 0.5
    order = np.zeros(env.d.n)
    order[p] = 0.74                                      # rounds to 0.5
    assert env._adjust_order(order, 25)[0][p] * 0.5 == 0.5
    order[p] = 0.24                                      # rounds to 0
    assert env._adjust_order(order, 25)[0][p] == 0
    order[p] = 500                                       # capped at 3 weeks of reference demand
    units = env._adjust_order(order, 25)[0][p] * 0.5
    assert units + env._units(None)[p] <= max(3 * env.ref(25)[p], 0.5 + env.ref(25)[p]) + 1e-9


def test_budget_reduction_removes_whole_packages_and_cost_stays_within_budget():
    env = make_env()
    env.reset(options={"start_week": 25})
    env.d.weekly_budget = 20.0
    n_steps, notes = env._adjust_order(np.full(env.d.n, 3.0), 25)
    cost = float((n_steps * env.d.step_size * env.price_at(25)).sum())
    assert 0 < cost <= 20.0
    assert np.allclose(n_steps, np.round(n_steps))
    assert any("budget" in n for n in notes)


# ---------------------------------------------------------------- storage: full fridge, freezer
def _full_fridge_env():
    env = make_env()
    env.reset(options={"start_week": 25})
    env.d.capacity["fridge"] = env.volumes()["fridge"] + 0.01      # fridge is full
    env.d.capacity["freezer"] = 500.0                              # plenty of freezer room
    return env


def test_freezable_product_is_bought_when_fridge_is_full_but_nonfreezable_is_cut():
    env = _full_fridge_env()
    chicken, milk = pid(env, "chicken_breast"), pid(env, "milk")
    assert env.d.freezable[chicken] and not env.d.freezable[milk]
    order = np.zeros(env.d.n)
    order[chicken], order[milk] = 2, 2
    n_steps, notes = env._adjust_order(order, 25)
    assert n_steps[chicken] == 2                                   # still purchasable
    assert n_steps[milk] == 0                                      # nowhere to put it
    chilled, frozen, overflow = env._place(n_steps * env.d.step_size, env.ref(25))
    assert not overflow and frozen[chicken] > 0
    assert chilled[chicken] * env.d.volume[chicken] <= 0.01 + CAP_TOL


def test_direct_to_freezer_lot_follows_thawing_rules():
    env = _full_fridge_env()
    chicken = pid(env, "chicken_breast")
    env.lots = [l for l in env.lots if l.p != chicken]             # no chilled chicken on hand
    env.ledger["initial"][chicken] = 0.0                           # keep the ledger consistent with the edited start state
    env.cash["initial_basis"] = env.basis_remaining()
    order = np.zeros(env.d.n)
    order[chicken] = 2
    _, _, _, _, info = env.step(order)
    assert info["order"][chicken] == 2
    for lot in env.lots:
        assert not (lot.frozen and lot.thawed), "a thawed lot must never be refrozen"
        if lot.p == chicken and lot.frozen:                        # went straight into the freezer, never chilled
            assert lot.chilled_days == 0 and lot.post_thaw_days == 0 and not lot.thawed and lot.frozen_days == DAYS
        if lot.p == chicken and lot.thawed:
            assert lot.chilled_days == 0                           # no chilled time before the first thaw
    assert_conserved(env)
    for s in STORAGES:
        assert env.max_volume_seen[s] <= env.d.capacity[s] + CAP_TOL, "thawing must never exceed capacity"


def test_thaw_keeps_counters_and_never_refreezes():
    env = make_env()
    env.reset(options={"start_week": 25})
    chicken = pid(env, "chicken_breast")
    env.lots = [Lot(chicken, 1.0, 4.5, frozen=True, chilled_days=1, frozen_days=10)]
    env.d.demand = env.d.demand.copy()
    env.d.demand[25, chicken] = 0.0                                # nothing eaten: isolate the storage rules
    env._thaw(env.d.demand[25] / DAYS, env.ref(25))
    thawed = [l for l in env.lots if l.thawed]
    frozen = [l for l in env.lots if l.frozen]
    assert thawed and frozen                                       # partial thaw only (just in time)
    t = thawed[0]
    assert (t.chilled_days, t.frozen_days, t.post_thaw_days) == (1, 10, 0)   # nothing reset
    assert t.units == pytest.approx(2 * env.ref(25)[chicken] / DAYS)
    assert (frozen[0].chilled_days, frozen[0].frozen_days) == (1, 10)
    assert sum(l.units for l in env.lots) == pytest.approx(1.0)
    # chicken has 3 chilled days; 1 used before freezing -> disposed when post_thaw_days reaches 2
    acc = dict(wasted=np.zeros(env.d.n), waste_cost=0.0, waste_units_quality=0.0, waste_units_safety_cutoff=0.0)
    env._age_and_dispose(acc)
    assert acc["wasted"].sum() == 0 and t.post_thaw_days == 1 and frozen[0].frozen_days == 11
    env._age_and_dispose(acc)
    assert acc["wasted"][chicken] == pytest.approx(t.units)        # thawed part disposed: 1 + 2 >= 3
    assert acc["waste_units_safety_cutoff"] == pytest.approx(t.units)
    assert frozen[0] in env.lots and frozen[0].frozen_days == 12   # frozen part is untouched


def test_thaw_only_into_free_space():
    env = make_env()
    env.reset(options={"start_week": 25})
    chicken = pid(env, "chicken_breast")
    env.lots = [Lot(chicken, 1.0, 4.5, frozen=True)]
    env.d.capacity["fridge"] = 1e-6                                # no room at all
    env._thaw(np.full(env.d.n, 1.0), env.ref(25))
    assert env.volumes()["fridge"] <= 1e-6 + CAP_TOL               # at most the free sliver was thawed
    assert sum(l.units for l in env.lots if l.frozen) > 0.99


# ---------------------------------------------------------------- consumption / disposal
def test_consumption_is_first_expiring_first_and_fractional():
    env = make_env()
    env.reset(options={"start_week": 25})
    milk = pid(env, "milk")
    env.lots = [Lot(milk, 1.0, 1.0, chilled_days=5), Lot(milk, 1.0, 2.0, chilled_days=0)]
    acc = dict(consumed=np.zeros(env.d.n), consumed_cost=0.0, stockout_value=0.0, stockout_units=0.0, stockout_days=0,
               stockout_p=np.zeros(env.d.n))
    demand = np.zeros(env.d.n)
    demand[milk] = 1.25
    env._serve(demand, env.price_at(25), acc)
    assert acc["consumed_cost"] == pytest.approx(1.0 * 1.0 + 0.25 * 2.0)   # older lot (cost 1.0) first
    assert [round(l.units, 2) for l in env.lots] == [0.75]
    demand[milk] = 2.0
    env._serve(demand, env.price_at(25), acc)
    assert acc["stockout_units"] == pytest.approx(1.25) and acc["stockout_days"] == 1
    assert acc["stockout_value"] == pytest.approx(1.25 * env.price_at(25)[milk])


def test_disposal_is_split_into_quality_and_safety_cutoff():
    env = make_env()
    env.reset(options={"start_week": 25})
    milk, bread = pid(env, "milk"), pid(env, "bread")
    env.lots = [Lot(milk, 1.0, 1.4, chilled_days=6), Lot(bread, 2.0, 3.0, chilled_days=4)]
    acc = dict(wasted=np.zeros(env.d.n), waste_cost=0.0, waste_units_quality=0.0, waste_units_safety_cutoff=0.0)
    env._age_and_dispose(acc)
    assert acc["waste_units_safety_cutoff"] == pytest.approx(1.0)  # milk: safety_conservative, 7-day limit reached
    assert acc["waste_units_quality"] == pytest.approx(2.0)        # bread: quality, 5-day limit reached
    assert acc["waste_cost"] == pytest.approx(1.4 + 6.0) and env.lots == []


# ---------------------------------------------------------------- terminal accounting
def _fresh_env_with(lots, week=60):
    env = make_env()
    env.reset(options={"start_week": 4})
    env.lots = lots
    return env, env.ref(week)


def test_terminal_long_life_within_two_week_buffer_is_not_charged():
    env, ref = _fresh_env_with([])
    rice = pid(env, "rice")
    env.lots = [Lot(rice, 1.5 * ref[rice], 2.5)]                   # 1.5 weeks of a 365-day product
    basis, credit, charge = env.terminal_valuation(60)
    assert basis == pytest.approx(1.5 * ref[rice] * 2.5) and charge == pytest.approx(0.0)


def test_terminal_stock_beyond_buffer_is_charged_in_full():
    env, ref = _fresh_env_with([])
    rice = pid(env, "rice")
    env.lots = [Lot(rice, 5 * ref[rice], 2.5)]
    _, _, charge = env.terminal_valuation(60)
    assert charge == pytest.approx((5 - CREDIT_BUFFER_WEEKS) * ref[rice] * 2.5)


def test_terminal_short_life_leftover_is_mostly_charged_and_frozen_counts_remaining_frozen_life():
    env, ref = _fresh_env_with([])
    chicken, tuna = pid(env, "chicken_breast"), pid(env, "canned_tuna")
    env.lots = [Lot(chicken, 0.5 * ref[chicken], 4.5, chilled_days=2)]      # 1 chilled day left -> credit 1/14
    _, _, charge = env.terminal_valuation(60)
    assert charge == pytest.approx(0.5 * ref[chicken] * 4.5 * (1 - 1 / 14))
    env.lots = [Lot(chicken, 0.5 * ref[chicken], 4.5, frozen=True, frozen_days=10)]   # 170 frozen days left -> full credit
    assert env.terminal_valuation(60)[2] == pytest.approx(0.0)
    env.lots = [Lot(tuna, 1.0 * ref[tuna], 1.3, chilled_days=100)]
    assert env.terminal_valuation(60)[2] == pytest.approx(0.0)


def _env_before_last_step(start=40):
    env = make_env()
    env.reset(options={"start_week": start})
    for _ in range(HORIZON - 1):
        env.step(np.zeros(env.d.n))
    return env


def test_very_large_last_week_purchase_is_not_free():
    base = _env_before_last_step()
    big, none = copy.deepcopy(base), copy.deepcopy(base)
    _, _, _, _, i_none = none.step(np.zeros(none.d.n))
    _, r_big, _, _, i_big = big.step(np.full(big.d.n, 1000.0))
    assert i_big["spend"] > 0 and i_big["spend"] <= big.d.weekly_budget
    assert i_big["terminal_inventory_basis"] > i_none["terminal_inventory_basis"]
    assert i_big["terminal_charge"] > i_none["terminal_charge"]    # leftovers (perishables, excess) are charged
    # cash spending is reported separately from the reward-side accounting
    assert i_big["episode_cash_spend"] == pytest.approx(i_big["spend"])
    assert i_none["episode_cash_spend"] == 0.0


def test_long_life_last_week_purchase_within_buffer_adds_no_terminal_charge():
    base = _env_before_last_step()
    beans = pid(base, "black_beans_canned")
    ref = base.ref(base.week)
    assert base._units(None)[beans] == 0.0 and 2 <= CREDIT_BUFFER_WEEKS * ref[beans]
    buy, none = copy.deepcopy(base), copy.deepcopy(base)
    order = np.zeros(base.d.n)
    order[beans] = 2                                               # 2 cans, less than the 2-week buffer
    _, _, _, _, i_none = none.step(np.zeros(none.d.n))
    _, _, _, _, i_buy = buy.step(order)
    assert i_buy["spend"] == pytest.approx(2 * base.price_at(base.week)[beans])
    assert i_buy["terminal_inventory_basis"] >= i_none["terminal_inventory_basis"]
    assert i_buy["terminal_charge"] == pytest.approx(i_none["terminal_charge"])    # beans leftovers are within the credit limit
    assert i_buy["stockout_units"] < i_none["stockout_units"]      # and the purchase did serve demand


def test_buying_nothing_scores_below_a_reference_rule():
    env = make_env()
    nothing = run_episode(env, 64, lambda e: np.zeros(e.d.n))
    average = run_episode(env, 64, lambda e: e.ref(e.week))
    assert average > nothing
    assert nothing < -STOCKOUT_FLOOR



# ---------------------------------------------------------------- terminal accounting: equal fulfilment, different leftovers
def _average_rule_state_before_last_step(start=30):
    """Env after 12 steps of the average_rule baseline (a training window), plus the policy's order for the last week."""
    from src.policies import AverageRule
    env = make_env()
    _, info = env.reset(options={"start_week": start})
    pol = AverageRule(env.d.product_ids, env.d.step_size, 1.0)
    for _ in range(HORIZON - 1):
        _, _, _, _, info = env.step(pol.act(info["obs_dict"]))
    return env, pol.act(info["obs_dict"])


def _extras_without_fulfilment_change(env, order_a, long_life: bool, n_items=6):
    """Products that need nothing this week (no order, no stockout under policy A); one extra package each."""
    probe = copy.deepcopy(env)
    _, _, _, _, info_a = probe.step(order_a)
    ok_products = [p for p in range(env.d.n)
                   if order_a[p] == 0 and info_a["stockout_by_product"][p] == 0 and env.d.allowed[p]
                   and (env.d.L_default[p] >= 180) == long_life and not env.d.sold_frozen[p]]
    ok_products.sort(key=lambda p: env.price_at(env.week)[p])
    extra = np.zeros(env.d.n)
    for p in ok_products[:n_items]:
        extra[p] = env.d.step_size[p]
    return extra


@pytest.mark.parametrize("long_life", [True, False])
def test_policy_with_unnecessary_last_week_purchases_gets_lower_reward(long_life):
    env, order_a = _average_rule_state_before_last_step()
    env.d.weekly_budget = 500.0                       # slack budget and no stock cap in the final week,
    env.stock_cap_weeks = 1e6                         # so the unnecessary extras are really bought
    extra = _extras_without_fulfilment_change(env, order_a, long_life)
    assert extra.sum() > 0
    a_env, b_env = copy.deepcopy(env), copy.deepcopy(env)
    _, r_a, _, _, i_a = a_env.step(order_a)
    _, r_b, _, _, i_b = b_env.step(order_a + extra)
    # equivalent demand fulfilment: same stockouts, same units eaten per product, same protein
    assert i_b["stockout_units"] == pytest.approx(i_a["stockout_units"])
    assert np.allclose(i_b["consumed_by_product"], i_a["consumed_by_product"])
    assert i_b["protein_g"] == pytest.approx(i_a["protein_g"])
    # B really bought more than A (cash spent), and its extras are left over or wasted
    assert i_b["spend"] > i_a["spend"]
    leftover_or_waste_b = i_b["terminal_inventory_basis"] + i_b["waste_cost"]
    leftover_or_waste_a = i_a["terminal_inventory_basis"] + i_a["waste_cost"]
    assert leftover_or_waste_b > leftover_or_waste_a
    # the policy with excessive leftover stock receives the lower reward, by exactly the extra costs / budget
    # (which lot is eaten can differ by a few cents of cost basis, so the consumed-cost difference is included)
    assert r_b < r_a
    extra_cost = ((i_b["terminal_charge"] - i_a["terminal_charge"]) + (i_b["waste_cost"] - i_a["waste_cost"])
                  + (i_b["consumed_cost"] - i_a["consumed_cost"]))
    assert extra_cost > 0
    assert r_a - r_b == pytest.approx(extra_cost / env.d.weekly_budget)
    assert (i_b["terminal_charge"] + i_b["waste_cost"]) > (i_a["terminal_charge"] + i_a["waste_cost"])
    # cash spending is reported separately and is higher for B
    assert i_b["episode_cash_spend"] > i_a["episode_cash_spend"]


# ---------------------------------------------------------------- shared diet rules (src/diets.py)
def test_diet_mask_comes_from_shared_diet_rules():
    import pandas as pd
    from src.diets import allowed_mask
    hh = pd.read_csv("data/households.csv").set_index("household_id")
    products = pd.read_csv("data/products.csv")
    for hid in ("hh_student", "hh_23", "hh_11"):               # omnivore, vegan, pescatarian
        env = make_env(household_id=hid)
        assert np.array_equal(env.d.allowed, allowed_mask(hh.loc[hid].to_dict(), products))
    vegan = make_env(household_id="hh_23")
    assert not vegan.d.allowed[pid(vegan, "milk")] and not vegan.d.allowed[pid(vegan, "frozen_pizza")]   # frozen_pizza is dairy-tagged
    vegan.reset(options={"start_week": 25})
    n_steps, _ = vegan._adjust_order(np.full(vegan.d.n, 2.0), 25)
    assert n_steps[~vegan.d.allowed].sum() == 0 and n_steps[vegan.d.allowed].sum() > 0
