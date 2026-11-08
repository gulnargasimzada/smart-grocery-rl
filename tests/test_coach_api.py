"""Tests for the Grocery Coach backend adapter (src/coach_api.py, src/coach_server.py).

Needs the git-ignored data/forecast_weekly.csv for most tests (skipped with the reason if missing). Test weeks 78-103 are never evaluated; the
adapter-vs-model equivalence runs on the validation episode (weeks 64-76).
"""
from __future__ import annotations

import copy
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from src import coach_api as ca
from src import coach_check as cc
from src.coach_api import CoachAdapter, lots_to_json
from src.env import CAP_TOL, GroceryEnv

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "forecast_weekly.csv").exists(),
                                reason="BLOCKER: real data/forecast_weekly.csv missing (git-ignored)")


@pytest.fixture(scope="module")
def adapter():
    return CoachAdapter()


def req(**kw):
    base = {"household_id": "hh_student", "region_id": "durham_nc", "week": 30, "priority": "balanced"}
    base.update(kw)
    return base


def order_vector(resp):
    v = np.zeros(36)
    ids = GroceryEnv().d.product_ids
    for it in resp["shopping_list"]:
        v[ids.index(it["product_id"])] = it["quantity"]
    return v


# ---------------------------------------------------------------- contract / schema
def test_response_has_the_documented_fields_and_is_json_serializable(adapter):
    r = adapter.recommend(req())
    json.dumps(r)
    assert r["schema_version"] == "1.0" and r["status"] == "ok" and r["errors"] == []
    assert set(r) >= {"request", "strategy", "shopping_list", "totals", "inventory", "forecast", "prices", "constraints", "warnings", "excluded_products", "notes"}
    s = r["strategy"]
    assert s["selected"] in ca.STRATEGIES and s["priority"] == "balanced" and s["rule"] and set(s["candidates"]) == {"optimization", "ppo_forecast"}
    assert {"name", "frozen_manifest"} <= set(s["model"])
    it = r["shopping_list"][0]
    assert set(it) >= {"product_id", "name", "unit", "quantity", "packages", "unit_price", "line_cost", "chilled_units", "freezer_units"}
    assert r["totals"]["total_cost"] == pytest.approx(sum(i["line_cost"] for i in r["shopping_list"]), abs=0.01)
    assert len(r["prices"]["items"]) == 36 and r["forecast"]["variant"] == "no_events" and r["forecast"]["method"] == "ridge"
    assert set(r["inventory"]["storage"]) == {"pantry", "fridge", "freezer"}


def test_cli_example_request_runs(adapter):
    r = adapter.recommend(ca.EXAMPLE_REQUEST)
    assert r["status"] == "ok" and r["request"]["priority"] == "balanced"


# ---------------------------------------------------------------- strategies and selection rules
def test_reduce_cost_waste_uses_optimization_and_equals_the_pure_model(adapter):
    r = adapter.recommend(req(priority="reduce_cost_waste", week=40))
    assert r["strategy"]["selected"] == "optimization" and not r["strategy"]["fallback"]
    env = adapter.build_state(adapter.parse(req(priority="reduce_cost_waste", week=40)))
    pol = adapter.optimization("hh_student", "durham_nc", "normal")
    pol.reset()
    assert np.allclose(order_vector(r), pol.act(env._obs_dict()))


def test_availability_uses_ppo_with_forecast_and_equals_the_pure_model(adapter):
    r = adapter.recommend(req(priority="availability", week=40, ppo_seed=1))
    assert r["strategy"]["selected"] == "ppo_forecast" and r["strategy"]["model"]["seed"] == 1
    p = adapter.parse(req(priority="availability", week=40, ppo_seed=1))
    env = adapter.build_state(p)
    expected = ca.action_to_order(adapter.ppo(1).act_vec(env._obs()), env.ref(40), env.on_hand_units(), env.d.step_size)
    final = env._adjust_order(expected, 40)[0] * env.d.step_size
    assert np.allclose(order_vector(r), final)


def test_balanced_follows_the_documented_rule(adapter):
    picks = set()
    for week in range(12, 100, 7):
        r = adapter.recommend(req(week=week))
        c = r["strategy"]["comparison"]
        o, q = r["strategy"]["candidates"]["optimization"]["projected_fill_rate"], r["strategy"]["candidates"]["ppo_forecast"]["projected_fill_rate"]
        expect = "optimization" if o >= q - ca.BALANCED_FILL_TOLERANCE - 1e-3 else "ppo_forecast"
        if abs((o - q) + ca.BALANCED_FILL_TOLERANCE) > 2e-3:                     # skip rounding-edge cases
            assert r["strategy"]["selected"] == expect
        assert c["tolerance"] == ca.BALANCED_FILL_TOLERANCE and "decision" in c
        picks.add(r["strategy"]["selected"])
    assert picks <= set(ca.STRATEGIES)


def test_the_two_plans_differ_between_priorities(adapter):
    a = order_vector(adapter.recommend(req(priority="reduce_cost_waste", week=50)))
    b = order_vector(adapter.recommend(req(priority="availability", week=50)))
    assert not np.allclose(a, b)


# ---------------------------------------------------------------- constraints
@pytest.mark.parametrize("priority", ca.PRIORITIES)
def test_plans_respect_all_environment_constraints(adapter, priority):
    for week in (12, 25, 40, 55, 70, 90, 103):
        r = adapter.recommend(req(week=week, priority=priority))
        assert r["status"] == "ok", r["errors"]
        c = r["constraints"]
        assert c["within_budget"] and c["whole_packages"] and c["diet_allergen_ok"] and c["stock_cap_ok"] and c["storage_ok"]
        assert r["totals"]["total_cost"] <= r["totals"]["weekly_budget"] + 1e-6
        if r["strategy"]["selected"] == "optimization":
            assert not c["adjusted_by_environment_pipeline"], "the MILP order is already feasible"
        elif c["adjusted_by_environment_pipeline"]:                # a PPO proposal may be trimmed by the pipeline; that must be reported
            assert c["pipeline_notes"]
        for s, v in r["inventory"]["storage"].items():
            assert v["used_l"] <= v["capacity_l"] + CAP_TOL


def test_given_inventory_is_used_and_reported(adapter):
    r = adapter.recommend(req(priority="reduce_cost_waste", inventory={"rice": 5.0, "milk": {"chilled": 2.0}, "chicken_breast": {"frozen": 1.0}}))
    inv = {i["product_id"]: i for i in r["inventory"]["items"]}
    assert inv["rice"]["on_hand_units"] == 5.0 and inv["milk"]["chilled_units"] == 2.0 and inv["chicken_breast"]["frozen_units"] == 1.0
    assert "rice" not in {i["product_id"] for i in r["shopping_list"]}                      # 5 kg on hand: nothing to buy
    assert not any(w["code"] == "default_inventory" for w in r["warnings"])
    assert any(w["code"] == "default_inventory" for w in adapter.recommend(req())["warnings"])


def test_diet_and_allergen_exclusions_are_respected_for_a_vegan_household(adapter):
    r = adapter.recommend(req(household_id="hh_23", priority="availability"))
    assert r["status"] == "ok" and r["excluded_products"]
    assert not ({i["product_id"] for i in r["shopping_list"]} & set(r["excluded_products"]))


# ---------------------------------------------------------------- unsupported scenarios and errors
def test_other_households_regions_and_budgets_fall_back_to_optimization_with_a_flag(adapter):
    for kw in ({"household_id": "hh_family"}, {"region_id": "san_francisco_ca"}, {"budget_scenario": "tight"}):
        r = adapter.recommend(req(priority="availability", **kw))
        assert r["status"] == "ok" and r["strategy"]["selected"] == "optimization" and r["strategy"]["fallback"] is True
        assert r["strategy"]["fallback_reason"] == "ppo_unsupported_scenario" and any(w["code"] == "ppo_unsupported_scenario" for w in r["warnings"])
        assert r["constraints"]["within_budget"]
    r = adapter.recommend(req(priority="reduce_cost_waste", household_id="hh_family"))
    assert r["strategy"]["fallback"] is False and not any(w["code"] == "ppo_unsupported_scenario" for w in r["warnings"])


@pytest.mark.parametrize("bad,code,status", [
    ({"forecast_variant": "with_events"}, "forecast_variant", "unsupported"),
    ({"storage_scenario": "limited"}, "storage_scenario", "unsupported"),
    ({"week": 3}, "week_out_of_range", "unsupported"),
    ({"week": 104}, "week_out_of_range", "unsupported"),
    ({"budget_scenario": "generous"}, "budget_scenario", "unsupported"),
    ({"schema_version": "2.0"}, "schema_version", "unsupported"),
    ({"priority": "cheapest"}, "invalid_priority", "error"),
    ({"week": "70"}, "invalid_week", "error"),
    ({"weeks_remaining": 0}, "invalid_weeks_remaining", "error"),
    ({"ppo_seed": 7}, "invalid_ppo_seed", "error"),
    ({"inventory": {"unicorn_meat": 1}}, "unknown_product", "error"),
    ({"inventory": {"milk": -1}}, "negative_quantity", "error"),
    ({"inventory": {"milk": "lots"}}, "invalid_number", "error"),
    ({"inventory": {"rice": {"frozen": 1}}}, "not_freezable", "error"),
    ({"inventory": {"rice": 500}}, "inventory_exceeds_capacity", "unsupported"),
    ({"inventory": {"milk": 1}, "lots": []}, "invalid_inventory", "error"),
    ({"household_id": "hh_nobody"}, "scenario_not_available", "unsupported"),
])
def test_unsupported_and_invalid_requests_are_flagged_clearly(adapter, bad, code, status):
    r = adapter.recommend(req(**bad))
    assert r["status"] == status and r["errors"][0]["code"] == code and r["errors"][0]["message"]
    assert "shopping_list" not in r
    json.dumps(r)


def test_non_object_request_is_an_error(adapter):
    assert adapter.recommend([1, 2])["errors"][0]["code"] == "invalid_request"


def test_missing_forecast_is_reported_not_replaced(adapter):
    from tests.mock_forecasts import mock_export_frame
    from src.forecast_table import ForecastTable
    frame = mock_export_frame()
    frame = frame[frame["week"] != 60]
    broken = CoachAdapter(table=ForecastTable(frame, adapter.ids))
    r = broken.recommend(req(week=60))
    assert r["status"] == "unsupported" and r["errors"][0]["code"] == "forecast_unavailable"
    assert broken.recommend(req(week=61))["status"] == "ok"


def test_partial_horizon_is_flagged_and_supported(adapter):
    r = adapter.recommend(req(priority="availability", week=60, weeks_remaining=4))
    assert r["status"] == "ok" and any(w["code"] == "partial_horizon" for w in r["warnings"])
    last = adapter.recommend(req(priority="reduce_cost_waste", week=60, weeks_remaining=1))
    assert last["status"] == "ok"


# ---------------------------------------------------------------- equivalence with the pure models (validation episode only)
def test_adapter_reproduces_the_pure_models_over_a_full_validation_episode(adapter):
    ref = cc.reference(adapter.table(), adapter)
    opt = cc.rolling_episode(adapter, "reduce_cost_waste")["metrics"]
    ppo = cc.rolling_episode(adapter, "availability")["metrics"]
    for got, want in ((opt, ref["optimization"]), (ppo, ref["ppo_forecast_seed0"])):
        for k in ("reward", "cash_spend", "waste_cost", "waste_units", "stockout_units", "stockout_value", "fill_rate_value"):
            assert got[k] == pytest.approx(want[k], rel=1e-9, abs=1e-9), k


def test_balanced_episode_runs_end_to_end_with_feasible_orders(adapter):
    r = cc.rolling_episode(adapter, "balanced", keep_responses=True)
    assert len(r["responses"]) == 13 and set(r["selected_per_week"]) <= set(ca.STRATEGIES)
    assert all(x["constraints"]["within_budget"] and x["constraints"]["storage_ok"] for x in r["responses"])
    assert np.isfinite(r["metrics"]["reward"])


def test_request_does_not_mutate_cached_state_between_calls(adapter):
    a = adapter.recommend(req(week=45, priority="reduce_cost_waste"))
    adapter.recommend(req(week=90, priority="availability", inventory={"rice": 2}))
    b = adapter.recommend(req(week=45, priority="reduce_cost_waste"))
    assert a == b


# ---------------------------------------------------------------- HTTP server
def test_http_server_serves_health_contract_and_recommendations(adapter):
    from src.coach_server import make_server
    server = make_server(0, adapter)
    port = server.server_address[1]
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert json.load(urllib.request.urlopen(base + "/health"))["status"] == "ok"
        contract = json.load(urllib.request.urlopen(base + "/contract"))
        assert contract["priorities"] == list(ca.PRIORITIES) and contract["ppo_supported_scenario"]["household_id"] == "hh_student"
        post = urllib.request.Request(base + "/recommend", data=json.dumps(req(priority="reduce_cost_waste")).encode(),
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(post) as resp:
            assert resp.headers["Access-Control-Allow-Origin"] == "*"
            body = json.load(resp)
        assert body == json.loads(json.dumps(adapter.recommend(req(priority="reduce_cost_waste"))))
        bad = urllib.request.Request(base + "/recommend", data=b"{not json", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400 and json.load(e.value)["errors"][0]["code"] == "invalid_json"
        unsupported = urllib.request.Request(base + "/recommend", data=json.dumps(req(week=2)).encode())
        assert json.load(urllib.request.urlopen(unsupported))["status"] == "unsupported"
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- frozen state untouched
def test_frozen_evaluation_state_is_untouched():
    from src import final_eval
    final_eval.verify_freeze()                                   # code, data, checkpoints, settings still match the manifest
    assert (ROOT / "results" / "final" / "test" / "report.md").exists()
