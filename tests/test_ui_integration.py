"""Checks that the static UI, its precomputed plans and the Python backend tell the same story.

- the UI's three priorities map to the same strategies as the backend (src/coach_api.py);
- results/ui/plans.json reproduces the frozen held-out results for the six methods (and carries the balanced-rule replay as a seventh, supplementary entry);
- the page script is valid JavaScript and contains no secrets;
- meal ingredients added on top of a plan obey whole packages, the weekly budget and storage space (fitExtras, run with node).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "ui" / "coach.html").read_text(encoding="utf-8")
PLANS = json.loads((ROOT / "results" / "ui" / "plans.json").read_text())
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def prio_map() -> dict:
    block = HTML[HTML.index("const PRIO={"):HTML.index("const PRIO_ORDER")]
    return {m.group(1): m.group(2) for m in re.finditer(r'\n\s+(\w+):\{title:"[^"]+",short:"[^"]+",method:"(\w+)"', block)}


def test_ui_priorities_map_to_the_same_strategies_as_the_backend():
    from src import coach_api
    ui = prio_map()
    assert ui == {"save": "optimization", "balanced": "balanced_rule", "stocked": "ppo_forecast"}
    backend = {"reduce_cost_waste": "optimization", "availability": "ppo_forecast"}          # fixed strategies of the backend priorities
    assert set(backend) <= set(coach_api.PRIORITIES) and "balanced" in coach_api.PRIORITIES
    assert ui["save"] == backend["reduce_cost_waste"] and ui["stocked"] == backend["availability"]
    assert "balanced_rule" in PLANS["methods"]                                                   # the backend's balanced rule, replayed


def test_the_page_says_it_is_a_static_demo_not_a_live_model_call():
    assert "Static demo: precomputed replays" in HTML and "Nothing here calls a live model" in HTML
    assert "STATIC DEMO" in HTML


def test_plans_json_reproduces_the_frozen_results_and_marks_the_balanced_rule_as_supplementary():
    final = pd.read_csv(ROOT / "results" / "final" / "test" / "mean_over_episodes.csv", index_col=0)
    ft = PLANS["final_test"]
    for k in final.index:
        for c in ("reward", "cash_spend", "waste_cost", "stockout_value", "fill_rate_value"):
            assert ft[k][c] == pytest.approx(final.loc[k, c], abs=1e-3), (k, c)
    assert set(PLANS["methods"]) == {"fixed", "average_rule", "forecast_rule", "optimization", "ppo_forecast", "ppo_no_forecast", "balanced_rule"}
    weeks = PLANS["methods"]["balanced_rule"]["weeks"]
    assert len(weeks) == 26 and {w["strategy"] for w in weeks.values()} <= {"optimization", "ppo_forecast"}
    assert "balanced_rule" in ft and "balanced_rule" not in final.index                          # not part of the frozen six-method table


@needs_node
def test_page_script_is_valid_javascript_and_contains_no_secrets(tmp_path):
    from ui_payload import load_payload, render_page
    page = render_page(load_payload(include_keys=False))
    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}|sk-ant-[0-9A-Za-z_-]{10,}", page)
    scripts = [body for attrs, body in re.findall(r"<script([^>]*)>(.*?)</script>", page, flags=re.S) if "application/json" not in attrs]
    assert scripts                                                    # the embedded data block (application/json) is not code
    for i, s in enumerate(scripts):
        f = tmp_path / f"s{i}.js"
        f.write_text(s)
        out = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[:500]


def run_fit_extras(case: dict) -> dict:
    start = HTML.index("function fitExtras(")
    end = HTML.index("const skipNote=")
    code = HTML[start:end]
    js = textwrap.dedent("""
        const c = %s;
        %s
        const out = fitExtras(c.prods, c.cap, c.pre, c.price, c.co, c.want, c.budget);
        console.log(JSON.stringify(out));
    """) % (json.dumps(case), code)
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[:500]
    return json.loads(r.stdout)


BASE = {"prods": [{"name": "Rice", "step": 1.0, "vol": 1.0, "store": "pantry"}, {"name": "Chicken", "step": 1.0, "vol": 0.7, "store": "fridge"},
                  {"name": "Bananas", "step": 0.5, "vol": 2.5, "store": "pantry"}],
        "cap": {"pantry": 100, "fridge": 100, "freezer": 20}, "pre": {"inv": [[], [], []]}, "price": [2.5, 4.5, 1.5], "co": [0, 0, 0],
        "want": [0, 0, 0], "budget": 75}


@needs_node
def test_meal_extras_are_rounded_up_to_whole_packages():
    r = run_fit_extras({**BASE, "want": [0.3, 1.2, 0.2]})
    assert r["extra"] == [1.0, 2.0, 0.5] and r["skipped"] == []


@needs_node
def test_meal_extras_never_exceed_the_weekly_budget():
    r = run_fit_extras({**BASE, "co": [0, 14, 0], "want": [1, 1, 0], "budget": 65})         # 14 chicken = $63; $2 left
    assert r["extra"] == [0, 0, 0] and [s["why"] for s in r["skipped"]] == ["budget", "budget"]
    r = run_fit_extras({**BASE, "co": [0, 12, 0], "want": [1, 1, 0], "budget": 58})         # $54 spent, $4 left: rice ($2.5) fits, chicken ($4.5) does not
    assert r["extra"] == [1.0, 0, 0] and r["skipped"][0]["why"] == "budget"


@needs_node
def test_meal_extras_never_exceed_storage_space():
    r = run_fit_extras({**BASE, "cap": {"pantry": 2.0, "fridge": 100, "freezer": 20}, "want": [3, 0, 0]})   # 3 kg of rice = 3 L, pantry has 2 L
    assert r["extra"] == [0, 0, 0] and r["skipped"][0]["why"] == "storage"
    r = run_fit_extras({**BASE, "cap": {"pantry": 10.0, "fridge": 100, "freezer": 20}, "pre": {"inv": [[{"q": 8.0}], [], []]}, "want": [3, 0, 0]})   # 8 L already stored
    assert r["extra"] == [0, 0, 0]
    r = run_fit_extras({**BASE, "cap": {"pantry": 10.0, "fridge": 100, "freezer": 20}, "pre": {"inv": [[{"q": 8.0}], [], []]}, "want": [2, 0, 0]})
    assert r["extra"] == [2.0, 0, 0]


@needs_node
def test_extras_are_added_in_order_until_the_budget_runs_out_and_nothing_is_added_partially():
    r = run_fit_extras({**BASE, "budget": 5.0, "want": [1, 0.5, 0]})                         # rice $2.5 fits; chicken $4.5 > $2.5 left
    assert r["extra"] == [1.0, 0, 0] and len(r["skipped"]) == 1


def test_payload_carries_package_and_storage_data_for_the_extras():
    from ui_payload import load_payload
    p = load_payload(include_keys=False)
    assert all({"step", "vol", "store"} <= set(x) for x in p["products"])
    assert {x["id"]: x["step"] for x in p["products"]}["bananas"] == 0.5 and {x["id"]: x["step"] for x in p["products"]}["rice"] == 1.0
    assert all(h["cap"] and set(h["cap"]) == {"pantry", "fridge", "freezer"} for h in p["households"])


@needs_node
def test_projected_fridge_score_has_all_inputs_for_the_precomputed_plans():
    """Week-1 'Projected fridge score' used to show NaN because the plan's `ef` had no spend / protein / stockout fields."""
    assert "ef:{waste_cost:W.waste_cost,spend:W.spend,protein_target_met:W.protein_target_met,stockout_days:W.stockout_days}" in HTML
    start = HTML.index("function fridgeScore(")
    end = HTML.index("\n}", start) + 2
    for method in ("optimization", "balanced_rule", "ppo_forecast"):
        w = PLANS["methods"][method]["weeks"]["78"]
        r = {k: w[k] for k in ("waste_cost", "spend", "protein_target_met", "stockout_days")}
        js = f"const budgetOf=()=>75,nRel=()=>20;{HTML[start:end]}\nconsole.log(JSON.stringify(fridgeScore({json.dumps(r)},{{}})));"
        out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30, check=True).stdout)
        assert isinstance(out["total"], int) and all(isinstance(p[1], (int, float)) and p[1] == p[1] for p in out["parts"]), (method, out)
