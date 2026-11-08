"""Tests for the evaluation framework: splits, boundaries, test-week guard, paired determinism, regression to the diagnostics."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from src import evaluate as ev
from src.diagnose_env import make_policy as diag_policy, run_episode as diag_run
from src.env import GroceryEnv
from src.evaluate import EpisodeSpec, episode_specs, evaluate_policies, run_episode, validate_spec
from src.policies import RLPolicy


def test_specs_stay_inside_their_split():
    for split, (lo, hi) in ev.SPLITS.items():
        for sp in episode_specs(split):
            assert lo <= sp.start_week and sp.end_week <= hi and sp.end_week - sp.start_week == 12
    assert [s.start_week for s in episode_specs("train")] == [4, 17, 30, 43, 51]
    assert max(s.end_week for s in episode_specs("train")) == 63
    assert [(s.start_week, s.end_week) for s in episode_specs("validation")] == [(64, 76)]
    assert [(s.start_week, s.end_week) for s in episode_specs("test")] == [(78, 90), (91, 103)]


def test_training_sampler_range_and_validation_complete_episodes():
    starts = ev.training_starts()
    assert starts[0] == 4 and starts[-1] == 51 and starts[-1] + 12 == 63
    assert ev.validation_starts(overlapping=True) == [64, 65]            # the only complete 13-week windows in 64-77


@pytest.mark.parametrize("split,start", [("train", 52), ("train", 3), ("validation", 52), ("validation", 66), ("validation", 78),
                                         ("test", 77), ("test", 92)])
def test_episodes_that_cross_a_boundary_are_rejected(split, start):
    with pytest.raises(ValueError, match="boundary"):
        validate_spec(EpisodeSpec(split, start))


def test_test_split_is_reserved():
    with pytest.raises(PermissionError, match="reserved"):
        evaluate_policies("test")
    with pytest.raises(PermissionError):
        evaluate_policies("test", confirm_final_test=False)
    monkey = [sys.argv]
    sys.argv = ["evaluate", "--split", "test"]
    try:
        with pytest.raises(SystemExit):
            ev.main()                                                    # the CLI does not offer the test split
    finally:
        sys.argv = monkey[0]


def test_spec_file_lists_all_splits_with_stable_hashes(tmp_path):
    path = ev.write_spec_file(tmp_path / "specs.json")
    data = json.loads(path.read_text())
    assert set(data) == {"train", "validation", "test"}
    assert data["test"]["hash"] == ev.spec_hash(episode_specs("test"))
    assert ev.spec_hash(episode_specs("train")) == ev.spec_hash(episode_specs("train"))


def test_paired_replay_is_deterministic_and_conditions_are_identical():
    spec = EpisodeSpec("validation", 64)
    a, b = run_episode(spec, "average_rule"), run_episode(spec, "average_rule")
    assert a == b
    f = run_episode(spec, "fixed")
    # identical conditions: the same recorded demand/prices -> the same demand value behind every fill rate
    demand_value = {}
    for name in ("buy_nothing", "fixed", "average_rule"):
        r = run_episode(spec, name)
        demand_value[name] = r["stockout_value"] / (1.0 - r["fill_rate_value"])
    assert max(demand_value.values()) == pytest.approx(min(demand_value.values()))


def test_all_requested_metrics_are_reported():
    ep, summary, _ = evaluate_policies("validation")
    for col in ["reward", "cash_spend", "net_cost", "waste_cost", "waste_units", "stockout_units", "stockout_value",
                "fill_rate_value", "protein_coverage", "terminal_basis", "terminal_credit", "terminal_charge"]:
        assert col in summary.columns and np.isfinite(summary[col]).all()
    assert np.allclose(summary["terminal_charge"], summary["terminal_basis"] - summary["terminal_credit"])


@pytest.mark.parametrize("start,split", [(30, "train"), (64, "validation")])
@pytest.mark.parametrize("name", ["buy_nothing", "fixed", "average_rule"])
def test_results_match_the_earlier_diagnostics(name, start, split):
    """Baseline results are preserved: the evaluation module reproduces src.diagnose_env on the same episodes."""
    new = run_episode(EpisodeSpec(split, start), name)
    env = GroceryEnv()
    old = diag_run(env, diag_policy(name, env), start)
    assert new["reward"] == pytest.approx(old["reward"])
    assert new["cash_spend"] == pytest.approx(old["spend"])
    assert new["waste_cost"] == pytest.approx(old["waste_cost"])
    assert new["stockout_value"] == pytest.approx(old["stockout_value"])
    assert new["net_cost"] == pytest.approx(old["net_cost"])


def test_rl_policy_interface_runs_on_the_same_episodes():
    spec = EpisodeSpec("validation", 64)
    rl_half = RLPolicy(lambda obs: np.full(36, 0.5), name="rl_half")
    r = run_episode(spec, rl_half)
    assert r["policy"] == "rl_half"
    assert r["reward"] == pytest.approx(run_episode(spec, "average_rule")["reward"])       # a = 0.5 is the Average Rule
    zero = run_episode(spec, RLPolicy(lambda obs: np.zeros(36)))
    assert zero["reward"] == pytest.approx(run_episode(spec, "buy_nothing")["reward"])
    ep, summary, _ = evaluate_policies("validation", policies=["fixed", rl_half])
    assert list(summary.index) == ["fixed", "rl_half"]
