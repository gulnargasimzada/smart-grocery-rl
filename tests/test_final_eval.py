"""Tests for the final evaluation pipeline. The held-out test split (weeks 78-103) is NEVER executed here:
the guard logic is tested with stubs, and the pipeline itself is exercised on the validation episode (dry run).
"""
from __future__ import annotations

import inspect
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import comparison as cmp
from src import final_eval as fe

ROOT = Path(__file__).resolve().parent.parent
needs_forecasts = pytest.mark.skipif(not (ROOT / "data" / "forecast_weekly.csv").exists(),
                                     reason="BLOCKER: real data/forecast_weekly.csv missing (git-ignored)")
needs_manifest = pytest.mark.skipif(not fe.MANIFEST.exists(), reason="run `python -m src.final_eval --freeze` first")


@needs_forecasts
@needs_manifest
def test_frozen_manifest_matches_the_current_state_and_records_the_settings():
    m = fe.verify_freeze()
    assert m["methods"]["optimization"]["delta"] == 0.4
    assert m["environment_constants"]["stockout_multiplier"] == 2.0
    assert [(e["start_week"], e["end_week"]) for e in m["evaluation"]["test_episodes"]] == [(78, 90), (91, 103)]
    ck = m["methods"]["ppo"]["checkpoints"]
    assert sorted(ck) == sorted(f"{a}_seed{s}" for a in ("forecast", "no_forecast") for s in (0, 1, 2))
    assert all(len(v["sha256"]) == 64 for v in ck.values())
    assert set(m["evaluation"]["metrics"]) >= {"cash_spend", "waste_units", "waste_cost", "stockout_units", "stockout_value", "fill_rate_value", "reward"}
    assert "data/forecast_weekly.csv" in m["data_sha256"] and "libraries" in m


@needs_forecasts
@needs_manifest
def test_changes_to_checkpoints_or_constants_are_detected(tmp_path, monkeypatch):
    for f in cmp.SELECTED.glob("*.zip"):
        shutil.copy(f, tmp_path / f.name)
    victim = tmp_path / "forecast_seed1.zip"
    victim.write_bytes(victim.read_bytes() + b"x")                        # one byte appended
    monkeypatch.setattr(cmp, "SELECTED", tmp_path)
    with pytest.raises(RuntimeError, match="checkpoint: forecast_seed1"):
        fe.verify_freeze()
    monkeypatch.undo()
    monkeypatch.setattr(fe, "STOCKOUT_MULT", 3.0)
    with pytest.raises(RuntimeError, match="environment_constants"):
        fe.verify_freeze()


@needs_forecasts
@needs_manifest
def test_held_out_run_is_refused_without_approval_and_without_running_anything(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(fe, "evaluate_split", lambda split: calls.append(split) or pytest.fail("evaluation must not run"))
    for bad in (None, "", "RUN-HELD-OUT-TEST-000000000000", "yes"):
        with pytest.raises(PermissionError):
            fe.run_test(bad)
    phrase = fe.approval_phrase(fe.load_manifest())
    existing = tmp_path / "test"
    existing.mkdir()
    monkeypatch.setattr(fe, "TEST_DIR", existing)
    with pytest.raises(FileExistsError):                                  # run once: never overwrite
        fe.run_test(phrase)
    assert calls == []
    monkeypatch.setattr(sys, "argv", ["final_eval", "--run-test"])        # CLI without the phrase
    with pytest.raises(PermissionError):
        fe.main()


@needs_forecasts
@needs_manifest
def test_approved_run_path_calls_the_same_pipeline_on_the_test_split_only(monkeypatch, tmp_path):
    """With the right phrase the code path is exercised with a stub: it asks for split 'test' exactly once and writes into TEST_DIR."""
    seen = []
    validation_runs = fe.evaluate_split("validation")                      # real, validation only

    def stub(split):
        seen.append(split)
        return validation_runs

    monkeypatch.setattr(fe, "evaluate_split", stub)
    monkeypatch.setattr(fe, "TEST_DIR", tmp_path / "test_out")
    text = fe.run_test(fe.approval_phrase(fe.load_manifest()))
    assert seen == ["test"] and (tmp_path / "test_out" / "report.md").exists() and "Held-out TEST" in text


def test_the_test_split_is_evaluated_in_exactly_one_place():
    src = inspect.getsource(fe)
    assert src.count('evaluate_split("test")') == 1 and 'evaluate_split("test")' in inspect.getsource(fe.run_test)
    assert 'evaluate_split("validation")' in inspect.getsource(fe.run_dry)


@needs_forecasts
def test_preflight_checks_forecast_metadata_for_the_test_weeks_without_running_them():
    fe.preflight(cmp.load_table(), "test")                                  # Ridge forecasts exist for weeks 78-103, trained earlier


@needs_forecasts
@needs_manifest
def test_dry_run_covers_all_methods_and_seeds_and_declares_no_winner(monkeypatch, tmp_path):
    monkeypatch.setattr(fe, "DRY_DIR", tmp_path)
    text = fe.run_dry()
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert set(runs["split"]) == {"validation"} and set(runs["start_week"]) == {64}
    expected = {"fixed", "average_rule", "forecast_rule", "optimization", *[f"ppo_{a}_seed{s}" for a in cmp.PPO_AGENTS for s in cmp.SEEDS]}
    assert set(runs["method"]) == expected
    t = pd.read_csv(tmp_path / "mean_over_episodes.csv", index_col=0)
    assert {"ppo_forecast_mean", "ppo_no_forecast_mean"} <= set(t.index)
    for m in fe.REPORT_METRICS:
        assert m in text
    ranks = pd.read_csv(tmp_path / "rankings_per_metric.csv", index_col=0)
    assert len(ranks) == len(fe.REPORT_METRICS) and ranks.shape[1] == 6
    assert "no overall winner" in text and "DRY RUN" in text
    assert "winner:" not in text.lower() and "best overall" not in text.lower()


def test_rankings_use_the_right_direction_per_metric():
    t = pd.DataFrame({m: 0.0 for m in cmp.METRICS}, index=fe.AGG_METHODS)
    t["waste_units"] = [5, 4, 3, 1, 2, 6]
    t["fill_rate_value"] = [0.5, 0.6, 0.7, 0.9, 0.8, 0.4]
    r = fe.rankings(t)
    assert r.loc["waste_units (lower better)", "#1"].startswith("Optimization")
    assert r.loc["fill_rate_value (higher better)", "#1"].startswith("Optimization")
    assert r.loc["fill_rate_value (higher better)", "#6"].startswith("PPO without forecast")
