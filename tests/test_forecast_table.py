"""Forecast interface tests with synthetic mock forecasts only (no ML output is generated or used)."""
from __future__ import annotations

import numpy as np
import pytest

from mock_forecasts import mock_frame
from src.env import GroceryEnv
from src.evaluate import EpisodeSpec, evaluate_policies, run_episode
from src.forecast_table import ForecastTable, ForecastUnavailable
from src.policies import ForecastRule

IDS = GroceryEnv().d.product_ids


def table(**kw) -> ForecastTable:
    return ForecastTable(mock_frame(**kw), IDS)


def test_valid_walk_forward_forecasts_are_returned_in_product_order():
    t = table()
    f = t.get("hh_student", 30, "no_events")
    assert f.shape == (36,) and np.all(f >= 0)
    assert np.allclose(t.get("hh_student", 30, "with_events"), 1.1 * f)
    assert t.missing_weeks("hh_student", range(4, 104)) == []


def test_in_sample_rows_are_never_used_and_never_replaced():
    t = table(valid="legacy_in_sample")
    with pytest.raises(ForecastUnavailable, match="out-of-sample"):
        t.get("hh_student", 30)                               # training/validation weeks of the legacy export are in-sample
    assert t.get("hh_student", 80).shape == (36,)             # test weeks are flagged out_of_sample
    assert t.missing_weeks("hh_student", range(60, 70)) == list(range(60, 70))
    with pytest.raises(ForecastUnavailable, match="unavailable for weeks"):
        t.require("hh_student", range(60, 70))


def test_file_without_validity_information_is_rejected_as_a_whole():
    t = table(valid="none")
    with pytest.raises(ForecastUnavailable, match="cannot verify"):
        t.get("hh_student", 90)


def test_forecast_source_in_sample_is_rejected():
    df = mock_frame()
    df["forecast_source"] = "walk_forward"
    df.loc[df["week"] == 40, "forecast_source"] = "in_sample"
    t = ForecastTable(df, IDS)
    with pytest.raises(ForecastUnavailable):
        t.get("hh_student", 40)
    assert t.get("hh_student", 41).shape == (36,)


def test_trained_through_week_must_precede_the_target_week():
    df = mock_frame()
    df.loc[df["week"] == 50, "trained_through_week"] = 50      # model saw week 50 itself
    t = ForecastTable(df, IDS)
    with pytest.raises(ForecastUnavailable):
        t.get("hh_student", 50)


def test_missing_and_invalid_values_raise_and_nothing_is_imputed():
    df = mock_frame()
    df = df[~((df["week"] == 60) & (df["product_id"] == "milk"))]
    df.loc[(df["week"] == 61) & (df["product_id"] == "rice"), "forecast_no_events"] = np.nan
    df.loc[(df["week"] == 62) & (df["product_id"] == "rice"), "forecast_no_events"] = -1.0
    t = ForecastTable(df, IDS)
    for w in (60, 61, 62):
        with pytest.raises(ForecastUnavailable):
            t.get("hh_student", w)
    with pytest.raises(ForecastUnavailable):
        t.get("hh_other", 70)


def test_actual_demand_columns_are_never_loaded(tmp_path):
    df = mock_frame()
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    t = ForecastTable.from_csv(path, IDS)
    assert 999.0 not in {v for v, _ in t._lookup.values()}
    only_actuals = df.drop(columns=["forecast_no_events", "forecast_with_events"])
    only_actuals.to_csv(path, index=False)
    with pytest.raises(ValueError, match="no forecast column"):
        ForecastTable.from_csv(path, IDS)
    with pytest.raises(ForecastUnavailable, match="not found"):
        ForecastTable.from_csv(tmp_path / "missing.csv", IDS)


def test_variants_are_separate():
    df = mock_frame().drop(columns=["forecast_with_events"])
    t = ForecastTable(df, IDS)
    assert t.get("hh_student", 30, "no_events").shape == (36,)
    with pytest.raises(ForecastUnavailable, match="with_events"):
        t.get("hh_student", 30, "with_events")
    with pytest.raises(ValueError):
        t.get("hh_student", 30, "bogus")


def test_legacy_column_names_are_mapped():
    df = mock_frame().rename(columns={"forecast_no_events": "forecast_units_no_events", "forecast_with_events": "forecast_units"})
    t = ForecastTable(df, IDS)
    assert np.allclose(t.get("hh_student", 30, "with_events"), 1.1 * t.get("hh_student", 30, "no_events"))


def test_env_uses_forecast_fn_and_fails_clearly_when_unavailable():
    t = table()
    env = GroceryEnv(forecast_fn=t.forecast_fn("hh_student"))
    obs, info = env.reset(options={"start_week": 30})
    assert obs[-1] == 1.0 and info["obs_dict"]["forecast"] is not None          # forecast_available flag
    bad = GroceryEnv(forecast_fn=table(valid="legacy_in_sample").forecast_fn("hh_student"))
    with pytest.raises(ForecastUnavailable):
        bad.reset(options={"start_week": 30})


def test_forecast_rule_needs_a_forecast_and_matches_average_rule_for_a_trailing_mean_forecast():
    env = GroceryEnv()
    _, info = env.reset(options={"start_week": 30})
    with pytest.raises(ForecastUnavailable):
        ForecastRule(env.d.product_ids, env.d.step_size).act(info["obs_dict"])   # no forecast in the env: no silent fallback
    spec = EpisodeSpec("validation", 64)
    t = table()
    fr = run_episode(spec, "forecast_rule", t)
    ar = run_episode(spec, "average_rule")
    assert fr["reward"] == pytest.approx(ar["reward"]) and fr["cash_spend"] == pytest.approx(ar["cash_spend"])


def test_evaluation_runs_non_forecast_baselines_and_reports_forecast_rule_unavailable():
    ep, summary, unavailable = evaluate_policies("validation")
    assert set(summary.index) == {"buy_nothing", "fixed", "average_rule"} and "forecast_rule" in unavailable
    ep, summary, unavailable = evaluate_policies("validation", table=table(valid="legacy_in_sample"))
    assert "forecast_rule" in unavailable and "unavailable for weeks" in unavailable["forecast_rule"]
    assert set(summary.index) == {"buy_nothing", "fixed", "average_rule"}
    ep, summary, unavailable = evaluate_policies("validation", table=table())
    assert not unavailable and "forecast_rule" in summary.index
