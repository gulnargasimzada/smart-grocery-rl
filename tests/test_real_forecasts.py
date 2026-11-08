"""Integration checks on the REAL exported forecasts (data/forecast_weekly.csv from `python -m src.forecast --export`).

The file is git-ignored. If it is missing these tests are skipped with the reason; that is a blocker for forecast-based
training, not a pass. No substitute forecasts are generated here and actual future demand is never used as a forecast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.env import GroceryEnv
from src.forecast_table import ForecastTable

PATH = Path(__file__).resolve().parent.parent / "data" / "forecast_weekly.csv"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason=f"BLOCKER: real forecast file {PATH} not found (git-ignored; export it with "
                                                           "`python -m src.forecast --export` on the ML side)")
PRODUCTS = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "products.csv")["product_id"].tolist()
HOUSEHOLDS = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "households.csv")["household_id"].tolist()


@pytest.fixture(scope="module")
def table():
    return ForecastTable.from_csv(PATH, PRODUCTS)


@pytest.mark.parametrize("variant", ["no_events", "with_events"])
def test_all_households_training_and_validation_weeks_available(table, variant):
    for hh in HOUSEHOLDS:
        assert table.missing_weeks(hh, range(4, 78), variant) == [], hh


def test_methods_and_cutoffs(table):
    df = pd.read_csv(PATH, usecols=["household_id", "week", "forecast_method", "model_trained_through_week", "split"])
    fb = df[df["forecast_method"] == "ma4_fallback"]
    ridge = df[df["forecast_method"] != "ma4_fallback"]
    assert set(fb["week"]) == set(range(4, 12))
    assert (ridge["model_trained_through_week"] < ridge["week"]).all()
    assert ridge[ridge["week"].between(64, 77)]["model_trained_through_week"].max() <= 63
    assert ridge[ridge["week"] >= 78]["model_trained_through_week"].max() <= 77


def test_fallback_weeks_equal_the_trailing_four_week_mean(table):
    """Correctness check using only past demand: weeks 4-11 must equal the mean of the 4 previous recorded weeks."""
    for hh in ("hh_student", "hh_family"):
        env = GroceryEnv(household_id=hh)
        for week in (4, 8, 11):
            assert np.allclose(table.get(hh, week, "no_events"), env.ref(week), atol=1e-3), (hh, week)


def test_forecasts_are_finite_nonnegative_and_ordered_in_the_observation(table):
    env = GroceryEnv(forecast_fn=table.forecast_fn("hh_student"))
    obs, info = env.reset(options={"start_week": 30})
    f = table.get("hh_student", 30)
    assert np.isfinite(f).all() and (f >= 0).all() and f.shape == (36,)
    assert list(info["obs_dict"]["forecast"]) == PRODUCTS
    assert np.allclose(obs[4 * 36:5 * 36], np.clip(f / np.maximum(env.ref(30), 1e-6), 0, 4), atol=1e-6)


def test_real_forecasts_reach_the_wrapper_observation_and_no_out_of_episode_requests(table):
    from src.rl_wrapper import OrderUpToWrapper
    base = table.forecast_fn("hh_student")
    for start in (4, 51, 64):                    # first training window (with MA4 fallback weeks), last training window, validation window
        requested = []
        env = OrderUpToWrapper(GroceryEnv(forecast_fn=lambda w: (requested.append(w), base(w))[1]))
        obs, _ = env.reset(options={"start_week": start})
        ref = env.unwrapped.ref(start)
        assert obs.shape == (190,) and obs[-1] == 1.0
        assert np.allclose(obs[4 * 36:5 * 36], np.clip(table.get("hh_student", start) / np.maximum(ref, 1e-6), 0, 4), atol=1e-6)
        done = False
        while not done:
            obs, _, done, _, info = env.step(np.full(36, 0.5))
        assert sorted(set(requested)) == list(range(start, start + 13))
        assert obs[-1] == 0.0
