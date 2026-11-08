"""Fast tests for the PPO pipeline pieces (no real training)."""
from __future__ import annotations

import numpy as np
import pytest

from mock_forecasts import mock_export_frame
from src import train_rl
from src.forecast_table import ForecastTable
from src.env import GroceryEnv


def test_symmetric_wrapper_maps_zero_to_the_average_rule_action():
    env = train_rl.SymmetricActionWrapper(train_rl.OrderUpToWrapper(GroceryEnv()))
    assert env.action_space.low.min() == -1 and env.action_space.high.max() == 1 and env.action_space.shape == (36,)
    assert np.allclose(env.action(np.zeros(36)), 0.5) and np.allclose(env.action(np.full(36, -1.0)), 0) and np.allclose(env.action(np.full(36, 1.0)), 1)
    assert np.allclose(env.action(np.full(36, 5.0)), 1.0)          # out-of-range values are clipped


def test_training_windows_start_in_weeks_12_to_51_and_end_by_week_63():
    env = train_rl.make_env(False, None)
    base = env
    while hasattr(base, "env"):
        base = base.env
    starts = set()
    for seed in range(60):
        env.reset(seed=seed)
        starts.add(base.start_week)
        assert 12 <= base.start_week <= 51 and base.start_week + 12 <= 63
    assert len(starts) > 10                                          # really sampled, not constant


def test_forecast_and_ablation_envs_have_the_same_observation_layout():
    table = ForecastTable(mock_export_frame(), GroceryEnv().d.product_ids)
    a, b = train_rl.make_env(True, table), train_rl.make_env(False, table)
    oa, _ = a.reset(seed=1)
    ob, _ = b.reset(seed=1)
    assert oa.shape == ob.shape == (190,) and oa[-1] == 1.0 and ob[-1] == 0.0
    assert np.all(ob[4 * 36:5 * 36] == 0) and np.any(oa[4 * 36:5 * 36] > 0)
    assert np.array_equal(oa[:4 * 36], ob[:4 * 36])                # everything except the forecast block is identical


def test_training_env_terminates_and_sb3_does_not_bootstrap():
    venv = train_rl.make_vec(False, None, 1, normalize_reward=False)
    venv.reset()
    for _ in range(13):
        _, _, dones, infos = venv.step([np.zeros(36, dtype=np.float32)])
    assert dones[0] and infos[0]["TimeLimit.truncated"] is False and "episode" in infos[0]


def test_tiny_training_run_writes_checkpoints_and_validation_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(train_rl, "CKPT", tmp_path)
    r = train_rl.train_one("no_forecast", seed=0, timesteps=2048, eval_every=2048, table=None)
    assert list(r["curve"]["timesteps"]) == [0, 2048] and not r["nan_or_inf"]
    assert (tmp_path / "no_forecast_seed0" / "step_0000000.zip").exists()
    assert np.isfinite(r["curve"]["val_reward"]).all()
