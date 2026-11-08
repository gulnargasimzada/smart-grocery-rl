"""PPO pilot: train with and without Ridge forecasts, select checkpoints on the VALIDATION split only.

Two agents under identical settings (same seeds, same budget, same hyperparameters, same 190-value observation layout):
    forecast     the observation contains the verified no_events Ridge forecasts (data/forecast_weekly.csv)
    no_forecast  ablation: forecast block is zeros and forecast_available = 0

Training episodes: 13 weeks, start week sampled from 12..51 (so every forecast is a Ridge prediction, never the MA4 fallback; episodes end by week 63).
Checkpoint selection and all reported numbers use the validation split (weeks 64-77) only. Test weeks 78-103 are never run; this module cannot
reach them (evaluation goes through src.evaluate, which refuses the test split).

Action space: the agent acts in [-1, 1]^36 (SB3 recommends symmetric Box spaces) and an affine map sends it to the existing normalized wrapper
action a = (x + 1) / 2 in [0, 1]^36. Consequence: x = 0 is a = 0.5, which is exactly the Average Rule baseline (1 week of cover). The initial policy is
therefore close to the Average Rule (this is an initial-policy choice, not a reward or constraint change; the step-0 checkpoint is evaluated and
reported so improvement over the initial policy is visible). `log_std_init` is -1 so the initial exploration noise is moderate.

Run:  python -m src.train_rl --timesteps 300000 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env import GroceryEnv
from src.evaluate import EpisodeSpec, run_episode
from src.forecast_table import ForecastTable
from src.policies import RLPolicy
from src.rl_wrapper import OrderUpToWrapper

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "rl_pilot"
CKPT = ROOT / "models_rl"                      # all checkpoints (untracked); only the selected ones are copied into results/
TRAIN_START_RANGE = (12, 51)                   # forecast-aware agent only sees Ridge predictions (weeks >= 12); windows end by week 63
SELECT_STARTS = [64, 65]                       # complete 13-week validation windows (they overlap; selection is correlated)
REPORT_START = 64                              # canonical validation episode used for the baseline comparison
HOUSEHOLD, REGION = "hh_student", "durham_nc"

HYPERPARAMETERS = dict(
    n_envs=4, n_steps=512, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=1.0, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, log_std_init=-1.0, net_arch=[64, 64], normalize_reward=True,
)


class SymmetricActionWrapper(gym.ActionWrapper):
    """Agent acts in [-1, 1]^n; the wrapped OrderUpToWrapper expects [0, 1]^n."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n = env.action_space.shape[0]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)

    def action(self, action):
        return (np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0) + 1.0) / 2.0


def load_table() -> ForecastTable:
    products = pd.read_csv(ROOT / "data" / "products.csv")["product_id"].tolist()
    table = ForecastTable.from_csv(ROOT / "data" / "forecast_weekly.csv", products)     # raises ForecastUnavailable if unusable
    table.require(HOUSEHOLD, range(TRAIN_START_RANGE[0], 78), "no_events")
    return table


def make_env(use_forecast: bool, table: ForecastTable | None):
    fn = table.forecast_fn(HOUSEHOLD, "no_events") if (use_forecast and table is not None) else None
    base = GroceryEnv(HOUSEHOLD, REGION, "normal", "default", forecast_fn=fn, start_range=TRAIN_START_RANGE)
    return Monitor(SymmetricActionWrapper(OrderUpToWrapper(base)))


def make_vec(use_forecast: bool, table, n_envs: int, normalize_reward: bool):
    venv = DummyVecEnv([lambda: make_env(use_forecast, table) for _ in range(n_envs)])
    if normalize_reward:      # scales the learning signal only (running return std); the environment reward itself is unchanged
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=HYPERPARAMETERS["gamma"])
    return venv


def policy_from_model(model: PPO, name: str, use_forecast: bool) -> RLPolicy:
    return RLPolicy(lambda obs: (np.clip(model.predict(obs, deterministic=True)[0], -1, 1) + 1.0) / 2.0, name=name,
                    uses_forecast=use_forecast)


def validate(model: PPO, name: str, use_forecast: bool, table, starts=SELECT_STARTS) -> dict:
    """Mean validation metrics over the given complete validation windows (deterministic policy)."""
    pol = policy_from_model(model, name, use_forecast)
    rows = [run_episode(EpisodeSpec("validation", s), pol, table if use_forecast else None, "no_events") for s in starts]
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("policy", "split", "start_week", "end_week")}
    return out


class ValidationCallback(BaseCallback):
    """Every `eval_every` timesteps: save a checkpoint and score it on the validation windows (never on test weeks)."""

    def __init__(self, name: str, use_forecast: bool, table, ckpt_dir: Path, eval_every: int):
        super().__init__()
        self.name, self.use_forecast, self.table = name, use_forecast, table
        self.ckpt_dir, self.eval_every = ckpt_dir, eval_every
        self.rows: list[dict] = []
        self._next = 0

    def _record(self):
        step = int(self.num_timesteps)
        path = self.ckpt_dir / f"step_{step:07d}.zip"
        self.model.save(path)
        val = validate(self.model, self.name, self.use_forecast, self.table)
        buf = self.model.ep_info_buffer
        train_r = float(np.mean([e["r"] for e in buf])) if len(buf) else float("nan")
        self.rows.append({"timesteps": step, "checkpoint": path.name, "train_ep_reward": train_r,
                          **{f"val_{k}": v for k, v in val.items()}})

    def _on_training_start(self):
        self._record()                               # step 0: the initial (near Average Rule) policy
        self._next = self.eval_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._record()
            self._next += self.eval_every
        return True

    def _on_training_end(self):
        if not self.rows or self.rows[-1]["timesteps"] != int(self.num_timesteps):
            self._record()


def train_one(agent: str, seed: int, timesteps: int, eval_every: int, table) -> dict:
    use_forecast = agent == "forecast"
    hp = HYPERPARAMETERS
    ckpt_dir = CKPT / f"{agent}_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    venv = make_vec(use_forecast, table, hp["n_envs"], hp["normalize_reward"])
    model = PPO("MlpPolicy", venv, n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
                learning_rate=hp["learning_rate"], gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"],
                ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"], seed=seed, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=hp["net_arch"], vf=hp["net_arch"]), log_std_init=hp["log_std_init"]), verbose=0)
    cb = ValidationCallback(f"ppo_{agent}", use_forecast, table, ckpt_dir, eval_every)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=cb)
    seconds = time.time() - t0
    curve = pd.DataFrame(cb.rows)
    curve.insert(0, "seed", seed)
    curve.insert(0, "agent", agent)
    best = curve.loc[curve["val_reward"].idxmax()]              # earliest checkpoint wins ties (idxmax returns the first)
    return {"agent": agent, "seed": seed, "seconds": seconds, "curve": curve, "selected": best.to_dict(),
            "nan_or_inf": bool(not np.isfinite(curve["val_reward"]).all())}


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO pilot (validation-only selection)")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--eval-every", type=int, default=20_480)       # multiple of n_envs * n_steps = 2048
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--agents", nargs="+", default=["forecast", "no_forecast"], choices=["forecast", "no_forecast"])
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    table = load_table()
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        runs = []
        for agent in args.agents:
            print(f"[train] agent={agent} seed={seed} timesteps={args.timesteps}", flush=True)
            r = train_one(agent, seed, args.timesteps, args.eval_every, table)
            print(f"        {r['seconds']:.0f}s, best validation checkpoint {r['selected']['checkpoint']} "
                  f"(val reward {r['selected']['val_reward']:.3f})", flush=True)
            runs.append(r)
        pd.concat([r["curve"] for r in runs], ignore_index=True).to_csv(OUT / f"learning_curves_seed{seed}.csv", index=False)
        meta = {"seed": seed, "timesteps": args.timesteps, "eval_every": args.eval_every, "train_start_range": TRAIN_START_RANGE,
                "select_starts": SELECT_STARTS, "hyperparameters": HYPERPARAMETERS,
                "runs": [{"agent": r["agent"], "seconds": round(r["seconds"], 1), "nan_or_inf": r["nan_or_inf"]} for r in runs]}
        (OUT / f"run_meta_seed{seed}.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"wrote learning_curves_seed{seed}.csv", flush=True)


if __name__ == "__main__":
    main()
