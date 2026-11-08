"""Gymnasium action wrapper for RL training: normalized action in [0, 1]^36 -> order quantities.

    target_p  = ACTION_MAX_COVER_WEEKS x a_p x ref_p            order-up-to level in purchase units
                (ref_p = trailing 4-week mean demand of product p, known history)
    deficit_p = max(0, target_p - on_hand_p)                    on_hand = chilled + frozen units
    order_p   = ceil(deficit_p / step_p) x step_p               whole packages, rounded UP when there is a deficit

a = 0 never buys; a = 0.5 orders up to one week of average demand (identical to the AverageRule baseline with 1 week of cover);
a = 1 orders up to two weeks. The proposed order then goes through the unchanged GroceryEnv pipeline (diet/allergen exclusions,
package rounding, stock cap, storage capacity incl. freezer placement, weekly budget). The reward, the observation and every
environment constant are untouched; `info["proposed_order"]` shows the order before the pipeline, `info["order"]` after it.

The observation space is the flat float32 Box of GroceryEnv and the action space is Box(0, 1, (36,)) float32, so the wrapped env can
be passed to Stable-Baselines3 PPO as it is (SB3 is not installed or used yet).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env import GroceryEnv

ACTION_MAX_COVER_WEEKS = 2.0     # a = 1 means "order up to this many weeks of reference demand" (docs/RL_TECHNICAL_DESIGN.md)
EPS = 1e-9


def action_to_order(action: np.ndarray, ref: np.ndarray, on_hand: np.ndarray, step_size: np.ndarray) -> np.ndarray:
    """Pure, deterministic conversion of a normalized action into a proposed order (purchase units, whole packages)."""
    a = np.clip(np.asarray(action, dtype=float), 0.0, 1.0)
    target = ACTION_MAX_COVER_WEEKS * a * ref
    deficit = target - on_hand
    n = np.where(deficit > EPS, np.ceil(deficit / step_size - EPS), 0.0)
    return n * step_size


class OrderUpToWrapper(gym.ActionWrapper):
    def __init__(self, env: GroceryEnv):
        super().__init__(env)
        n = env.unwrapped.d.n
        self.action_space = spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)

    def action(self, action):
        base: GroceryEnv = self.env.unwrapped
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape != (base.d.n,):
            raise ValueError(f"action must have shape ({base.d.n},), got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("action contains NaN or infinite values")
        order = action_to_order(a, base.ref(base.week), base.on_hand_units(), base.d.step_size)
        self._proposed = order
        return order

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        info["proposed_order"] = self._proposed
        return obs, reward, terminated, truncated, info
