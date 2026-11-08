"""Baseline shopping policies for GroceryEnv (no learning).

Every policy exposes  act(obs_dict) -> np.ndarray  (order quantities in purchase units, shape (n_products,)) and reset().
`obs_dict` is `info["obs_dict"]` from GroceryEnv (inventory per product, week, ref_demand = trailing 4-week mean demand).
Policies only use information available at decision time. The environment still applies diet, stock cap, storage and budget.

Package-aware: purchases are whole packages, so the baselines return multiples of the package step themselves.

Policies
--------
BuyNothing     diagnostic floor, never a real baseline
Fixed          "same quantities every week": a standing order whose long-run average equals the mean weekly demand of the
               TRAINING weeks 4-63. A fractional need is accumulated and a package is delivered whenever a full step is
               reached (same schedule every week, independent of the inventory).
FixedRounded   diagnostic variant: the same weekly quantity rounded to the nearest package by the environment
               (items below half a package are never bought)
ForecastRule   "buy the ML forecast minus current stock": order up to the forecast demand of the week, in whole packages (rounded up).
               Needs a verified out-of-sample forecast (`obs["forecast"]`); raises ForecastUnavailable otherwise, never a fallback.
RLPolicy       interface for a trained agent: wraps a `predict(obs_vector) -> action in [0, 1]^36` callable. It acts in the normalized
               action space of `OrderUpToWrapper`; no agent is trained yet.
AverageRule    "buy the recent average consumption": order up to `cover_weeks` x the trailing 4-week mean demand
               (net of stock on hand), in whole packages (rounded up when a deficit exists)
"""
from __future__ import annotations

import numpy as np

from src.forecast_table import ForecastUnavailable

EPS = 1e-9


def on_hand(obs: dict, product_ids: list[str]) -> np.ndarray:
    inv = obs["inventory"]
    return np.array([inv[p]["chilled"] + inv[p]["frozen"] for p in product_ids])


class Policy:
    name = "policy"
    normalized_action = False      # True for policies that act in the normalized RL action space
    uses_forecast = False          # True if the policy (or the observation it was trained on) needs verified forecasts

    def reset(self) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class BuyNothing(Policy):
    name = "buy_nothing"

    def __init__(self, n_products: int):
        self.n = n_products

    def act(self, obs: dict) -> np.ndarray:
        return np.zeros(self.n)


class Fixed(Policy):
    name = "fixed"

    def __init__(self, weekly_plan: np.ndarray, step_size: np.ndarray):
        self.plan = np.asarray(weekly_plan, float)
        self.step = np.asarray(step_size, float)
        self.reset()

    def reset(self) -> None:
        self.acc = np.zeros_like(self.plan)

    def act(self, obs: dict) -> np.ndarray:
        self.acc += self.plan
        n = np.floor(self.acc / self.step + EPS)
        order = n * self.step
        self.acc -= order
        return order


class FixedRounded(Policy):
    name = "fixed_rounded"

    def __init__(self, weekly_plan: np.ndarray):
        self.plan = np.asarray(weekly_plan, float)

    def act(self, obs: dict) -> np.ndarray:
        return self.plan.copy()


class AverageRule(Policy):
    name = "average_rule"

    def __init__(self, product_ids: list[str], step_size: np.ndarray, cover_weeks: float = 1.0):
        self.ids = product_ids
        self.step = np.asarray(step_size, float)
        self.cover = cover_weeks

    def act(self, obs: dict) -> np.ndarray:
        need = self.cover * np.asarray(obs["ref_demand"]) - on_hand(obs, self.ids)
        n = np.where(need > EPS, np.ceil(need / self.step - EPS), 0.0)
        return n * self.step


class ForecastRule(Policy):
    name = "forecast_rule"
    uses_forecast = True

    def __init__(self, product_ids: list[str], step_size: np.ndarray):
        self.ids = product_ids
        self.step = np.asarray(step_size, float)

    def act(self, obs: dict) -> np.ndarray:
        fc = obs.get("forecast")
        if fc is None:
            raise ForecastUnavailable("forecast_rule needs a forecast, but the environment has no verified forecast")
        need = np.array([fc[p] for p in self.ids]) - on_hand(obs, self.ids)
        n = np.where(need > EPS, np.ceil(need / self.step - EPS), 0.0)
        return n * self.step


class RLPolicy(Policy):
    """Interface for a trained agent. `predict(obs_vector)` must return an action in [0, 1]^n (for example
    `lambda obs: model.predict(obs, deterministic=True)[0]` for a Stable-Baselines3 model). Evaluated on the OrderUpToWrapper."""
    name = "rl"
    normalized_action = True

    def __init__(self, predict, name: str = "rl", uses_forecast: bool = False):
        self.predict = predict
        self.name = name
        self.uses_forecast = uses_forecast     # True if the agent was trained with the forecast block in its observation

    def act(self, obs: dict) -> np.ndarray:  # pragma: no cover - the evaluation runner uses act_vec
        raise TypeError("RLPolicy acts on the flat observation vector: use act_vec")

    def act_vec(self, obs_vec: np.ndarray) -> np.ndarray:
        return np.asarray(self.predict(obs_vec), dtype=float)
