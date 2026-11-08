"""Weekly demand forecasts for the RL side: a strict, read-only table interface.

This module only READS a forecast file that the ML side produces. It never trains or generates forecasts, never reads
`src/forecast.py`, and never falls back to actual demand.

Rules (see docs/RL_EVALUATION.md):
- A forecast for week t is used only if it is verifiably out-of-sample: the file must say which target weeks the generating model
  was fitted on (`trained_through_week < week`), or flag the row with `out_of_sample == 1`. A file with neither column is rejected
  as a whole, because in-sample predictions cannot be told apart. Rows with `forecast_source == "in_sample"` are rejected too.
- Missing, invalid or non-finite forecasts raise `ForecastUnavailable` with the reason. Nothing is imputed, and the actual-demand
  columns of the file (`actual_units`, `demand_units`) are never loaded.
- Two variants: `no_events` (guest/away flags unknown) and `with_events` (the household knows guest/away weeks in advance).

Accepted columns
----------------
required: household_id, week, product_id
values:   forecast_no_events, forecast_with_events           (canonical)
          forecast_units_no_events, forecast_units           (legacy names of data/forecast_weekly.csv; `forecast_units` = with events)
validity: trained_through_week (int)  and/or  out_of_sample (0/1);  optional: forecast_source
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("no_events", "with_events")
_VALUE_ALIASES = {
    "no_events": ("forecast_no_events", "forecast_units_no_events"),
    "with_events": ("forecast_with_events", "forecast_units"),
}
_USECOLS = {"household_id", "week", "product_id", "trained_through_week", "out_of_sample", "forecast_source",
            *[c for names in _VALUE_ALIASES.values() for c in names]}   # actual-demand columns are deliberately not loaded


class ForecastUnavailable(RuntimeError):
    """A forecast-dependent component needs a forecast that does not exist or is not verifiably out-of-sample."""


class ForecastTable:
    def __init__(self, frame: pd.DataFrame, product_ids: list[str]):
        self.product_ids = list(product_ids)
        self.problem: str | None = None            # why the whole table is unusable, if it is
        df = frame[[c for c in frame.columns if c in _USECOLS]].copy()
        missing = {"household_id", "week", "product_id"} - set(df.columns)
        if missing:
            raise ValueError(f"forecast table is missing required columns: {sorted(missing)}")
        self.values: dict[str, str] = {}
        for variant, names in _VALUE_ALIASES.items():
            found = [c for c in names if c in df.columns]
            if found:
                self.values[variant] = found[0]
        if not self.values:
            raise ValueError("forecast table has no forecast column (forecast_no_events / forecast_with_events)")
        valid = pd.Series(True, index=df.index)
        if "trained_through_week" in df.columns:
            valid &= df["trained_through_week"] < df["week"]
        elif "out_of_sample" in df.columns:
            valid &= df["out_of_sample"].astype(int) == 1
        else:
            self.problem = ("cannot verify that forecasts are out-of-sample: the file has neither `trained_through_week` "
                            "nor `out_of_sample`")
            valid &= False
        if "trained_through_week" in df.columns and "out_of_sample" in df.columns:
            valid &= df["out_of_sample"].astype(int) == 1
        if "forecast_source" in df.columns:
            valid &= df["forecast_source"].astype(str).str.lower() != "in_sample"
        df["_valid"] = valid.to_numpy()
        self._lookup: dict[tuple[str, str, int, str], tuple[float, bool]] = {}
        for variant, col in self.values.items():
            for hh, wk, pid, val, ok in zip(df["household_id"], df["week"].astype(int), df["product_id"], df[col], df["_valid"]):
                self._lookup[(variant, hh, wk, pid)] = (float(val), bool(ok))     # type: ignore[index]

    @classmethod
    def from_csv(cls, path: str | Path, product_ids: list[str]) -> "ForecastTable":
        path = Path(path)
        if not path.exists():
            raise ForecastUnavailable(f"forecast file not found: {path}")
        return cls(pd.read_csv(path, usecols=lambda c: c in _USECOLS), product_ids)

    def has_variant(self, variant: str) -> bool:
        return variant in self.values

    def get(self, household_id: str, week: int, variant: str = "no_events") -> np.ndarray:
        """Forecast vector (purchase units, products.csv order) for one household and week, or ForecastUnavailable."""
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if variant not in self.values:
            raise ForecastUnavailable(f"forecast variant {variant!r} is not in the file")
        if self.problem:
            raise ForecastUnavailable(self.problem)
        out = np.empty(len(self.product_ids))
        for i, pid in enumerate(self.product_ids):
            entry = self._lookup.get((variant, household_id, int(week), pid))
            if entry is None:
                raise ForecastUnavailable(f"no {variant} forecast for {household_id} / {pid} / week {week}")
            value, ok = entry
            if not ok:
                raise ForecastUnavailable(f"{variant} forecast for {household_id} / week {week} is not verifiably out-of-sample")
            if not np.isfinite(value) or value < 0:
                raise ForecastUnavailable(f"invalid forecast value for {household_id} / {pid} / week {week}")
            out[i] = value
        return out

    def missing_weeks(self, household_id: str, weeks, variant: str = "no_events") -> list[int]:
        """Weeks in `weeks` for which `get` would fail (empty list = fully covered)."""
        bad = []
        for w in weeks:
            try:
                self.get(household_id, int(w), variant)
            except ForecastUnavailable:
                bad.append(int(w))
        return bad

    def require(self, household_id: str, weeks, variant: str = "no_events") -> None:
        bad = self.missing_weeks(household_id, weeks, variant)
        if bad:
            reason = self.problem or "missing or not verifiably out-of-sample"
            raise ForecastUnavailable(f"{variant} forecasts for {household_id} unavailable for weeks {bad[:6]}"
                                      f"{'...' if len(bad) > 6 else ''} ({len(bad)} weeks): {reason}")

    def forecast_fn(self, household_id: str, variant: str = "no_events"):
        """Callable for GroceryEnv(forecast_fn=...): week -> forecast vector. Raises ForecastUnavailable when invalid."""
        return lambda week: self.get(household_id, int(week), variant)
