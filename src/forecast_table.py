"""Weekly demand forecasts for the RL side: a strict, read-only table interface.

This module only READS a forecast file that the ML side produces. It never trains or generates forecasts, never reads
`src/forecast.py`, and never falls back to actual demand.

Rules (see docs/RL_EVALUATION.md):
- A forecast for week t is used only if it is verifiably out-of-sample. Ridge rows are checked against the training-cutoff metadata
  `model_trained_through_week` (alias `trained_through_week`): the model must have been fitted on target weeks strictly before `week`.
  When that metadata column exists it decides; the legacy `out_of_sample` column (1 only from week 78) is IGNORED, because it is 0 for
  valid walk-forward predictions of weeks before 78. `out_of_sample` is used only by old files that have no cutoff column at all.
- `forecast_method == "ma4_fallback"` rows (weeks 4-11, too little history to fit a model) are historical 4-week moving averages, not Ridge
  predictions. They use only earlier weeks, so they are accepted, but they are reported as such (`method()`, `fallback_weeks()`); pass
  `allow_fallback=False` to refuse them.
- A file with neither cutoff metadata nor `out_of_sample` is rejected as a whole. `forecast_source == "in_sample"` rows are rejected too.
- Missing, invalid or non-finite forecasts raise `ForecastUnavailable` with the reason. Nothing is imputed, and the actual-demand columns
  (`actual_units`, `demand_units`) are never loaded.
- Two variants: `no_events` (guest/away flags unknown; the default) and `with_events` (the household knows guest/away weeks in advance).

Accepted columns (data/forecast_weekly.csv from `python -m src.forecast --export`)
----------------------------------------------------------------------------------
required: household_id, week, product_id
values:   forecast_units_no_events (= no_events), forecast_units (= with_events); canonical aliases forecast_no_events / forecast_with_events
validity: model_trained_through_week (NaN for fallback rows) or trained_through_week; forecast_method; optional split, forecast_source;
          legacy out_of_sample
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
FALLBACK = "ma4_fallback"
_CUTOFF_COLUMNS = ("model_trained_through_week", "trained_through_week")
_USECOLS = {"household_id", "week", "product_id", "out_of_sample", "forecast_source", "forecast_method", *_CUTOFF_COLUMNS,
            *[c for names in _VALUE_ALIASES.values() for c in names]}   # actual-demand columns are deliberately not loaded


class ForecastUnavailable(RuntimeError):
    """A forecast-dependent component needs a forecast that does not exist or is not verifiably out-of-sample."""


class ForecastTable:
    def __init__(self, frame: pd.DataFrame, product_ids: list[str], allow_fallback: bool = True):
        self.product_ids = list(product_ids)
        self.allow_fallback = allow_fallback
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
        method = df["forecast_method"].fillna("").astype(str) if "forecast_method" in df.columns else pd.Series("", index=df.index)
        is_fallback = method == FALLBACK
        cutoff_cols = [c for c in _CUTOFF_COLUMNS if c in df.columns]
        if cutoff_cols:
            cutoff = df[cutoff_cols[0]]
            ridge_ok = cutoff < df["week"]                        # NaN compares False: a model row without a cutoff is rejected
            valid &= (is_fallback & self.allow_fallback) | (~is_fallback & ridge_ok)
        elif "out_of_sample" in df.columns:                       # old files without cutoff metadata
            valid &= (df["out_of_sample"].astype(int) == 1) & ~is_fallback
        else:
            self.problem = ("cannot verify that forecasts are out-of-sample: the file has neither `model_trained_through_week` "
                            "nor `out_of_sample`")
            valid &= False
        if "forecast_source" in df.columns:
            valid &= df["forecast_source"].astype(str).str.lower() != "in_sample"
        df["_valid"] = valid.to_numpy()
        df["_method"] = np.where(is_fallback, FALLBACK, "ridge")
        self._lookup: dict[tuple[str, str, int, str], tuple[float, bool, str]] = {}
        for variant, col in self.values.items():
            for hh, wk, pid, val, ok, m in zip(df["household_id"], df["week"].astype(int), df["product_id"], df[col], df["_valid"],
                                               df["_method"]):
                self._lookup[(variant, hh, wk, pid)] = (float(val), bool(ok), m)

    @classmethod
    def from_csv(cls, path: str | Path, product_ids: list[str], allow_fallback: bool = True) -> "ForecastTable":
        path = Path(path)
        if not path.exists():
            raise ForecastUnavailable(f"forecast file not found: {path}")
        return cls(pd.read_csv(path, usecols=lambda c: c in _USECOLS), product_ids, allow_fallback)

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
            value, ok, _ = entry
            if not ok:
                raise ForecastUnavailable(f"{variant} forecast for {household_id} / week {week} is not verifiably out-of-sample"
                                          f"{'' if self.allow_fallback else ' (or is a refused ma4_fallback)'}")
            if not np.isfinite(value) or value < 0:
                raise ForecastUnavailable(f"invalid forecast value for {household_id} / {pid} / week {week}")
            out[i] = value
        return out

    def method(self, household_id: str, week: int, variant: str = "no_events") -> str:
        """'ridge' or 'ma4_fallback' (historical 4-week moving average, not a Ridge prediction) for a valid forecast."""
        self.get(household_id, week, variant)
        return self._lookup[(variant, household_id, int(week), self.product_ids[0])][2]

    def fallback_weeks(self, household_id: str, weeks, variant: str = "no_events") -> list[int]:
        return [int(w) for w in weeks if self.method(household_id, int(w), variant) == FALLBACK]

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
