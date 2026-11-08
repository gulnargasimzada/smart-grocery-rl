"""Synthetic mock forecasts for tests (never real ML output)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.env import GroceryData


def mock_frame(household_id="hh_student", scale=1.0, valid="trained_through_week", weeks=range(4, 104)) -> pd.DataFrame:
    """Forecast = trailing 4-week mean demand x scale. `valid` selects how out-of-sample status is encoded."""
    d = GroceryData(household_id, "durham_nc", "normal", "default")
    rows = []
    for w in weeks:
        ref = d.demand[w - 4:w].mean(axis=0) * scale
        for i, pid in enumerate(d.product_ids):
            rows.append({"household_id": household_id, "week": w, "product_id": pid,
                         "forecast_no_events": ref[i], "forecast_with_events": ref[i] * 1.1,
                         "actual_units": 999.0})                    # must never be read
    df = pd.DataFrame(rows)
    if valid == "trained_through_week":
        df["trained_through_week"] = df["week"] - 1                 # walk-forward: fitted on weeks before each target week
    elif valid == "legacy_in_sample":
        df["out_of_sample"] = (df["week"] >= 78).astype(int)        # like data/forecast_weekly.csv: only test weeks are honest
    elif valid != "none":
        raise ValueError(valid)
    return df


def mock_export_frame(households=("hh_student",), tilt=0.0) -> pd.DataFrame:
    """Synthetic frame with exactly the columns of `python -m src.forecast --export` (data/forecast_weekly.csv):
    weeks 4-11 ma4_fallback (cutoff NaN), 12-63 walk-forward (cutoff = week-1), 64-77 cutoff 63, 78-103 cutoff 77;
    legacy out_of_sample = 1 only from week 78. Values are the trailing 4-week mean (x1.1 with events), optionally tilted per product."""
    rows = []
    for hh in households:
        d = GroceryData(hh, "durham_nc", "normal", "default")
        for w in range(4, 104):
            ref = d.demand[w - 4:w].mean(axis=0) * (1.0 + tilt * np.arange(d.n))
            for i, pid in enumerate(d.product_ids):
                fallback = w < 12
                rows.append({
                    "household_id": hh, "week": w, "product_id": pid, "actual_units": 999.0,
                    "forecast_units": ref[i] * 1.1, "forecast_units_no_events": ref[i],
                    "split": "train" if w < 64 else "validation" if w < 78 else "test",
                    "forecast_method": "ma4_fallback" if fallback else "ridge",
                    "model_trained_through_week": np.nan if fallback else (w - 1 if w < 64 else 63 if w < 78 else 77),
                    "out_of_sample": int(w >= 78),
                })
    return pd.DataFrame(rows)
