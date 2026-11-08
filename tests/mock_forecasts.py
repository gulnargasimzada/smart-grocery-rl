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
