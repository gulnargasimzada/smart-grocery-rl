"""Demand forecasting (Person 1).

Goal: predict next week's demand (purchase units, see data/products.csv) for every household and product.
Step 1: simple baselines + the evaluation that every later model is compared against.
Step 2: XGBoost model. It predicts the RATIO  demand / reference  (reference = mean of the last 8 weeks) with
        sample weights = reference, so the training loss equals the absolute error in purchase units (= our MAE).
        Prediction = ratio * reference. Products a household can never eat have reference 0 and get prediction 0.

Step 3: Ridge regression (simple ML baseline) and a validation set to compare the models fairly.
        Split of the weeks (set by the team captain):  TRAIN 0-63 | VALIDATION 64-77 | TEST 78-103.
        Weeks 0-3 cannot be predicted (a forecast needs 4 weeks of history), so the model is trained on target weeks 4-63.
        Models and parameters (Ridge alpha, XGBoost depth / trees) are chosen on the validation weeks only.
        The chosen models are then retrained on weeks 4-77 and scored ONCE on the test weeks.

Step 4: export + API for the team.
        python -m src.forecast            validation comparison, then test results
        python -m src.forecast --export   also writes data/forecast_weekly.csv and trains + saves the final models
                                          (models/forecaster_with_events.json, models/forecaster_no_events.json)
        predict_next_week(history, household_id, week, guest=None, away=None)  -> {product_id: units}
        guest/away=None means "unknown": the model WITHOUT guest/away flags is used.

Run from the repo root:  python -m src.forecast
Time-based split: the test weeks 78-103 are never used for training or for choosing parameters.
Every prediction for week t uses only information from weeks before t, plus the event flags of week t
(holiday is always known; guest/away weeks are known only if the household tells us in advance, so we also
train a model WITHOUT the guest/away flags to show the honest result when they are not known).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import CATEGORY_KCAL_SHARE, HOLIDAY_WEEKS

TRAIN_WEEKS = 78          # first TEST week (weeks before it are train + validation)
VAL_START = 64           # first VALIDATION week (train targets are MIN_HISTORY .. VAL_START-1)
KEYS = ["household_id", "product_id"]
MIN_HISTORY = 4          # first week that has 4 weeks of history
DIET_CODE = {"omnivore": 0, "pescatarian": 1, "vegetarian": 2, "vegan": 3}


def load_weekly(path: str = "data/demand_weekly.csv") -> pd.DataFrame:
    """Weekly demand, sorted so that rolling windows per household and product are correct."""
    weekly = pd.read_csv(path)
    return weekly.sort_values(KEYS + ["week"]).reset_index(drop=True)


def add_baselines(weekly: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Adds baseline predictions for week t using only weeks before t.
    pred_last_week: demand of the previous week
    pred_ma4: mean demand of the previous `window` weeks"""
    out = weekly.copy()
    g = out.groupby(KEYS)["demand_units"]
    out["pred_last_week"] = g.shift(1)
    out["pred_ma4"] = g.transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
    return out


def event_label(df: pd.DataFrame) -> pd.Series:
    """One label per row: away > guest > holiday > normal (a week can only be away or guest, never both)."""
    label = pd.Series("normal", index=df.index)
    label[df["holiday_week"] == 1] = "holiday"
    label[df["guest_week"] == 1] = "guest"
    label[df["away_week"] == 1] = "away"
    return label


def score(df: pd.DataFrame, pred_col: str) -> dict:
    """MAE in purchase units and WMAPE = sum|error| / sum(actual) (a % that is comparable across products)."""
    err = (df[pred_col] - df["demand_units"]).abs()
    return {"MAE": err.mean(), "WMAPE_%": 100 * err.sum() / df["demand_units"].sum()}


def evaluate(df: pd.DataFrame, pred_cols: list, test_start: int = TRAIN_WEEKS, test_end: int | None = None) -> pd.DataFrame:
    """Scores on the weeks test_start .. test_end-1 (default: to the last week), overall and by event type."""
    test = df[df["week"] >= test_start]
    if test_end is not None:
        test = test[test["week"] < test_end]
    test = test.dropna(subset=pred_cols).copy()
    test["event"] = event_label(test)
    rows = []
    for col in pred_cols:
        rows.append({"model": col, "slice": "all", "rows": len(test), **score(test, col)})
        for ev, part in test.groupby("event"):
            rows.append({"model": col, "slice": ev, "rows": len(part), **score(part, col)})
    return pd.DataFrame(rows)


def summary_table(res: pd.DataFrame) -> pd.DataFrame:
    """One row per model: MAE and WMAPE overall, and WMAPE per event type."""
    allrows = res[res["slice"] == "all"].set_index("model")
    out = allrows[["MAE", "WMAPE_%"]].copy()
    for ev in ("normal", "guest", "holiday", "away"):
        out[f"{ev}_%"] = res[res["slice"] == ev].set_index("model")["WMAPE_%"]
    return out.round({"MAE": 4, "WMAPE_%": 1, "normal_%": 1, "guest_%": 1, "holiday_%": 1, "away_%": 1})


# ---------------------------------------------------------------------------------------------------------
# Step 2: features + XGBoost
# ---------------------------------------------------------------------------------------------------------
LAG_FEATURES = ["lag1", "lag2", "lag3", "lag4", "lag52", "ma4", "ma8", "std4"]
HISTORY_EVENT_FEATURES = ["away_last4", "guest_last4", "prev_away", "prev_guest"]
SEASON_FEATURES = ["week_sin", "week_cos"]
STATIC_FEATURES = ["n_adults", "n_children", "eats_out_share", "diet_code", "kcal_per_adult_day",
                   "category_code", "product_code", "kcal_per_unit"]
HOLIDAY_FEATURES = ["holiday_week"]
EVENT_FEATURES = ["guest_week", "away_week"]
FEATURES_WITH_EVENTS = (LAG_FEATURES + HISTORY_EVENT_FEATURES + SEASON_FEATURES + STATIC_FEATURES
                        + HOLIDAY_FEATURES + EVENT_FEATURES)
FEATURES_NO_EVENTS = [f for f in FEATURES_WITH_EVENTS if f not in EVENT_FEATURES]


def build_features(weekly: pd.DataFrame, households: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """One row per household x product x week. All lag/rolling features use weeks before `week` only."""
    df = weekly.sort_values(KEYS + ["week"]).reset_index(drop=True).copy()
    g = df.groupby(KEYS)["demand_units"]
    for k in (1, 2, 3, 4, 52):
        df[f"lag{k}"] = g.shift(k)
    shifted = g.shift(1)
    roll = shifted.groupby([df["household_id"], df["product_id"]])
    df["ma4"] = roll.transform(lambda s: s.rolling(4, min_periods=4).mean())
    df["ma8"] = roll.transform(lambda s: s.rolling(8, min_periods=4).mean())
    df["std4"] = roll.transform(lambda s: s.rolling(4, min_periods=4).std())

    for flag in ("away_week", "guest_week"):                      # what happened in the previous weeks
        gf = df.groupby(KEYS)[flag]
        df[f"prev_{flag.split('_')[0]}"] = gf.shift(1)
        df[f"{flag.split('_')[0]}_last4"] = sum(gf.shift(k).fillna(0) for k in (1, 2, 3, 4))

    angle = 2 * np.pi * (df["week"] % 52) / 52
    df["week_sin"], df["week_cos"] = np.sin(angle), np.cos(angle)

    hh = households.set_index("household_id")
    for col in ("n_adults", "n_children", "eats_out_share", "kcal_per_adult_day"):
        df[col] = df["household_id"].map(hh[col])
    df["diet_code"] = df["household_id"].map(hh["diet"].map(DIET_CODE))
    pr = products.set_index("product_id")
    df["category_code"] = df["product_id"].map(pr["category"].astype("category").cat.codes)
    df["product_code"] = df["product_id"].map({p: i for i, p in enumerate(products["product_id"])})
    df["kcal_per_unit"] = df["product_id"].map(pr["kcal_per_unit"])

    df["reference"] = df["ma8"]                                   # level the ratio is measured against
    return df


XGB_GRID = [{"n_estimators": n, "max_depth": d} for n in (200, 400) for d in (4, 6)]
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
VARIANTS = {"no_guest_away": FEATURES_NO_EVENTS, "with_events": FEATURES_WITH_EVENTS}


def make_model(n_estimators: int = 400, max_depth: int = 6):
    """XGBoost if it is installed, otherwise a scikit-learn model with the same interface (only for testing)."""
    try:
        from xgboost import XGBRegressor
        return "xgboost", XGBRegressor(objective="reg:absoluteerror", n_estimators=n_estimators, learning_rate=0.05,
                                       max_depth=max_depth, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                       tree_method="hist", random_state=0, n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "sklearn-fallback", HistGradientBoostingRegressor(loss="absolute_error", max_iter=n_estimators // 2,
                                                                 learning_rate=0.08, max_depth=max_depth, random_state=0)


def training_rows(df: pd.DataFrame, first_week: int, end_week: int) -> pd.DataFrame:
    return df[(df["week"] >= first_week) & (df["week"] < end_week) & (df["reference"] > 0)]


def fit_forecaster(df: pd.DataFrame, features: list, train_weeks: int = TRAIN_WEEKS, first_week: int = MIN_HISTORY,
                   params: dict | None = None):
    """Trains XGBoost on target weeks first_week .. train_weeks-1. Returns (model name, fitted model)."""
    train = training_rows(df, first_week, train_weeks)
    name, model = make_model(**(params or {}))
    ratio = train["demand_units"] / train["reference"]
    model.fit(train[features], ratio, sample_weight=train["reference"])
    return name, model


def to_units(ratio: pd.Series, df: pd.DataFrame) -> pd.Series:
    """ratio * reference. Rows without a usable reference (no history / never eaten) get NaN / 0."""
    pred = (ratio * df["reference"]).clip(lower=0)
    pred[df["reference"] == 0] = 0.0
    pred[df["reference"].isna()] = np.nan
    return pred


def predict_units(df: pd.DataFrame, model, features: list) -> pd.Series:
    """XGBoost prediction in purchase units."""
    return to_units(pd.Series(model.predict(df[features]), index=df.index), df)


# --- Ridge regression: a simple linear baseline -------------------------------------------------------------
def ridge_matrix(df: pd.DataFrame, include_events: bool) -> pd.DataFrame:
    """Scale-free features for the linear model: lags and averages are divided by the reference level."""
    safe_ref = df["reference"].where(df["reference"] > 0)
    x = pd.DataFrame(index=df.index)
    for c in ("lag1", "lag2", "lag3", "lag4", "ma4"):
        x[f"{c}_ratio"] = (df[c] / safe_ref).fillna(1.0)
    x["std4_ratio"] = (df["std4"] / safe_ref).fillna(0.0)
    for c in SEASON_FEATURES + HISTORY_EVENT_FEATURES + HOLIDAY_FEATURES + (EVENT_FEATURES if include_events else []):
        x[c] = df[c].fillna(0.0)
    cat = pd.Categorical(df["category_code"], categories=range(len(CATEGORY_KCAL_SHARE)))
    return pd.concat([x, pd.get_dummies(cat, prefix="cat").astype(float).set_axis(df.index)], axis=1)


def fit_ridge(df: pd.DataFrame, alpha: float, include_events: bool, first_week: int = MIN_HISTORY,
              end_week: int = TRAIN_WEEKS):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = training_rows(df, first_week, end_week)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(ridge_matrix(train, include_events), train["demand_units"] / train["reference"],
              ridge__sample_weight=train["reference"])
    return model


def ridge_units(df: pd.DataFrame, model, include_events: bool) -> pd.Series:
    return to_units(pd.Series(model.predict(ridge_matrix(df, include_events)), index=df.index), df)


# --- Model comparison with a validation set ------------------------------------------------------------------
def val_score(df: pd.DataFrame, pred: pd.Series) -> dict:
    rows = df[(df["week"] >= VAL_START) & (df["week"] < TRAIN_WEEKS)].copy()
    rows["pred"] = pred.loc[rows.index]
    return score(rows.dropna(subset=["pred"]), "pred")


def compare_models(df: pd.DataFrame):
    """Chooses the Ridge alpha and the XGBoost settings on the VALIDATION weeks (models trained on TRAIN weeks only),
    then retrains the chosen models on train + validation and fills the pred_* columns used for the test results.
    Returns (df, validation table, best settings)."""
    val_rows, best = [], {}
    for base in ("pred_last_week", "pred_ma4"):
        val_rows.append({"model": base, "setting": "-", "chosen": "", **val_score(df, df[base])})
    for variant, feats in VARIANTS.items():
        events = variant == "with_events"

        scored = []
        for alpha in RIDGE_ALPHAS:
            model = fit_ridge(df, alpha, events, end_week=VAL_START)
            scored.append((val_score(df, ridge_units(df, model, events)), alpha))
        best_alpha = min(scored, key=lambda t: t[0]["MAE"])[1]
        for sc, alpha in scored:
            val_rows.append({"model": f"ridge_{variant}", "setting": f"alpha={alpha}",
                             "chosen": "*" if alpha == best_alpha else "", **sc})
        df[f"pred_ridge_{variant}"] = ridge_units(df, fit_ridge(df, best_alpha, events, end_week=TRAIN_WEEKS), events)

        scored = []
        for params in XGB_GRID:
            _, model = fit_forecaster(df, feats, train_weeks=VAL_START, params=params)
            scored.append((val_score(df, predict_units(df, model, feats)), params))
        best_params = min(scored, key=lambda t: t[0]["MAE"])[1]
        for sc, params in scored:
            val_rows.append({"model": f"xgb_{variant}", "setting": f"trees={params['n_estimators']},depth={params['max_depth']}",
                             "chosen": "*" if params == best_params else "", **sc})
        _, final = fit_forecaster(df, feats, train_weeks=TRAIN_WEEKS, params=best_params)
        df[f"pred_xgb_{variant}"] = predict_units(df, final, feats)
        best[variant] = best_params
    return df, pd.DataFrame(val_rows), best


MODEL_WITH_EVENTS = "models/forecaster_with_events.json"
MODEL_NO_EVENTS = "models/forecaster_no_events.json"
MODEL_EVAL_WITH_EVENTS = "models/eval/forecaster_with_events.json"   # trained on weeks before TRAIN_WEEKS only
MODEL_EVAL_NO_EVENTS = "models/eval/forecaster_no_events.json"


def save_forecaster(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_model"):                 # XGBoost native format
        model.save_model(path)
    else:                                            # scikit-learn fallback (testing only)
        with open(path, "wb") as f:
            pickle.dump(model, f)


def load_forecaster(path: str):
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(path)
        return model
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def predict_next_week(history: pd.DataFrame, household_id: str, week: int, guest=None, away=None,
                      model=None, households: pd.DataFrame | None = None,
                      products: pd.DataFrame | None = None, for_evaluation: bool = False) -> dict:
    """Forecast of `week` for ONE household: {product_id: demand in purchase units}.

    history: weekly demand rows (columns of data/demand_weekly.csv) that contain at least the weeks before `week`
             for this household (weeks up to week-1; the more the better, the model uses up to 52 weeks back).
    guest / away: True / False if the household knows about guests or a trip next week, None if unknown.
                  None uses the model without these flags; True/False uses the model with them.
    model: optional fitted model; by default the saved final model is loaded from models/.
    for_evaluation: True loads the models in models/eval/ (trained on weeks 4-77 only). Use them whenever you measure
                    accuracy or evaluate a policy on the test weeks 78-103; the default models saw those weeks.
    """
    households = pd.read_csv("data/households.csv") if households is None else households
    products = pd.read_csv("data/products.csv") if products is None else products
    use_events = guest is not None or away is not None
    features = FEATURES_WITH_EVENTS if use_events else FEATURES_NO_EVENTS
    if model is None:
        if for_evaluation:
            model = load_forecaster(MODEL_EVAL_WITH_EVENTS if use_events else MODEL_EVAL_NO_EVENTS)
        else:
            model = load_forecaster(MODEL_WITH_EVENTS if use_events else MODEL_NO_EVENTS)

    hist = history[(history["household_id"] == household_id) & (history["week"] < week)]
    if hist["week"].max() != week - 1:
        raise ValueError(f"history must end at week {week - 1} (it ends at {hist['week'].max()})")
    target = pd.DataFrame({"household_id": household_id, "week": week, "product_id": products["product_id"],
                           "guest_week": int(bool(guest)), "away_week": int(bool(away)),
                           "holiday_week": int((week % 52) in HOLIDAY_WEEKS), "demand_units": np.nan})
    rows = pd.concat([hist, target], ignore_index=True)
    feats = build_features(rows, households, products)
    now = feats[feats["week"] == week]
    pred = predict_units(now, model, features)
    return dict(zip(now["product_id"], pred.round(4)))


def export(df: pd.DataFrame, best: dict) -> None:
    """data/forecast_weekly.csv: one row per household, product and week >= MIN_HISTORY.
    Predictions come from models trained on weeks < TRAIN_WEEKS, so only out_of_sample == 1 rows are honest."""
    out = df[df["week"] >= MIN_HISTORY][KEYS + ["week", "demand_units"]].copy()
    out["forecast_units"] = df.loc[out.index, "pred_xgb_with_events"].round(4)
    out["forecast_units_no_events"] = df.loc[out.index, "pred_xgb_no_guest_away"].round(4)
    out["out_of_sample"] = (out["week"] >= TRAIN_WEEKS).astype(int)
    out = out.rename(columns={"demand_units": "actual_units"})
    out = out[["household_id", "week", "product_id", "actual_units", "forecast_units",
               "forecast_units_no_events", "out_of_sample"]].sort_values(["household_id", "week", "product_id"])
    out.to_csv("data/forecast_weekly.csv", index=False)
    print(f"\nWrote data/forecast_weekly.csv: {len(out)} rows (out_of_sample rows: {int(out['out_of_sample'].sum())})")

    for label, variant, path, eval_path in (("with events", "with_events", MODEL_WITH_EVENTS, MODEL_EVAL_WITH_EVENTS),
                                            ("no events", "no_guest_away", MODEL_NO_EVENTS, MODEL_EVAL_NO_EVENTS)):
        n_weeks = int(df["week"].max()) + 1
        _, final = fit_forecaster(df, VARIANTS[variant], train_weeks=n_weeks, params=best[variant])   # ALL weeks
        save_forecaster(final, path)
        print(f"Saved final model ({label}, settings {best[variant]}, trained on target weeks {MIN_HISTORY}-{n_weeks - 1}): {path}")
        _, ev = fit_forecaster(df, VARIANTS[variant], train_weeks=TRAIN_WEEKS, params=best[variant])  # weeks before the test
        save_forecaster(ev, eval_path)
        print(f"Saved evaluation model ({label}, trained on target weeks {MIN_HISTORY}-{TRAIN_WEEKS - 1}): {eval_path}")


def main(do_export: bool = False) -> None:
    weekly = load_weekly()
    households = pd.read_csv("data/households.csv")
    products = pd.read_csv("data/products.csv")
    df = add_baselines(build_features(weekly, households, products))
    print(f"rows: {len(df)} | households: {df['household_id'].nunique()} | products: {df['product_id'].nunique()}")
    print(f"split of target weeks: train {MIN_HISTORY}-{VAL_START - 1} | validation {VAL_START}-{TRAIN_WEEKS - 1} "
          f"| test {TRAIN_WEEKS}-{int(df['week'].max())} ({int((df['week'] >= TRAIN_WEEKS).sum())} rows)")

    df, val_table, best = compare_models(df)
    print("\nVALIDATION weeks (models trained on train weeks only; * = chosen setting; lower is better):")
    print(val_table.round({"MAE": 4, "WMAPE_%": 1}).to_string(index=False))

    test_models = ["pred_last_week", "pred_ma4", "pred_ridge_no_guest_away", "pred_xgb_no_guest_away",
                   "pred_ridge_with_events", "pred_xgb_with_events"]
    print(f"\nTEST weeks {TRAIN_WEEKS}-{int(df['week'].max())} (chosen settings, retrained on weeks {MIN_HISTORY}-{TRAIN_WEEKS - 1}; "
          "columns after WMAPE_% = error by week type):")
    print(summary_table(evaluate(df, test_models)).to_string())
    if do_export:
        export(df, best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="write data/forecast_weekly.csv and save the final models")
    main(ap.parse_args().export)