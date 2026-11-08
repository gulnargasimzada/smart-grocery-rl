"""Demand forecasting (Person 1).

Goal: predict next week's demand (purchase units, see data/products.csv) for every household and product.
Step 1: simple baselines + the evaluation that every later model is compared against.
Step 2: XGBoost model. It predicts the RATIO  demand / reference  (reference = mean of the last 8 weeks) with
        sample weights = reference, so the training loss equals the absolute error in purchase units (= our MAE).
        Prediction = ratio * reference. Products a household can never eat have reference 0 and get prediction 0.

Run from the repo root:  python -m src.forecast
Time-based split: weeks 0-79 are history/training, weeks 80-103 are the test period (never used for training).
Every prediction for week t uses only information from weeks before t, plus the event flags of week t
(holiday is always known; guest/away weeks are known only if the household tells us in advance, so we also
train a model WITHOUT the guest/away flags to show the honest result when they are not known).
"""
import numpy as np
import pandas as pd

TRAIN_WEEKS = 80
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


def evaluate(df: pd.DataFrame, pred_cols: list, test_start: int = TRAIN_WEEKS) -> pd.DataFrame:
    """Scores on the test weeks, overall and by event type."""
    test = df[(df["week"] >= test_start)].dropna(subset=pred_cols).copy()
    test["event"] = event_label(test)
    rows = []
    for col in pred_cols:
        rows.append({"model": col, "slice": "all", "rows": len(test), **score(test, col)})
        for ev, part in test.groupby("event"):
            rows.append({"model": col, "slice": ev, "rows": len(part), **score(part, col)})
    return pd.DataFrame(rows)


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


def make_model():
    """XGBoost if it is installed, otherwise a scikit-learn model with the same interface (only for testing)."""
    try:
        from xgboost import XGBRegressor
        return "xgboost", XGBRegressor(objective="reg:absoluteerror", n_estimators=400, learning_rate=0.05,
                                       max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                       tree_method="hist", random_state=0, n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "sklearn-fallback", HistGradientBoostingRegressor(loss="absolute_error", max_iter=300,
                                                                 learning_rate=0.08, max_depth=6, random_state=0)


def fit_forecaster(df: pd.DataFrame, features: list, train_weeks: int = TRAIN_WEEKS):
    """Trains on target weeks MIN_HISTORY .. train_weeks-1. Returns (model name, fitted model)."""
    train = df[(df["week"] >= MIN_HISTORY) & (df["week"] < train_weeks) & (df["reference"] > 0)]
    name, model = make_model()
    ratio = train["demand_units"] / train["reference"]
    model.fit(train[features], ratio, sample_weight=train["reference"])
    return name, model


def predict_units(df: pd.DataFrame, model, features: list) -> pd.Series:
    """Predicted demand in purchase units. Rows without a usable reference (no history / never eaten) get NaN / 0."""
    ratio = pd.Series(model.predict(df[features]), index=df.index)
    pred = (ratio * df["reference"]).clip(lower=0)
    pred[df["reference"] == 0] = 0.0
    pred[df["reference"].isna()] = np.nan
    return pred


def main() -> None:
    weekly = load_weekly()
    households = pd.read_csv("data/households.csv")
    products = pd.read_csv("data/products.csv")
    df = add_baselines(build_features(weekly, households, products))
    n_test = int((df["week"] >= TRAIN_WEEKS).sum())
    print(f"rows: {len(df)} | households: {df['household_id'].nunique()} | products: {df['product_id'].nunique()} "
          f"| test rows (weeks {TRAIN_WEEKS}-{int(df['week'].max())}): {n_test}")

    model_name = None
    for label, features in (("pred_xgb_no_guest_away", FEATURES_NO_EVENTS),
                            ("pred_xgb_with_events", FEATURES_WITH_EVENTS)):
        model_name, model = fit_forecaster(df, features)
        df[label] = predict_units(df, model, features)
    print(f"model: {model_name} | trained on target weeks {MIN_HISTORY}-{TRAIN_WEEKS - 1}")

    res = evaluate(df, ["pred_last_week", "pred_ma4", "pred_xgb_no_guest_away", "pred_xgb_with_events"])
    print("\nResults on the test weeks (lower is better):")
    print(res.round({"MAE": 4, "WMAPE_%": 1}).to_string(index=False))


if __name__ == "__main__":
    main()