"""Builds the data payload for the Grocery Coach interface (ui/coach.html).

Shared by app.py (Streamlit) and build_static.py (static site for Netlify), so both show the same data.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

from src.diets import allowed_mask

try:  # optional settings from a .env file in the repo root
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "results.json"
PLANS = ROOT / "results" / "plans.json"            # optional: the RL agent's order per test week (docs/UI_INTEGRATION.md)
OFFICIAL = ROOT / "results" / "ui" / "plans.json"  # written by: python export_ui_plans.py (the team's frozen models, week by week)
FORECAST = DATA / "forecast_weekly.csv"            # written by: python -m src.forecast --export
UI = ROOT / "ui" / "coach.html"
DATA_FILES = ("products.csv", "households.csv", "recipes.csv", "demand_weekly.csv")
EXTRA_FILES = ("price_weekly_synthetic.csv", "price_history.csv", "products_nutrition_source.csv", "forecast_weekly.csv")


def week_split() -> dict:
    """Train / validation / test weeks, taken from src/forecast.py so the app matches the ML and RL work."""
    try:
        from src import forecast as f
        return {"train_start": int(f.MIN_HISTORY), "val_start": int(f.VAL_START), "test_start": int(f.TRAIN_WEEKS)}
    except Exception:  # forecast.py not on this branch yet: same values as the team's agreed split
        return {"train_start": 4, "val_start": 64, "test_start": 78}


def load_forecast(n_weeks: int):
    """{household: {product: [units per week, None before the forecast starts]}} from data/forecast_weekly.csv.

    Uses `forecast_units_no_events` (guest/away weeks unknown in advance), the honest choice for the app.
    Weeks >= test_start come from models that never trained on them (out_of_sample == 1)."""
    if not FORECAST.exists():
        return None
    fw = pd.read_csv(FORECAST)
    col = "forecast_units_no_events" if "forecast_units_no_events" in fw.columns else "forecast_units"
    out = {}
    for (hh, pid), g in fw.groupby(["household_id", "product_id"]):
        series = [None] * n_weeks
        for w, v in zip(g["week"], g[col]):
            if 0 <= int(w) < n_weeks and pd.notna(v):
                series[int(w)] = round(float(v), 4)
        out.setdefault(hh, {})[pid] = series
    return out


def load_official():
    """The team's six frozen methods replayed week by week on the held-out test episodes (export_ui_plans.py)."""
    if not OFFICIAL.exists():
        return None
    try:
        return json.loads(OFFICIAL.read_text())
    except json.JSONDecodeError:
        return None


def forecast_label() -> str:
    if not FORECAST.exists():
        return ""
    try:
        methods = set(pd.read_csv(FORECAST, usecols=["forecast_method"])["forecast_method"].dropna().astype(str))
    except ValueError:
        return "ML forecast"
    return "Ridge forecast" if any("ridge" in m for m in methods) else "XGBoost forecast" if any("xgb" in m for m in methods) else "ML forecast"


def load_plans():
    """{household: {week: {"order": {product: units}, "notes": [...]}}} from results/plans.json, if present."""
    if not PLANS.exists():
        return None
    try:
        raw = json.loads(PLANS.read_text())
    except json.JSONDecodeError:
        return None
    rows = raw if isinstance(raw, list) else [raw]
    out = {}
    for r in rows:
        hh = r.get("household")
        for wk, plan in (r.get("weeks") or {}).items():
            out.setdefault(hh, {})[int(wk)] = {"order": plan.get("order", {}), "notes": plan.get("notes", [])}
    return out or None


def split_list(value) -> list:
    """'milk;egg' or 'milk, egg' -> ['milk', 'egg']; empty or NaN -> []."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [x.strip().lower() for x in re.split(r"[;,|]", str(value)) if x.strip()]


def num(value, default=0.0) -> float:
    try:
        v = float(value)
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def load_payload(include_keys: bool = True) -> dict:
    """Build the JSON the interface reads.

    include_keys=False leaves out API keys (used for the public static build).
    """
    products = pd.read_csv(DATA / "products.csv")
    households = pd.read_csv(DATA / "households.csv")
    recipes = pd.read_csv(DATA / "recipes.csv")
    weekly = pd.read_csv(DATA / "demand_weekly.csv")

    grams = {}
    source = DATA / "products_nutrition_source.csv"
    if source.exists():
        src = pd.read_csv(source)
        if {"product_id", "grams_per_unit"} <= set(src.columns):
            grams = dict(zip(src["product_id"], src["grams_per_unit"]))

    product_rows = [
        {
            "id": r.product_id, "name": r.name, "category": r.category, "unit": str(r.unit),
            "price": num(r.price_usd), "shelf": int(num(r.shelf_life_days, 7)),
            "kcal": num(getattr(r, "kcal_per_unit", 0)), "protein": num(getattr(r, "protein_g_per_unit", 0)),
            "fiber": num(getattr(r, "fiber_g_per_unit", 0)), "carbs": num(getattr(r, "carbs_g_per_unit", 0)),
            "fat": num(getattr(r, "fat_g_per_unit", 0)), "sugar": num(getattr(r, "sugar_g_per_unit", 0)),
            "sodium": num(getattr(r, "sodium_mg_per_unit", 0)),
            "allergens": split_list(getattr(r, "allergens", "")), "grams": num(grams.get(r.product_id, 0)),
        }
        for r in products.itertuples(index=False)
    ]

    household_rows = []
    for h in households.to_dict("records"):
        allowed = allowed_mask(h, products)
        household_rows.append({
            "id": h["household_id"], "persona": h["persona"], "n_people": int(num(h.get("n_people"), 1)),
            "n_adults": int(num(h.get("n_adults"), 1)), "n_children": int(num(h.get("n_children"), 0)),
            "diet": str(h.get("diet") or "omnivore"), "exclusions": split_list(h.get("allergen_exclusions")),
            "eats_out": num(h.get("eats_out_share")), "budget": num(h.get("weekly_budget_usd")),
            "protein_target": num(h.get("weekly_protein_target_g")), "weekend_boost": num(h.get("weekend_boost"), 1),
            "allowed": [pid for pid, ok in zip(products["product_id"], allowed) if ok],
        })

    recipe_rows = []
    for r in recipes.itertuples(index=False):
        ingredients = {}
        for part in str(r.ingredients).split(";"):
            if ":" in part:
                pid, qty = part.split(":", 1)
                ingredients[pid.strip()] = num(qty)
        recipe_rows.append({
            "id": r.recipe_id, "name": r.name, "meal_type": r.meal_type, "servings": int(num(r.servings, 1)) or 1,
            "prep_time_min": int(num(r.prep_time_min)), "ingredients": ingredients, "tags": split_list(r.tags),
        })

    # demand[household][product] = list of weekly demand in purchase units, week 0..N-1
    demand, flags = {}, {}
    for hh_id, g in weekly.groupby("household_id"):
        piv = g.pivot_table(index="week", columns="product_id", values="demand_units", aggfunc="sum").sort_index()
        demand[hh_id] = {pid: [round(float(v), 4) for v in piv[pid].fillna(0)] for pid in piv.columns}
        wk = g.groupby("week").max(numeric_only=True).sort_index()
        flags[hh_id] = {k: [int(v) for v in wk[k + "_week"]] for k in ("guest", "away", "holiday") if k + "_week" in wk}

    # Sale weeks (synthetic weekly prices): promos[product] = discount share per week, 0 = regular price
    promos = {}
    weekly_prices = DATA / "price_weekly_synthetic.csv"
    if weekly_prices.exists():
        wp = pd.read_csv(weekly_prices).sort_values("week")
        for pid, g in wp.groupby("product_id"):
            depth = (1 - g["price_usd"] / g["regular_price_usd"]).where(g["is_promo"].astype(str) == "True", 0)
            promos[pid] = [round(float(v), 3) for v in depth.clip(lower=0).fillna(0)]

    # Real shelf prices (Open Prices): one summary per product, in USD per purchase unit
    real_prices = {}
    history = DATA / "price_history.csv"
    if history.exists():
        ph = pd.read_csv(history)
        paid = ph["promo_price_usd"].fillna(ph["regular_price_usd"])
        ph = ph.assign(paid=paid)
        for pid, g in ph.groupby("product_id"):
            real_prices[pid] = {
                "n": int(len(g)), "stores": int(g["store_name"].nunique()),
                "median": round(float(g["regular_price_usd"].median()), 2),
                "low": round(float(g["paid"].min()), 2), "high": round(float(g["regular_price_usd"].max()), 2),
                "first": str(g["date"].min()), "last": str(g["date"].max()),
            }

    results = None
    if RESULTS.exists():
        try:
            results = json.loads(RESULTS.read_text())
        except json.JSONDecodeError:
            results = None

    n_weeks = int(weekly["week"].max()) + 1
    forecast = load_forecast(n_weeks)
    return {
        "split": week_split(), "forecast": forecast, "plans": load_plans(), "official": load_official(),
        "forecast_model": forecast_label() if forecast else "",
        "products": product_rows, "households": household_rows, "recipes": recipe_rows,
        "demand": demand, "flags": flags, "promos": promos, "real_prices": real_prices,
        "results": results, "meals": live_meals(),
        # Barcode scanner sources beyond Open Food Facts (optional; leave unset to skip them).
        # FDC_API_KEY: free key from https://fdc.nal.usda.gov/api-key-signup ; BARCODELOOKUP_API_KEY: paid.
        # Note: keys are sent to the browser, so use them for local demos only.
        "fdc_key": os.environ.get("FDC_API_KEY", "") if include_keys else "",
        "bl_key": os.environ.get("BARCODELOOKUP_API_KEY", "") if include_keys else "",
        # Plain-language AI explanations: URL of the Netlify Function (see netlify/functions/explain.mjs).
        "explain_url": os.environ.get("EXPLAIN_URL", ""),
    }


def live_meals():
    """Hook for src/meals.py (Person 4).

    Once the environment can report the current pantry, return
        {household_id: suggest_meals(inventory, household, recipes, products)}
    and the Meals tab shows those picks and explanations at the top.
    Until then the interface ranks recipes.csv itself with the same rules (use-it-first, code-computed nutrition).
    """
    return None



def render_page(payload: dict) -> str:
    """Return the interface as one HTML document with the payload embedded."""
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    body = UI.read_text(encoding="utf-8").replace("__COACH_DATA__", data, 1)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head>'
        f"<body>{body}</body></html>"
    )
