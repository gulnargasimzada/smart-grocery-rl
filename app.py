"""Smart Grocery Coach - Streamlit app (Person 4).

Run from the repo root:
    streamlit run app.py

How it works
- Python loads data/*.csv and, when it exists, results/results.json (see INTERFACES.md sections 2 and 9).
- Everything is packed into one JSON payload and handed to the designed interface in ui/coach.html,
  which Streamlit shows full-width through components.html.
- Until src/evaluate.py writes results/results.json, the interface runs a clearly labelled demo
  simulation of the four policies in the browser, on the same weekly demand data.
- Tabs: Results, Pantry & shopping list, Use-it-first meals, Barcode lookup (Open Food Facts).
- Data v2: diets and allergens come from src/diets.py (the same rules the demand model and constraints use),
  sale weeks from data/price_weekly_synthetic.csv and real store prices from data/price_history.csv.
"""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.diets import allowed_mask

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "results.json"
UI = ROOT / "ui" / "coach.html"
APP_HEIGHT = 1500  # px; the interface scrolls inside this frame on smaller screens

st.set_page_config(page_title="Smart Grocery Coach", layout="wide", initial_sidebar_state="collapsed")

# Let the designed interface fill the page: hide Streamlit's header, footer and padding.
st.markdown(
    """
    <style>
      header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"] {display: none !important;}
      .block-container {padding: 0 !important; max-width: 100% !important;}
      [data-testid="stAppViewContainer"] > .main {padding: 0 !important;}
      iframe {border: 0; display: block;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


@st.cache_data(show_spinner=False)
def load_payload(mtimes: tuple) -> dict:
    """Build the JSON the interface reads. `mtimes` busts the cache when a data file changes."""
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

    return {
        "products": product_rows, "households": household_rows, "recipes": recipe_rows,
        "demand": demand, "flags": flags, "promos": promos, "real_prices": real_prices,
        "results": results, "meals": live_meals(),
    }


def live_meals():
    """Hook for src/meals.py (Person 4).

    Once the environment can report the current pantry, return
        {household_id: suggest_meals(inventory, household, recipes, products)}
    and the Meals tab shows those picks and explanations at the top.
    Until then the interface ranks recipes.csv itself with the same rules (use-it-first, code-computed nutrition).
    """
    return None


def main():
    files = [DATA / f for f in ("products.csv", "households.csv", "recipes.csv", "demand_weekly.csv")]
    extra = [DATA / f for f in ("price_weekly_synthetic.csv", "price_history.csv", "products_nutrition_source.csv")]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        st.error(f"Missing data files: {', '.join(missing)}. Run `python -m src.data_gen` from the repo root.")
        return
    mtimes = tuple(f.stat().st_mtime for f in files + extra + [RESULTS] if f.exists())
    payload = load_payload(mtimes)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    body = UI.read_text(encoding="utf-8").replace("__COACH_DATA__", data, 1)
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"></head>'
        f"<body>{body}</body></html>"
    )
    components.html(page, height=APP_HEIGHT, scrolling=True)


if __name__ == "__main__":
    main()
