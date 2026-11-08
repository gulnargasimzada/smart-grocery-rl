"""Validates all data files (data v2). Run from the repo root:  python -m src.check_data

Checks products, households, recipes, (if present) the regional budgets and the demand history, and that they agree
with each other. Weekly prices are checked separately by src/check_prices.py.
Prints per-recipe nutrition, cost and allergens, and how many recipes each household can actually use.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import CATEGORY_KCAL_SHARE, WITHIN_CATEGORY_WEIGHT
from src.diets import DIET_EXCLUDED_TAGS, PRODUCT_TAGS, allowed_mask, parse_list

ALLERGENS = {"milk", "egg", "gluten", "peanut", "soy", "fish"}
MEAL_TYPES = {"breakfast", "lunch", "dinner", "snack"}
PRODUCT_COLS = ["product_id", "name", "category", "unit", "price_usd", "shelf_life_days", "kcal_per_unit",
                "protein_g_per_unit", "fiber_g_per_unit", "carbs_g_per_unit", "fat_g_per_unit",
                "sugar_g_per_unit", "sodium_mg_per_unit", "allergens"]
NUMERIC_PRODUCT_COLS = PRODUCT_COLS[4:13]
HOUSEHOLD_COLS = ["household_id", "persona", "n_adults", "n_children", "n_people", "diet", "allergen_exclusions",
                  "eats_out_share", "kcal_per_adult_day", "weekly_budget_usd", "weekly_protein_target_g",
                  "weekend_boost", "demand_noise", "guest_prob", "description"]
RECIPE_COLS = ["recipe_id", "name", "meal_type", "servings", "prep_time_min", "ingredients", "tags"]
DIET_TAGS = ["vegan", "vegetarian", "pescatarian"]       # a diet tag is a CLAIM: the recipe must fit that diet

errors, warnings = [], []


def check(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def missing_columns(df: pd.DataFrame, cols: list, name: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    check(not missing, f"{name}: missing columns {missing}")
    return bool(missing)


def check_products(products: pd.DataFrame) -> None:
    if missing_columns(products, PRODUCT_COLS, "products.csv"):
        return
    ids = products["product_id"]
    check(ids.is_unique, "products.csv: duplicate product_id")
    check(all(i == i.lower() and " " not in i for i in ids), "products.csv: product_id must be lowercase, no spaces")
    for c in NUMERIC_PRODUCT_COLS:
        check(pd.to_numeric(products[c], errors="coerce").notna().all(), f"products.csv: {c} has missing/non-numeric values")
    check((products["price_usd"] > 0).all(), "products.csv: price_usd must be > 0")
    check((products["kcal_per_unit"] > 0).all(), "products.csv: kcal_per_unit must be > 0")
    check((products["shelf_life_days"] > 0).all(), "products.csv: shelf_life_days must be > 0")
    bad_cat = sorted(set(products["category"]) - set(CATEGORY_KCAL_SHARE))
    check(not bad_cat, f"products.csv: categories not known to demand.py: {bad_cat}")
    for r in products.itertuples():
        unknown = parse_list(r.allergens) - ALLERGENS
        check(not unknown, f"products.csv: {r.product_id} has unknown allergens {sorted(unknown)}")
        check(r.product_id in WITHIN_CATEGORY_WEIGHT, f"demand.py: no WITHIN_CATEGORY_WEIGHT for {r.product_id}")
        atwater = 4 * r.protein_g_per_unit + 4 * r.carbs_g_per_unit + 9 * r.fat_g_per_unit
        if abs(atwater - r.kcal_per_unit) > 0.35 * r.kcal_per_unit:
            warnings.append(f"{r.product_id}: macros give {atwater:.0f} kcal vs listed {r.kcal_per_unit:.0f}")
    extra = sorted(set(WITHIN_CATEGORY_WEIGHT) - set(ids))
    check(not extra, f"demand.py has weights for products that are not in products.csv: {extra}")
    extra = sorted(set(PRODUCT_TAGS) - set(ids))
    check(not extra, f"diets.py has tags for products that are not in products.csv: {extra}")


def check_households(households: pd.DataFrame, products: pd.DataFrame) -> None:
    if missing_columns(households, HOUSEHOLD_COLS, "households.csv"):
        return
    check(households["household_id"].is_unique, "households.csv: duplicate household_id")
    for h in households.to_dict("records"):
        hid = h["household_id"]
        check(h["diet"] in DIET_EXCLUDED_TAGS, f"{hid}: unknown diet '{h['diet']}'")
        unknown = parse_list(h["allergen_exclusions"]) - ALLERGENS
        check(not unknown, f"{hid}: unknown allergens {sorted(unknown)}")
        check(h["n_people"] == h["n_adults"] + h["n_children"], f"{hid}: n_people != n_adults + n_children")
        check(h["n_adults"] >= 1, f"{hid}: needs at least one adult")
        check(h["weekly_budget_usd"] > 0 and h["weekly_protein_target_g"] > 0, f"{hid}: budget and protein target must be > 0")
        check(0 <= h["eats_out_share"] < 1, f"{hid}: eats_out_share must be in [0, 1)")
        check(0 <= h["guest_prob"] < 1, f"{hid}: guest_prob must be in [0, 1)")
        if h["diet"] in DIET_EXCLUDED_TAGS:
            n_allowed = int(allowed_mask(h, products).sum())
            check(n_allowed >= 20, f"{hid}: only {n_allowed} products are allowed (diet + allergens)")


def parse_ingredients(text: str) -> dict:
    out = {}
    for part in str(text).split(";"):
        pid, qty = part.split(":")
        out[pid.strip()] = float(qty)
    return out


def recipe_diet(ingredient_ids) -> str:
    tags = {PRODUCT_TAGS.get(i) for i in ingredient_ids} - {None}
    if not tags:
        return "vegan"
    if "meat" in tags:
        return "omnivore"
    return "vegetarian" if "fish" not in tags else "pescatarian"


def check_recipes(recipes: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    if missing_columns(recipes, RECIPE_COLS, "recipes.csv"):
        return pd.DataFrame()
    check(recipes["recipe_id"].is_unique, "recipes.csv: duplicate recipe_id")
    p = products.set_index("product_id")
    rows = []
    for r in recipes.itertuples():
        check(r.meal_type in MEAL_TYPES, f"{r.recipe_id}: unknown meal_type '{r.meal_type}'")
        check(r.servings > 0, f"{r.recipe_id}: servings must be > 0")
        try:
            ing = parse_ingredients(r.ingredients)
        except ValueError:
            errors.append(f"{r.recipe_id}: ingredients must look like 'product_id:qty;product_id:qty'")
            continue
        unknown = [i for i in ing if i not in p.index]
        check(not unknown, f"{r.recipe_id}: unknown products {unknown}")
        check(all(q > 0 for q in ing.values()), f"{r.recipe_id}: quantities must be > 0")
        if unknown:
            continue
        kcal = sum(q * p.loc[i, "kcal_per_unit"] for i, q in ing.items()) / r.servings
        protein = sum(q * p.loc[i, "protein_g_per_unit"] for i, q in ing.items()) / r.servings
        cost = sum(q * p.loc[i, "price_usd"] for i, q in ing.items()) / r.servings
        allergens = sorted(set().union(*[parse_list(p.loc[i, "allergens"]) for i in ing]))
        diet = recipe_diet(ing)
        tags = {t.strip() for t in str(r.tags).split(";")}
        for t in DIET_TAGS:
            if t in tags:
                fits = {"vegan": diet == "vegan", "vegetarian": diet in ("vegan", "vegetarian"),
                        "pescatarian": diet != "omnivore"}[t]
                check(fits, f"{r.recipe_id}: tagged '{t}' but its ingredients make it {diet}")
        if not (150 <= kcal <= 900):
            warnings.append(f"{r.recipe_id}: {kcal:.0f} kcal per serving looks unusual")
        rows.append({"recipe": r.recipe_id, "servings": r.servings, "kcal/serv": round(kcal),
                     "protein_g/serv": round(protein, 1), "cost/serv": round(cost, 2),
                     "allergens": ",".join(allergens), "ingredients": set(ing), "meal_type": r.meal_type})
    return pd.DataFrame(rows)


def check_recipe_coverage(households: pd.DataFrame, products: pd.DataFrame, table: pd.DataFrame) -> None:
    ids = products["product_id"].tolist()
    counts = {}
    for h in households.to_dict("records"):
        if h.get("diet") not in DIET_EXCLUDED_TAGS:
            continue
        allowed = {pid for pid, ok in zip(ids, allowed_mask(h, products)) if ok}
        usable = table[table["ingredients"].apply(lambda s: s <= allowed)]
        counts[h["household_id"]] = len(usable)
        missing = sorted({"breakfast", "lunch", "dinner"} - set(usable["meal_type"]))
        if missing:
            warnings.append(f"{h['household_id']}: no usable recipe for {missing}")
    if counts:
        worst = min(counts, key=counts.get)
        print(f"usable recipes per household (diet + allergens): min {counts[worst]} ({worst}) | "
              f"median {int(np.median(list(counts.values())))} | max {max(counts.values())}")
        few = [h for h, c in counts.items() if c < 5]
        if few:
            warnings.append(f"households with fewer than 5 usable recipes: {few}")


def check_demand(demand: pd.DataFrame, households: pd.DataFrame, products: pd.DataFrame) -> None:
    need = ["household_id", "week", "guest_week", "away_week", "holiday_week", "product_id", "demand_units"]
    if missing_columns(demand, need, "demand_weekly.csv"):
        return
    check(set(demand["household_id"]) == set(households["household_id"]),
          "demand_weekly.csv: households differ from households.csv (run python -m src.data_gen)")
    check(set(demand["product_id"]) == set(products["product_id"]),
          "demand_weekly.csv: products differ from products.csv (run python -m src.data_gen)")
    n_weeks = int(demand["week"].max()) + 1
    expected_rows = demand["household_id"].nunique() * n_weeks * demand["product_id"].nunique()
    check(len(demand) == expected_rows, f"demand_weekly.csv: {len(demand)} rows, expected {expected_rows} (complete grid)")
    check(demand["demand_units"].notna().all() and (demand["demand_units"] >= 0).all(),
          "demand_weekly.csv: demand_units has NaN or negative values")
    totals = demand.groupby(["household_id", "product_id"])["demand_units"].sum().unstack()
    for h in households.to_dict("records"):
        if h["household_id"] not in totals.index or h.get("diet") not in DIET_EXCLUDED_TAGS:
            continue
        ok = allowed_mask(h, products)
        row = totals.loc[h["household_id"], products["product_id"]].to_numpy()
        check((row[~ok] == 0).all(), f"{h['household_id']}: demand exists for products excluded by diet/allergen")
        check((row[ok] > 0).all(), f"{h['household_id']}: allowed products with zero demand")
    print(f"demand_weekly.csv: {len(demand)} rows | {n_weeks} weeks | all checks on the grid and exclusions done")


def check_regional_budgets(budgets: pd.DataFrame, households: pd.DataFrame) -> None:
    need = ["household_id", "region_id", "weekly_budget_usd"]
    if missing_columns(budgets, need, "household_budgets_regional.csv"):
        return
    check(set(budgets["household_id"]) == set(households["household_id"]),
          "household_budgets_regional.csv: households differ from households.csv (run python -m src.households_gen)")
    n_regions = budgets["region_id"].nunique()
    check(len(budgets) == len(households) * n_regions and not budgets.duplicated(["household_id", "region_id"]).any(),
          f"household_budgets_regional.csv: expected one row per household and region ({len(households)} x {n_regions})")
    check((budgets["weekly_budget_usd"] > 0).all(), "household_budgets_regional.csv: budgets must be > 0")
    print(f"household_budgets_regional.csv: {len(budgets)} rows | {n_regions} regions: {sorted(budgets['region_id'].unique())}")


def main() -> None:
    products = pd.read_csv("data/products.csv")
    households = pd.read_csv("data/households.csv")
    recipes = pd.read_csv("data/recipes.csv")
    print(f"products: {len(products)} | households: {len(households)} | recipes: {len(recipes)}")

    check_products(products)
    check_households(households, products)
    table = check_recipes(recipes, products)
    if not table.empty:
        print(table.drop(columns=["ingredients", "meal_type"]).to_string(index=False))
        check_recipe_coverage(households, products, table)
    if Path("data/household_budgets_regional.csv").exists():
        check_regional_budgets(pd.read_csv("data/household_budgets_regional.csv"), households)
    if Path("data/demand_weekly.csv").exists():
        check_demand(pd.read_csv("data/demand_weekly.csv"), households, products)
    else:
        warnings.append("data/demand_weekly.csv not found (run python -m src.data_gen)")

    for w in warnings:
        print("WARNING:", w)
    if errors:
        print(f"\nFAILED: {len(errors)} problem(s)")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()