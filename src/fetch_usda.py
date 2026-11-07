"""Fills the nutrition columns of data/products.csv from USDA FoodData Central (SR Legacy).

Run from the repo root:
    python -m src.fetch_usda
or, if you already downloaded the zip manually:
    python -m src.fetch_usda --zip path/to/FoodData_Central_sr_legacy_food_csv_2018-04.zip

Outputs:
    data/products.csv                   (updated: USDA-based nutrition per purchase unit)
    data/products_nutrition_source.csv  (audit trail: which USDA food was used for each product)

USDA values are per 100 g of edible food. Per purchase unit we compute:
    value_per_unit = value_per_100g * grams_per_unit * edible_fraction / 100
"""
import argparse
import difflib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

SR_LEGACY_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip"
RAW_DIR = Path("data/usda_raw")
PRODUCTS_PATH = Path("data/products.csv")
SOURCE_PATH = Path("data/products_nutrition_source.csv")

# product_id -> (USDA description query, grams per purchase unit, edible fraction of those grams)
# Query format: 'prefix' or 'prefix + keyword + keyword' (see find_food).
# grams_per_unit and edible_fraction are OUR ASSUMPTIONS (documented here, easy to change).
MAPPING = {
    "milk":           ("Milk, whole, 3.25% milkfat", 1032, 1.00),                      # 1 L (about 244 g per cup)
    "eggs":           ("Egg, whole, raw, fresh", 600, 1.00),                           # 12 x 50 g (large egg, edible part)
    "chicken_breast": ("Chicken, broilers or fryers, breast + meat only + raw", 500, 1.00),
    "rice":           ("Rice, white, long-grain, regular, raw, unenriched", 1000, 1.00),
    "bread":          ("Bread, white, commercially prepared", 450, 1.00),              # assumed 450 g loaf
    "bananas":        ("Bananas, raw", 1000, 0.64),                                    # peel is not eaten
    "apples":         ("Apples, raw, with skin", 1000, 0.90),                          # core/stem not eaten
    "tomatoes":       ("Tomatoes, red, ripe, raw, year round average", 500, 0.91),
    "spinach":        ("Spinach, raw", 200, 1.00),                                     # packaged, trimmed
    "lentils":        ("Lentils, raw", 500, 1.00),
    "yogurt":         ("Yogurt, plain, whole milk", 500, 1.00),
    "oats":           ("Cereals, oats, regular and quick, not fortified, dry", 500, 1.00),
}

# product column -> USDA nutrient_nbr candidates (first one found is used)
NUTRIENT_NBR = {
    "kcal_per_unit":      [208],          # Energy (kcal)
    "protein_g_per_unit": [203],
    "fat_g_per_unit":     [204],
    "carbs_g_per_unit":   [205],          # Carbohydrate, by difference
    "fiber_g_per_unit":   [291],
    "sugar_g_per_unit":   [269.3, 269],   # Sugars, total
    "sodium_mg_per_unit": [307],
}

COLUMN_ORDER = ["product_id", "name", "category", "unit", "price_usd", "shelf_life_days",
                "kcal_per_unit", "protein_g_per_unit", "fiber_g_per_unit", "carbs_g_per_unit",
                "fat_g_per_unit", "sugar_g_per_unit", "sodium_mg_per_unit", "allergens"]


def get_zip(local_zip: str | None) -> Path:
    if local_zip:
        return Path(local_zip)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "sr_legacy_csv.zip"
    if not target.exists():
        print(f"Downloading {SR_LEGACY_URL} ...")
        urllib.request.urlretrieve(SR_LEGACY_URL, target)
    return target


def read_table(zf: zipfile.ZipFile, basename: str, usecols: list) -> pd.DataFrame:
    name = next((n for n in zf.namelist() if n == basename or n.endswith("/" + basename)), None)
    if name is None:
        raise SystemExit(f"{basename} not found inside the zip")
    with zf.open(name) as f:
        return pd.read_csv(f, usecols=usecols)


def find_food(food: pd.DataFrame, query: str) -> pd.Series:
    """query = 'prefix' or 'prefix + keyword + keyword': description must start with the prefix
    and contain every keyword (case-insensitive). Exact prefix match wins, else the shortest description."""
    prefix, *keywords = [part.strip() for part in query.split("+")]
    desc = food["description"].str.lower()
    mask = desc.str.startswith(prefix.lower())
    for kw in keywords:
        mask &= desc.str.contains(kw.lower(), regex=False)
    hits = food[mask]
    if hits.empty:
        close = difflib.get_close_matches(query, food["description"].tolist(), n=8, cutoff=0.4)
        raise SystemExit(f"No USDA food matches '{query}'. Closest: {close}")
    exact = hits[hits["description"].str.lower() == prefix.lower()]
    if not exact.empty:
        return exact.iloc[0]
    return hits.assign(n=hits["description"].str.len()).sort_values(["n", "fdc_id"]).iloc[0]


def main(local_zip: str | None = None) -> None:
    with zipfile.ZipFile(get_zip(local_zip)) as zf:
        food = read_table(zf, "food.csv", ["fdc_id", "description"])
        nutrient = read_table(zf, "nutrient.csv", ["id", "nutrient_nbr"])
        food_nutrient = read_table(zf, "food_nutrient.csv", ["fdc_id", "nutrient_id", "amount"])

    nutrient["nbr"] = pd.to_numeric(nutrient["nutrient_nbr"], errors="coerce").round(1)
    nbr_to_id = dict(zip(nutrient["nbr"], nutrient["id"]))

    products = pd.read_csv(PRODUCTS_PATH)
    old = products.set_index("product_id")[["kcal_per_unit", "protein_g_per_unit"]].copy()
    audit_rows, new_cols = [], {c: [] for c in NUTRIENT_NBR}

    for pid in products["product_id"]:
        query, grams, edible = MAPPING[pid]
        row = find_food(food, query)
        fid = int(row["fdc_id"])
        amounts = food_nutrient[food_nutrient["fdc_id"] == fid].set_index("nutrient_id")["amount"]
        audit = {"product_id": pid, "usda_fdc_id": fid, "usda_description": row["description"],
                 "usda_url": f"https://fdc.nal.usda.gov/food-details/{fid}/nutrients",
                 "grams_per_unit": grams, "edible_fraction": edible}
        for col, nbrs in NUTRIENT_NBR.items():
            per100 = next((amounts[nbr_to_id[round(n, 1)]] for n in nbrs
                           if round(n, 1) in nbr_to_id and nbr_to_id[round(n, 1)] in amounts.index), None)
            if per100 is None:
                raise SystemExit(f"{pid}: nutrient {nbrs} missing for USDA food {fid} ({row['description']})")
            audit[col.replace("_per_unit", "_per_100g")] = per100
            value = per100 * grams * edible / 100
            new_cols[col].append(round(value, 0 if col == "kcal_per_unit" else 1))
        audit_rows.append(audit)
        print(f"{pid:15s} -> {row['description']}  (fdc_id {fid})")

    for col, values in new_cols.items():
        products[col] = values
    products = products[COLUMN_ORDER]
    products.to_csv(PRODUCTS_PATH, index=False)
    pd.DataFrame(audit_rows).to_csv(SOURCE_PATH, index=False)

    # Compare with the old placeholder numbers and run sanity checks.
    print("\nOld placeholder vs USDA-based values (per purchase unit):")
    for r in products.itertuples():
        o = old.loc[r.product_id]
        print(f"  {r.product_id:15s} kcal {o['kcal_per_unit']:>6.0f} -> {r.kcal_per_unit:>6.0f} | "
              f"protein {o['protein_g_per_unit']:>5.0f} -> {r.protein_g_per_unit:>5.1f} g")
    for r in products.itertuples():
        atwater = 4 * r.protein_g_per_unit + 4 * r.carbs_g_per_unit + 9 * r.fat_g_per_unit
        if abs(atwater - r.kcal_per_unit) > 0.30 * r.kcal_per_unit:
            print(f"  WARNING {r.product_id}: macros give {atwater:.0f} kcal vs listed {r.kcal_per_unit:.0f}")
        if r.fiber_g_per_unit > r.carbs_g_per_unit or r.sugar_g_per_unit > r.carbs_g_per_unit:
            print(f"  WARNING {r.product_id}: fiber or sugar exceeds carbohydrates")
    print(f"\nWrote {PRODUCTS_PATH} and {SOURCE_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="path to a locally downloaded SR Legacy CSV zip")
    main(ap.parse_args().zip)