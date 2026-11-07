"""Data check. Run from the repo root after editing any CSV in data/:
    python -m src.check_data
"""
import pandas as pd

products = pd.read_csv("data/products.csv")
households = pd.read_csv("data/households.csv")
recipes = pd.read_csv("data/recipes.csv")

assert products.product_id.is_unique, "duplicate product_id"
assert (products.price_usd > 0).all() and (products.shelf_life_days > 0).all()
print(f"products: {len(products)} | households: {len(households)} | recipes: {len(recipes)}")

p = products.set_index("product_id")
rows = []
for _, r in recipes.iterrows():
    kcal = protein = cost = 0.0
    allergens = set()
    for item in r["ingredients"].split(";"):
        pid, qty = item.split(":")
        assert pid in p.index, f"{r['recipe_id']}: unknown product '{pid}'"
        qty = float(qty)
        kcal += qty * p.loc[pid, "kcal_per_unit"]
        protein += qty * p.loc[pid, "protein_g_per_unit"]
        cost += qty * p.loc[pid, "price_usd"]
        a = p.loc[pid, "allergens"]
        if isinstance(a, str):
            allergens.add(a)
    s = r["servings"]
    rows.append({
        "recipe": r["recipe_id"],
        "servings": s,
        "kcal/serv": round(kcal / s),
        "protein_g/serv": round(protein / s, 1),
        "cost/serv": round(cost / s, 2),
        "allergens": ",".join(sorted(allergens)),
    })

print(pd.DataFrame(rows).to_string(index=False))
print("All checks passed.")