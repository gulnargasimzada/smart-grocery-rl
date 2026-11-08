"""Creates data/households.csv (30 diverse households) from archetypes.
Run from the repo root:  python -m src.households_gen
Also writes data/household_budgets_regional.csv (same households, one budget per region, from the regional prices).

Every household gets: size, diet, allergens, how much it eats out, weekly budget and protein target.
Budgets are CALIBRATED: budget = expected normal-week grocery cost * tier factor (tight / normal / comfortable),
so the budget constraint is meaningful for every household.
All numbers are PLACEHOLDERS to be checked later by the business teammate.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import DEFAULT_SEED, draw_taste, expected_weekly_cost

SEED = 7
REGIONAL_PRICES = "data/price_weekly_regional_synthetic.csv"   # columns: region_id, week, product_id, regular_price_usd, ...
DEFAULT_REGION = "durham_nc"      # households.csv budgets use products.csv prices (= this region, see sync_reference_prices)
TRAIN_WEEKS = 80                  # reference price = mean regular price over weeks 0-79
PROTEIN_CHILD_G_DAY = 40     # protein targets per day are placeholders (see ARCHETYPES)

TIER_FACTOR = {"tight": (1.10, 1.20), "normal": (1.20, 1.35), "comfortable": (1.35, 1.55)}
ALLERGEN_WEIGHTS = {"peanut": 0.25, "milk": 0.20, "gluten": 0.20, "soy": 0.10, "fish": 0.15, "egg": 0.10}
ALLERGY_PROB = 0.20

# key: label, number of generated households, adults range, children range, kcal per adult per day,
#      protein g per adult per day, diet probabilities, eats-out share range, budget tier,
#      weekend boost range, demand noise range, guest-week probability range
ARCHETYPES = {
    "student":      dict(label="Student", count=4, adults=(1, 1), children=(0, 0), kcal=2000, protein=65,
                         diets={"omnivore": 0.7, "vegetarian": 0.2, "pescatarian": 0.1}, eats_out=(0.20, 0.35),
                         tier="tight", weekend=(1.25, 1.40), noise=(0.25, 0.35), guest=(0.04, 0.08)),
    "couple":       dict(label="Working couple", count=4, adults=(2, 2), children=(0, 0), kcal=2000, protein=65,
                         diets={"omnivore": 0.75, "pescatarian": 0.15, "vegetarian": 0.10}, eats_out=(0.20, 0.30),
                         tier="normal", weekend=(1.15, 1.30), noise=(0.18, 0.25), guest=(0.05, 0.08)),
    "family":       dict(label="Family with children", count=5, adults=(2, 2), children=(1, 3), kcal=2000, protein=65,
                         diets={"omnivore": 0.85, "vegetarian": 0.15}, eats_out=(0.08, 0.18),
                         tier="normal", weekend=(1.30, 1.45), noise=(0.28, 0.35), guest=(0.06, 0.10)),
    "athlete":      dict(label="Athlete", count=3, adults=(1, 2), children=(0, 0), kcal=2800, protein=110,
                         diets={"omnivore": 0.8, "pescatarian": 0.2}, eats_out=(0.10, 0.20),
                         tier="normal", weekend=(1.05, 1.15), noise=(0.15, 0.22), guest=(0.03, 0.06)),
    "vegetarian":   dict(label="Vegetarian household", count=3, adults=(1, 2), children=(0, 0), kcal=2000, protein=60,
                         diets={"vegetarian": 1.0}, eats_out=(0.15, 0.30),
                         tier="normal", weekend=(1.15, 1.30), noise=(0.20, 0.28), guest=(0.04, 0.08)),
    "vegan":        dict(label="Vegan household", count=2, adults=(1, 2), children=(0, 0), kcal=2000, protein=55,
                         diets={"vegan": 1.0}, eats_out=(0.15, 0.30),
                         tier="tight", weekend=(1.15, 1.30), noise=(0.22, 0.30), guest=(0.04, 0.07)),
    "senior":       dict(label="Senior couple", count=3, adults=(2, 2), children=(0, 0), kcal=1800, protein=70,
                         diets={"omnivore": 1.0}, eats_out=(0.05, 0.12),
                         tier="tight", weekend=(1.05, 1.12), noise=(0.12, 0.18), guest=(0.03, 0.06)),
    "roommates":    dict(label="Shared student house", count=2, adults=(3, 4), children=(0, 0), kcal=2200, protein=65,
                         diets={"omnivore": 0.8, "vegetarian": 0.2}, eats_out=(0.25, 0.40),
                         tier="tight", weekend=(1.20, 1.40), noise=(0.30, 0.40), guest=(0.06, 0.10)),
    "professional": dict(label="Single professional", count=1, adults=(1, 1), children=(0, 0), kcal=2000, protein=65,
                         diets={"omnivore": 1.0}, eats_out=(0.35, 0.50),
                         tier="comfortable", weekend=(1.15, 1.30), noise=(0.22, 0.30), guest=(0.04, 0.08)),
}

# The three households of data v1 keep their ids (so nothing that referenced them breaks).
LEGACY = [
    dict(household_id="hh_student", arch="student", n_adults=1, n_children=0, diet="omnivore",
         eats_out_share=0.25, weekend_boost=1.30, demand_noise=0.25, guest_prob=0.06),
    dict(household_id="hh_couple", arch="couple", n_adults=2, n_children=0, diet="omnivore",
         eats_out_share=0.25, weekend_boost=1.20, demand_noise=0.20, guest_prob=0.06),
    dict(household_id="hh_family", arch="family", n_adults=2, n_children=2, diet="omnivore",
         eats_out_share=0.12, weekend_boost=1.35, demand_noise=0.30, guest_prob=0.06),
]


def pick(rng, options: dict) -> str:
    keys = list(options)
    return str(rng.choice(keys, p=np.array(list(options.values())) / sum(options.values())))


def uniform(rng, lo_hi, digits=2) -> float:
    return round(float(rng.uniform(*lo_hi)), digits)


def draw_allergen(rng, diet: str) -> str:
    if rng.random() >= ALLERGY_PROB:
        return ""
    skip = {"vegan": {"milk", "egg", "fish"}, "vegetarian": {"fish"}}.get(diet, set())
    options = {a: w for a, w in ALLERGEN_WEIGHTS.items() if a not in skip}
    return pick(rng, options)


def describe(arch: dict, row: dict, budget: int) -> str:
    parts = [f"{arch['label']}: {row['n_adults']} adult(s)"]
    if row["n_children"]:
        parts.append(f"{row['n_children']} child(ren)")
    text = ", ".join(parts)
    if row["diet"] != "omnivore":
        text += f", {row['diet']}"
    if row["allergen_exclusions"]:
        text += f", allergy: {row['allergen_exclusions']}"
    return f"{text}; {arch['tier']} budget (${budget}/week)"


def regional_reference_prices(products: pd.DataFrame) -> dict:
    """region_id -> reference price per product (same order as products). Empty dict if the file is missing."""
    if not Path(REGIONAL_PRICES).exists():
        return {}
    prices = pd.read_csv(REGIONAL_PRICES)
    col = "regular_price_usd" if "regular_price_usd" in prices.columns else "price_usd"
    prices = prices[prices["week"] < TRAIN_WEEKS]
    ref = prices.groupby(["region_id", "product_id"])[col].mean().unstack("product_id")
    missing = sorted(set(products["product_id"]) - set(ref.columns))
    if missing:
        raise SystemExit(f"{REGIONAL_PRICES}: no prices for {missing}")
    return {r: ref.loc[r, products["product_id"]].to_numpy() for r in ref.index}


def main() -> None:
    rng = np.random.default_rng(SEED)
    products = pd.read_csv("data/products.csv")

    specs = []
    for leg in LEGACY:
        specs.append({**leg, "allergen_exclusions": ""})
    for key, a in ARCHETYPES.items():
        for _ in range(a["count"]):
            diet = pick(rng, a["diets"])
            specs.append(dict(
                arch=key, n_adults=int(rng.integers(a["adults"][0], a["adults"][1] + 1)),
                n_children=int(rng.integers(a["children"][0], a["children"][1] + 1)), diet=diet,
                eats_out_share=uniform(rng, a["eats_out"]), weekend_boost=uniform(rng, a["weekend"]),
                demand_noise=uniform(rng, a["noise"]), guest_prob=uniform(rng, a["guest"]),
                allergen_exclusions=draw_allergen(rng, diet)))

    regional = regional_reference_prices(products)
    regional_rows = []
    rows = []
    for i, s in enumerate(specs):
        a = ARCHETYPES[s["arch"]]
        row = {"household_id": s.get("household_id", f"hh_{i + 1:02d}"), "persona": a["label"],
               "n_adults": s["n_adults"], "n_children": s["n_children"],
               "n_people": s["n_adults"] + s["n_children"], "diet": s["diet"],
               "allergen_exclusions": s["allergen_exclusions"], "eats_out_share": s["eats_out_share"],
               "kcal_per_adult_day": a["kcal"]}
        taste = draw_taste(DEFAULT_SEED + i, len(products))      # same taste as data_gen will use for this household
        cost = expected_weekly_cost({**row}, products, taste)
        tier_factor = uniform(rng, TIER_FACTOR[a["tier"]])      # drawn once: tightness of the household in every region
        budget = int(max(20, 5 * round(cost * tier_factor / 5)))
        protein_week = (s["n_adults"] * a["protein"] + s["n_children"] * PROTEIN_CHILD_G_DAY) * 7 \
            * (1 - s["eats_out_share"])
        row.update({"weekly_budget_usd": budget, "weekly_protein_target_g": int(10 * round(protein_week / 10)),
                    "weekend_boost": s["weekend_boost"], "demand_noise": s["demand_noise"],
                    "guest_prob": s["guest_prob"], "description": describe(a, {**s}, budget)})
        rows.append(row)
        for region, ref_price in regional.items():              # same household, same taste, same tier, regional prices
            region_cost = expected_weekly_cost({**row}, products.assign(price_usd=ref_price), taste)
            regional_rows.append({"household_id": row["household_id"], "region_id": region,
                                  "weekly_budget_usd": int(max(20, 5 * round(region_cost * tier_factor / 5)))})

    out = pd.DataFrame(rows)
    out.to_csv("data/households.csv", index=False)
    print(f"Wrote data/households.csv with {len(out)} households")
    if regional_rows:
        reg = pd.DataFrame(regional_rows)
        reg.to_csv("data/household_budgets_regional.csv", index=False)
        print(f"Wrote data/household_budgets_regional.csv: {len(reg)} rows "
              f"({reg['household_id'].nunique()} households x {reg['region_id'].nunique()} regions)")
        print("Average weekly budget by region:")
        print(reg.groupby("region_id")["weekly_budget_usd"].mean().round(1).to_string())
    print(out[["household_id", "persona", "n_people", "diet", "allergen_exclusions", "weekly_budget_usd",
               "weekly_protein_target_g"]].to_string(index=False))


if __name__ == "__main__":
    main()