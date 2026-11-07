"""Diet and allergen rules shared by the data generator, the constraints layer and the meal planner.

- A household has a diet (omnivore / pescatarian / vegetarian / vegan) and a list of allergens.
- A product is allowed for a household if neither its diet tag nor its allergens are excluded.
Allergens are HARD exclusions (safety). Diets are lifestyle exclusions.
"""
import numpy as np
import pandas as pd

# Animal-origin tag of each product (products without a tag are plant based).
PRODUCT_TAGS = {
    "chicken_breast": "meat", "ground_beef": "meat",
    "salmon_fillet": "fish", "canned_tuna": "fish",
    "milk": "dairy", "yogurt": "dairy", "cheddar_cheese": "dairy", "butter": "dairy", "frozen_pizza": "dairy",
    "eggs": "egg",
}

DIET_EXCLUDED_TAGS = {
    "omnivore": set(),
    "pescatarian": {"meat"},
    "vegetarian": {"meat", "fish"},
    "vegan": {"meat", "fish", "dairy", "egg"},
}


def parse_list(value) -> set:
    """'peanut;soy' -> {'peanut', 'soy'}; empty / NaN -> empty set. ',' is accepted as a separator too."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {part.strip().lower() for part in str(value).replace(",", ";").split(";") if part.strip()}


def allowed_mask(household: dict, products: pd.DataFrame) -> np.ndarray:
    """Boolean array (one entry per product row) = product may be bought/eaten by this household."""
    excluded_tags = DIET_EXCLUDED_TAGS[str(household.get("diet", "omnivore"))]
    excluded_allergens = parse_list(household.get("allergen_exclusions"))
    mask = []
    for r in products.itertuples():
        tag_ok = PRODUCT_TAGS.get(r.product_id) not in excluded_tags
        allergen_ok = not (parse_list(r.allergens) & excluded_allergens)
        mask.append(tag_ok and allergen_ok)
    return np.array(mask, dtype=bool)