"""Real U.S. grocery price history from the Open Prices dataset (Open Food Facts).

Run from the repo root:
    python -m src.fetch_open_prices                 # downloads the dataset snapshot (~80 MB) if not cached
    python -m src.fetch_open_prices --input path/to/prices.parquet

Source: https://huggingface.co/datasets/openfoodfacts/open-prices (licence: ODbL 1.0, credit Open Prices).
Output: data/price_history.csv  (one row per accepted real price observation)
        data/price_history_meta.json (snapshot info, counts, rejection reasons)

Matching is deliberately strict. A record is kept only if ALL of these hold:
  1. U.S. store (OSM country code US) and currency USD; not flagged as a duplicate by Open Prices.
  2. Product name rule: the name matches the product's include pattern and none of its exclude patterns.
  3. Category rule: Open Food Facts category tags contain one required tag and none of the excluded tags
     (loose produce without a barcode must carry one of the listed raw category tags instead).
  4. Package-size rule: the package size is known, unambiguous (the numeric field and the label text agree)
     and inside the product's accepted range, so the price can be converted to our purchase unit.
  5. Not a bundle (twin / variety pack) and not a packaged product from a warehouse club (Costco, Sam's Club,
     BJ's), whose bundle prices do not match the per-unit package size in Open Food Facts.
  6. The record matches exactly one of our products (otherwise it is ambiguous and rejected).
  7. The converted price is not an outlier: more than 3x away from the product's median (products with
     at least 5 records), or more than 5x away from the products.csv price (products with fewer records,
     a wide sanity bound against price-entry errors).
Nothing is imputed: a missing regular price stays missing, unknown sizes or counts are rejected.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URL = "https://huggingface.co/datasets/openfoodfacts/open-prices/resolve/main/prices.parquet"
DATASET_API = "https://huggingface.co/api/datasets/openfoodfacts/open-prices"
RAW_PATH = Path("data/open_prices_raw/prices.parquet")
OUT_PATH = Path("data/price_history.csv")
META_PATH = Path("data/price_history_meta.json")
PRODUCTS_PATH = Path("data/products.csv")

OUTLIER_FACTOR = 3.0      # reject prices more than 3x above or below the product median
OUTLIER_MIN_N = 5         # the outlier check needs at least this many observations of a product
CATALOG_FACTOR = 5.0      # products with < OUTLIER_MIN_N records: must be within 5x of the products.csv price
SIZE_TOLERANCE = 0.05     # numeric size and label text must agree within 5 %

COLUMNS = [
    "id", "type", "product_code", "product_name", "category_tag", "price", "price_is_discounted",
    "price_without_discount", "discount_type", "price_per", "currency", "date", "duplicate_of", "source",
    "proof_type", "product_product_name", "product_product_quantity", "product_product_quantity_unit",
    "product_quantity", "product_categories_tags", "location_osm_id", "location_osm_type",
    "location_osm_name", "location_osm_brand", "location_osm_address_city",
    "location_osm_address_postcode", "location_osm_address_country_code",
]

OUTPUT_COLUMNS = [
    "product_id", "unit", "date", "regular_price_usd", "promo_price_usd", "price_is_discounted",
    "discount_type", "package_price_usd", "package_regular_price_usd", "package_size", "package_unit",
    "units_per_package", "source_product_name", "product_code", "category_tag", "store_name", "store_brand",
    "store_city", "store_postcode", "store_osm_id", "store_osm_type", "proof_type", "source_app",
    "source", "source_price_id", "match_type",
]


# --------------------------------------------------------------------------------------------------
# Matching rules, one per product in data/products.csv
# kind: "mass" (g), "volume" (ml), "count" (pieces), "loaf" (one loaf, size in g only checked)
# --------------------------------------------------------------------------------------------------
@dataclass
class Rule:
    kind: str
    std_qty: float                     # size of our purchase unit in g / ml / pieces
    size_range: tuple[float, float]    # accepted package size (same base unit)
    name_any: str = ""                 # regex the product name must match (case-insensitive)
    name_none: str = ""                # regex the product name must not match
    cats_any: tuple[str, ...] = ()     # OFF category tags, at least one required
    cats_none: tuple[str, ...] = ()    # OFF category tags that disqualify the record
    raw_tags: tuple[str, ...] = ()     # loose produce (no barcode): accepted category_tag values
    raw_price_per: tuple[str, ...] = ()  # accepted price_per for loose produce
    scale: bool = True                 # convert by size; False = one package is one purchase unit
    extra: dict = field(default_factory=dict)


FLAVOURS = r"chocolate|vanilla|strawberr|blueberr|raspberr|peach|cherry|mango|banana|honey|maple|cinnamon|caramel|mocha|coffee|fruit"
MULTIPACK = r"\b\d+\s*(?:-\s*)?(?:pk|pack|packs|cans|ct|count)\b|\bmulti-?pack\b|\b\d+\s*x\s*\d"
# Warehouse clubs sell bundles under barcodes whose Open Food Facts size is one unit (e.g. a $18.79 "5 oz"
# tuna = a multipack), so their packaged-product prices cannot be converted reliably. Loose produce is kept.
WAREHOUSE_CLUBS = r"costco|sam'?s club|\bbj'?s\b"
BUNDLE = r"twin pack|two pack|\b2\s*-?\s*pack\b|variety pack|\bbundle\b"
WHOLESALE = r"\b(?:[2-9]\d|\d{3})\s*(?:lb|lbs|pounds?)\b|\bcase\b|\b\d{2,}\s*ct\b"   # 20+ lb sacks, cases
PREPARED = ("en:meals", "en:sandwiches", "en:snacks", "en:sweet-snacks", "en:salty-snacks", "en:desserts",
            "en:confectioneries", "en:sauces", "en:soups", "en:prepared-salads")

RULES: dict[str, Rule] = {
    "milk": Rule("volume", 1000, (900, 3800),
                 name_any=r"\bmilk\b",
                 name_none=FLAVOURS + r"|almond|oat|soy|coconut|cashew|rice|hazelnut|pea\b|plant|non-?dairy|"
                           r"condensed|evaporated|powder|dry|buttermilk|goat|kefir|cream|shake|protein|tea|latte|"
                           r"eggnog|creamer|ice",
                 cats_any=("en:milks", "en:cow-milks"),
                 cats_none=("en:flavoured-milks", "en:milk-substitutes", "en:plant-based-foods-and-beverages",
                            "en:powdered-milks", "en:condensed-milks", "en:evaporated-milks", "en:chocolates",
                            "en:creams", "en:fermented-milk-products", "en:goat-milks") + PREPARED),
    "eggs": Rule("count", 12, (6, 24),
                 name_any=r"\beggs?\b",
                 name_none=r"white|liquid|substitute|noodle|salad|sandwich|burrito|bites|patty|roll|nog|chocolate|"
                           r"plant|vegan|boiled|frittata|scramble|pasta|bagel|dye|duck|quail",
                 cats_any=("en:chicken-eggs", "en:fresh-eggs", "en:eggs"),
                 cats_none=PREPARED + ("en:egg-whites", "en:liquid-eggs", "en:duck-eggs", "en:quail-eggs")),
    "chicken_breast": Rule("mass", 500, (300, 3000),
                           name_any=r"chicken breasts?|breast fillets?",
                           name_none=r"bread|nugget|tender|strip|cooked|grilled|rotisserie|roast|smoked|deli|"
                                     r"slice|lunch|sandwich|wrap|bites|patt|marinat|season|fried|crispy|stuffed|"
                                     r"popcorn|soup|broth|salad|dog|cat|plant|meatless|ground|chunk|canned|skewer|glaze|honey|buffalo|"
                                     r"shredded|diced|fritter",
                           cats_any=("en:chicken-breasts", "en:chicken-breast-fillets", "en:chickens"),
                           cats_none=PREPARED + ("en:breaded-products", "en:cooked-poultries", "en:poultry-nuggets",
                                                 "en:chicken-preparations", "en:meat-preparations",
                                                 "en:breaded-chicken", "en:sliced-chicken", "en:canned-foods",
                                                 "en:canned-meats", "en:ground-meats")),
    "rice": Rule("mass", 1000, (450, 5000),
                 name_any=r"\brice\b",
                 name_none=r"cake|crisp|cracker|noodle|vinegar|milk|bowl|pudding|krispie|cereal|flour|season|pilaf|"
                           r"mix|fried|spanish|mexican|cauliflower|chip|bran|wine|paper|ready|microwav|cup|instant|"
                           r"heat|sauce|bar\b|syrup|sushi|risotto|broccoli|chicken|bean|cooked|frozen|steam|wild|black|"
                           r"purple|forbidden|arborio|carnaroli|red rice|sticky|glutinous|sprouted|blend|medley",
                 cats_any=("en:rices",),
                 cats_none=PREPARED + ("en:rice-dishes", "en:cooked-rices", "en:precooked-rices",
                                       "en:frozen-foods", "en:rice-flours", "en:rice-cakes", "en:puffed-rice-cakes")),
    "bread": Rule("loaf", 1, (400, 1000), scale=False,
                  name_any=r"\bbread\b|\bloaf\b",
                  name_none=r"crumb|stuffing|garlic|pita|naan|flat|tortilla|corn ?bread|banana|zucchini|pumpkin|"
                            r"pudding|mix|stick|bowl|crouton|bun|roll|bagel|muffin|cinnamon|raisin|brioche|sweet|"
                            r"gluten.?free|keto|cake|frozen|toast",
                  cats_any=("en:sliced-breads", "en:white-breads", "en:wheat-breads", "en:wholemeal-breads",
                            "en:multigrain-sliced-breads", "en:sandwich-breads", "en:breads"),
                  cats_none=PREPARED + ("en:flatbreads", "en:buns", "en:sweet-breads", "en:breadcrumbs",
                                        "en:bread-mixes", "en:bagel-breads", "en:bagels", "en:brioches",
                                        "en:gluten-free-breads", "en:frozen-foods", "en:viennoiseries")),
    "bananas": Rule("mass", 1000, (0, 0), raw_tags=("en:bananas",), raw_price_per=("KILOGRAM",)),
    "apples": Rule("mass", 1000, (0, 0),
                   raw_tags=("en:apples", "en:fuji-apples", "en:gala-apples", "en:granny-smith-apples",
                             "en:golden-delicious-apples", "en:honeycrisp-apples", "en:red-delicious-apples",
                             "en:pink-lady-apples"),
                   raw_price_per=("KILOGRAM",)),
    "tomatoes": Rule("mass", 500, (0, 0), raw_tags=("en:tomatoes", "en:roma-tomatoes", "en:vine-tomatoes"),
                     raw_price_per=("KILOGRAM",)),
    "spinach": Rule("mass", 200, (140, 600),
                    name_any=r"spinach",
                    name_none=r"frozen|chopped|creamed|dip|pasta|tortilla|wrap|artichoke|souffl|pie|pizza|soup|"
                              r"ravioli|mix|blend|spring|kale|arugula|chard|salad kit|bites|cheese|feta|canned",
                    cats_any=("en:spinachs", "en:baby-spinachs", "en:spinach-leaves"),
                    cats_none=PREPARED + ("en:frozen-foods", "en:frozen-vegetables", "en:canned-foods",
                                          "en:frozen-plant-based-foods"),
                    raw_tags=("en:spinachs",), raw_price_per=("KILOGRAM",)),
    "lentils": Rule("mass", 500, (300, 2300),
                    name_any=r"lentils?",
                    name_none=r"soup|chip|pasta|cracker|puff|snack|curry|dal\b|dahl|stew|meal|cooked|steamed|"
                              r"ready|canned|sprouted|flour|pilaf|salad|burger|bites",
                    cats_any=("en:lentils", "en:green-lentils", "en:red-lentils", "en:brown-lentils",
                              "en:dried-lentils", "en:black-lentils"),
                    cats_none=PREPARED + ("en:canned-foods", "en:canned-legumes", "en:lentil-soups",
                                          "en:pastas", "en:flours")),
    "yogurt": Rule("mass", 500, (400, 1000),
                   name_any=r"yogh?urt",
                   name_none=FLAVOURS + r"|lemon|key lime|coconut|almond|oat|soy|cashew|plant|dairy.?free|drink|"
                                        r"smoothie|dip|coated|frozen|pretzel|raisin|bar\b|tube|kids|dressing|"
                                        r"sauce|granola|parfait|cup|stick|favor|quadratini|skyr|icelandic|kefir|flavou?r|blended|"
                                        r"squeez|danimals|candy|light\s*(?:&|and)\s*fit|lime|pie|whip|mousse",
                   cats_any=("en:yogurts", "en:plain-yogurts", "en:greek-style-yogurts", "en:natural-yogurts"),
                   cats_none=("en:meals", "en:snacks", "en:sweet-snacks", "en:confectioneries", "en:fruit-yogurts", "en:flavoured-yogurts", "en:dairy-substitutes",
                                         "en:plant-based-yogurts", "en:drinkable-yogurts",
                                         "en:fermented-dairy-desserts-with-fruits", "en:frozen-yogurts")),
    "oats": Rule("mass", 500, (250, 2300),
                 name_any=r"\boats\b|oatmeal|\bporridge\b",
                 name_none=FLAVOURS + r"|apple|brown sugar|raisin|milk|bar\b|bars|cookie|granola|flour|bran|"
                                      r"overnight|cup|bites|cereal|squares|crisp|cluster|protein|packet|pouch|"
                                      r"instant|flavor|variety|creamer|bread|muffin|pancake|clusters",
                 cats_any=("en:rolled-oats", "en:oat-flakes", "en:porridge-oats", "en:oats", "en:steel-cut-oats",
                           "en:quick-oats", "en:rolled-flakes"),
                 cats_none=("en:snacks", "en:sweet-snacks", "en:biscuits", "en:cereal-bars", "en:granolas",
                            "en:mueslis", "en:flavoured-oatmeals", "en:instant-oatmeals", "en:oat-flours",
                            "en:plant-based-beverages")),
    "tofu": Rule("mass", 400, (300, 600),
                 name_any=r"\btofu\b",
                 name_none=r"bake|smoke|marinat|season|fried|teriyaki|bites|nugget|scramble|skin|pudding|dessert|"
                           r"noodle|burger|sesame|sauce|soup|cubes|puffs|flavou?r|sriracha|curry|ginger",
                 cats_any=("en:tofu", "en:tofus", "en:plain-tofu", "xx:tofu", "en:firm-tofus", "en:silken-tofus"),
                 cats_none=PREPARED + ("en:meat-preparations", "en:smoked-tofus", "en:flavoured-tofus")),
    "carrots": Rule("mass", 1000, (300, 2500),
                    name_any=r"carrots?",
                    name_none=r"juice|cake|chip|frozen|canned|soup|puree|baby food|pouch|peas|mix|blend|"
                              r"dip|slaw|muffin|bread|glazed|steam|celery|ranch|hummus|snack pack|cooked",
                    cats_any=("en:carrots", "en:baby-carrots", "en:fresh-carrots"),
                    cats_none=PREPARED + ("en:frozen-foods", "en:canned-foods", "en:beverages",
                                          "en:frozen-vegetables", "en:juices-and-nectars", "en:baby-foods",
                                          "en:mixes-of-vegetables")),
    "potatoes": Rule("mass", 2000, (1000, 5000),
                     name_any=r"potato(?:es)?",
                     name_none=r"sweet|chip|fries|tots|mash|hash|flake|salad|soup|crisp|wedge|skins|bread|starch|"
                               r"flour|pancake|frozen|dehydrat|instant|scallop|au gratin|roast|seasoned|canned|"
                               r"bun|roll|snack|stick",
                     cats_any=("en:potatoes", "en:fresh-potatoes", "en:russet-potatoes", "en:red-potatoes",
                               "en:yellow-potatoes", "en:white-potatoes"),
                     cats_none=PREPARED + ("en:sweet-potatoes", "en:chips-and-fries", "en:crisps",
                                           "en:frozen-foods", "en:potato-crisps", "en:mashed-potatoes",
                                           "en:potato-preparations", "en:dehydrated-potatoes", "en:canned-foods"),
                     raw_tags=("en:potatoes", "en:russet-potatoes", "en:red-potatoes", "en:yellow-potatoes",
                               "en:white-potatoes"),
                     raw_price_per=("KILOGRAM",)),
    "pasta": Rule("mass", 500, (340, 2300),
                  name_any=r"spaghetti|penne|rotini|fusilli|macaroni|linguine|fettuccine|farfalle|rigatoni|elbows?\b|"
                           r"\bpasta\b|ziti|angel hair|bow ?ties?|capellini|orecchiette|bucatini|cavatappi|"
                           r"conchiglie|shells\b|orzo|vermicelli",
                  name_none=r"sauce|salad|cheese|dinner|meal|chip|frozen|fresh|filled|ravioli|tortellini|gnocchi|"
                            r"lasagn|chickpea|lentil|bean|rice|gluten|cauliflower|protein|veggie|zucchini|squash|"
                            r"konjac|shirataki|egg noodle|kit|soup|bake|skillet|cup|instant|alfredo|pesto|"
                            r"meatball|bowl|entree|marie calend|seasoning|organic|whole wheat|whole grain|legume|keto|low.?carb",
                  cats_any=("en:pastas", "en:dry-pastas", "en:durum-wheat-pastas", "en:spaghetti",
                            "en:durum-wheat-semolina-pastas"),
                  cats_none=PREPARED + ("en:pasta-dishes", "en:fresh-pastas", "en:stuffed-pastas",
                                        "en:gluten-free-pastas", "en:legume-pastas", "en:frozen-foods",
                                        "en:canned-foods", "en:pasta-sauces", "en:wholemeal-pastas")),
    "cheddar_cheese": Rule("mass", 200, (150, 1000),
                           name_any=r"cheddar",
                           name_none=r"cracker|chip|popcorn|sauce|soup|bread|biscuit|bagel|goldfish|puff|snack|mac|"
                                     r"powder|dip|spread|crisps|pretzel|jalape|bacon|bites|whisps|curl|crunch|"
                                     r"sausage|burger|potato|plant|vegan|dairy.?free|cheez|nacho|cups|tray|"
                                     r"cracker cuts|string|stick|blend|jack|colby|fiesta|mexican|queso|mozzarella|asadero|"
                                     r"pepper|marble|swiss|gouda|parmesan",
                           cats_any=("en:cheddar-cheese", "en:cheddars", "en:white-cheddar-cheeses",
                                     "en:cheeses-from-england", "en:cow-cheeses", "en:cheeses"),
                           cats_none=PREPARED + ("en:appetizers", "en:crackers", "en:cheese-spreads",
                                                 "en:processed-cheeses", "en:cheese-sauces",
                                                 "en:plant-based-foods-and-beverages", "en:dairy-substitutes")),
    "peanut_butter": Rule("mass", 340, (250, 1400),
                          name_any=r"peanut butter",
                          name_none=r"cup|cookie|cracker|\bbar\b|bars|chip|pretzel|filled|powder|ice|candy|wafer|"
                                    r"sandwich|granola|chocolate|honey|cereal|protein|puff|bites|banana|jelly|grape|"
                                    r"strawberr|whip|squeeze|pouch|cluster|oat|pie|fudge|treat|dog|cat|maple|"
                                    r"cinnamon|blend|spread with|snack|almond|cashew",
                          cats_any=("en:peanut-butters", "en:smooth-peanut-butters", "en:crunchy-peanut-butters"),
                          cats_none=("en:snacks", "en:sweet-snacks", "en:confectioneries", "en:biscuits",
                                     "en:cereal-bars", "en:chocolate-spreads", "en:meals", "en:sandwiches",
                                     "en:powdered-peanut-butters")),
    "salmon_fillet": Rule("mass", 400, (150, 1500),
                          name_any=r"salmon",
                          name_none=r"smoke|lox|canned|pouch|jerky|burger|cake|patt|sushi|salad|dip|spread|bites|"
                                    r"nugget|cream cheese|bagel|oil|dog|cat|treat|pink salmon|chunk|flakes|roe|"
                                    r"caviar|poke|cured|gravlax|teriyaki|bowl|meal|sauce|marinat|season|glaze|herb|citrus|lemon|"
                                    r"butter|dill|honey|bourbon|chipotle|blacken|cajun|pesto|garlic|bbq|maple|"
                                    r"crust|stuffed|skin|roll|biscu|feline|food|cedar|plank",
                          cats_any=("en:salmon-fillets", "en:salmons", "en:atlantic-salmons",
                                    "en:fresh-salmon-fillets", "en:frozen-salmon-fillets"),
                          cats_none=PREPARED + ("en:smoked-fishes", "en:smoked-salmons", "en:canned-foods",
                                                "en:canned-fishes", "en:fish-preparations", "en:cooked-fishes")),
    "ground_beef": Rule("mass", 500, (300, 2500),
                        name_any=r"ground beef|beef,? ground|ground chuck|ground round|ground sirloin",
                        name_none=r"plant|meatless|impossible|beyond|meal|taco|burrito|chili|sauce|cooked|crumble|"
                                  r"patt|burger|meatball|lasagn|pasta|soup|pie|dog|cat|turkey|pork|chicken|bison|"
                                  r"lamb|blend|seasoned|frozen",
                        cats_any=("en:ground-beef", "en:ground-beef-meats", "en:ground-meats", "en:beef"),
                        cats_none=PREPARED + ("en:plant-based-foods-and-beverages", "en:meat-alternatives",
                                              "en:meat-analogues", "en:cooked-meats", "en:frozen-foods",
                                              "en:burgers", "en:meatballs")),
    "canned_tuna": Rule("mass", 142, (110, 180),
                        name_any=r"\btuna\b",
                        name_none=r"salad|kit|pouch|sandwich|jerky|steak|creation|spread|lunch|dip|sushi|helper|"
                                  r"cat|dog|infused|flavou?r|season|lemon|pepper|spicy|sriracha|jalape|herb|garlic|"
                                  r"teriyaki|ranch|smoke|bowl|cracker|pasta|noodle|casserole|sauce|olive oil|"
                                  r"oil\b|fillet|ventresca|belly|" + MULTIPACK,
                        cats_any=("en:canned-tunas", "en:canned-light-tunas", "en:canned-albacore-tunas"),
                        cats_none=("en:prepared-salads", "en:meals", "en:sandwiches", "en:snacks",
                                   "en:tunas-in-olive-oil", "en:tunas-in-oil", "en:pet-food")),
    "chickpeas_canned": Rule("mass", 425, (380, 480),
                             name_any=r"chick ?peas|garbanzo",
                             name_none=r"hummus|snack|roast|crunch|pasta|chip|puff|flour|dried|dry|curry|masala|"
                                       r"salad|soup|seasoned|bites|spread|dip|aquafaba|cracker|" + MULTIPACK,
                             cats_any=("en:canned-chickpeas", "en:canned-legumes", "en:canned-common-beans"),
                             cats_none=PREPARED + ("en:hummus", "en:dried-chickpeas", "en:roasted-chickpeas",
                                                   "en:dips", "en:spreads")),
    "black_beans_canned": Rule("mass", 425, (380, 480),
                               name_any=r"black beans",
                               name_none=r"chip|soup|refried|rice|dip|burrito|salsa|pasta|seasoned|spic|chili|"
                                         r"burger|corn|dried|dry|bowl|cracker|snack|cuban|frijoles negros refritos|"
                                         r"" + MULTIPACK,
                               cats_any=("en:canned-black-beans", "en:canned-common-beans", "en:canned-legumes"),
                               cats_none=PREPARED + ("en:refried-beans", "en:dried-beans", "en:dried-legumes",
                                                     "en:dips")),
    "tortillas": Rule("count", 10, (6, 30),
                      name_any=r"tortillas?",
                      name_none=r"chip|strip|soup|crisp|snack|bowl|kit|corn|maiz|ma[ií]z|whole wheat|low.?carb|"
                                r"keto|gluten|spinach|tomato|almond|cassava|protein|street taco|mini|taco shell|"
                                r"tostada|wrap|española|espanola|egg|potato",
                      cats_any=("en:flour-tortillas", "en:wheat-tortillas", "en:tortillas", "en:flatbreads"),
                      cats_none=PREPARED + ("en:crisps", "en:chips-and-fries", "en:corn-tortillas",
                                            "en:corn-chips", "en:tortilla-chips", "en:taco-shells")),
    "butter": Rule("mass", 225, (200, 1000),
                   name_any=r"\bbutter\b",
                   name_none=r"peanut|almond|nut|cashew|sunflower|seed|cookie|apple|cocoa|shea|cup|popcorn|cracker|"
                             r"garlic|biscuit|croissant|chicken|sauce|cake|bean|bread|pie|toffee|milk|squash|"
                             r"caramel|fudge|candy|spray|substitute|vegan|plant|margarine|spread|buttery|"
                             r"pecan|lettuce|crunch|bar\b|scotch|rum|herb|honey|cinnamon|ghee|oil|dairy.?free|"
                             r"brown butter|beurre|chips|mints|truffle|pretzel|waffle|pancake|lobster|baby|"
                             r"cream|brickle|frosting|icing|olive|tahini|soy|pumpkin|believe|whipped|spreadable|light",
                   cats_any=("en:butters", "en:dairy-butters", "en:salted-butters", "en:unsalted-butters",
                             "en:sweet-cream-butters", "en:butter"),
                   cats_none=("en:plant-based-spreads", "en:margarines", "en:nut-butters", "en:legume-butters",
                              "en:snacks", "en:sweet-snacks", "en:biscuits", "en:confectioneries",
                              "en:plant-based-foods-and-beverages", "en:ghee", "en:clarified-butters",
                              "en:light-butters")),
    "plant_milk": Rule("volume", 1000, (900, 3800),
                       name_any=r"soy ?milk|soymilk|almond ?milk|almondmilk|oat ?milk|oatmilk|"
                                r"(?:soy|almond|oat)\b.*\b(?:beverage|drink)",
                       name_none=FLAVOURS + r"|creamer|yogurt|ice|frozen|coffee|protein|shake|kids|barista|"
                                            r"latte|nog|matcha|chai|dessert|cookie|cereal|chip|bar\b|"
                                            r"powder|cheese|butter",
                       cats_any=("en:plant-based-milk-alternatives", "en:soy-milks", "en:almond-milks",
                                 "en:oat-milks", "en:milk-substitutes", "en:soy-beverages", "en:almond-beverages",
                                 "en:oat-based-drinks"),
                       cats_none=PREPARED + ("en:creamers", "en:plant-based-creamers", "en:frozen-desserts",
                                             "en:yogurts", "en:plant-based-yogurts", "en:flavoured-plant-milks",
                                             "en:chocolate-plant-milks", "en:coffee-creamers")),
    "oranges": Rule("mass", 1000, (0, 0), name_none=r"mand|clementine|tangerine|satsuma|juice|blood|cara|sumo|citrus|kumquat",
                    raw_tags=("en:oranges", "en:navel-oranges", "en:valencia-oranges"),
                    raw_price_per=("KILOGRAM",)),
    "avocados": Rule("count", 4, (3, 6),
                     name_any=r"avocados?\b",
                     name_none=r"oil|guac|dip|spread|bite|piece|chunk|pulp|chip|toast|salsa|dressing|mayo|"
                               r"frozen|mini|sliced|mash|smash|baby",
                     cats_any=("en:avocados", "en:hass-avocados"),
                     cats_none=PREPARED + ("en:guacamoles", "en:avocado-oils", "en:frozen-foods",
                                           "en:vegetable-oils", "en:dips"),
                     raw_tags=("en:avocados", "en:hass-avocados"), raw_price_per=("UNIT",)),
    "broccoli": Rule("mass", 500, (250, 1500),
                     name_any=r"broccoli",
                     name_none=r"frozen|cheese|sauce|rice|slaw|soup|salad|stir|mix|blend|pasta|chicken|beef|"
                               r"cauliflower|carrot|steam|tots|bites|casserole|pizza|kit|puree|canned|rabe|"
                               r"broccolini|sprouts|seed",
                     cats_any=("en:broccoli", "en:broccoli-florets", "en:broccoli-crowns", "en:fresh-broccoli"),
                     cats_none=PREPARED + ("en:frozen-foods", "en:frozen-vegetables", "en:frozen-plant-based-foods",
                                           "en:canned-foods", "en:mixes-of-vegetables", "en:pasta-dishes"),
                     raw_tags=("en:broccoli",), raw_price_per=("KILOGRAM",)),
    "onions": Rule("mass", 1000, (500, 5000),
                   name_any=r"onions?\b",
                   name_none=r"ring|soup|powder|dip|fried|crisp|chip|green|pearl|pickl|flake|minced|sauce|salt|"
                             r"seasoning|jam|relish|chive|scallion|spring|leek|shallot|garlic|frozen|diced|chopped|"
                             r"pepper|caramel|sour cream|bagel|bread|cracker|burger",
                   cats_any=("en:onions", "en:yellow-onions", "en:white-onions", "en:red-onions", "en:sweet-onions"),
                   cats_none=PREPARED + ("en:frozen-foods", "en:green-onions", "en:onion-powders", "en:spices",
                                         "en:condiments", "en:pickles", "en:shallots"),
                   raw_tags=("en:onions", "en:yellow-onions", "en:white-onions", "en:red-onions", "en:sweet-onions"),
                   raw_price_per=("KILOGRAM",)),
    "bell_peppers": Rule("count", 3, (2, 6),
                         name_any=r"bell peppers?|sweet peppers?",
                         name_none=r"roast|stuff|jar|pickl|chip|sauce|frozen|mini|hot|jalap|chili|chile|flake|"
                                   r"powder|stir|mix|blend|onion|diced|sliced|strips|dip|relish|banana|seed",
                         cats_any=("en:bell-peppers", "en:sweet-peppers", "en:red-bell-peppers",
                                   "en:mixed-bell-peppers", "en:green-bell-peppers", "en:yellow-bell-peppers"),
                         cats_none=PREPARED + ("en:frozen-foods", "en:canned-foods", "en:roasted-peppers",
                                               "en:pickles", "en:spices", "en:chili-peppers"),
                         raw_tags=("en:bell-peppers", "en:red-bell-peppers", "en:green-bell-peppers",
                                   "en:yellow-bell-peppers", "en:orange-bell-peppers", "en:mixed-bell-peppers"),
                         raw_price_per=("UNIT",)),
    "frozen_vegetables": Rule("mass", 1000, (300, 2500),
                              name_any=r"vegetables|veggies|medley|mixed|stir.?fry|california|normandy|winter blend|"
                                       r"peas (?:and|&) carrots|mixed vegetables",
                              name_none=r"sauce|season|butter|cheese|rice|pasta|noodle|fries|tots|potato|steak|meal|"
                                        r"bowl|pizza|avocado|fruit|berr|mango|chicken|beef|shrimp|pork|soup|"
                                        r"dumpling|egg roll|spring roll|burger|patt|nugget|lasagn|pie|casserole|"
                                        r"breaded|tempura|fried|riced|mashed|pot sticker|potsticker|wrap|burrito|"
                                        r"onion ring|teriyaki|gravy|alfredo|garlic|herb|lemon|glaze|honey|maple|"
                                        r"cream|quinoa|grain|chips|canned|dip|tamale|hash|soup|broth|chopped|"
                                        r"\bpeas\b(?!\s*(?:and|&)\s*carrots)|\bcorn\b|green beans|spinach|edamame",
                              cats_any=("en:frozen-vegetables",),
                              cats_none=PREPARED + ("en:canned-foods", "en:canned-vegetables", "en:frozen-fries",
                                                    "en:frozen-fruits", "en:potatoes-and-their-products",
                                                    "en:frozen-meals", "en:frozen-potatoes",
                                                    "en:cooking-helpers")),
    "olive_oil": Rule("volume", 500, (250, 3000),
                      name_any=r"olive oil",
                      name_none=r"spray|cooking spray|\bin (?:extra virgin )?olive oil|tuna|sardine|anchov|dressing|"
                                r"mayo|chip|cracker|bread|pasta|sauce|pesto|soap|infused|garlic|truffle|blend|"
                                r"vinaigrette|popcorn|hummus|season|tapenade|artichoke|tomato|lemon|basil|chili|"
                                r"rosemary|herb|pomace|light|mist|canola|vegetable|avocado|sunflower|flavou?r|"
                                r"with|marinat|olives\b|dipping|crisps",
                      cats_any=("en:olive-oils", "en:extra-virgin-olive-oils", "en:virgin-olive-oils"),
                      cats_none=PREPARED + ("en:canned-foods", "en:fishes-and-their-products", "en:seafood",
                                            "en:flavoured-olive-oils", "en:olive-pomace-oils", "en:cooking-sprays",
                                            "en:oil-blends", "en:salad-dressings")),
    "orange_juice": Rule("volume", 1000, (900, 3800),
                         name_any=r"orange juice|\boj\b",
                         name_none=r"blend|cocktail|drink|punch|soda|concentrate|frozen|sparkling|mango|pineapple|"
                                   r"carrot|smoothie|cranberry|mix|banana|strawberr|tangerine|mandarin|ginger|"
                                   r"turmeric|lemonade|beverage|light|less sugar|kids|shot|cold.?pressed|probiotic|"
                                   r"" + MULTIPACK,
                         cats_any=("en:orange-juices", "en:squeezed-orange-juices", "en:orange-juices-with-pulp",
                                   "en:orange-juices-without-pulp", "en:not-from-concentrate-orange-juices"),
                         cats_none=PREPARED + ("en:fruit-nectars", "en:juice-drinks", "en:concentrated-juices",
                                               "en:orange-juices-from-concentrate", "en:multifruit-juices",
                                               "en:fruit-juice-blends", "en:smoothies", "en:frozen-foods",
                                               "en:sodas")),
    "breakfast_cereal": Rule("mass", 400, (200, 1000),
                             name_any=r".",
                             name_none=r"\bbar\b|bars|oatmeal|\boats\b|rolled|porridge|grits|cream of wheat|"
                                       r"farina|snack|bites|cup|bowl|milk|treats|cookie|cracker|"
                                       r"instant|hot cereal|granola|muesli|baby|infant|toddler|" + MULTIPACK,
                             cats_any=("en:breakfast-cereals",),
                             cats_none=("en:rolled-oats", "en:oat-flakes", "en:porridge-oats", "en:porridges",
                                        "en:mueslis", "en:granolas", "en:cereal-bars", "en:snacks",
                                        "en:sweet-snacks", "en:biscuits", "en:baby-foods", "en:hot-cereals",
                                        "en:oatmeals", "en:instant-oatmeals", "en:flavoured-oatmeals",
                                        "en:quick-oats", "en:steel-cut-oats", "en:oats")),
    "frozen_pizza": Rule("mass", 400, (300, 900),
                         name_any=r"pizza",
                         name_none=r"sauce|crust|dough|bites|roll|snack|kit|mini|pocket|bagel|cracker|chip|"
                                   r"seasoning|flour|stuffed|french bread|pizzeria style rolls|pretzel|cheese sticks|"
                                   r"cauliflower|gluten|keto|" + MULTIPACK,
                         cats_any=("en:frozen-pizzas",),
                         cats_none=("en:pizza-sauces", "en:pizza-doughs", "en:pizza-crusts", "en:snacks",
                                    "en:appetizers", "en:pizza-rolls", "en:gluten-free-pizzas")),
}


# --------------------------------------------------------------------------------------------------
# Package size parsing
# --------------------------------------------------------------------------------------------------
MASS_G = {"g": 1, "gr": 1, "gram": 1, "grams": 1, "kg": 1000, "kilogram": 1000, "kilograms": 1000,
          "oz": 28.349523125, "ounce": 28.349523125, "ounces": 28.349523125,
          "lb": 453.59237, "lbs": 453.59237, "pound": 453.59237, "pounds": 453.59237}
VOL_ML = {"ml": 1, "milliliter": 1, "millilitre": 1, "cl": 10, "l": 1000, "liter": 1000, "litre": 1000,
          "liters": 1000, "litres": 1000, "fl oz": 29.5735295625, "floz": 29.5735295625,
          "fl. oz": 29.5735295625, "fluid ounce": 29.5735295625, "fluid ounces": 29.5735295625,
          "gal": 3785.411784, "gallon": 3785.411784, "gallons": 3785.411784,
          "qt": 946.352946, "quart": 946.352946, "quarts": 946.352946,
          "pt": 473.176473, "pint": 473.176473, "pints": 473.176473}
SIZE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(fl\.?\s*oz|fluid ounces?|kilograms?|grams?|kg|gr|g|ounces?|oz|pounds?|lbs?|"
    r"millilit(?:er|re)s?|ml|cl|lit(?:er|re)s?|l|gallons?|gal|quarts?|qt|pints?|pt)\b", re.I)
COUNT_RE = re.compile(r"(\d+)\s*(?:-\s*)?(?:ct|count|pk|pack|pcs|pieces?|units?|eggs|tortillas|avocados|peppers)\b",
                      re.I)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_size_text(text: str | None) -> tuple[float, str, bool] | None:
    """Parse a label like '20 oz (567g)', '1 lb 8 oz', 'half gallon', '32 fl oz'.
    Returns (value, 'g'|'ml', plain_oz_used) or None. Text in parentheses is ignored (it repeats the size).
    Multi-packs ('4 x 5 oz') return None: the price would be for several packages."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = re.sub(r"\(.*?\)", " ", text.lower()).strip()
    if re.search(r"\d\s*[x×]\s*\d", t):
        return None
    if re.fullmatch(r"(?:1\s*)?half\s*gallon", t) or t in {"1/2 gallon", "1/2 gal"}:
        return VOL_ML["gallon"] / 2, "ml", False
    if t in {"gallon", "1 gallon", "one gallon"}:
        return VOL_ML["gallon"], "ml", False
    pairs = SIZE_RE.findall(t)
    if not pairs:
        return None
    mass = vol = 0.0
    plain_oz = False
    for value, unit in pairs:
        u = re.sub(r"\s+", " ", unit.lower().replace(".", ". ").replace(". ", " ")).strip()
        u = "fl oz" if u.startswith("fl") else u
        if u in VOL_ML:
            vol += _num(value) * VOL_ML[u]
        elif u in MASS_G:
            mass += _num(value) * MASS_G[u]
            plain_oz = plain_oz or u in {"oz", "ounce", "ounces"}
    if mass and vol:
        return None                                  # mixed dimensions: ambiguous
    return (mass, "g", plain_oz) if mass else (vol, "ml", False)


def package_size(row, kind: str) -> tuple[float | None, str]:
    """Package size in the rule's base unit (g, ml or pieces). Returns (size, reason_if_rejected)."""
    text = row["product_quantity"]
    if kind == "count":
        counts = set()
        for src in (text, row["name"]):
            if isinstance(src, str):
                counts |= {int(c) for c in COUNT_RE.findall(src)}
                low = src.lower()
                if re.search(r"half\s*dozen|1/2\s*dozen", low):
                    counts.add(6)
                elif re.search(r"\b(?:1\s*)?dozen\b", low):
                    counts.add(12)
                elif re.search(r"\b18\s*eggs?\b|\b1\.5\s*dozen", low):
                    counts.add(18)
        if isinstance(text, str) and re.fullmatch(r"\s*\d+\s*", text):
            counts.add(int(text))
        counts.discard(0)
        if len(counts) == 1:
            return float(counts.pop()), ""
        return None, "count_missing" if not counts else "count_conflict"

    want = "ml" if kind == "volume" else "g"
    off_q, off_u = row["product_product_quantity"], row["product_product_quantity_unit"]
    off = (float(off_q), off_u) if pd.notna(off_q) and off_q > 0 and off_u in ("g", "ml") else None
    txt = parse_size_text(text)
    if txt and txt[1] == "g" and want == "ml" and txt[2]:
        # US labels often write "64 oz" for liquids; read plain ounces as fluid ounces for liquid products
        txt = (txt[0] / MASS_G["oz"] * VOL_ML["fl oz"], "ml", False)
    if off and txt:
        if off[1] != txt[1] or abs(off[0] - txt[0]) > SIZE_TOLERANCE * max(off[0], txt[0]):
            return None, "size_conflict"
    size = off or (txt[:2] if txt else None)
    if size is None:
        return None, "size_missing"
    if size[1] != want:
        return None, "size_wrong_dimension"
    return size[0], ""


# --------------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------------
def download(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {}
    try:
        with urllib.request.urlopen(DATASET_API, timeout=60) as r:
            info["dataset_last_modified"] = json.load(r).get("lastModified")
    except Exception as e:  # metadata is optional
        info["dataset_last_modified"] = None
        print(f"warning: could not read dataset metadata ({e})")
    if not path.exists():
        print(f"downloading {DATASET_URL} -> {path}")
        urllib.request.urlretrieve(DATASET_URL, path)
    return info


def tags_of(x) -> set[str]:
    return set(x) if x is not None and not isinstance(x, float) else set()


def match_product(row) -> list[str]:
    """Product ids whose name + category rules accept this record (size is checked later)."""
    hits = []
    name, tags = row["name"], row["tags"]
    if row["type"] == "CATEGORY":
        if name and re.search(WHOLESALE, name, re.I):
            return hits
        for pid, r in RULES.items():
            if row["category_tag"] in r.raw_tags and not (name and r.name_none and re.search(r.name_none, name, re.I)):
                hits.append(pid)
        return hits
    if not name:
        return hits
    if re.search(BUNDLE, name, re.I):
        return hits
    for pid, r in RULES.items():
        if not r.name_any or not r.cats_any:
            continue
        if not re.search(r.name_any, name, re.I):
            continue
        if r.name_none and re.search(r.name_none, name, re.I):
            continue
        if not tags & set(r.cats_any) or tags & set(r.cats_none):
            continue
        hits.append(pid)
    return hits


def convert(row, pid: str) -> tuple[dict | None, str]:
    """Convert one matched record to our purchase unit. Returns (fields, rejection_reason)."""
    r = RULES[pid]
    price = float(row["price"])
    pwd = float(row["price_without_discount"]) if pd.notna(row["price_without_discount"]) else np.nan
    discounted = bool(row["price_is_discounted"]) if pd.notna(row["price_is_discounted"]) else False
    if not price > 0:
        return None, "price_not_positive"
    store = f"{row['location_osm_name'] or ''} {row['location_osm_brand'] or ''}"
    if row["type"] == "PRODUCT" and re.search(WAREHOUSE_CLUBS, store, re.I):
        return None, "warehouse_club_pack_size_unreliable"
    if not np.isnan(pwd) and not pwd > price:
        return None, "regular_not_above_promo"

    if row["type"] == "CATEGORY":
        per = row["price_per"]
        if per not in r.raw_price_per:
            return None, f"raw_price_per_{str(per).lower()}_not_convertible"
        if per == "KILOGRAM":
            factor, size, unit = r.std_qty / 1000.0, 1000.0, "g"
        else:  # UNIT, only allowed for count products
            factor, size, unit = r.std_qty, 1.0, "piece"
    else:
        size, why = package_size(row, r.kind)
        if size is None:
            return None, why
        lo, hi = r.size_range
        if not lo <= size <= hi:
            return None, "size_out_of_range"
        unit = {"mass": "g", "loaf": "g", "volume": "ml", "count": "piece"}[r.kind]
        factor = r.std_qty / size if r.scale else 1.0

    if discounted:
        promo_pkg, regular_pkg = price, pwd          # regular stays NaN if Open Prices does not have it
    else:
        promo_pkg, regular_pkg = np.nan, price
    return {
        "regular_price_usd": regular_pkg * factor,
        "promo_price_usd": promo_pkg * factor,
        "price_is_discounted": discounted,
        "package_price_usd": price,
        "package_regular_price_usd": regular_pkg,
        "package_size": size,
        "package_unit": unit,
        "units_per_package": 1.0 / factor,
    }, ""


def build(raw: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict[str, int] = {"raw_rows": len(raw)}
    d = raw[(raw["currency"] == "USD") & (raw["location_osm_address_country_code"] == "US")].copy()
    stats["us_usd_rows"] = len(d)
    stats["flagged_duplicate_of"] = int(d["duplicate_of"].notna().sum())
    d = d[d["duplicate_of"].isna()]
    d["name"] = d["product_product_name"].fillna(d["product_name"]).fillna("").astype(str).str.strip()
    d["tags"] = d["product_categories_tags"].apply(tags_of)

    rows, reasons = [], {}
    unit_of = products.set_index("product_id")["unit"]
    for row in d.to_dict("records"):
        hits = match_product(row)
        if not hits:
            continue
        if len(hits) > 1:
            reasons["ambiguous_multiple_products"] = reasons.get("ambiguous_multiple_products", 0) + 1
            continue
        pid = hits[0]
        fields, why = convert(row, pid)
        if fields is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        rows.append({
            "product_id": pid, "unit": unit_of[pid], "date": str(row["date"]),
            **fields,
            "discount_type": row["discount_type"],
            "source_product_name": row["name"] or None,
            "product_code": row["product_code"], "category_tag": row["category_tag"],
            "store_name": row["location_osm_name"], "store_brand": row["location_osm_brand"],
            "store_city": row["location_osm_address_city"], "store_postcode": row["location_osm_address_postcode"],
            "store_osm_id": row["location_osm_id"], "store_osm_type": row["location_osm_type"],
            "proof_type": row["proof_type"], "source_app": row["source"],
            "source": "open_prices", "source_price_id": row["id"],
            "match_type": "loose_produce" if row["type"] == "CATEGORY" else "barcode_product",
        })
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    stats["matched_before_dedup"] = len(out)

    # Exact duplicates: same product, item, store, day and prices (several people uploading the same tag)
    key = ["product_id", "date", "store_osm_id", "product_code", "category_tag",
           "package_price_usd", "package_regular_price_usd"]
    out = out.sort_values("source_price_id")
    dup = out.duplicated(subset=key, keep="first")
    stats["exact_duplicates_removed"] = int(dup.sum())
    out = out[~dup]

    # Outliers: converted reference price (regular, else promo) more than 3x away from the product median
    ref = out["regular_price_usd"].fillna(out["promo_price_usd"])
    logp = np.log(ref)
    med = logp.groupby(out["product_id"]).transform("median")
    n = out.groupby("product_id")["product_id"].transform("size")
    bad = (n >= OUTLIER_MIN_N) & ((logp - med).abs() > np.log(OUTLIER_FACTOR))
    reasons["price_outlier"] = int(bad.sum())
    out, ref = out[~bad], ref[~bad]

    # Sanity bound for products with too few observations for the outlier check (catches price-entry errors)
    cat = out["product_id"].map(products.set_index("product_id")["price_usd"])
    n = out.groupby("product_id")["product_id"].transform("size")
    implausible = (n < OUTLIER_MIN_N) & ((np.log(ref) - np.log(cat)).abs() > np.log(CATALOG_FACTOR))
    reasons["implausible_vs_catalog_price"] = int(implausible.sum())
    out = out[~implausible]

    for c in ["regular_price_usd", "promo_price_usd"]:
        out[c] = out[c].round(4)
    out["units_per_package"] = out["units_per_package"].round(4)
    out = out.sort_values(["product_id", "date", "source_price_id"]).reset_index(drop=True)
    stats["rows_written"] = len(out)
    stats["rejections_after_match"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="local copy of prices.parquet (skips the download)")
    args = ap.parse_args()

    products = pd.read_csv(PRODUCTS_PATH)
    missing = set(products["product_id"]) - set(RULES)
    assert not missing, f"no matching rule for: {sorted(missing)}"

    info = {}
    path = args.input
    if path is None:
        info = download(RAW_PATH)
        path = RAW_PATH
    raw = pd.read_parquet(path, columns=COLUMNS)
    out, stats = build(raw, products)
    out.to_csv(OUT_PATH, index=False)

    coverage = out.groupby("product_id").agg(n=("date", "size"), stores=("store_osm_id", "nunique"),
                                             first=("date", "min"), last=("date", "max"))
    meta = {
        "source": "Open Prices (Open Food Facts), https://huggingface.co/datasets/openfoodfacts/open-prices",
        "licence": "ODbL 1.0",
        "retrieved_from": DATASET_URL,
        **info,
        "input_file": str(path),
        **stats,
        "products_with_data": int(coverage.shape[0]),
        "products_total": int(len(products)),
        "per_product": {pid: {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
                        for pid, r in coverage.to_dict("index").items()},
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "per_product"}, indent=2))
    print(coverage.to_string())
    print(f"\nwrote {OUT_PATH} ({len(out)} rows, {coverage.shape[0]}/{len(products)} products)")


if __name__ == "__main__":
    main()
