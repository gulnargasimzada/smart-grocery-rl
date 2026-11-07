# Price data

Two price datasets for the RL environment and the optimization step. They are kept in **separate files**:
real observations are never mixed with synthetic values.

| File | Kind | Rows | Made by |
|---|---|---:|---|
| `data/price_history.csv` | **Real** U.S. shelf and receipt prices from Open Prices | 1,134 | `python -m src.fetch_open_prices` |
| `data/price_history_meta.json` | Snapshot date, counts, rejection reasons, per-product coverage | – | same script |
| `data/price_weekly_synthetic.csv` | **Synthetic** weekly prices, 36 products x 104 weeks | 3,744 | `python -m src.generate_synthetic_prices` |

All prices are in **USD per purchase unit** (the `unit` column of `data/products.csv`, e.g. milk = 1 L,
eggs = 12 count), so they can be multiplied directly with order quantities.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.fetch_open_prices            # downloads the Open Prices snapshot (~80 MB) to data/open_prices_raw/ (git-ignored)
python -m src.generate_synthetic_prices    # needs data/price_history.csv and data/demand_weekly.csv
python -m src.check_prices                 # validation; exits with status 1 if a check fails
```

`python -m src.fetch_open_prices --input path/to/prices.parquet` reuses a local copy.
Open Prices is updated daily, so a fresh download can add new observations. The snapshot used for the
committed file is recorded in `data/price_history_meta.json` (`dataset_last_modified`).
The synthetic file is fully reproducible (fixed seed 42): the same `price_history.csv` always gives the same file.

## 1. Real prices: `data/price_history.csv`

Source: [Open Prices](https://prices.openfoodfacts.org/) by Open Food Facts,
[Hugging Face dataset](https://huggingface.co/datasets/openfoodfacts/open-prices), licence
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1.0/). Crowdsourced prices from price tags and receipts.
**Credit Open Prices when using this file; derived databases must stay under ODbL.**

### How records are selected

A record is kept only if all of these rules pass (rules for every product are in `RULES` in `src/fetch_open_prices.py`):

1. **U.S. and USD:** store country code `US` and currency `USD`. Records that Open Prices flags as duplicates (`duplicate_of`) are dropped.
2. **Product name:** the name matches the product's include pattern and none of its exclude patterns.
   For example, milk excludes chocolate, plant-based, condensed, powdered and buttermilk; yogurt must be unflavoured; salmon must be an unseasoned fillet.
3. **Category:** the Open Food Facts category tags contain one required tag (for example `en:milks`, `en:canned-tunas`) and no excluded tag (meals, snacks, frozen, plant-based, and so on).
   Loose produce without a barcode must carry one of the listed raw category tags (for example `en:bananas`, `en:roma-tomatoes`).
4. **Package size:** the size must be known and in the product's accepted range.
   The numeric size and the label text must agree within 5 %, so a conflict such as "1 lb 8 oz" stored as 453 g is rejected.
   Count products (eggs, tortillas, avocados, bell peppers) need an explicit count on the label.
   Liquids need a volume, because converting grams to millilitres would need an assumed density.
5. **Not a bundle:** twin packs and variety packs are rejected. Packaged products from warehouse clubs (Costco, Sam's Club, BJ's) are rejected too, because their bundle prices do not match the single-unit size in Open Food Facts.
6. **Unambiguous:** the record must match exactly one of our 36 products.
7. **Not an outlier:** for products with at least 5 records, the price must be within 3x of the product median.
   For products with fewer records, it must be within 5x of the placeholder price in `products.csv`, a wide guard against entry errors.

### Conversion to purchase units

- **Packaged products:** `price per purchase unit = package price x (purchase-unit size / package size)`.
  For example, a 1-gallon (3,785 ml) jug of milk at $5.99 becomes $1.58 per 1 L.
  Bread is priced per loaf (400–1,000 g) without scaling.
- **Loose produce priced per kg:** `price per kg x kg per purchase unit`.
  Produce priced per piece is used only for count products (avocados x 4, bell peppers x 3).
  Per-piece prices of produce sold by weight are rejected: the piece weight is unknown and is not guessed.

Exact duplicates are removed, keeping the lowest Open Prices id. A duplicate is the same product, store, day, item and prices, typically several people uploading the same price tag.

Nothing is imputed. If a price is discounted but Open Prices has no regular price, `regular_price_usd` stays empty (11 rows).

### Columns

| Column | Meaning |
|---|---|
| `product_id`, `unit` | our product and its purchase unit (`products.csv`) |
| `date` | date the price was observed (original Open Prices date) |
| `regular_price_usd` | regular (non-promotional) price per purchase unit; empty if unknown |
| `promo_price_usd` | promotional price per purchase unit; empty if the price was not discounted |
| `price_is_discounted`, `discount_type` | original Open Prices discount flag and type |
| `package_price_usd`, `package_regular_price_usd` | original prices of the package as recorded |
| `package_size`, `package_unit`, `units_per_package` | package size (g, ml or piece; 1000 g for per-kg produce) and how many purchase units it holds |
| `source_product_name`, `product_code`, `category_tag` | name, barcode (packaged) or raw category (loose produce) |
| `store_name`, `store_brand`, `store_city`, `store_postcode`, `store_osm_id`, `store_osm_type` | store information (OpenStreetMap) |
| `proof_type` | `PRICE_TAG` or `RECEIPT` |
| `source_app` | upload tool recorded by Open Prices |
| `source`, `source_price_id` | `open_prices` and the Open Prices price id (traceable to the original record) |
| `match_type` | `barcode_product` or `loose_produce` |

### Coverage

The table covers 1,134 observations from 97 stores in 40 cities, dated 2024-07-13 to 2026-10-31. Of these, 1,000 are barcode products and 134 are loose produce; 1,006 come from price tags and 128 from receipts.

| product_id | unit | real obs | stores | dates | median real regular price | products.csv price | synthetic baseline (source) |
|---|---|---:|---:|---|---:|---:|---|
| milk | 1 L | 40 | 21 | 2024-10-06 – 2026-10-06 | $2.11 | $1.40 | $2.11 (Open Prices) |
| eggs | 12 count | 54 | 16 | 2025-05-22 – 2026-10-25 | $8.99 | $3.50 | $8.99 (Open Prices) |
| chicken_breast | 500 g | 0 | 0 | – | – | $4.50 | $4.50 (products.csv) |
| rice | 1 kg | 38 | 9 | 2025-09-22 – 2026-09-15 | $9.31 | $2.50 | $9.31 (Open Prices) |
| bread | 1 loaf | 99 | 22 | 2024-07-13 – 2026-10-31 | $6.69 | $3.00 | $6.69 (Open Prices) |
| bananas | 1 kg | 61 | 34 | 2024-07-19 – 2026-10-30 | $1.94 | $1.50 | $1.94 (Open Prices) |
| apples | 1 kg | 14 | 7 | 2025-09-28 – 2026-02-22 | $4.39 | $3.00 | $4.39 (Open Prices) |
| tomatoes | 500 g | 9 | 7 | 2025-10-04 – 2026-10-06 | $1.46 | $2.50 | $1.46 (Open Prices) |
| spinach | 200 g | 2 | 1 | 2025-11-07 – 2025-11-07 | $5.30 | $2.50 | $2.50 (products.csv) |
| lentils | 500 g | 5 | 2 | 2025-09-22 – 2026-09-12 | $2.75 | $2.00 | $2.00 (products.csv) |
| yogurt | 500 g | 20 | 8 | 2025-05-19 – 2026-10-25 | $4.33 | $3.00 | $4.33 (Open Prices) |
| oats | 500 g | 40 | 8 | 2024-12-05 – 2026-10-31 | $4.65 | $2.50 | $4.65 (Open Prices) |
| tofu | 400 g | 21 | 10 | 2025-10-11 – 2026-09-25 | $3.44 | $2.50 | $3.44 (Open Prices) |
| carrots | 1 kg | 20 | 10 | 2025-08-22 – 2026-09-13 | $2.19 | $1.80 | $2.19 (Open Prices) |
| potatoes | 2 kg | 13 | 8 | 2025-09-26 – 2026-03-08 | $3.92 | $3.50 | $3.92 (Open Prices) |
| pasta | 500 g | 94 | 14 | 2025-09-22 – 2026-08-22 | $3.08 | $1.60 | $3.08 (Open Prices) |
| cheddar_cheese | 200 g | 40 | 13 | 2024-07-13 – 2026-10-17 | $4.86 | $3.20 | $4.86 (Open Prices) |
| peanut_butter | 340 g | 163 | 32 | 2025-09-12 – 2026-10-18 | $3.30 | $3.00 | $3.30 (Open Prices) |
| salmon_fillet | 400 g | 0 | 0 | – | – | $9.00 | $9.00 (products.csv) |
| ground_beef | 500 g | 9 | 5 | 2025-04-27 – 2026-01-23 | $8.82 | $5.50 | $8.82 (Open Prices) |
| canned_tuna | 1 can 142 g | 9 | 3 | 2026-07-19 – 2026-09-15 | $5.73 | $1.30 | $5.73 (Open Prices) |
| chickpeas_canned | 1 can 425 g | 9 | 2 | 2025-09-22 – 2026-08-20 | $1.93 | $1.10 | $1.10 (products.csv) |
| black_beans_canned | 1 can 425 g | 5 | 1 | 2025-09-22 – 2026-08-11 | $2.89 | $1.10 | $1.10 (products.csv) |
| tortillas | 10 count | 0 | 0 | – | – | $2.80 | $2.80 (products.csv) |
| butter | 225 g | 38 | 13 | 2025-02-22 – 2026-10-17 | $6.96 | $3.80 | $6.96 (Open Prices) |
| plant_milk | 1 L | 69 | 19 | 2025-08-11 – 2026-10-25 | $3.90 | $2.20 | $3.90 (Open Prices) |
| oranges | 1 kg | 4 | 2 | 2025-09-28 – 2026-09-27 | $7.70 | $2.50 | $2.50 (products.csv) |
| avocados | 4 count | 31 | 16 | 2025-09-17 – 2026-10-17 | $6.00 | $4.00 | $6.00 (Open Prices) |
| broccoli | 500 g | 2 | 2 | 2025-11-29 – 2026-01-22 | $1.64 | $2.20 | $2.20 (products.csv) |
| onions | 1 kg | 11 | 6 | 2025-12-23 – 2026-03-14 | $1.96 | $1.60 | $1.96 (Open Prices) |
| bell_peppers | 3 count | 1 | 1 | 2026-09-20 – 2026-09-20 | $2.97 | $3.00 | $3.00 (products.csv) |
| frozen_vegetables | 1 kg | 6 | 3 | 2025-12-13 – 2026-01-30 | $6.70 | $3.00 | $6.70 (Open Prices) |
| olive_oil | 500 ml | 47 | 14 | 2024-10-18 – 2026-10-11 | $14.99 | $7.00 | $14.99 (Open Prices) |
| orange_juice | 1 L | 2 | 1 | 2025-11-12 – 2025-11-12 | $3.88 | $3.00 | $3.00 (products.csv) |
| breakfast_cereal | 400 g | 149 | 15 | 2024-07-13 – 2026-10-18 | $7.05 | $3.80 | $7.05 (Open Prices) |
| frozen_pizza | 1 pizza 400 g | 9 | 5 | 2025-10-25 – 2026-10-31 | $10.45 | $5.00 | $10.45 (Open Prices) |

Real observations exist for 33 of 36 products. **No reliable match** was found for `chicken_breast`, `salmon_fillet` and `tortillas`:
- chicken breast is mostly sold by weight without a barcode size;
- salmon in the data is seasoned or ready-to-cook;
- tortilla packs rarely state a count.

## 2. Synthetic prices: `data/price_weekly_synthetic.csv`

One row per product and week: weeks 0–103, the same weeks as `demand_weekly.csv`, with week 0 in early January.
Products appear in `products.csv` row order. **Every row is synthetic** (`is_synthetic = True`, `data_source = synthetic`).

### Model (`src/generate_synthetic_prices.py`, seed 42)

- **Baseline:** the median real regular price, if the product has at least 5 real observations from at least 3 stores (25 products).
  Otherwise the baseline is `price_usd` from `products.csv` (11 products). The baseline is the price level at the middle of the two-year horizon.
- **Inflation:** a product-specific annual rate drawn around a category rate, from 2.0 % (produce) to 3.5 % (protein).
- **Seasonality:** a 52-week sine wave with a random phase per product, ±8 % for fruit and vegetables and ±1–3 % otherwise.
- **Cost shocks:** slow AR(1) noise.
- **Sticky shelf prices:** the regular price changes only on occasional price-change weeks or after more than 8 % drift.
  Produce changes about every 3 weeks; packaged goods change every 12–20 weeks.
- **Promotions:** temporary discounts of 1 or 2 weeks, starting with a category-specific weekly probability of 5–10 %, doubled in holiday weeks 46 and 51.
  Depth is 10–35 %, and each promotion is followed by at least one full-price week. Promotions cover 9.3 % of product-weeks.

### Columns

| Column | Meaning |
|---|---|
| `product_id`, `week`, `week_of_year`, `unit` | product, week index (0–103), `week % 52`, purchase unit |
| `regular_price_usd` | regular shelf price per purchase unit |
| `promo_price_usd`, `promo_depth` | promotional price and discount fraction (empty when there is no promotion) |
| `is_promo` | promotion this week |
| `price_usd` | price paid this week: the promo price in promotion weeks, else the regular price |
| `baseline_price_usd`, `baseline_source`, `n_real_obs` | calibration level, `open_prices_median` or `products_csv`, and the number of real observations |
| `is_synthetic`, `data_source`, `seed` | always `True`, `synthetic`, `42` |

## Validation (`python -m src.check_prices`)

**Real data:**
- schema; product ids and units match `products.csv`;
- no missing required values; no duplicate ids or observations;
- valid dates; positive prices and sizes;
- a promo price exists exactly when the price is discounted, and it is below the regular price;
- prices within 0.1x–10x of `products.csv`.

**Synthetic data:**
- 36 x 104 rows; one row per product and week; weeks equal those of `demand_weekly.csv`;
- units match; no missing values outside the promo columns; every row labelled synthetic;
- promo price and depth present exactly in promo weeks, below the regular price, with depth 5–50 %;
- `price_usd` is consistent with the promo and regular prices; promotions last 1–2 weeks;
- regular prices within 0.7x–1.4x of the baseline.

**Separation:** the real file has only `open_prices` rows and the synthetic file only `synthetic` rows.

## Limitations

- **Geographic and store bias:** about 75 % of the real observations come from the San Francisco Bay Area (Mountain View, Sunnyvale, Los Altos, Palo Alto).
  Safeway and Whole Foods dominate. Real prices are therefore higher than a U.S. national average and over-represent organic and premium items (for example eggs at $8.99 per dozen).
- **Small samples:** several products have fewer than 10 observations, and the time span is short (mostly 2025–2026).
  The real data is **not** a weekly time series and cannot be used directly as one; it is used for price levels only.
- **Size effects:** prices are scaled linearly to our purchase unit. Bulk packages (for example a gallon of milk) are cheaper per unit than our standard unit would be, and small packages are dearer.
- **Variant mix:** a product groups several real variants (for example `plant_milk` combines unflavoured soy, almond and oat milk; `rice` combines white, brown, jasmine and basmati), so the spread within a product is wide.
- **Placeholders:** `products.csv` prices are placeholders (see `INTERFACES.md`). They are used as the baseline for 11 products and as a wide sanity bound.
  The real medians are often 1.5–4x higher. Household budgets in `households.csv` were calibrated on the placeholder prices, so switching the environment to these prices changes the economics. That needs a team decision (see `docs/INTERFACES_PRICES_PROPOSAL.md`).
- **Synthetic is not real:** inflation rates, seasonality, promotion frequency and depth are plausible assumptions, not estimates from data.
  Results that use the synthetic file must be reported as simulated.
- **Upstream:** Open Prices is crowdsourced and updated daily. Labels and categories can be wrong, and a new download may change the matched set.

## 3. Regional synthetic prices: `data/price_weekly_regional_synthetic.csv`

**All regional differences are synthetic assumptions, not verified local prices.** For the demo only.

`python -m src.generate_regional_prices` reads `data/price_weekly_synthetic.csv` (unchanged) and writes
5 regions x 36 products x 104 weeks = 18,720 rows. The run is reproducible (seed 42).

Each regional price is the base price times a fixed multiplier per region and product:
`multiplier = region index x exp(category adjustment + product noise)`, with the noise drawn from Normal(0, 0.025) once per (region, product).
Weekly fluctuations and promotion weeks and depths are inherited from the base file, so they are the same in every region; only the price level changes.
The base file counts as index 1.00. Its baselines lean toward Bay Area prices (see Limitations), so the indices are relative, not national.

| region_id | Region | Index | Category adjustments (log points) | Multiplier range |
|---|---|---:|---|---|
| `durham_nc` | Durham, NC | 0.95 | protein −0.02, dairy +0.01, fats_oils +0.02 | 0.91–1.00 |
| `san_francisco_ca` | San Francisco, CA | 1.20 | fruit/vegetables −0.04, protein +0.03, beverages +0.02 | 1.13–1.26 |
| `atlanta_ga` | Atlanta, GA | 0.97 | protein −0.02, fruit +0.01, beverages −0.02 | 0.92–1.00 |
| `new_york_ny` | New York, NY | 1.16 | fruit/vegetables +0.03, protein +0.03, processed +0.02 | 1.10–1.29 |
| `chicago_il` | Chicago, IL | 1.03 | dairy −0.03, grains −0.01, protein −0.01 | 0.98–1.08 |

The indices follow the usual ordering of cost-of-living grocery indices: SF and NYC are highest and the Southeast lowest. The category adjustments are illustrative.

Columns:

| Column | Meaning |
|---|---|
| `region_id`, `region_name` | region |
| `week`, `product_id`, `unit` | as in the base file |
| `regular_price_usd`, `promo_price_usd`, `is_promo`, `price_usd` | as in the base file, scaled by the multiplier (promo price empty outside promotions) |
| `regional_multiplier` | the multiplier applied |
| `source` | always `synthetic_regional` |

`python -m src.check_prices` also validates this file:
- row count; no duplicate (region, week, product) rows;
- units match `products.csv`; no missing values;
- positive prices; promo price below regular;
- `price_usd` is consistent with the promo and regular prices;
- promotion weeks equal the base file's; regular price = base x multiplier.

### Selecting prices by location (RL and Optimization)

No module uses this file yet. Suggested use, without changing the interfaces:

```python
regional = pd.read_csv("data/price_weekly_regional_synthetic.csv")
products = pd.read_csv("data/products.csv")
r = regional[regional["region_id"] == region_id]                       # e.g. "durham_nc"
# week x product matrix of the price actually paid, columns in products.csv order (INTERFACES convention)
prices = r.pivot(index="week", columns="product_id", values="price_usd")[products["product_id"]].to_numpy()
week_prices = prices[week]                                             # shape (n_products,)
```

- **RL (`GroceryEnv`):** take `region_id` as a setting, use `week_prices` for `spend` and `waste_cost`, and optionally add `prices` and `is_promo` to `obs`.
- **Optimization (`apply_constraints`):** compute order cost as `order @ week_prices` instead of using `products.csv` `price_usd`.
- **Weeks:** use `price_usd` for what is paid and `regular_price_usd` for "regular" comparisons. Use the same `week` index as the demand data, `week % 104` if an episode runs longer.
- **Households:** budgets in `households.csv` are not regional. Comparing regions changes only prices, not budgets or demand.
