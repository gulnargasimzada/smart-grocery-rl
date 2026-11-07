# Proposed INTERFACES.md changes: price data

**Status: proposal, not applied.** `INTERFACES.md` has not been changed. Please review; once the team agrees,
the owner adds the text below to `INTERFACES.md`. Details of the datasets: `docs/PRICE_DATA.md`.

## 1. Add to the table in section 2 (Data files)

| File | Owner | Columns |
|---|---|---|
| `price_history.csv` | price data | `product_id, unit, date, regular_price_usd, promo_price_usd, price_is_discounted, discount_type, package_price_usd, package_regular_price_usd, package_size, package_unit, units_per_package, source_product_name, product_code, category_tag, store_name, store_brand, store_city, store_postcode, store_osm_id, store_osm_type, proof_type, source_app, source, source_price_id, match_type` (**real**, Open Prices) |
| `price_weekly_synthetic.csv` | price data, generated | `product_id, week, week_of_year, unit, regular_price_usd, promo_price_usd, is_promo, promo_depth, price_usd, baseline_price_usd, baseline_source, n_real_obs, is_synthetic, data_source, seed` (**synthetic**) |

Suggested text below the table:

> Regenerate the price files with `python -m src.fetch_open_prices` and `python -m src.generate_synthetic_prices`; validate them with `python -m src.check_prices`.
> Real (`price_history.csv`) and synthetic (`price_weekly_synthetic.csv`) prices are never mixed in one file.

## 2. Add to section 1 (Conventions)

> **Prices** are USD per purchase unit. `products.csv` `price_usd` is a static reference price.
> Week-dependent prices come from `price_weekly_synthetic.csv` (`price_usd` = the price actually paid that week, `regular_price_usd` = the price without promotion).

## 3. Add to section 11 (File ownership)

> Price data: `src/fetch_open_prices.py`, `src/generate_synthetic_prices.py`, `src/check_prices.py`, `data/price_history.csv`, `data/price_history_meta.json`, `data/price_weekly_synthetic.csv`, `docs/PRICE_DATA.md`.

## 4. Open questions for the team (no change made)

1. **Should the environment use weekly prices?** If yes, `GroceryEnv` would read `price_usd[product, week]` instead of the static `products.csv` price.
   `info["spend"]` and `info["waste_cost"]` would then use the week's price, and `obs` could add `prices` (and optionally `is_promo`) so the agent can learn to buy on promotion.
   This is a Person 2 decision.
2. **Should `apply_constraints` take a price vector?** Today it computes cost from `products.csv`. A `prices: np.ndarray | None = None` argument would keep the current behaviour by default.
   This is a Person 3 decision.
3. **Price level vs. budgets:** the real (Bay Area) medians are often 1.5–4x the placeholder `price_usd`, and `households.csv` budgets were calibrated on the placeholders.
   The options are:
   - (a) keep the placeholders and use only the synthetic *relative* movements (`price_usd / baseline_price_usd`);
   - (b) recalibrate the budgets with the new prices;
   - (c) update `products.csv` `price_usd` after the business teammate verifies the real prices.

   This is a Person 1 decision together with the team.
