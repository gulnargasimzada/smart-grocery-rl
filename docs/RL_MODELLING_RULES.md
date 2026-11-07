# RL modelling rules (Step 2 onward)

Rules the RL environment, baselines and evaluation must follow. They do not change any existing data.

## 1. Purchases are in whole packages; consumption is fractional
- Quantities are in purchase units (`unit` in `products.csv`, e.g. 1 L, 12 count, 1 can, 500 g).
- **Orders must respect indivisible package quantities where applicable.** Packaged and counted items
  (eggs, cans, bread, tortillas, frozen pizza, ...) can only be bought in whole units. The order is rounded
  to the package step *before* the budget is checked, so the cost is the cost of what is actually bought.
  Loose items sold by weight may use a smaller step (for example 0.25 kg). The step per product is
  defined in the environment code (not in `products.csv`) when the environment is built.
- **Consumption may be fractional.** Weekly demand in `demand_weekly.csv` is continuous (for example 0.51 kg).
  Unused fractions stay in inventory.
- **Inventory carries over between weeks** with ageing by shelf life (`shelf_life_days`). Expired units are waste.
  Demand not served from stock is a stockout.

## 2. Budgets, regions and prices
- Budget scenario per episode: `normal` (P90, default), `challenging` (P75), `tight` (original budget),
  from `data/household_region_budgets.csv`. All budgets are synthetic.
- Prices come from `data/price_weekly_regional_synthetic.csv` for the household's region and the current week.
- Report results per scenario and per region. Do not mix scenarios in one headline number.

## 3. Evaluation must not reward memorising synthetic demand
- Budgets were calibrated on weeks 0-77. **Training uses weeks 0-77 only; evaluation uses held-out weeks 78-103.**
  Weeks 78-103 must never be used for training, tuning or model selection.
- Use **independent random seeds** for evaluation, different from the training seeds and not reused for
  hyperparameter selection. Report mean and std over seeds (see `INTERFACES.md` section 9).
- Compare RL with the baselines (`fixed`, `average_rule`, `forecast_rule`) on the same weeks, seeds,
  region and budget scenario.
- Demand is synthetic: results are simulated, not real savings.
