# Findings: GroceryEnv v0 diagnostics (hand-written; numbers come from `report.md`)

Scope: hh_student, Durham, normal budget ($75/week, synthetic), default storage, 13-week episodes. Training windows (start weeks 4, 17, 30, 43, 51)
and one validation window (start week 64) only. Recorded-demand replay, so results are deterministic and paired, not independent samples.
Reproduce with `python -m src.diagnose_env`.

Disclosure: when GroceryEnv v0 was first checked, a rough sanity run also printed returns for the windows starting in weeks 78 and 91. Nothing was
tuned from it, and the diagnostic script now refuses any episode that ends after week 77.

## 1. The 65-80 stockout units
They came from the plain "order the trailing mean demand every week" rule (`naive_ref`, 76.8 units per episode, fill rate 59% by value).
- **Main cause: package rounding (86% of the stockouts disappear when package sizes are removed).** Orders are rounded to the nearest package, so an item
  whose weekly need is under half a package is never bought. 16 of 36 products had a fill rate under 10%
  (breakfast cereal, rice, salmon, olive oil, eggs, ground beef, oats, avocados, bread, tofu, pasta).
  This is a policy problem (the policy must think in packages), not an environment bug.
- Shelf life: about 5%. Stock cap: under 2%. Budget and storage/thawing: 0%.
- What remains with everything relaxed (5.3 units) is demand above the trailing mean with no safety stock.

## 2. Package-aware baselines (`src/policies.py`)
| Policy (train / validation) | Reward | Cash spend | Waste cost | Stockout units | Fill rate (value) | Protein coverage |
|---|---|---|---|---|---|---|
| buy_nothing | -28.1 / -29.8 | 0 | 0.2 | 213 / 228 | 3% | 0.04 / 0.05 |
| fixed (standing order) | -15.4 / -16.1 | 699 / 733 | 58 / 47 | 54 / 61 | 74% | 1.04 / 1.12 |
| average_rule (1 week cover) | -14.8 / -15.3 | 906 / 926 | 146 / 134 | 23 / 23 | 88% | 1.28 / 1.44 |

- `average_rule` has the best reward on both splits, but it spends about $70/week of the $75 budget and wastes about 16% of what it buys (simulated).
  More cover (1.5 and 2 weeks) is worse on both splits: more waste, no fewer stockouts. Fixed spends less but leaves 26% of demand value unserved.
- Protein coverage above 1.0 means the recorded demand itself exceeds the 340 g target; it is not a sign of over-buying.
- For `average_rule` the stockouts come mainly from the **budget** (66% disappear with an unlimited budget) and **shelf life** (51%), because it reorders
  many whole packages at once and perishables expire. For `fixed`: package size (71%), shelf life (18%), budget (8%).
  Storage capacity and thawing never bind for any of the policies with the default capacities.

## 3. Environment adjustments considered
- **Kept:** budget-reduction priority = most overstocked first (stock after the order, in weeks of demand). I also tried "stock before the order" because the
  first rule repeatedly drops slow items with large packages (olive oil, butter) when the budget binds. It lowered stockout days but gave a **worse reward on
  both training and validation windows** (average_rule: -15.4 vs -14.8 train, -16.4 vs -15.3 validation), so it was not adopted.
- No change to the reward, storage rules or constants was made.

## 4. Open points for review
- The budget binds for a sensible baseline. Whether $75/week (P90 of normal-week recorded-demand cost) is the right difficulty once packages are whole is a
  decision for later; the budget files stay unchanged.
- Waste of about $146 per episode for `average_rule` is simulated disposal under quality/conservative shelf lives, not a food-safety statement.
