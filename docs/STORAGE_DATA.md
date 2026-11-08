# Storage data for the RL simulation (SYNTHETIC)

Files in `data_rl/`, built by `python -m src.build_storage_data` and validated by
`python -m src.build_storage_data --check` (exit 1 on failure). The original `data/*.csv` files are only read.

**Everything here is a synthetic assumption for the demo. None of it is verified food-safety guidance.**
Do not present these durations as safety advice.

## `storage_rules.csv` (36 products, `products.csv` row order)
| Column | Meaning |
|---|---|
| `default_storage` | `pantry`, `fridge` or `freezer`: where a purchase goes unless the agent chooses otherwise |
| `allowed_storage` | `;`-separated storage types the product may use |
| `freezing_supported` | True if the product can be stored in the freezer (for frozen products this means freezer only) |
| `volume_l_per_unit` | Approximate packed volume of one purchase unit in litres, including packaging |
| `shelf_life_{pantry,fridge,freezer}_days` | Planning shelf life per storage type; empty when the storage is not allowed |
| `shelf_life_kind` | `quality` or `safety_conservative` (below) |
| `source_status` | Always `synthetic_assumption` |

### Quality durations vs. safety limits
- `quality`: the duration describes when taste or texture declines (produce, bakery, pantry staples, hard cheese, butter). It is a planning
  estimate. The simulation treats the end of this period as spoilage/waste to keep the model simple, not as a verified safety cutoff.
- `safety_conservative`: raw meat, fish, eggs, milk, yogurt and tofu. The number is a deliberately conservative cutoff for the simulation:
  the product is discarded at the limit and the limit is never extended. It is an assumption, not a verified safety rule.
  For these products the freezer duration is a quality estimate for frozen storage.
- Shelf life at the default storage equals `shelf_life_days` in `products.csv` (checked), so existing code stays consistent.
- Not modelled: opened packages, cooked leftovers, thawing, re-freezing, and partial-unit freshness.

## `household_storage.csv` (30 households)
Capacities are in **litres of usable volume** (`capacity_unit = litres_usable`), the same unit as `volume_l_per_unit`.
Formula: `(base + per_person x n_people) x U(0.9, 1.1)`, rounded to 5 L, seed 42 (reproducible).

| Storage | base L | per person L |
|---|---:|---:|
| fridge | 90 | 55 |
| freezer | 25 | 17 |
| pantry | 110 | 55 |

Diet, budget and region do not affect capacity.

## Checks (`--check`)
- Coverage: all 36 products and all 30 households exactly once; product order matches `products.csv`.
- No missing values (shelf life may be empty only for storage types that are not allowed).
- Consistency: default storage is allowed; shelf life exists exactly for allowed storage; `freezing_supported` matches the freezer
  entries; freezer life is longer than fridge life; default shelf life equals `products.csv`; volumes positive and at most 5 L; one capacity unit.
- Capacity sanity: capacity is at least 2x the P95 weekly demand volume (by default storage) for every household (currently >= 4x).
- Reproducibility: both CSVs equal a fresh rebuild.

## Assumptions to review
Volumes and capacities are rough estimates (for example a loaf = 2.5 L, a 12-egg carton = 0.9 L). Capacity is generous, so storage
will rarely be the binding constraint with these values. Tighten `CAPACITY_MODEL` in `src/build_storage_data.py` if you want it to bind.
