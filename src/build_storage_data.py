"""Build synthetic storage data for the RL simulation into data_rl/.

Outputs
-------
data_rl/storage_rules.csv     one row per product (36): storage types, shelf life by storage, volume
data_rl/household_storage.csv one row per household (30): fridge / freezer / pantry capacity

ALL VALUES ARE SYNTHETIC ASSUMPTIONS for the demo. They are NOT verified food-safety guidance.
See docs/STORAGE_DATA.md. Original data/*.csv files are only read, never written.

Run:    python -m src.build_storage_data          # (re)build both files
Check:  python -m src.build_storage_data --check  # validation, exit 1 on failure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "data_rl"
SEED = 42
STORAGES = ["pantry", "fridge", "freezer"]

# product_id: (default, volume_l_per_unit, pantry_days, fridge_days, freezer_days, kind)
# Days are None when that storage is not allowed. Freezer days only where freezing is supported.
# kind: "safety_conservative" = raw/perishable animal products where the simulation must discard
#       at the limit and never extend it (conservative ASSUMPTION, not a verified rule);
#       "quality" = durations describe quality (taste, texture), food is not discarded for safety reasons.
# The shelf life at the default storage equals shelf_life_days in products.csv (kept consistent).
RULES: dict[str, tuple] = {
    "milk":               ("fridge",  1.10, None, 7,    None, "safety_conservative"),
    "eggs":               ("fridge",  0.90, None, 21,   None, "safety_conservative"),
    "chicken_breast":     ("fridge",  0.70, None, 3,    180,  "safety_conservative"),
    "rice":               ("pantry",  1.00, 365,  None, None, "quality"),
    "bread":              ("pantry",  2.50, 5,    10,   90,   "quality"),
    "bananas":            ("pantry",  2.50, 5,    8,    None, "quality"),
    "apples":             ("fridge",  2.00, 7,    21,   None, "quality"),
    "tomatoes":           ("pantry",  0.90, 6,    10,   None, "quality"),
    "spinach":            ("fridge",  1.20, None, 5,    None, "quality"),
    "lentils":            ("pantry",  0.60, 365,  None, None, "quality"),
    "yogurt":             ("fridge",  0.55, None, 14,   None, "safety_conservative"),
    "oats":               ("pantry",  1.10, 365,  None, None, "quality"),
    "tofu":               ("fridge",  0.45, None, 10,   90,   "safety_conservative"),
    "carrots":            ("fridge",  1.80, None, 21,   None, "quality"),
    "potatoes":           ("pantry",  2.60, 30,   None, None, "quality"),
    "pasta":              ("pantry",  1.40, 365,  None, None, "quality"),
    "cheddar_cheese":     ("fridge",  0.20, None, 30,   180,  "quality"),
    "peanut_butter":      ("pantry",  0.30, 180,  None, None, "quality"),
    "salmon_fillet":      ("fridge",  0.45, None, 3,    120,  "safety_conservative"),
    "ground_beef":        ("fridge",  0.60, None, 3,    120,  "safety_conservative"),
    "canned_tuna":        ("pantry",  0.15, 730,  None, None, "quality"),
    "chickpeas_canned":   ("pantry",  0.45, 730,  None, None, "quality"),
    "black_beans_canned": ("pantry",  0.45, 730,  None, None, "quality"),
    "tortillas":          ("pantry",  0.90, 14,   30,   180,  "quality"),
    "butter":             ("fridge",  0.25, None, 60,   180,  "quality"),
    "plant_milk":         ("fridge",  1.10, None, 14,   None, "quality"),
    "oranges":            ("fridge",  1.80, 7,    14,   None, "quality"),
    "avocados":           ("pantry",  0.90, 4,    7,    None, "quality"),
    "broccoli":           ("fridge",  1.50, None, 7,    None, "quality"),
    "onions":             ("pantry",  1.70, 30,   None, None, "quality"),
    "bell_peppers":       ("fridge",  1.30, None, 7,    None, "quality"),
    "frozen_vegetables":  ("freezer", 2.00, None, None, 180,  "quality"),
    "olive_oil":          ("pantry",  0.60, 540,  None, None, "quality"),
    "orange_juice":       ("fridge",  1.10, None, 10,   None, "quality"),
    "breakfast_cereal":   ("pantry",  3.00, 180,  None, None, "quality"),
    "frozen_pizza":       ("freezer", 1.80, None, None, 180,  "quality"),
}
# Items that are sold frozen: the freezer is their only storage, "freezing supported" means already frozen.
SOLD_FROZEN = {"frozen_vegetables", "frozen_pizza"}

RULE_COLUMNS = [
    "product_id", "default_storage", "allowed_storage", "freezing_supported",
    "volume_l_per_unit", "shelf_life_pantry_days", "shelf_life_fridge_days", "shelf_life_freezer_days",
    "shelf_life_kind", "source_status", "notes",
]
HH_COLUMNS = [
    "household_id", "n_people", "fridge_l", "freezer_l", "pantry_l", "capacity_unit", "source_status",
]
# Usable volume in litres = base + per-person x n_people, times a seeded +-10% jitter, rounded to 5 L.
CAPACITY_MODEL = {"fridge": (90, 55), "freezer": (25, 17), "pantry": (110, 55)}


def build_rules() -> pd.DataFrame:
    products = pd.read_csv(DATA / "products.csv")
    rows = []
    for pid in products["product_id"]:
        default, vol, p, f, z, kind = RULES[pid]
        allowed = [s for s, d in zip(STORAGES, (p, f, z)) if d is not None]
        freezing = z is not None
        note = {
            "safety_conservative": "Limit is a conservative simulation cutoff (assumption, not verified food-safety guidance). Never extend.",
            "quality": "Durations are quality/planning assumptions, not safety limits.",
        }[kind]
        if pid in SOLD_FROZEN:
            note = "Sold frozen; freezer only. " + note
        rows.append({
            "product_id": pid, "default_storage": default, "allowed_storage": ";".join(allowed),
            "freezing_supported": freezing, "volume_l_per_unit": vol,
            "shelf_life_pantry_days": p, "shelf_life_fridge_days": f, "shelf_life_freezer_days": z,
            "shelf_life_kind": kind, "source_status": "synthetic_assumption", "notes": note,
        })
    df = pd.DataFrame(rows, columns=RULE_COLUMNS)
    for c in ["shelf_life_pantry_days", "shelf_life_fridge_days", "shelf_life_freezer_days"]:
        df[c] = df[c].astype("Int64")
    return df


def build_households() -> pd.DataFrame:
    hh = pd.read_csv(DATA / "households.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for hid, n in zip(hh["household_id"], hh["n_people"]):
        row = {"household_id": hid, "n_people": int(n)}
        for kind, (base, per) in CAPACITY_MODEL.items():
            jitter = rng.uniform(0.9, 1.1)
            row[f"{kind}_l"] = int(round((base + per * n) * jitter / 5) * 5)
        row["capacity_unit"] = "litres_usable"
        row["source_status"] = "synthetic_assumption"
        rows.append(row)
    return pd.DataFrame(rows, columns=HH_COLUMNS)


def check() -> int:
    fails: list[str] = []

    def ok(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    products = pd.read_csv(DATA / "products.csv")
    hh = pd.read_csv(DATA / "households.csv")
    rules = pd.read_csv(OUT / "storage_rules.csv")
    cap = pd.read_csv(OUT / "household_storage.csv")

    # Coverage
    ok(list(rules.columns) == RULE_COLUMNS, "storage_rules columns match schema")
    ok(list(cap.columns) == HH_COLUMNS, "household_storage columns match schema")
    ok(len(rules) == 36 and set(rules["product_id"]) == set(products["product_id"]) and not rules["product_id"].duplicated().any(),
       "storage_rules covers all 36 products exactly once")
    ok(len(cap) == 30 and set(cap["household_id"]) == set(hh["household_id"]) and not cap["household_id"].duplicated().any(),
       "household_storage covers all 30 households exactly once")
    ok(list(rules["product_id"]) == list(products["product_id"]), "storage_rules keeps products.csv row order")

    # Missing values (empty shelf-life cells are only allowed for storages that are not allowed)
    required = [c for c in RULE_COLUMNS if not c.startswith("shelf_life_") or c == "shelf_life_kind"]
    ok(rules[required].isna().sum().sum() == 0, "no missing values in required rule columns")
    ok(cap.isna().sum().sum() == 0, "no missing values in household_storage")

    # Consistency
    allowed = rules["allowed_storage"].str.split(";")
    ok(all(set(a) <= set(STORAGES) and len(a) >= 1 for a in allowed), "allowed storage types are valid and non-empty")
    ok(all(d in a for d, a in zip(rules["default_storage"], allowed)), "default storage is an allowed storage")
    days_ok = True
    for s in STORAGES:
        col = rules[f"shelf_life_{s}_days"]
        has = col.notna()
        days_ok &= bool((has == allowed.apply(lambda a: s in a)).all()) and bool((col.dropna() > 0).all())
    ok(days_ok, "shelf life is present (>0) exactly for allowed storage types")
    ok((rules["freezing_supported"] == rules["shelf_life_freezer_days"].notna()).all(),
       "freezing_supported matches freezer shelf life")
    ok(rules["freezing_supported"].eq(allowed.apply(lambda a: "freezer" in a)).all(),
       "freezing_supported matches freezer in allowed_storage")
    default_days = [rules.loc[i, f"shelf_life_{d}_days"] for i, d in enumerate(rules["default_storage"])]
    ok(list(default_days) == list(products["shelf_life_days"]), "shelf life at default storage equals products.csv shelf_life_days")
    ok((rules["volume_l_per_unit"] > 0).all() and rules["volume_l_per_unit"].max() <= 5, "volumes are positive and plausible (<= 5 L per unit)")
    ok(rules["shelf_life_kind"].isin(["quality", "safety_conservative"]).all(), "shelf_life_kind is quality or safety_conservative")
    ok((rules["source_status"] == "synthetic_assumption").all() and (cap["source_status"] == "synthetic_assumption").all(),
       "everything is labelled synthetic_assumption")
    both = rules.dropna(subset=["shelf_life_fridge_days", "shelf_life_freezer_days"])
    ok((both["shelf_life_freezer_days"] > both["shelf_life_fridge_days"]).all(), "freezer life exceeds fridge life where both exist")
    ok((cap[["fridge_l", "freezer_l", "pantry_l"]] > 0).all().all(), "capacities are positive")
    ok((cap["capacity_unit"] == "litres_usable").all(), "one capacity unit (litres_usable) everywhere")
    ok((cap["n_people"].values == hh.set_index("household_id").loc[cap["household_id"], "n_people"].values).all(),
       "n_people matches households.csv")

    # Capacity sanity: P95 weekly demand volume should fit in total capacity with headroom,
    # and the default-storage mix of one week should fit in each storage.
    vol = rules.set_index("product_id")
    d = pd.read_csv(DATA / "demand_weekly.csv")
    d["default"] = d["product_id"].map(vol["default_storage"])
    d["litres"] = d["demand_units"] * d["product_id"].map(vol["volume_l_per_unit"])
    wk = d.groupby(["household_id", "week", "default"])["litres"].sum().unstack(fill_value=0)
    p95 = wk.groupby("household_id").quantile(0.95)
    c = cap.set_index("household_id")
    for s in STORAGES:
        head = (c[f"{s}_l"] / p95[s]).min()
        ok(head >= 2.0, f"{s}: capacity >= 2x P95 weekly demand volume in every household (min headroom {head:.1f}x)")
    # Reproducibility
    ok(build_rules().to_csv(index=False) == (OUT / "storage_rules.csv").read_text(), "storage_rules.csv equals a fresh rebuild")
    ok(build_households().to_csv(index=False) == (OUT / "household_storage.csv").read_text(), "household_storage.csv equals a fresh rebuild (seeded)")
    return len(fails)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        n = check()
        print(f"{n} check(s) failed" if n else "all checks passed")
        sys.exit(1 if n else 0)
    OUT.mkdir(exist_ok=True)
    build_rules().to_csv(OUT / "storage_rules.csv", index=False)
    build_households().to_csv(OUT / "household_storage.csv", index=False)
    print("wrote data_rl/storage_rules.csv and data_rl/household_storage.csv")


if __name__ == "__main__":
    main()
