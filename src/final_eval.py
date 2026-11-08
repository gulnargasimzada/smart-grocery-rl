"""Final evaluation pipeline: six methods on the two held-out test episodes (weeks 78-90 and 91-103), prepared but NOT executed.

Methods (frozen, nothing is retrained or retuned): Fixed Rule, Average Rule, Forecast Rule (Ridge), Traditional Optimization (MILP, delta 0.4) and
PPO with / without Ridge forecasts (the six selected checkpoints; every seed reported, plus the mean over seeds).

Commands
--------
python -m src.final_eval --freeze            write results/final/freeze_manifest.json (hashes of code, data, checkpoints, forecasts, settings)
python -m src.final_eval --verify            check that everything still matches the manifest (no evaluation is run)
python -m src.final_eval --dry-run           run the SAME pipeline on the validation episode -> results/final/dry_run_validation/ (never the test split)
python -m src.final_eval --run-test --approve-test <phrase>
                                             the one-time held-out run. Refuses unless (1) the manifest verifies, (2) the approval phrase matches the
                                             manifest, (3) results/final/test/ does not exist yet. The phrase is printed by --verify. It is a safeguard against
                                             accidental runs: it must only be used after the project owner explicitly approves the test run.

The report gives every metric separately and each method's rank per metric; it never declares an overall winner. Results are deterministic
replays of one recorded trajectory (two episodes): no confidence intervals or significance claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import comparison as cmp
from src.env import CREDIT_BUFFER_WEEKS, CREDIT_DAYS, PROTEIN_WEIGHT, STOCK_CAP_WEEKS, STOCKOUT_MULT
from src.evaluate import EpisodeSpec, episode_specs, run_episode, spec_hash, validate_spec

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "final"
MANIFEST = OUT / "freeze_manifest.json"
TEST_DIR = OUT / "test"
DRY_DIR = OUT / "dry_run_validation"
VARIANT = cmp.VARIANT
CODE_FILES = ["src/env.py", "src/evaluate.py", "src/policies.py", "src/rl_wrapper.py", "src/forecast_table.py", "src/optimization.py",
              "src/optimization_eval.py", "src/train_rl.py", "src/comparison.py", "src/final_eval.py", "src/diets.py"]
DATA_FILES = ["data/products.csv", "data/households.csv", "data/demand_weekly.csv", "data/price_weekly_regional_synthetic.csv",
              "data/household_region_budgets.csv", "data_rl/storage_rules.csv", "data_rl/household_storage.csv", "data/forecast_weekly.csv"]
REPORT_METRICS = ["cash_spend", "net_cost", "waste_units", "waste_cost", "stockout_units", "stockout_value", "fill_rate_value", "reward",
                  "protein_coverage", "terminal_charge"]
AGG_METHODS = ["fixed", "average_rule", "forecast_rule", "optimization", "ppo_forecast_mean", "ppo_no_forecast_mean"]
LABELS = {"fixed": "Fixed Rule", "average_rule": "Average Rule", "forecast_rule": "Forecast Rule (Ridge)",
          "optimization": "Optimization (MILP)", "ppo_forecast_mean": "PPO with forecast (mean of 3 seeds)",
          "ppo_no_forecast_mean": "PPO without forecast (mean of 3 seeds)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def library_versions() -> dict:
    import gymnasium, pandas, scipy, stable_baselines3, torch
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__}


def build_manifest() -> dict:
    specs = episode_specs("test")
    return {
        "purpose": "Frozen state for the one-time held-out test run (weeks 78-103). Nothing below may change before the run.",
        "base_git_commit": git("rev-parse", "HEAD"),
        "methods": {
            "fixed": "standing order of the mean weekly demand of training weeks 4-63",
            "average_rule": "order up to 1 week of trailing 4-week mean demand, whole packages",
            "forecast_rule": "order up to the Ridge no_events forecast, whole packages",
            "optimization": {"policy": "weekly MILP (src/optimization.py)", "delta": cmp.delta_from_tuning(),
                             "delta_chosen_on": "training windows only (results/optimization/tuned.json)"},
            "ppo": {"checkpoints": {f"{a}_seed{s}": {"path": f"results/rl_pilot/selected/{a}_seed{s}.zip",
                                                      "sha256": sha256(cmp.SELECTED / f"{a}_seed{s}.zip")}
                                    for a in cmp.PPO_AGENTS for s in cmp.SEEDS},
                    "selection": "best mean validation reward on windows 64 and 65 (results/rl_pilot/); no further training or tuning",
                    "forecast_variant": VARIANT},
        },
        "environment_constants": {"stockout_multiplier": STOCKOUT_MULT, "protein_weight": PROTEIN_WEIGHT, "stock_cap_weeks": STOCK_CAP_WEEKS,
                                  "terminal_credit_days": CREDIT_DAYS, "terminal_credit_buffer_weeks": CREDIT_BUFFER_WEEKS},
        "evaluation": {"household": "hh_student", "region": "durham_nc", "budget_scenario": "normal", "storage_scenario": "default",
                       "horizon_weeks": 13, "test_episodes": [{"start_week": s.start_week, "end_week": s.end_week} for s in specs],
                       "test_spec_hash": spec_hash(specs), "forecast_variant": VARIANT,
                       "replay": "deterministic, paired replay of the recorded demand and prices (one trajectory)",
                       "metrics": REPORT_METRICS},
        "code_sha256": {f: sha256(ROOT / f) for f in CODE_FILES},
        "data_sha256": {f: sha256(ROOT / f) for f in DATA_FILES},
        "validation_reference_sha256": {f: sha256(ROOT / f) for f in ("results/comparison/comparison_table.csv",)},
        "libraries": library_versions(),
    }


def manifest_hash(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "libraries"}      # library patch levels are recorded but do not block the run
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def approval_phrase(manifest: dict) -> str:
    return f"RUN-HELD-OUT-TEST-{manifest_hash(manifest)[:12]}"


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} not found: run `python -m src.final_eval --freeze` first")
    return json.loads(MANIFEST.read_text())


def verify_freeze() -> dict:
    """Returns the stored manifest if every hashed file and setting still matches, else raises RuntimeError listing the differences."""
    stored = load_manifest()
    current = build_manifest()
    diffs = []
    for key in ("code_sha256", "data_sha256", "validation_reference_sha256"):
        for f, h in stored[key].items():
            if current[key].get(f) != h:
                diffs.append(f"{key}: {f}")
    for k in ("checkpoints",):
        for name, entry in stored["methods"]["ppo"][k].items():
            if current["methods"]["ppo"][k][name]["sha256"] != entry["sha256"]:
                diffs.append(f"checkpoint: {name}")
    for k in ("environment_constants", "evaluation"):
        if stored[k] != current[k]:
            diffs.append(k)
    if stored["methods"]["optimization"] != current["methods"]["optimization"]:
        diffs.append("optimization settings")
    if diffs:
        raise RuntimeError("frozen state changed since --freeze: " + "; ".join(diffs))
    return stored


def preflight(table, split: str) -> None:
    """Metadata-only checks before a run: forecasts exist for every episode week and were produced by models trained strictly earlier."""
    for sp in episode_specs(split):
        table.require(sp.household_id, sp.weeks, VARIANT)
        for w in sp.weeks:
            assert table.method(sp.household_id, w, VARIANT) == "ridge", f"week {w} is not a Ridge forecast"


# ------------------------------------------------------------------------------------------ evaluation
def evaluate_split(split: str) -> pd.DataFrame:
    """One row per (method, episode) for the given split. Used by the dry run (validation) and, unchanged, by the held-out run (test)."""
    table = cmp.load_table()
    specs = episode_specs(split)
    for sp in specs:
        validate_spec(sp)
    preflight(table, split)
    rows = []
    for name, method in cmp.build_methods().items():
        for sp in specs:
            r = run_episode(sp, method, table if cmp.uses_forecast(method) else None, VARIANT)
            r["method"] = name
            rows.append(r)
    return pd.DataFrame(rows)


def per_episode_tables(runs: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(s): cmp.aggregate(g.reset_index(drop=True)) for s, g in runs.groupby("start_week")}


def mean_over_episodes(runs: pd.DataFrame) -> pd.DataFrame:
    m = runs.groupby("method", sort=False)[cmp.METRICS].mean().reset_index()
    return cmp.aggregate(m)


def rankings(t: pd.DataFrame) -> pd.DataFrame:
    """For each metric the six aggregated methods ordered best to worst. Deliberately no overall score or winner."""
    rows = {}
    for m in REPORT_METRICS:
        higher = m in cmp.HIGHER_IS_BETTER
        order = t.loc[AGG_METHODS, m].sort_values(ascending=not higher).index
        rows[m + (" (higher better)" if higher else " (lower better)")] = [f"{LABELS[o].split(' (')[0]} {t.loc[o, m]:.2f}" for o in order]
    return pd.DataFrame(rows, index=[f"#{i}" for i in range(1, len(AGG_METHODS) + 1)]).T


def write_report(runs: pd.DataFrame, out_dir: Path, split: str, manifest_note: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = episode_specs(split)
    mean_t = mean_over_episodes(runs)
    per_ep = per_episode_tables(runs)
    runs.to_csv(out_dir / "runs.csv", index=False)
    mean_t.to_csv(out_dir / "mean_over_episodes.csv")
    for s, t in per_ep.items():
        t.to_csv(out_dir / f"episode_{s}.csv")
    pct = cmp.vs_average_rule(mean_t)
    pct.to_csv(out_dir / "vs_average_rule_pct.csv")
    ranks = rankings(mean_t)
    ranks.to_csv(out_dir / "rankings_per_metric.csv")
    cols = ["cash_spend", "net_cost", "waste_units", "waste_cost", "stockout_units", "stockout_value", "fill_rate_value", "reward",
            "protein_coverage", "terminal_charge"]
    title = "Held-out TEST evaluation" if split == "test" else "DRY RUN of the final pipeline on the VALIDATION episode (not a result)"
    lines = [
        f"# {title}",
        "",
        manifest_note,
        f"Episodes: {[(s.start_week, s.end_week) for s in specs]} (spec hash `{spec_hash(specs)}`). hh_student, durham_nc, normal budget (synthetic), default storage, "
        f"stockout multiplier {STOCKOUT_MULT}. Forecasts: Ridge `{VARIANT}`, identical for every forecast-dependent method. PPO checkpoints and the Optimization delta "
        "are frozen (see `results/final/freeze_manifest.json`). Deterministic replay of one recorded trajectory: no confidence intervals or significance claims. "
        "Metrics are reported separately; **no overall winner is declared**.",
        "",
        "## Mean over the episodes",
        cmp.md(mean_t[cols]),
        "",
        "## Rank of the six methods on each metric (best to worst)",
        cmp.md(ranks),
        "",
        "## Change relative to the Average Rule (%)",
        "Positive is better for reward, fill rate, protein coverage; negative is better for spend, net cost, waste, stockouts, terminal charge.",
        cmp.md(pct[cols].drop(index="average_rule"), "{:+.1f}"),
        "",
    ]
    for s, t in per_ep.items():
        lines += [f"## Episode weeks {s}-{s + 12}", cmp.md(t[cols]), ""]
    lines += ["## PPO seeds", "Individual seeds appear in the tables above (`ppo_*_seed0/1/2`); the `_mean` rows are the mean over the three seeds.", ""]
    text = "\n".join(lines)
    (out_dir / "report.md").write_text(text)
    return text


def run_dry() -> str:
    manifest = verify_freeze()
    runs = evaluate_split("validation")
    return write_report(runs, DRY_DIR, "validation",
                        f"Pipeline check only, run on validation week 64-76 with the frozen state (manifest hash `{manifest_hash(manifest)[:12]}`). "
                        "PPO validation numbers are optimistic because the checkpoints were selected on these weeks.")


def run_test(approval: str | None) -> str:
    manifest = verify_freeze()
    if approval != approval_phrase(manifest):
        raise PermissionError("the held-out test run needs the explicit approval phrase from `--verify`; it must only be used after the project "
                              "owner has approved the run")
    if TEST_DIR.exists():
        raise FileExistsError(f"{TEST_DIR} already exists: the held-out test is run once; results must not be overwritten")
    runs = evaluate_split("test")
    return write_report(runs, TEST_DIR, "test",
                        f"One-time held-out run with the frozen state (manifest hash `{manifest_hash(manifest)[:12]}`, base commit `{manifest['base_git_commit'][:7]}`).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run-test", action="store_true")
    ap.add_argument("--approve-test", default=None, help="approval phrase printed by --verify (only after explicit owner approval)")
    args = ap.parse_args()
    if args.freeze:
        OUT.mkdir(parents=True, exist_ok=True)
        m = build_manifest()
        MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST.relative_to(ROOT)} (manifest hash {manifest_hash(m)[:12]})")
    elif args.verify:
        m = verify_freeze()
        print(f"frozen state verified. manifest hash {manifest_hash(m)[:12]}\napproval phrase for the held-out run: {approval_phrase(m)}")
    elif args.dry_run:
        print(run_dry())
    else:
        print(run_test(args.approve_test))


if __name__ == "__main__":
    main()
