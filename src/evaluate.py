"""Policy evaluation under identical conditions (deterministic, paired replay of the recorded demand and prices).

Splits (target weeks) and 13-week episodes that never cross a boundary:
    train       weeks  4-63   episodes start 4..51 (training sampler); the fixed evaluation set starts at 4, 17, 30, 43, 51
    validation  weeks 64-77   complete 13-week episodes only: start 64 (weeks 64-76); start 65 is also complete but overlaps it
    test        weeks 78-103  two fixed episodes: 78-90 and 91-103 -- RESERVED, see below

Every policy runs the same list of episode specs (household, region, budget scenario, storage scenario, start week, horizon), so
demand, prices, budget, capacities and the initial inventory are identical. Results are simulated. There is ONE recorded demand
trajectory, so episodes and policy differences are not independent samples: no confidence intervals or significance claims are made.

The test split is reserved for the final evaluation after model selection. `evaluate_policies("test", ...)` refuses to run unless
`confirm_final_test=True`, and the command line does not offer it.

Run:  python -m src.evaluate --split validation        (or --split train)
      python -m src.evaluate --split validation --forecast-csv PATH --forecast-variant no_events
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.env import GroceryEnv
from src.forecast_table import ForecastTable, ForecastUnavailable
from src.policies import AverageRule, BuyNothing, Fixed, ForecastRule, Policy
from src.rl_wrapper import OrderUpToWrapper

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "evaluation"
HORIZON = 13
SPLITS = {"train": (4, 63), "validation": (64, 77), "test": (78, 103)}      # inclusive target weeks
TRAIN_START_RANGE = (4, 51)
TRAIN_EVAL_STARTS = [4, 17, 30, 43, 51]
TEST_STARTS = [78, 91]
FIXED_PLAN_WEEKS = (4, 63)           # the Fixed baseline is built from training weeks only
BASELINES = ("buy_nothing", "fixed", "average_rule", "forecast_rule")
METRICS = ["reward", "cash_spend", "net_cost", "waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days",
           "fill_rate_value", "protein_coverage", "terminal_basis", "terminal_credit", "terminal_charge"]


@dataclass(frozen=True)
class EpisodeSpec:
    split: str
    start_week: int
    household_id: str = "hh_student"
    region_id: str = "durham_nc"
    budget_scenario: str = "normal"
    storage_scenario: str = "default"
    horizon: int = HORIZON

    @property
    def end_week(self) -> int:
        return self.start_week + self.horizon - 1

    @property
    def weeks(self) -> range:
        return range(self.start_week, self.end_week + 1)


def validate_spec(spec: EpisodeSpec) -> None:
    """An episode must lie completely inside its split."""
    if spec.split not in SPLITS:
        raise ValueError(f"unknown split {spec.split!r}")
    lo, hi = SPLITS[spec.split]
    if spec.start_week < lo or spec.end_week > hi:
        raise ValueError(f"episode weeks {spec.start_week}-{spec.end_week} cross the {spec.split} boundary {lo}-{hi}")


def training_starts() -> list[int]:
    """All start weeks a training sampler may use (every window ends by week 63)."""
    lo, hi = TRAIN_START_RANGE
    return list(range(lo, hi + 1))


def validation_starts(overlapping: bool = False) -> list[int]:
    lo, hi = SPLITS["validation"]
    complete = [s for s in range(lo, hi + 1) if s + HORIZON - 1 <= hi]        # 64, 65
    return complete if overlapping else complete[:1]


def episode_specs(split: str, **scenario) -> list[EpisodeSpec]:
    starts = {"train": TRAIN_EVAL_STARTS, "validation": validation_starts(), "test": TEST_STARTS}[split]
    specs = [EpisodeSpec(split, s, **scenario) for s in starts]
    for sp in specs:
        validate_spec(sp)
    return specs


def spec_hash(specs: list[EpisodeSpec]) -> str:
    return hashlib.sha256(json.dumps([asdict(s) for s in specs], sort_keys=True).encode()).hexdigest()[:12]


def write_spec_file(path: Path = OUT / "specs.json") -> Path:
    """Writes the fixed episode lists (weeks only; nothing is run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {s: {"hash": spec_hash(episode_specs(s)), "episodes": [asdict(e) for e in episode_specs(s)]} for s in SPLITS}
    path.write_text(json.dumps(payload, indent=2))
    return path


# ---------------------------------------------------------------------------------------------- running
def make_policy(name: str, env: GroceryEnv) -> Policy:
    d = env.d
    lo, hi = FIXED_PLAN_WEEKS
    if name == "buy_nothing":
        return BuyNothing(d.n)
    if name == "fixed":
        return Fixed(d.demand[lo:hi + 1].mean(axis=0), d.step_size)
    if name == "average_rule":
        return AverageRule(d.product_ids, d.step_size, 1.0)
    if name == "forecast_rule":
        return ForecastRule(d.product_ids, d.step_size)
    raise ValueError(f"unknown policy {name!r}")


def forecast_status(table: ForecastTable | None, specs: list[EpisodeSpec], variant: str) -> str | None:
    """None if forecast_rule can run on all specs, otherwise the reason it cannot."""
    if table is None:
        return "no forecast table supplied"
    try:
        for sp in specs:
            table.require(sp.household_id, range(sp.start_week, min(sp.end_week + 1, SPLITS["test"][1]) + 1), variant)
    except ForecastUnavailable as e:
        return str(e)
    return None


def run_episode(spec: EpisodeSpec, policy: Policy | str, table: ForecastTable | None = None, variant: str = "no_events") -> dict:
    validate_spec(spec)
    if isinstance(policy, str):
        policy = make_policy(policy, GroceryEnv(spec.household_id, spec.region_id, spec.budget_scenario, spec.storage_scenario,
                                                spec.horizon))
    # the forecast is attached only for policies that depend on it, so non-forecast baselines never touch the forecast file
    forecast_fn = table.forecast_fn(spec.household_id, variant) if (table is not None and policy.uses_forecast) else None
    env = GroceryEnv(spec.household_id, spec.region_id, spec.budget_scenario, spec.storage_scenario, spec.horizon, forecast_fn)
    base = env
    if policy.normalized_action:
        env = OrderUpToWrapper(base)
    obs, info = env.reset(options={"start_week": spec.start_week})
    policy.reset()
    obs_dict, done = info["obs_dict"], False
    tot = dict.fromkeys(["reward", "cash_spend", "waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days",
                         "demand_value"], 0.0)
    protein = []
    while not done:
        week = base.week
        action = policy.act_vec(obs) if policy.normalized_action else policy.act(obs_dict)
        obs, reward, _, done, info = env.step(action)
        obs_dict = info["obs_dict"]
        tot["reward"] += reward
        tot["cash_spend"] += info["spend"]
        for k in ("waste_cost", "waste_units", "stockout_units", "stockout_value", "stockout_days"):
            tot[k] += info[k]
        tot["demand_value"] += float((base.d.demand[week] * base.price_at(week)).sum())
        protein.append(info["protein_target_met"])
    row = {k: tot[k] for k in METRICS if k in tot}
    row.update(net_cost=info["episode_net_cost"], fill_rate_value=1.0 - tot["stockout_value"] / tot["demand_value"],
               protein_coverage=float(np.mean(protein)), terminal_basis=info["terminal_inventory_basis"],
               terminal_credit=info["terminal_inventory_credit"], terminal_charge=info["terminal_charge"])
    return {"policy": policy.name, "split": spec.split, "start_week": spec.start_week, "end_week": spec.end_week,
            **{k: row[k] for k in METRICS}}


def evaluate_policies(split: str, policies: list[str | Policy] | None = None, table: ForecastTable | None = None,
                      variant: str = "no_events", confirm_final_test: bool = False, **scenario):
    """Runs every policy on the same episode list. Returns (episode rows, mean-per-policy table, unavailable {name: reason})."""
    if split == "test" and not confirm_final_test:
        raise PermissionError("the test split is reserved for the final evaluation; pass confirm_final_test=True only then")
    specs = episode_specs(split, **scenario)
    unavailable: dict[str, str] = {}
    rows = []
    for pol in (policies if policies is not None else list(BASELINES)):
        name = pol if isinstance(pol, str) else pol.name
        if name == "forecast_rule" or (not isinstance(pol, str) and pol.uses_forecast):
            reason = forecast_status(table, specs, variant)
            if reason:
                unavailable[name] = reason
                continue
        for sp in specs:
            rows.append(run_episode(sp, pol, table, variant))
    episodes = pd.DataFrame(rows)
    summary = episodes.groupby("policy", sort=False)[METRICS].mean() if len(episodes) else pd.DataFrame(columns=METRICS)
    return episodes, summary, unavailable


# ---------------------------------------------------------------------------------------------- report
def md(df: pd.DataFrame) -> str:
    d = df.reset_index()
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        rows.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |")
    return "\n".join(rows)


def write_report(split: str, episodes: pd.DataFrame, summary: pd.DataFrame, unavailable: dict, variant: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = episode_specs(split)
    episodes.to_csv(OUT / f"{split}_episodes.csv", index=False)
    summary.to_csv(OUT / f"{split}_summary.csv")
    lines = [
        f"# Policy evaluation: {split} split",
        "",
        f"hh_student, durham_nc, normal budget (synthetic), default storage; {len(specs)} episode(s) of {HORIZON} weeks, "
        f"starting in weeks {[s.start_week for s in specs]} (spec hash `{spec_hash(specs)}`).",
        "Deterministic, paired replay of the recorded demand and prices: every policy sees the same episodes. This is one recorded "
        "trajectory, so the episodes are not independent samples; no confidence intervals or significance claims are made. Simulated results.",
        "`reward` is the training signal; `cash_spend` is money paid; `net_cost` = consumed + waste + terminal charge; "
        "`fill_rate_value` is price-weighted; terminal columns value the leftover inventory (basis, credit, charge).",
        "",
        f"## Mean per {HORIZON}-week episode",
        md(summary),
        "",
        "## Per episode",
        md(episodes.set_index(["policy", "start_week"])[["end_week"] + METRICS]),
        "",
    ]
    for name, reason in unavailable.items():
        lines.append(f"- `{name}` was NOT run ({variant} forecasts): {reason}")
    text = "\n".join(lines) + "\n"
    (OUT / f"report_{split}.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["train", "validation"], required=True)     # the test split is deliberately not offered
    ap.add_argument("--forecast-csv", default=None, help="out-of-sample forecast file from the ML side (see docs/RL_EVALUATION.md)")
    ap.add_argument("--forecast-variant", choices=["no_events", "with_events"], default="no_events")
    args = ap.parse_args()
    table = None
    if args.forecast_csv:
        products = pd.read_csv(ROOT / "data" / "products.csv")["product_id"].tolist()
        try:
            table = ForecastTable.from_csv(args.forecast_csv, products)
        except (ForecastUnavailable, ValueError) as e:
            print(f"forecast file rejected: {e}")
    episodes, summary, unavailable = evaluate_policies(args.split, table=table, variant=args.forecast_variant)
    write_spec_file()
    print(write_report(args.split, episodes, summary, unavailable, args.forecast_variant))


if __name__ == "__main__":
    main()
