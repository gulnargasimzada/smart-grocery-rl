"""Report for the PPO pilot: select checkpoints on the validation windows, compare with the baselines, plot learning curves.

Reads results/rl_pilot/learning_curves_seed*.csv (written by src.train_rl) and the checkpoints in models_rl/.
Only validation episodes (weeks 64-77) are evaluated. The test split is never touched.

Run:  python -m src.report_rl
"""
from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.evaluate import EpisodeSpec, run_episode
from src.forecast_table import ForecastTable
from src.policies import RLPolicy
from src.train_rl import CKPT, OUT, REPORT_START, SELECT_STARTS, policy_from_model

ROOT = Path(__file__).resolve().parent.parent
COLS = ["reward", "cash_spend", "waste_cost", "waste_units", "stockout_units", "stockout_value", "fill_rate_value", "protein_coverage"]
BASELINES = ["fixed", "average_rule", "forecast_rule"]


def md(df: pd.DataFrame) -> str:
    d = df.reset_index()
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        rows.append("| " + " | ".join(f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |")
    return "\n".join(rows)


def run_policy(policy, starts, table):
    rows = [run_episode(EpisodeSpec("validation", s), policy, table, "no_events") for s in starts]
    return {c: float(np.mean([r[c] for r in rows])) for c in COLS}


def main() -> None:
    curves = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "learning_curves_seed*.csv")))], ignore_index=True)
    metas = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(OUT / "run_meta_seed*.json")))]
    products = pd.read_csv(ROOT / "data" / "products.csv")["product_id"].tolist()
    table = ForecastTable.from_csv(ROOT / "data" / "forecast_weekly.csv", products)

    # ---- baselines on the same validation windows
    base = {b: {"canonical": run_policy(b, [REPORT_START], table if b == "forecast_rule" else None),
                "selection": run_policy(b, SELECT_STARTS, table if b == "forecast_rule" else None)} for b in BASELINES}

    # ---- selected / final checkpoints
    sel_rows, sel_dir = [], OUT / "selected"
    sel_dir.mkdir(exist_ok=True)
    for (agent, seed), g in curves.groupby(["agent", "seed"]):
        use_fc = agent == "forecast"
        best = g.loc[g["val_reward"].idxmax()]                 # earliest wins ties
        final = g.iloc[-1]
        for tag, row in (("selected", best), ("final", final), ("initial", g.iloc[0])):
            model = PPO.load(CKPT / f"{agent}_seed{seed}" / row["checkpoint"], device="cpu")
            pol = policy_from_model(model, f"ppo_{agent}", use_fc)
            res = run_policy(pol, [REPORT_START], table if use_fc else None)
            sel_res = run_policy(pol, SELECT_STARTS, table if use_fc else None)
            sel_rows.append({"agent": agent, "seed": seed, "checkpoint": tag, "timesteps": int(row["timesteps"]),
                             **res, "reward_both_windows": sel_res["reward"]})
            if tag == "selected":
                shutil.copy(CKPT / f"{agent}_seed{seed}" / row["checkpoint"], sel_dir / f"{agent}_seed{seed}.zip")
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(OUT / "checkpoint_results.csv", index=False)

    mean = sel.groupby(["agent", "checkpoint"])[COLS + ["reward_both_windows", "timesteps"]].mean()
    comp = pd.DataFrame({f"{b}": base[b]["canonical"] for b in BASELINES}).T
    comp_sel = pd.DataFrame({f"{b}": base[b]["selection"] for b in BASELINES}).T
    avg_r = base["average_rule"]["canonical"]["reward"]
    avg_r_sel = base["average_rule"]["selection"]["reward"]

    # ---- learning curves
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"forecast": "tab:blue", "no_forecast": "tab:orange"}
    for agent, g in curves.groupby("agent"):
        for seed, h in g.groupby("seed"):
            ax[0].plot(h["timesteps"], h["val_reward"], color=colors[agent], alpha=0.35, lw=1)
            ax[1].plot(h["timesteps"], h["train_ep_reward"], color=colors[agent], alpha=0.35, lw=1)
        m = g.groupby("timesteps")[["val_reward", "train_ep_reward"]].mean()
        ax[0].plot(m.index, m["val_reward"], color=colors[agent], lw=2.2, label=f"PPO {agent} (mean of seeds)")
        ax[1].plot(m.index, m["train_ep_reward"], color=colors[agent], lw=2.2, label=f"PPO {agent} (mean of seeds)")
    for b, c in zip(BASELINES, ["gray", "green", "red"]):
        ax[0].axhline(base[b]["selection"]["reward"], color=c, ls="--", lw=1.2, label=b)
    ax[0].set(title="Validation reward (mean of windows 64 and 65)", xlabel="training timesteps", ylabel="13-week episode reward")
    ax[1].set(title="Training episode reward (start weeks 12-51, stochastic policy)", xlabel="training timesteps")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=130)

    # ---- verdict
    verdict = []
    for agent in ("forecast", "no_forecast"):
        s = sel[(sel["agent"] == agent) & (sel["checkpoint"] == "selected")]
        wins = int((s["reward"] > avg_r).sum())
        verdict.append(f"- PPO {agent}: selected checkpoints beat the Average Rule (reward {avg_r:.2f}) in {wins} of {len(s)} seeds on the "
                       f"canonical validation episode (mean {s['reward'].mean():.2f}, best {s['reward'].max():.2f}, worst {s['reward'].min():.2f}).")
    times = {m["seed"]: {r["agent"]: r["seconds"] for r in m["runs"]} for m in metas}
    hp = metas[0]["hyperparameters"]
    lines = [
        "# PPO pilot: validation results (selection on validation windows only; test weeks never run)",
        "",
        f"hh_student, durham_nc, normal budget, default storage, 13-week episodes. Training start weeks {metas[0]['train_start_range'][0]}-"
        f"{metas[0]['train_start_range'][1]}; seeds {sorted(times)}; {metas[0]['timesteps']} timesteps per run (identical for both agents). "
        "Checkpoints every ~20k steps, plus step 0.",
        f"Checkpoint selection = best mean validation reward over the two complete windows starting in weeks {SELECT_STARTS} (they overlap, so this "
        f"is effectively one trajectory). The table below reports the canonical validation episode (weeks {REPORT_START}-{REPORT_START + 12}). "
        "Because checkpoints were chosen on these same weeks, the selected numbers are optimistic; `final` (last checkpoint) and `initial` "
        "(step 0) are shown for comparison. Deterministic replay of one recorded trajectory: no confidence intervals or significance claims.",
        "",
        "## Verdict",
        *verdict,
        "",
        f"## Baselines, canonical validation episode (weeks {REPORT_START}-{REPORT_START + 12})",
        md(comp),
        "",
        "## PPO checkpoints, canonical validation episode (mean over seeds)",
        md(mean),
        "",
        "## PPO per seed",
        md(sel.set_index(["agent", "seed", "checkpoint"])),
        "",
        f"## Baselines on the selection windows {SELECT_STARTS} (mean of the two windows; the reference lines of the plot)",
        md(comp_sel),
        "",
        "![learning curves](learning_curves.png)",
        "",
        "## Training setup",
        f"- Hyperparameters: {json.dumps(hp)}",
        "- Training wall time per run (seconds): " + json.dumps(times),
        "- Observation: 190 values; forecast agent gets the verified no_events Ridge forecasts, ablation gets zeros and forecast_available = 0.",
        "- Action: [-1, 1]^36 mapped to the normalized wrapper action a = (x + 1) / 2, so x = 0 equals the Average Rule; the initial policy is close to it.",
        "- Termination: the episode ends with `terminated=True` (finite horizon with a terminal charge), so PPO does not bootstrap after step 13.",
        "- Stability: " + ("non-finite validation values seen" if any(r["nan_or_inf"] for m in metas for r in m["runs"]) else
                          "no NaN/inf in any validation score; see the curves for variance between seeds."),
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
