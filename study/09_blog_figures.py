#!/usr/bin/env python3
"""Three figures for the blog writeup -- not part of the core analysis
pipeline, just presentation of results 04/08 already established. Reuses
the same color palette and mark conventions as 04_analysis.py.

Writes to ../figures/:
    probe_type_matrix.png/pdf       -- accuracy by probe type, agree vs disagree
    auroc_progression.png/pdf       -- confidence -> confidence+D, with CIs
    atomic_null_result.png/pdf      -- AND-consistent vs inconsistent, overlapping CIs

Usage:
    .venv/bin/python3 09_blog_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY_DIR = Path(__file__).parent
ROOT = STUDY_DIR.parent
FIGURES_DIR = ROOT / "figures"

AGREE_COLOR = "#0072B2"
DISAGREE_COLOR = "#D55E00"
NEUTRAL_COLOR = "#4D4D4D"

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#888888",
        "font.size": 11,
    }
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def save_fig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"saved {FIGURES_DIR / name}.{{png,pdf}}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def figure_probe_type_matrix() -> None:
    """Accuracy when each probe agrees vs. disagrees with P0 -- the figure
    that shows A/B/C are flat and D stands out."""
    rows = load_jsonl(ROOT / "data" / "healthbench" / "jev_judgments.jsonl")
    df = pd.DataFrame(rows)
    agg = df.groupby(["pair_id", "formulation"]).agg(p_met=("p_met", "mean"), ground_truth=("ground_truth", "first")).reset_index()
    agg["verdict"] = np.where(agg["p_met"] > 0.5, "MET", "UNMET")
    wide = agg.pivot(index="pair_id", columns="formulation", values="verdict")
    gt = agg.drop_duplicates("pair_id").set_index("pair_id")["ground_truth"]
    p0_correct = wide["P0"] == gt

    probes = [("A", "paraphrase"), ("B", "paraphrase"), ("C", "negation"), ("D", "subcriterion*")]
    agree_acc, agree_ci, disagree_acc, disagree_ci, labels = [], [], [], [], []
    for probe, rel in probes:
        matches = wide[probe] == wide["P0"]
        a, d = p0_correct[matches], p0_correct[~matches]
        agree_acc.append(a.mean())
        agree_ci.append(wilson_ci(int(a.sum()), len(a)))
        disagree_acc.append(d.mean())
        disagree_ci.append(wilson_ci(int(d.sum()), len(d)))
        labels.append(f"{probe}\n({rel})\nn={len(a)}/{len(d)}")

    x = np.arange(len(probes))
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.32
    for offset, accs, cis, color, name in (
        (-width / 2, agree_acc, agree_ci, AGREE_COLOR, "probe agrees with P0"),
        (width / 2, disagree_acc, disagree_ci, DISAGREE_COLOR, "probe disagrees with P0"),
    ):
        yerr_lo = [a - lo for a, (lo, hi) in zip(accs, cis)]
        yerr_hi = [hi - a for a, (lo, hi) in zip(accs, cis)]
        ax.bar(x + offset, accs, width=width, color=color, label=name, alpha=0.9)
        ax.errorbar(x + offset, accs, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("P0 accuracy vs. physician ground truth")
    ax.set_title("Accuracy by probe type: A/B/C are flat, D stands out")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save_fig(fig, "probe_type_matrix")


def figure_auroc_progression() -> None:
    """Confidence alone -> confidence + D-disagreement, with the bootstrap CI."""
    fig, ax = plt.subplots(figsize=(7, 5))
    models = ["Confidence\nalone", "D-disagreement\nalone", "Confidence +\nD-disagreement"]
    aurocs = [0.714, 0.728, 0.804]
    # 95% CI on the Model1->Model3 improvement was [0.047, 0.139] (bootstrap, see 04_analysis.py);
    # shown here as an error bar on Model 3 only, since that's the comparison that matters.
    x = np.arange(len(models))
    colors = [NEUTRAL_COLOR, NEUTRAL_COLOR, DISAGREE_COLOR]
    bars = ax.bar(x, aurocs, color=colors, width=0.5)
    ax.errorbar(x[2], aurocs[2], yerr=[[0.090 - 0.047], [0.139 - 0.090]], fmt="none", color="black", capsize=4, linewidth=1.3)
    ax.axhline(0.5, color="#AAAAAA", linestyle="--", linewidth=1, label="chance (0.5)")
    for xi, v in zip(x, aurocs):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("AUROC (predicting whether P0 is wrong)")
    ax.set_title("Adding D-disagreement to confidence: +0.090 AUROC\n(bootstrap 95% CI [0.047, 0.139], cross-validated)")
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "auroc_progression")


def figure_atomic_null_result() -> None:
    """AND-consistent vs. inconsistent accuracy, with Wilson CIs -- visually
    showing why this result is NOT a replication: the bars point the wrong
    way and the error bars swallow the difference whole."""
    atomic_path = ROOT / "data" / "healthbench" / "atomic_judgments.jsonl"
    jev_path = ROOT / "data" / "healthbench" / "jev_judgments.jsonl"

    p0_rows = [r for r in load_jsonl(jev_path) if r["formulation"] == "P0"]
    p0 = pd.DataFrame(p0_rows).groupby("pair_id").agg(p0_p_met=("p_met", "mean"), ground_truth=("ground_truth", "first")).reset_index()
    p0["p0_verdict"] = np.where(p0["p0_p_met"] > 0.5, "MET", "UNMET")
    p0["p0_correct"] = p0["p0_verdict"] == p0["ground_truth"]
    p0 = p0.set_index("pair_id")

    atomic = pd.DataFrame(load_jsonl(atomic_path))
    agg = atomic.groupby(["pair_id", "atomic_idx"]).agg(p_met=("p_met", "mean")).reset_index()
    agg["verdict"] = np.where(agg["p_met"] > 0.5, "MET", "UNMET")
    and_verdict = agg.groupby("pair_id")["verdict"].agg(lambda v: "MET" if (v == "MET").all() else "UNMET")

    joined = p0.join(and_verdict.rename("and_verdict"), how="inner")
    inconsistent = joined["and_verdict"] != joined["p0_verdict"]

    groups = [("AND-consistent", joined.loc[~inconsistent, "p0_correct"]), ("AND-inconsistent", joined.loc[inconsistent, "p0_correct"])]
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (label, s) in enumerate(groups):
        acc = s.mean()
        lo, hi = wilson_ci(int(s.sum()), len(s))
        color = AGREE_COLOR if i == 0 else DISAGREE_COLOR
        ax.bar(i, acc, color=color, width=0.5)
        ax.errorbar(i, acc, yerr=[[acc - lo], [hi - acc]], fmt="none", color="black", capsize=5, linewidth=1.3)
        ax.annotate(f"{acc:.1%}\n(n={len(s)})", (i, hi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("P0 accuracy vs. physician ground truth")
    ax.set_title("Atomic decomposition: overlapping error bars = no reliable effect\n(n=113 -- too small to separate these)")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    save_fig(fig, "atomic_null_result")


def main() -> None:
    figure_probe_type_matrix()
    figure_auroc_progression()
    figure_atomic_null_result()


if __name__ == "__main__":
    main()
