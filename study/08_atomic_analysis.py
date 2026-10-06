#!/usr/bin/env python3
"""Analysis: does decomposition inconsistency predict P0 error, beyond confidence?

Reads: ../data/healthbench/atomic_judgments.jsonl   (written by 07)
       ../data/healthbench/jev_judgments.jsonl       (for P0's own answer + ground truth)
       ../data/healthbench/atomic_decomposition.jsonl (for n_atomic per criterion)
Writes: printed tables only -- n=113 is too small for the figure polish
        04_analysis.py has; tables are more honest at this sample size.

This is the cleaner follow-up to Probe D (see study/EVOLUTION.md section 9):
instead of asking an LLM to guess a "consequence" of a criterion (which
usually turned out to be a narrower sub-clause, not a full-scope
implication), this stage uses an EXPLICIT, pre-registered decomposition
into atomic AND-combined requirements, frozen BEFORE any Jev data was
touched (06_decompose_criteria.py). That gives an exact, checkable
prediction a fuzzy "implication" never had:

    compound_verdict should equal AND(atomic verdicts)

if Jev is reasoning consistently about the compound criterion and its own
stated pieces.

Two signals, per the request to keep continuous probabilities, not just
the binary AND:
    and_inconsistent  = P0's verdict != AND(atomic verdicts)              (binary)
    compound_p_soft   = product(atomic p_met_i)   -- independence approx  (continuous)
    gap               = p0_p_met - compound_p_soft                        (continuous, signed)

No new ground truth is used or needed: the test is whether these signals
predict P0 being wrong against the SAME physician label already used
throughout this study.

Usage:
    .venv/bin/python3 08_atomic_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).parent
ROOT = STUDY_DIR.parent
ATOMIC_JUDGMENTS_PATH = ROOT / "data" / "healthbench" / "atomic_judgments.jsonl"
JUDGMENTS_PATH = ROOT / "data" / "healthbench" / "jev_judgments.jsonl"
DECOMPOSITION_PATH = ROOT / "data" / "healthbench" / "atomic_decomposition.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_p0_table() -> pd.DataFrame:
    """P0's own answer per pair, aggregated over its 3 repeats -- reuses the
    exact same jev_judgments.jsonl rows 03_jev_inference.py already wrote;
    nothing re-asked."""
    rows = [r for r in load_jsonl(JUDGMENTS_PATH) if r["formulation"] == "P0"]
    df = pd.DataFrame(rows)
    agg = df.groupby("pair_id").agg(p0_p_met=("p_met", "mean"), ground_truth=("ground_truth", "first")).reset_index()
    agg["p0_verdict"] = np.where(agg["p0_p_met"] > 0.5, "MET", "UNMET")
    agg["p0_confidence"] = (2 * (agg["p0_p_met"] - 0.5).abs()).clip(0, 1)
    agg["p0_correct"] = agg["p0_verdict"] == agg["ground_truth"]
    return agg.set_index("pair_id")


def load_atomic_table() -> pd.DataFrame:
    """One row per pair: AND(atomic verdicts), the independence-approximation
    soft compound probability, and the repeat-noise check, aggregated over
    each atomic requirement's 3 repeats."""
    df = pd.DataFrame(load_jsonl(ATOMIC_JUDGMENTS_PATH))
    agg = (
        df.groupby(["pair_id", "atomic_idx"])
        .agg(p_met=("p_met", "mean"), p_met_std=("p_met", "std"), n_repeats=("p_met", "size"))
        .reset_index()
    )
    agg["p_met_std"] = agg["p_met_std"].fillna(0.0)
    agg["verdict"] = np.where(agg["p_met"] > 0.5, "MET", "UNMET")

    rows = []
    for pair_id, group in agg.groupby("pair_id"):
        group = group.sort_values("atomic_idx")
        and_verdict = "MET" if (group["verdict"] == "MET").all() else "UNMET"
        compound_p_soft = group["p_met"].prod()  # independence approximation: P(A and B and ...) = prod(P_i)
        rows.append(
            {
                "pair_id": pair_id,
                "and_verdict": and_verdict,
                "compound_p_soft": compound_p_soft,
                "mean_atomic_std": group["p_met_std"].mean(),
                "n_atomic": len(group),
            }
        )
    return pd.DataFrame(rows).set_index("pair_id")


def report_repeat_noise(atomic: pd.DataFrame, p0: pd.DataFrame) -> None:
    mean_within_std = atomic["mean_atomic_std"].mean()
    shift = (p0["p0_p_met"] - atomic["compound_p_soft"]).abs().mean()
    print(
        f"repeat noise check: mean within-atomic std = {mean_within_std:.3f} "
        f"(from 3 repeats) vs. mean |P0 - compound_p_soft| shift = {shift:.3f}"
    )
    if mean_within_std >= shift:
        print("  WARNING: repeat noise is comparable to the signal -- treat results cautiously at this n")
    print()


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def outcome_analysis(pairs: pd.DataFrame) -> None:
    print("=" * 70)
    print("OUTCOME: does AND-inconsistency / the soft-probability gap predict")
    print("P0 being wrong against the EXISTING physician label? (n=113 -- small, say so)")
    print("=" * 70)

    inconsistent = pairs["and_verdict"] != pairs["p0_verdict"]
    for label, mask in (("AND-consistent", ~inconsistent), ("AND-inconsistent", inconsistent)):
        sub = pairs[mask]
        if len(sub):
            k = int(sub["p0_correct"].sum())
            lo, hi = wilson_ci(k, len(sub))
            print(f"  {label:<18} n={len(sub):3d}  accuracy={sub['p0_correct'].mean():.1%}  (95% CI [{lo:.1%}, {hi:.1%}])")
        else:
            print(f"  {label:<18} n=0")

    print(f"\n  gap = P0's own p_met minus the independence-approximation compound_p_soft")
    print(f"  mean |gap| when P0 correct:   {pairs.loc[pairs['p0_correct'], 'gap'].abs().mean():.3f}")
    print(f"  mean |gap| when P0 incorrect: {pairs.loc[~pairs['p0_correct'], 'gap'].abs().mean():.3f}")
    print()


def model_comparison(pairs: pd.DataFrame) -> None:
    import statsmodels.api as sm
    from scipy.stats import chi2
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    print("=" * 70)
    print("MODEL COMPARISON: confidence vs. the soft-probability gap (continuous, signed)")
    print("=" * 70)

    y = (~pairs["p0_correct"]).astype(int).values
    confidence = pairs["p0_confidence"].astype(float).values
    gap = pairs["gap"].astype(float).values
    abs_gap = np.abs(gap)

    def fit_report(name, X):
        X = sm.add_constant(X)
        m = sm.Logit(y, X).fit(disp=0)
        pred = m.predict(X)
        print(
            f"  {name:<42} llf={m.llf:7.2f}  AUROC={roc_auc_score(y, pred):.3f}  "
            f"AUPRC={average_precision_score(y, pred):.3f}  log_loss={log_loss(y, pred):.3f}  "
            f"Brier={brier_score_loss(y, pred):.3f}"
        )
        return m

    print("\n[Standalone]")
    m1 = fit_report("Model 1: confidence only", confidence)
    fit_report("compound_p_soft's own confidence alone", np.abs(pairs["compound_p_soft"].values - 0.5))
    fit_report("|gap| alone", abs_gap)
    fit_report("signed gap alone", gap)

    print("\n[Combined]")
    m3 = fit_report("Model 3: confidence + |gap|", np.column_stack([confidence, abs_gap]))
    fit_report("confidence + signed gap", np.column_stack([confidence, gap]))

    lr_stat = 2 * (m3.llf - m1.llf)
    p_value = chi2.sf(lr_stat, df=1)
    print(f"\nLikelihood-ratio test, Model 3 vs. Model 1: LR={lr_stat:.3f}, df=1, p={p_value:.5f}")
    print(f"  (n={len(y)} -- a single df=1 LR test is reasonably trustworthy even at this n, but treat the")
    print(f"   p-value as suggestive, not confirmatory, until replicated on a larger decomposed set)")

    print("\nBootstrap (2000 resamples): AUROC(Model 3) - AUROC(Model 1)")
    rng = np.random.RandomState(20261004)
    n = len(y)
    diffs = []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        yb, cb, gb = y[idx], confidence[idx], abs_gap[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            X1b, X3b = sm.add_constant(cb), sm.add_constant(np.column_stack([cb, gb]))
            m1b = sm.Logit(yb, X1b).fit(disp=0)
            m3b = sm.Logit(yb, X3b).fit(disp=0)
            diffs.append(roc_auc_score(yb, m3b.predict(X3b)) - roc_auc_score(yb, m1b.predict(X1b)))
        except Exception:
            continue
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  n valid resamples={len(diffs)}, mean diff={diffs.mean():.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")
    print("  (CI excludes 0)" if lo > 0 else "  (CI includes 0 -- NOT conclusive at this sample size)")
    print()


def cross_validated_comparison(pairs: pd.DataFrame, n_folds: int = 5, n_repeats: int = 20) -> None:
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    print("=" * 70)
    print(f"CROSS-VALIDATED ({n_folds}-fold, {n_repeats} repeats, pooled out-of-fold) -- n=113, interpret cautiously")
    print("=" * 70)

    y = (~pairs["p0_correct"]).astype(int).values
    confidence = pairs["p0_confidence"].astype(float).values
    abs_gap = np.abs(pairs["gap"].astype(float).values)

    aurocs_m1, aurocs_m3 = [], []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=20261004 + repeat)
        oof_m1, oof_m3, oof_y = [], [], []
        for train_idx, test_idx in skf.split(confidence, y):
            X1_train = sm.add_constant(confidence[train_idx])
            X3_train = sm.add_constant(np.column_stack([confidence[train_idx], abs_gap[train_idx]]))
            X1_test = sm.add_constant(confidence[test_idx], has_constant="add")
            X3_test = sm.add_constant(np.column_stack([confidence[test_idx], abs_gap[test_idx]]), has_constant="add")
            try:
                m1 = sm.Logit(y[train_idx], X1_train).fit(disp=0)
                m3 = sm.Logit(y[train_idx], X3_train).fit(disp=0)
            except Exception:
                continue
            oof_m1.extend(m1.predict(X1_test))
            oof_m3.extend(m3.predict(X3_test))
            oof_y.extend(y[test_idx])
        if len(set(oof_y)) < 2:
            continue
        aurocs_m1.append(roc_auc_score(oof_y, oof_m1))
        aurocs_m3.append(roc_auc_score(oof_y, oof_m3))

    aurocs_m1, aurocs_m3 = np.array(aurocs_m1), np.array(aurocs_m3)
    print(f"\nOut-of-fold AUROC, Model 1 (confidence only):   mean={aurocs_m1.mean():.3f} (sd={aurocs_m1.std():.3f})")
    print(f"Out-of-fold AUROC, Model 3 (confidence + |gap|): mean={aurocs_m3.mean():.3f} (sd={aurocs_m3.std():.3f})")
    diff = aurocs_m3 - aurocs_m1
    print(f"Paired difference: mean={diff.mean():.3f}, 95% range=[{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]")
    print(f"(fold size here is ~{len(y) // n_folds} pairs -- noisier than the 397-pair Probe-D cross-validation)")
    print()


def main() -> None:
    if not ATOMIC_JUDGMENTS_PATH.exists():
        raise SystemExit(f"Missing {ATOMIC_JUDGMENTS_PATH} -- run 07_atomic_jev_inference.py first.")

    p0 = load_p0_table()
    atomic = load_atomic_table()
    pairs = p0.join(atomic, how="inner")
    pairs["gap"] = pairs["p0_p_met"] - pairs["compound_p_soft"]
    print(f"loaded {len(pairs)} pairs with both P0 and a full atomic decomposition\n")

    report_repeat_noise(atomic, p0)
    outcome_analysis(pairs)
    model_comparison(pairs)
    cross_validated_comparison(pairs)


if __name__ == "__main__":
    main()
