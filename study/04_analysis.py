#!/usr/bin/env python3
"""Analysis: does probe disagreement predict P0 correctness beyond confidence?

Reads: ../data/healthbench/jev_judgments.jsonl (written by 03_jev_inference.py)
Writes: ../figures/*.png, and prints the headline numbers.

This is the direct test of:
    H0: disagreement carries no information about correctness beyond Jev's
        own stated confidence.
    H1: disagreement carries incremental information, especially where
        confidence is already high.

Needs only Jev's own answers + ground truth -- no escalation call, no
second judge (see 05_escalation.py for that separate, optional question).

Two analyses:
  1. OUTCOME-level (needs ground truth): accuracy of P0's verdict, split by
     confidence bucket x whether any probe (A-D) disagreed with P0.
  2. COHERENCE-level (needs no ground truth at all -- a pure Jev
     self-consistency diagnostic): does Probe C (negation) actually behave
     like a negation of P0, and does Probe D actually behave like something
     P0 entails?

NAMING NOTE (read before trusting any "implication" label below): Probe D
was GENERATED as an attempted implication, but audit_d_scope_and_filtered_
rerun() found 21/23 of the criteria behind its disagreement cases are
"partial_scope" -- D usually captures one clause of a criterion with
several implicit requirements, not a full-scope consequence of the whole
thing. So "D (implication)" below means "D, generated to attempt an
implication, scope not established" -- NOT a validated logical relationship.
The empirical result (D adds information beyond confidence, cross-
validated) stands regardless; what's unresolved is WHY -- logical
inconsistency, criterion decomposition, or just relative question
difficulty. Call it a subcriterion / implication-like probe, not an
implication, until that's settled (e.g. via explicit atomic decomposition
with its own ground truth -- see conversation notes, not yet implemented
here).

Usage:
    .venv/bin/python3 04_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).parent
ROOT = STUDY_DIR.parent
JUDGMENTS_PATH = ROOT / "data" / "healthbench" / "jev_judgments.jsonl"
PAIRS_PATH = ROOT / "data" / "healthbench" / "pairs.jsonl"
PROBES_PATH = ROOT / "data" / "healthbench" / "criterion_probes.jsonl"
FIGURES_DIR = ROOT / "figures"
ENV_PATH = ROOT / ".env"

CONFIDENCE_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CONFIDENCE_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

# Only A (direct predicate) and B (semantic paraphrase) are supposed to
# preserve P0's verdict -- that's the actual "redundant measurement"
# H0/H1 is about. C (negation) is SUPPOSED to flip relative to P0 when
# Jev is coherent, and D (subcriterion/implication-like) is a one-directional relationship,
# not a same-verdict expectation -- counting either as "disagreement" in
# the same sense as A/B inflates the signal with cases where Jev did
# exactly what it should (confirmed empirically: in this run, Probe C
# disagreed with P0 97.5% of the time in the high-confidence bucket,
# which is the negation working, not instability). C and D's own logical
# relationships are checked separately in coherence_analysis().
PARAPHRASE_PROBES = ("A", "B")

# Fixed categorical colors (Okabe-Ito, colorblind-safe) -- a solid default
# since this project has no brand palette.md. If one exists, swap these two
# hex values and re-run scripts/validate_palette.js from the dataviz skill
# before publishing; don't eyeball it.
AGREE_COLOR = "#0072B2"  # blue
DISAGREE_COLOR = "#D55E00"  # orange/vermillion
GUIDE_COLOR = "#888888"  # neutral gray for reference lines -- never data identity
NEUTRAL_COLOR = "#4D4D4D"  # single-series bars
LOW_N_THRESHOLD = 5  # cells thinner than this get visually de-emphasized, never hidden

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
    """Wilson score interval for a binomial proportion -- behaves sanely at
    small n and at p near 0 or 1, unlike a normal approximation, which is
    exactly our situation (some confidence buckets have single-digit n)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def save_fig(fig: plt.Figure, name: str) -> None:
    """Both a web-weight PNG and a vector PDF for print -- a reviewer or a
    print layout needs the latter; a blog embed needs the former."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"saved {FIGURES_DIR / name}.{{png,pdf}}")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_judgments() -> pd.DataFrame:
    return pd.DataFrame(load_jsonl(JUDGMENTS_PATH))


def aggregate_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse N_REPEATS independent calls per (pair_id, formulation) into
    one mean p_met (and its spread) -- see 03_jev_inference.py's docstring
    for why repeats exist at all: a single Noul call is a noisy point
    estimate (Jev's answers are documented as not bit-deterministic), so a
    between-formulation difference only means something once it's compared
    against the within-formulation spread computed here."""
    agg = df.groupby(["pair_id", "formulation"]).agg(
        p_met=("p_met", "mean"),
        p_met_std=("p_met", "std"),
        n_repeats=("p_met", "size"),
        ground_truth=("ground_truth", "first"),
    ).reset_index()
    agg["p_met_std"] = agg["p_met_std"].fillna(0.0)
    agg["verdict"] = np.where(agg["p_met"] > 0.5, "MET", "UNMET")
    return agg


def report_repeat_noise(df: pd.DataFrame, agg: pd.DataFrame) -> None:
    """Sanity check: is the spread WITHIN a formulation (repeat noise) small
    relative to the typical shift BETWEEN formulations (the thing we're
    trying to measure)? If not, N_REPEATS is too low to trust the rest."""
    mean_within_std = agg["p_met_std"].mean()
    p0 = agg[agg["formulation"] == "P0"].set_index("pair_id")["p_met"]
    shifts = []
    for probe in ("A", "B", "C", "D"):
        probe_p = agg[agg["formulation"] == probe].set_index("pair_id")["p_met"]
        shifts.append((probe_p - p0).abs().mean())
    mean_between_shift = sum(shifts) / len(shifts)
    print(
        f"repeat noise check: mean within-formulation std = {mean_within_std:.3f} "
        f"(from {agg['n_repeats'].iloc[0]} repeats) vs. mean |P0 - probe| shift = "
        f"{mean_between_shift:.3f}"
    )
    if mean_within_std >= mean_between_shift:
        print(
            "  WARNING: repeat noise is comparable to or larger than the between-formulation "
            "shift -- increase N_REPEATS in 03_jev_inference.py before trusting disagreement counts."
        )
    print()


def build_pair_table(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per pair_id: P0's own verdict/confidence/p_met, plus each
    probe's verdict/p_met, plus derived disagreement_count and correctness.
    `agg` is already one row per (pair_id, formulation) -- see aggregate_repeats().

    Pivoting p_met (float) and verdict (str) together in one pivot(values=[...])
    call silently demotes the float columns to object dtype -- arithmetic on
    them still "works" (elementwise on python floats), which is why nothing
    looked wrong until statsmodels did a strict numpy dtype check and failed
    on np.isfinite(object array). Pivoting them separately keeps real float64."""
    wide_p = agg.pivot(index="pair_id", columns="formulation", values="p_met")
    wide_v = agg.pivot(index="pair_id", columns="formulation", values="verdict")
    ground_truth = agg.drop_duplicates("pair_id").set_index("pair_id")["ground_truth"]

    out = pd.DataFrame(index=wide_p.index)
    out["ground_truth"] = ground_truth
    out["p0_p_met"] = wide_p["P0"].astype(float)
    out["p0_verdict"] = wide_v["P0"]
    out["p0_confidence"] = (2 * (out["p0_p_met"] - 0.5).abs()).clip(0, 1)
    out["p0_correct"] = out["p0_verdict"] == out["ground_truth"]

    for probe in ("A", "B", "C", "D"):
        out[f"{probe}_p_met"] = wide_p[probe].astype(float)
        out[f"{probe}_verdict"] = wide_v[probe]

    out["disagreement_count"] = sum(
        (out[f"{probe}_verdict"] != out["p0_verdict"]).astype(int) for probe in PARAPHRASE_PROBES
    )
    out["any_disagreement"] = out["disagreement_count"] > 0

    # C is a negation, so it naturally sits far from P0 -- project it onto the
    # same "P(this criterion is MET)" scale (1 - P(C)) before pooling it with
    # A/B as a genuine ensemble of equivalent estimates. D is excluded: it's a
    # one-directional subcriterion/implication-like relationship, not an equivalence, so it doesn't belong
    # in a spread/mean computed as if all members estimate the same quantity.
    out["C_proj_p_met"] = 1 - out["C_p_met"]
    ensemble = out[["A_p_met", "B_p_met", "C_proj_p_met"]]
    out["probe_mean"] = ensemble.mean(axis=1)
    out["probe_spread"] = ensemble.std(axis=1)
    out["probe_mean_distance"] = (out["probe_mean"] - 0.5).abs()
    out["probe_mean_verdict"] = np.where(out["probe_mean"] > 0.5, "MET", "UNMET")
    out["probe_mean_correct"] = out["probe_mean_verdict"] == out["ground_truth"]
    return out


def outcome_analysis(pairs: pd.DataFrame) -> None:
    print("=" * 70)
    print("OUTCOME ANALYSIS: does disagreement predict P0 correctness,")
    print("beyond P0's own confidence?")
    print("=" * 70)

    pairs["confidence_bucket"] = pd.cut(pairs["p0_confidence"], CONFIDENCE_BINS, labels=CONFIDENCE_LABELS)

    table = (
        pairs.groupby(["confidence_bucket", "any_disagreement"])
        .agg(n=("p0_correct", "size"), accuracy=("p0_correct", "mean"))
        .reset_index()
    )
    print(table.to_string(index=False))

    # The headline comparison: within the top confidence bucket only --
    # matched to the SAME boundary as the table above (pd.cut's bucket),
    # not a separately recomputed ">= 0.8" filter, which disagreed with
    # the table at the boundary (pd.cut's bins are right-inclusive, so a
    # pair at exactly 0.8 confidence falls in "0.6-0.8", not "0.8-1.0").
    high_conf = pairs[pairs["confidence_bucket"] == CONFIDENCE_LABELS[-1]]
    n_agree = (~high_conf["any_disagreement"]).sum()
    n_disagree = high_conf["any_disagreement"].sum()
    if n_agree and n_disagree:
        agree_acc = high_conf.loc[~high_conf["any_disagreement"], "p0_correct"].mean()
        disagree_acc = high_conf.loc[high_conf["any_disagreement"], "p0_correct"].mean()
        print(
            f"\nHigh-confidence ({CONFIDENCE_LABELS[-1]}) pairs: accuracy when probes agree = "
            f"{agree_acc:.3f} (n={n_agree}), when any probe disagrees = "
            f"{disagree_acc:.3f} (n={n_disagree})"
        )
        if min(n_agree, n_disagree) < LOW_N_THRESHOLD:
            print(f"-> too few pairs in one group (threshold={LOW_N_THRESHOLD}) to say anything from this split alone")
        else:
            print("-> H1 supported on this sample" if agree_acc > disagree_acc else "-> no support for H1 on this sample")
    else:
        print(
            f"\nHigh-confidence ({CONFIDENCE_LABELS[-1]}) pairs: n_agree={n_agree}, n_disagree={n_disagree} "
            "-- one group is empty, can't compare"
        )

    table["k"] = (table["accuracy"] * table["n"]).round().astype(int)
    table["ci_lo"], table["ci_hi"] = zip(*table.apply(lambda r: wilson_ci(r["k"], r["n"]), axis=1))

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = {label: i for i, label in enumerate(CONFIDENCE_LABELS)}
    offsets = {False: -0.04, True: 0.04}  # nudge the two series apart so error bars don't overlap
    for disagree, group in table.groupby("any_disagreement"):
        color = DISAGREE_COLOR if disagree else AGREE_COLOR
        label = "any probe disagrees" if disagree else "all probes agree"
        xs = [x_positions[str(b)] + offsets[disagree] for b in group["confidence_bucket"]]
        yerr_lo = group["accuracy"] - group["ci_lo"]
        yerr_hi = group["ci_hi"] - group["accuracy"]
        # thin low-n points instead of hiding them -- never silently drop a cell
        alphas = [1.0 if n >= LOW_N_THRESHOLD else 0.35 for n in group["n"]]
        for x, y, lo, hi, n, a in zip(xs, group["accuracy"], yerr_lo, yerr_hi, group["n"], alphas):
            ax.errorbar(x, y, yerr=[[lo], [hi]], fmt="o", color=color, alpha=a, capsize=3, markersize=7)
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 9), fontsize=8, color="#666666")
        ax.plot(xs, group["accuracy"], "-", color=color, alpha=0.5, linewidth=1.5, label=label)
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()))
    ax.set_xlabel("P0 confidence bucket")
    ax.set_ylabel("P0 accuracy vs. physician ground truth")
    ax.set_title("Does probe disagreement predict P0 errors, beyond confidence?")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    save_fig(fig, "accuracy_by_confidence_and_disagreement")

    counts = pairs["disagreement_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=NEUTRAL_COLOR, width=0.6)
    for bar, count in zip(bars, counts.values):
        ax.annotate(
            str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9,
        )
    ax.set_xlabel(f"number of paraphrase probes (of {','.join(PARAPHRASE_PROBES)}) disagreeing with P0")
    ax.set_ylabel("number of pairs")
    ax.set_title("Distribution of probe disagreement")
    fig.tight_layout()
    save_fig(fig, "disagreement_count_histogram")


def formulation_summary(pairs: pd.DataFrame) -> None:
    """Plain descriptive stats, requested separately from the H0/H1 test:
    how is confidence distributed per formulation, how often does each
    probe's verdict match P0's (and the physician label), and how does our
    overall Jev accuracy compare to the paper's reported HealthBench
    number (a ballpark replication check, not an exact one -- different
    sample, different framing: we use Noul, the paper's headline HealthBench
    number is Jev Choice)."""
    print("=" * 70)
    print("FORMULATION SUMMARY")
    print("=" * 70)

    print("\nP(MET) distribution per formulation:")
    for f in ("P0", "A", "B", "C", "D"):
        p = pairs[f"{f}_p_met"] if f != "P0" else pairs["p0_p_met"]
        print(f"  {f}: mean={p.mean():.3f} std={p.std():.3f} min={p.min():.3f} max={p.max():.3f}")

    print("\nHow often does each probe's verdict match P0's (agreement rate):")
    for f in ("A", "B", "C", "D"):
        match_rate = (pairs[f"{f}_verdict"] == pairs["p0_verdict"]).mean()
        print(f"  {f} vs P0: {match_rate:.1%} agree" + ("  <- expected to DISAGREE (negation)" if f == "C" else ""))

    print("\nHow often does each formulation's OWN verdict match the physician label:")
    for f in ("P0", "A", "B", "C", "D"):
        verdict_col = "p0_verdict" if f == "P0" else f"{f}_verdict"
        acc = (pairs[verdict_col] == pairs["ground_truth"]).mean()
        print(f"  {f}: {acc:.1%}")

    overall_acc = pairs["p0_correct"].mean()
    try:
        from sklearn.metrics import cohen_kappa_score

        kappa = cohen_kappa_score(pairs["p0_verdict"], pairs["ground_truth"])
        kappa_str = f"{kappa:.3f}"
    except ImportError:
        kappa_str = "(scikit-learn not installed)"
    print(
        f"\nReplication check vs. Rao & Callison-Burch's reported HealthBench numbers "
        f"(ballpark only -- different sample, our Noul framing vs. their headline Jev Choice):\n"
        f"  this run:  P0 accuracy = {overall_acc:.1%}, Cohen's kappa = {kappa_str}, n={len(pairs)} pairs\n"
        f"  the paper: Jev Choice accuracy = 77.1%, Cohen's kappa = 0.54, n=406 pairs"
    )
    print()


def coherence_analysis(pairs: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("COHERENCE ANALYSIS: does Jev's own answers obey the logical")
    print("relationship each probe was built to have with P0? (no ground truth needed)")
    print("=" * 70)

    corr = pairs["p0_p_met"].corr(pairs["C_p_met"])
    print(f"\nProbe C (negation): correlation(P0 p_met, Probe C p_met) = {corr:.3f}  (expect negative)")

    met_mask = pairs["p0_verdict"] == "MET"
    implied_rate = (pairs.loc[met_mask, "D_p_met"] >= pairs.loc[met_mask, "p0_p_met"]).mean()
    print(
        f"Probe D (subcriterion/implication-like, scope not validated -- see audit below): "
        f"when P0=MET, P(D) >= P(P0) in {implied_rate:.1%} of pairs (n={met_mask.sum()}; "
        f"expect well above 50% if a full-scope implication relationship holds in Jev's answers)"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    # the reference line is a guide, not data -- it recedes in neutral gray;
    # the actual data carries the one identity color in the chart
    ax.plot([0, 1], [1, 0], "--", color=GUIDE_COLOR, linewidth=1.2, label="perfect negation (P(C)=1-P(P0))", zorder=1)
    ax.scatter(pairs["p0_p_met"], pairs["C_p_met"], alpha=0.25, s=14, color=AGREE_COLOR, edgecolors="none", zorder=2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("P0 P(MET)")
    ax.set_ylabel("Probe C P(MET)")
    ax.set_title(f"Negation coherence (n={len(pairs)}, r={corr:.2f})")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "negation_coherence_scatter")


def probe_behavior_exploration(pairs: pd.DataFrame) -> None:
    """Reframes the question from "does disagreement predict error?" (which
    the outcome analysis already answered: not reliably) to "what does
    disagreement actually measure?" Four checks, each testing a specific
    alternative explanation rather than the escalation hypothesis directly:

    1. Confidence-amplifier check: do CORRECT P0 verdicts show MORE spread
       across the probe ensemble (different phrasings converge on the same
       right answer via different paths) and INCORRECT verdicts show LESS
       (consistently, confidently wrong)? If so, spread predicts correctness
       in the OPPOSITE direction from the original hypothesis.
    2. The confidence x disagreement grid, with disagreement as a CONTINUOUS
       spread (not a binary verdict-flip) -- the binary version is why the
       0.8-1.0 bucket showed zero disagreement at all: a flip across 0.5 is
       a lossy way to measure instability when nothing is forced to cross
       that exact line. The money cell is high-confidence x high-spread:
       if THAT'S accurate, the original escalation idea is strongly
       falsified, not just unsupported.
    3. Mean-vs-spread baseline: is accuracy really tracking spread, or is it
       tracking the ensemble MEAN (effectively a free 3-way ensemble vote),
       which would mean "disagreement" was never the active ingredient.
    4. Per-probe-type matrix: accuracy when THAT SPECIFIC probe agrees vs.
       disagrees with P0, kept separate instead of collapsed into one
       "any disagreement" number -- paraphrase-type and logic-type probes
       may behave in opposite directions, which a collapsed number hides.
    """
    print("\n" + "=" * 70)
    print("PROBE BEHAVIOR EXPLORATION: what does disagreement actually measure?")
    print("=" * 70)

    print("\n[1] Confidence-amplifier check: probe spread, correct vs. incorrect P0 verdicts")
    for correct, label in ((True, "P0 correct"), (False, "P0 incorrect")):
        subset = pairs[pairs["p0_correct"] == correct]
        print(f"  {label} (n={len(subset)}): mean probe spread (std of A,B,1-C) = {subset['probe_spread'].mean():.3f}")

    print("\n[2] Confidence x disagreement grid (disagreement = continuous probe spread, median split)")
    median_spread = pairs["probe_spread"].median()
    pairs["spread_level"] = np.where(pairs["probe_spread"] >= median_spread, "high", "low")
    pairs["confidence_bucket"] = pd.cut(pairs["p0_confidence"], CONFIDENCE_BINS, labels=CONFIDENCE_LABELS)
    grid = (
        pairs.groupby(["confidence_bucket", "spread_level"])
        .agg(n=("p0_correct", "size"), accuracy=("p0_correct", "mean"))
        .reset_index()
    )
    print(grid.pivot(index="confidence_bucket", columns="spread_level", values=["n", "accuracy"]).to_string())
    high_conf_high_spread = pairs[(pairs["confidence_bucket"] == CONFIDENCE_LABELS[-1]) & (pairs["spread_level"] == "high")]
    high_conf_low_spread = pairs[(pairs["confidence_bucket"] == CONFIDENCE_LABELS[-1]) & (pairs["spread_level"] == "low")]
    print(
        f"\n  the money cell -- high confidence + high spread: "
        f"n={len(high_conf_high_spread)}, accuracy={high_conf_high_spread['p0_correct'].mean():.1%}"
        if len(high_conf_high_spread)
        else "\n  the money cell -- high confidence + high spread: n=0, empty even with continuous spread"
    )
    print(
        f"  high confidence + low spread:  n={len(high_conf_low_spread)}, "
        f"accuracy={high_conf_low_spread['p0_correct'].mean():.1%}"
        if len(high_conf_low_spread)
        else "  high confidence + low spread: n=0"
    )

    print("\n[3] Mean-vs-spread baseline: which one is actually doing the work?")
    print(f"  corr(probe_spread, probe_mean_distance_from_0.5) = {pairs['probe_spread'].corr(pairs['probe_mean_distance']):.3f}")
    print(f"  P0 accuracy alone:                        {pairs['p0_correct'].mean():.1%}")
    print(f"  3-probe ensemble MEAN accuracy (A,B,1-C):  {pairs['probe_mean_correct'].mean():.1%}")
    median_dist = pairs["probe_mean_distance"].median()
    for dist_level, dist_mask in (("far from 0.5 (confident mean)", pairs["probe_mean_distance"] >= median_dist),
                                   ("near 0.5 (unsure mean)", pairs["probe_mean_distance"] < median_dist)):
        sub = pairs[dist_mask]
        by_spread = sub.groupby("spread_level")["p0_correct"].agg(["mean", "size"])
        print(f"  mean {dist_level} (n={len(sub)}): accuracy by spread level ->")
        for level, row in by_spread.iterrows():
            print(f"    spread={level}: accuracy={row['mean']:.1%} (n={int(row['size'])})")

    print("\n[4] Per-probe-type matrix: accuracy when THAT probe agrees vs. disagrees with P0")
    print(f"  {'probe':<6} {'relationship':<14} {'acc. agree':>12} {'(n)':>6} {'acc. disagree':>15} {'(n)':>6}")
    relationships = {"A": "paraphrase", "B": "paraphrase", "C": "negation", "D": "subcriterion*"}
    for probe, rel in relationships.items():
        matches = pairs[f"{probe}_verdict"] == pairs["p0_verdict"]
        agree_acc = pairs.loc[matches, "p0_correct"].mean()
        disagree_acc = pairs.loc[~matches, "p0_correct"].mean()
        print(
            f"  {probe:<6} {rel:<14} {agree_acc:>11.1%} {matches.sum():>6} "
            f"{disagree_acc:>14.1%} {(~matches).sum():>6}"
        )
    print("  * generated as an attempted implication; audit found most are really a narrower subcriterion -- see audit_d_scope_and_filtered_rerun()")
    print()


def check_compound_criterion_confound(pairs: pd.DataFrame) -> None:
    """Probe D for a multi-step criterion ("Step 1: ...? Step 2: ...?") can
    end up as a paraphrase of just ONE step rather than a consequence of the
    whole compound criterion -- confirmed by inspecting real examples. If
    that scope mismatch were driving the D-disagreement effect, the effect
    should be much larger for compound criteria than simple ones. It isn't
    (checked below), which is evidence against "it's just an artifact of
    partial-scope extraction on multi-step criteria" -- not proof the probe
    is a perfect equivalence, but evidence the effect isn't purely that."""
    print("\n[Confound check] Is the D-disagreement effect just an artifact of")
    print("probe D testing one step of a multi-step criterion, not the whole thing?")
    pairs_raw = load_jsonl(PAIRS_PATH)
    p0_by_pair = {p["pair_id"]: p["p0"] for p in pairs_raw}
    is_compound = pairs.index.to_series().map(lambda pid: "Step 1" in p0_by_pair[pid])

    for compound_val, label in ((True, "compound (multi-step)"), (False, "simple (single-clause)")):
        mask = is_compound == compound_val
        d_disagree = pairs["D_verdict"] != pairs["p0_verdict"]
        agree = pairs.loc[mask & ~d_disagree, "p0_correct"]
        disagree = pairs.loc[mask & d_disagree, "p0_correct"]
        print(
            f"  {label} (n={mask.sum()}): D agrees acc={agree.mean():.1%} (n={len(agree)}), "
            f"D disagrees acc={disagree.mean():.1%} (n={len(disagree)})"
        )
    print()


def model_comparison_analysis(pairs: pd.DataFrame) -> None:
    """The decisive test, per the reframed hypothesis:
    H1: logical inconsistency between semantically related probes predicts
        judgment error, BEYOND the model's reported confidence.
    Three nested logistic regressions predicting P0 error (1=wrong):
        Model 1: error ~ confidence
        Model 2: error ~ D_disagreement
        Model 3: error ~ confidence + D_disagreement
    If Model 3 doesn't improve on Model 1, D adds nothing confidence didn't
    already have. Continuous confidence throughout -- no arbitrary buckets.
    """
    import statsmodels.api as sm
    from scipy.stats import chi2
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    print("\n" + "=" * 70)
    print("MODEL COMPARISON: does D_disagreement add information beyond confidence?")
    print("=" * 70)

    y = (~pairs["p0_correct"]).astype(int).values  # 1 = error
    confidence = pairs["p0_confidence"].values
    d_disagree = (pairs["D_verdict"] != pairs["p0_verdict"]).astype(int).values

    X1 = sm.add_constant(confidence)
    X2 = sm.add_constant(d_disagree)
    X3 = sm.add_constant(np.column_stack([confidence, d_disagree]))

    m1 = sm.Logit(y, X1).fit(disp=0)
    m2 = sm.Logit(y, X2).fit(disp=0)
    m3 = sm.Logit(y, X3).fit(disp=0)

    results = {}
    for name, m, X in (("Model 1 (confidence only)", m1, X1), ("Model 2 (D_disagree only)", m2, X2), ("Model 3 (both)", m3, X3)):
        pred = m.predict(X)
        results[name] = pred
        print(
            f"\n{name}: llf={m.llf:.2f}\n"
            f"  AUROC={roc_auc_score(y, pred):.3f}  AUPRC={average_precision_score(y, pred):.3f}  "
            f"log_loss={log_loss(y, pred):.3f}  Brier={brier_score_loss(y, pred):.3f}"
        )

    lr_stat = 2 * (m3.llf - m1.llf)
    p_value = chi2.sf(lr_stat, df=1)
    print(
        f"\nLikelihood-ratio test, Model 3 vs. Model 1 (does adding D_disagreement help?):\n"
        f"  LR statistic={lr_stat:.3f}, df=1, p={p_value:.5f}"
    )
    print(f"  D_disagreement coefficient in Model 3: {m3.params[2]:.3f} (p={m3.pvalues[2]:.5f})")

    print("\nBootstrap (2000 resamples, refit each time): AUROC(Model 3) - AUROC(Model 1)")
    rng = np.random.RandomState(20261003)
    n = len(y)
    diffs = []
    for _ in range(2000):
        idx = rng.randint(0, n, n)
        yb, cb, db = y[idx], confidence[idx], d_disagree[idx]
        if len(np.unique(yb)) < 2:
            continue
        X1b, X3b = sm.add_constant(cb), sm.add_constant(np.column_stack([cb, db]))
        try:
            m1b = sm.Logit(yb, X1b).fit(disp=0)
            m3b = sm.Logit(yb, X3b).fit(disp=0)
            diffs.append(roc_auc_score(yb, m3b.predict(X3b)) - roc_auc_score(yb, m1b.predict(X1b)))
        except Exception:
            continue
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  n valid resamples={len(diffs)}, mean diff={diffs.mean():.3f}, 95% CI=[{lo:.3f}, {hi:.3f}]")
    print("  (CI excludes 0 -> Model 3 reliably beats Model 1 on this sample)" if lo > 0 else "  (CI includes 0 -> not conclusive)")
    print()


def standalone_and_direction_analysis(pairs: pd.DataFrame) -> None:
    """Resolves the critical ambiguity: is D just a BETTER classifier on its
    own (interpretation B: "use D instead of P0", not a relational signal),
    or does the P0/D RELATIONSHIP carry information neither has alone
    (interpretation A: a genuine consistency diagnostic)? If D standalone
    already matches the combined model, that's B. If D standalone is much
    weaker than confidence+D combined, that's A.

    Also replaces the binary agree/disagree with the continuous SIGNED
    difference (D_p_met - P0_p_met) to check whether direction matters:
    P0=.20/D=.90 and P0=.90/D=.20 both have |diff|=.70 but may not carry
    the same information.
    """
    import statsmodels.api as sm
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    print("=" * 70)
    print("STANDALONE vs. RELATIONAL: is D just a better classifier, or is the")
    print("P0/D relationship doing something neither one does alone?")
    print("=" * 70)

    y = (~pairs["p0_correct"]).astype(int).values
    p0_raw = pairs["p0_p_met"].astype(float).values
    d_raw = pairs["D_p_met"].astype(float).values
    confidence = pairs["p0_confidence"].astype(float).values
    diff_signed = d_raw - p0_raw
    diff_abs = np.abs(diff_signed)

    def fit_report(name, X):
        X = sm.add_constant(X)
        m = sm.Logit(y, X).fit(disp=0)
        pred = m.predict(X)
        print(
            f"  {name:<42} AUROC={roc_auc_score(y, pred):.3f}  AUPRC={average_precision_score(y, pred):.3f}  "
            f"Brier={brier_score_loss(y, pred):.3f}"
        )
        return m, pred

    print("\n[Standalone predictors of P0 error -- is D alone already a strong classifier?]")
    fit_report("P0 confidence alone (Model 1 again)", confidence)
    fit_report("P0 raw probability alone", p0_raw)
    fit_report("D raw probability alone", d_raw)
    fit_report("D's own confidence alone (|D-0.5|)", np.abs(d_raw - 0.5))

    print("\n[Joint continuous models]")
    fit_report("P0 raw + D raw (both probabilities)", np.column_stack([p0_raw, d_raw]))
    fit_report("confidence + signed diff (D - P0)", np.column_stack([confidence, diff_signed]))
    fit_report("confidence + |diff| (magnitude only)", np.column_stack([confidence, diff_abs]))

    print("\n[Does direction matter? signed diff vs. |diff|, alone]")
    fit_report("signed diff (D - P0) alone", diff_signed)
    fit_report("|diff| alone", diff_abs)

    print(
        "\n  Interpretation: if 'D raw probability alone' is close to the combined models' AUROC, "
        "D is simply a better classifier (use its answer directly). If it's much weaker than the "
        "joint/combined models, the P0-D RELATIONSHIP carries information neither has alone."
    )
    print()


def cross_validated_comparison(pairs: pd.DataFrame, n_folds: int = 5, n_repeats: int = 20) -> None:
    """Models 1-3 so far were fit and evaluated on the SAME 397 pairs -- that
    shows association in this sample, not that it generalizes out of sample.
    Repeated stratified k-fold: fit on train folds only, predict on the
    held-out fold, pool every fold's held-out predictions (never evaluated
    on data their own fold's model was trained on), report AUROC on the
    pooled out-of-fold predictions, averaged over n_repeats different fold
    splits to reduce split-to-split noise at this sample size."""
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    print("=" * 70)
    print(f"CROSS-VALIDATED COMPARISON ({n_folds}-fold stratified, {n_repeats} repeats, pooled out-of-fold)")
    print("=" * 70)

    y = (~pairs["p0_correct"]).astype(int).values
    confidence = pairs["p0_confidence"].astype(float).values
    d_disagree = (pairs["D_verdict"] != pairs["p0_verdict"]).astype(int).values

    aurocs_m1, aurocs_m3 = [], []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=20261003 + repeat)
        oof_m1, oof_m3, oof_y = [], [], []
        for train_idx, test_idx in skf.split(confidence, y):
            X1_train = sm.add_constant(confidence[train_idx])
            X3_train = sm.add_constant(np.column_stack([confidence[train_idx], d_disagree[train_idx]]))
            X1_test = sm.add_constant(confidence[test_idx], has_constant="add")
            X3_test = sm.add_constant(np.column_stack([confidence[test_idx], d_disagree[test_idx]]), has_constant="add")
            m1 = sm.Logit(y[train_idx], X1_train).fit(disp=0)
            m3 = sm.Logit(y[train_idx], X3_train).fit(disp=0)
            oof_m1.extend(m1.predict(X1_test))
            oof_m3.extend(m3.predict(X3_test))
            oof_y.extend(y[test_idx])
        aurocs_m1.append(roc_auc_score(oof_y, oof_m1))
        aurocs_m3.append(roc_auc_score(oof_y, oof_m3))

    aurocs_m1, aurocs_m3 = np.array(aurocs_m1), np.array(aurocs_m3)
    print(f"\nOut-of-fold AUROC, Model 1 (confidence only):        mean={aurocs_m1.mean():.3f} (sd={aurocs_m1.std():.3f})")
    print(f"Out-of-fold AUROC, Model 3 (confidence + D_disagree): mean={aurocs_m3.mean():.3f} (sd={aurocs_m3.std():.3f})")
    diff = aurocs_m3 - aurocs_m1
    print(
        f"Paired difference across {n_repeats} repeats: mean={diff.mean():.3f}, "
        f"95% range=[{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]"
    )
    print(
        "-> improvement survives held-out evaluation, not just in-sample fit"
        if diff.mean() > 0 and np.percentile(diff, 2.5) > 0
        else "-> improvement SHRINKS or disappears out-of-fold -- in-sample result may not generalize"
    )
    print()


def paired_error_table(pairs: pd.DataFrame) -> None:
    """Separates the 71 D-disagreement pairs into the phenomenon we care
    about (D disagrees AND P0 is wrong) vs. where the signal fails (D
    disagrees but P0 was right anyway) -- with the actual P0/D criterion
    text for a sample of each, not just the aggregate counts."""
    print("\n" + "=" * 70)
    print("PAIRED ERROR TABLE: D-disagreement cases, split by whether P0 was actually wrong")
    print("=" * 70)

    pairs_raw = load_jsonl(PAIRS_PATH)
    p0_by_pair = {p["pair_id"]: p["p0"] for p in pairs_raw}
    probes_by_p0 = {p["p0"]: p["probes"]["probe_d"] for p in load_jsonl(PROBES_PATH)}

    d_disagree = pairs["D_verdict"] != pairs["p0_verdict"]
    group_wrong = pairs[d_disagree & ~pairs["p0_correct"]]
    group_correct = pairs[d_disagree & pairs["p0_correct"]]
    print(f"\nD disagrees + P0 WRONG (the phenomenon):   n={len(group_wrong)}")
    print(f"D disagrees + P0 was actually CORRECT (signal fails): n={len(group_correct)}")

    for label, group in (("D disagrees + P0 WRONG", group_wrong), ("D disagrees + P0 correct (fails)", group_correct)):
        print(f"\n-- {label}, up to 3 examples --")
        for pid in group.index[:3]:
            p0_text = p0_by_pair[pid]
            print(f"  P0 (truncated): {p0_text[:140]}")
            print(f"  D  (truncated): {probes_by_p0[p0_text][:140]}")
            print(
                f"  P(P0)={pairs.loc[pid, 'p0_p_met']:.2f}  P(D)={pairs.loc[pid, 'D_p_met']:.2f}  "
                f"gold={pairs.loc[pid, 'ground_truth']}"
            )
    print()


SCOPE_AUDIT_PATH = ROOT / "data" / "healthbench" / "d_scope_audit.jsonl"

SCOPE_AUDIT_PROMPT = """\
Original criterion (P0):
\"\"\"{p0}\"\"\"

Derived statement (D), intended as a logical consequence of P0 ("if P0 is \
satisfied, then D holds"):
\"\"\"{d}\"\"\"

Does D represent a consequence of the ENTIRE P0 criterion, or does it only \
cover PART of a compound requirement (e.g. P0 has multiple steps/clauses \
joined by "and", but D only follows from one of them)?

Respond with exactly one label:
- "full_scope": D is a consequence of everything P0 requires.
- "partial_scope": D only follows from one clause of a multi-part P0.
- "invalid": D is not actually implied by P0 at all.

Respond with ONLY the label, no other text."""


def audit_d_scope_and_filtered_rerun(pairs: pd.DataFrame) -> None:
    """The methodological check that has to happen before any write-up: is
    Probe D a full-scope consequence of P0, or does it often only cover one
    clause of a compound criterion (confirmed as a real failure mode by
    inspecting examples earlier)? Classifies the (P0, D) text pairs behind
    the D-disagreement cases -- only 23 unique criteria, not all 71 pairs,
    since many pairs share the same criterion text -- then re-checks
    whether the accuracy gap survives when scope-mismatched/invalid
    criteria are excluded entirely."""
    import os

    import anthropic
    from dotenv import load_dotenv

    print("=" * 70)
    print("SCOPE AUDIT: is D a full consequence of P0, or does it only cover")
    print("part of a compound criterion? (classifying unique criteria, not pairs)")
    print("=" * 70)

    pairs_raw = load_jsonl(PAIRS_PATH)
    p0_by_pair = {p["pair_id"]: p["p0"] for p in pairs_raw}
    probes_by_p0 = {p["p0"]: p["probes"]["probe_d"] for p in load_jsonl(PROBES_PATH)}

    d_disagree_mask = pairs["D_verdict"] != pairs["p0_verdict"]
    disagreeing_p0_texts = {p0_by_pair[pid] for pid in pairs.index[d_disagree_mask]}
    print(f"\n{len(disagreeing_p0_texts)} unique criteria behind {d_disagree_mask.sum()} D-disagreement pairs")

    cached = {r["p0"]: r["scope"] for r in load_jsonl(SCOPE_AUDIT_PATH)} if SCOPE_AUDIT_PATH.exists() else {}
    to_classify = [p0 for p0 in disagreeing_p0_texts if p0 not in cached]
    if to_classify:
        load_dotenv(ENV_PATH)
        api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        with SCOPE_AUDIT_PATH.open("a", encoding="utf-8") as out:
            for p0 in to_classify:
                msg = client.messages.create(
                    model="claude-sonnet-5",
                    max_tokens=16,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": SCOPE_AUDIT_PROMPT.format(p0=p0, d=probes_by_p0[p0])}],
                )
                scope = next(b.text for b in msg.content if b.type == "text").strip().strip('"').lower()
                cached[p0] = scope
                out.write(json.dumps({"p0": p0, "scope": scope}, ensure_ascii=False) + "\n")
                out.flush()

    from collections import Counter

    print("\nScope classification of the 23 criteria:", dict(Counter(cached[p0] for p0 in disagreeing_p0_texts)))

    pairs = pairs.copy()
    pairs["p0_text"] = [p0_by_pair[pid] for pid in pairs.index]
    pairs["d_scope"] = pairs["p0_text"].map(lambda p0: cached.get(p0, "full_scope"))

    clean = pairs[pairs["d_scope"] == "full_scope"]
    d_disagree_clean = clean["D_verdict"] != clean["p0_verdict"]
    print(f"\nFiltered to full_scope criteria only: n={len(clean)} pairs (of {len(pairs)})")
    agree_acc = clean.loc[~d_disagree_clean, "p0_correct"].mean()
    disagree_acc = clean.loc[d_disagree_clean, "p0_correct"].mean()
    print(
        f"  D agrees:    acc={agree_acc:.1%} (n={(~d_disagree_clean).sum()})\n"
        f"  D disagrees: acc={disagree_acc:.1%} (n={d_disagree_clean.sum()})"
    )
    print(
        "-> effect survives excluding scope-mismatched/invalid criteria"
        if d_disagree_clean.sum() and disagree_acc < agree_acc - 0.1
        else "-> effect weakens substantially once scope-mismatched criteria are excluded"
    )
    print()


def main() -> None:
    if not JUDGMENTS_PATH.exists():
        raise SystemExit(f"Missing {JUDGMENTS_PATH} -- run 03_jev_inference.py first (needs TYPESAFE_API_KEY).")

    df = load_judgments()
    agg = aggregate_repeats(df)
    report_repeat_noise(df, agg)

    pairs = build_pair_table(agg)
    print(f"loaded {len(pairs)} pairs with all 5 formulations judged\n")

    formulation_summary(pairs)
    outcome_analysis(pairs)
    coherence_analysis(pairs)
    probe_behavior_exploration(pairs)
    check_compound_criterion_confound(pairs)
    model_comparison_analysis(pairs)
    standalone_and_direction_analysis(pairs)
    cross_validated_comparison(pairs)
    paired_error_table(pairs)
    audit_d_scope_and_filtered_rerun(pairs)


if __name__ == "__main__":
    main()
