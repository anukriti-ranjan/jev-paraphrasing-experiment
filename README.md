# Redundant decision probes as a Jev uncertainty signal

Tests whether asking Jev the same binary HealthBench criterion through
several reformulations gives information about correctness beyond Jev's
own stated confidence.

```
H0: probe disagreement carries no information about correctness beyond
    Jev's own confidence.
H1: it does, especially on verdicts where confidence is already high.
```

**Read [`EVOLUTION.md`](study/EVOLUTION.md) for the full story — what was tried,
what broke, what was found, and why the conclusion changed twice.** 
This README is just how to run the code.

## Pipeline

Run in order — each stage reads an earlier stage's output file.

| # | Script | Reads | Writes | Needs |
|---|---|---|---|---|
| 1 | `01_prepare_dataset.py` | `../data/raw/healthbench_meta.jsonl` | `../data/healthbench/pairs.jsonl`, `criteria.jsonl` | nothing |
| 2 | `02_paraphrase.py` | `criteria.jsonl` | `criterion_probes.jsonl` | `ANTHROPIC_AUTH_TOKEN` |
| 3 | `03_jev_inference.py` | `pairs.jsonl`, `criterion_probes.jsonl` | `jev_judgments.jsonl` | `TYPESAFE_API_KEY` |
| 4 | `04_analysis.py` | `jev_judgments.jsonl` | `../figures/*.png`, `*.pdf` + printed tables | nothing |
| 5 | `05_escalation.py` | `jev_judgments.jsonl`, `pairs.jsonl` | `escalation_judgments.jsonl` | `ANTHROPIC_AUTH_TOKEN` |
| 6 | `06_decompose_criteria.py` | `criteria.jsonl` | `atomic_decomposition.jsonl` | `ANTHROPIC_AUTH_TOKEN` |
| 7 | `07_atomic_jev_inference.py` | `atomic_decomposition.jsonl`, `pairs.jsonl` | `atomic_judgments.jsonl` | `TYPESAFE_API_KEY` |
| 8 | `08_atomic_analysis.py` | `atomic_judgments.jsonl`, `jev_judgments.jsonl` | printed tables only | nothing |
| — | `09_blog_figures.py` | the above outputs | `../figures/*.png`, `*.pdf` | nothing |

Stage 9 is presentation only — it re-renders a few results from Stages 4
and 8 as the specific figures `BLOG_POST.md` embeds. It's not part of the
experiment itself.

Every script is independently resumable: rerunning it skips whatever rows
are already in its output file, so a partial failure never costs you the
whole batch.

## Setup

```bash
git clone https://github.com/delip/autorubric    # into jevVsllms/, alongside study/
cd study
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python3 anthropic python-dotenv matplotlib pandas numpy statsmodels scikit-learn
uv pip install --python .venv/bin/python3 -e "../autorubric[typesafe]"
```

Credentials, read from a `.env` file in the project root (one level
above `study/`):

```
ANTHROPIC_AUTH_TOKEN=...   # Stages 2, 5, 6
TYPESAFE_API_KEY=...       # Stages 3, 7
TYPESAFE_BASE_URL=...      # optional, only if not using TypeSafe's default endpoint
```

## Running it

```bash
mkdir -p ../data/raw
curl -o ../data/raw/healthbench_meta.jsonl \
  https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_meta_eval.jsonl

cd study
.venv/bin/python3 01_prepare_dataset.py
.venv/bin/python3 02_paraphrase.py
.venv/bin/python3 03_jev_inference.py
.venv/bin/python3 04_analysis.py
.venv/bin/python3 05_escalation.py
.venv/bin/python3 06_decompose_criteria.py
.venv/bin/python3 07_atomic_jev_inference.py
.venv/bin/python3 08_atomic_analysis.py
.venv/bin/python3 09_blog_figures.py
```

1. **Builds the pairs dataset locally.** Groups HealthBench's raw
   physician meta-eval rows into units and pairs, drops tied-vote pairs,
   takes a fixed-seed random sample (397 pairs / 200 units / 30 unique
   criteria). Deterministic — rerunning with the same seed reproduces the
   same files byte-for-byte.
2. **Generates and validates four probes per criterion** (direct
   predicate, paraphrase, negation, a fourth probe originally intended as
   an implication — see `EVOLUTION.md` section 9 for why that label turned
   out to be too strong). One call produces all four variants; a second,
   different model classifies each probe's actual relationship to P0 and
   flags mismatches. Non-deterministic — rerunning produces different text
   and possibly different flags.
3. **Runs Jev** on every `(pair, formulation)`, 3 times each (5,955
   isolated calls) — same `state` every time, only the question text
   changes, and no call ever sees more than one phrasing of a criterion.
   The repeats exist because Jev's answers are documented as not
   bit-deterministic.
4. **Runs the hypothesis test**: accuracy of Jev's P0 verdict, split by
   confidence and by probe disagreement; a ground-truth-free coherence
   check; then a full statistical pass (standalone-vs-relational check,
   nested logistic models with AUROC/AUPRC/Brier, a likelihood-ratio test,
   bootstrap confidence intervals, and repeated cross-validation) once a
   real per-probe-type effect shows up.
5. **Tests disagreement as a practical escalation signal** — not required
   for H0/H1, which Stage 4 already answers from Jev + ground truth alone.
6. **Decomposes each criterion into its atomic AND-combined requirements**
   — frozen before any Jev data is touched, specifically so later stages
   can't select on an outcome this stage hasn't seen yet. A second model
   checks each decomposition is actually complete and non-overlapping; in
   this run, only 8 of 30 criteria passed cleanly.
7. **Runs Jev on each atomic requirement**, reusing P0's existing answer
   rather than re-asking it.
8. **Tests whether decomposition inconsistency predicts error**, against
   the same physician ground truth used throughout — same statistical
   rigor as Stage 4. On this run (113 pairs, 8 criteria), the result does
   not replicate Stage 4's finding; see `EVOLUTION.md` section 10 for why
   that's a useful result in itself, not just a smaller one.

Stage 2 makes ~150 Anthropic calls; Stage 3 is the long-running one at
~5,955 Jev calls. Both write incrementally, so it's safe to `Ctrl-C` and
resume later.

## Results

The short version: a naive "ask it four ways and count disagreement"
signal doesn't work. Breaking the probes apart by type surfaced a real,
statistically significant effect for one specific probe — confirmed with
cross-validation and a standalone-vs-relational check — but an audit
found that probe wasn't behaving the way it was supposed to, and a
cleaner replacement (explicit criterion decomposition, Stages 6–8) did
not reproduce the effect at the sample size available.

