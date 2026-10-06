#!/usr/bin/env python3
"""Dataset preparation: HealthBench's raw physician meta-eval -> Jev-ready pairs.

------------------------------------------------------------------------
1. WHAT THE ORIGINAL DATA LOOKS LIKE
------------------------------------------------------------------------
Source: ../data/raw/healthbench_meta.jsonl (OpenAI's public release, 29,511
lines). Each line is ONE PHYSICIAN-LABELED (unit, criterion) ROW:

    {
      "prompt_id": "...",            # identifies the conversation
      "completion_id": "...",        # identifies which model's answer to it
      "prompt": [{"role": "user", "content": "..."}, ...],
      "completion": "...",           # the model's answer -- what's being judged
      "rubric": "...",               # the criterion text, e.g. "The completion
                                      #   must be tailored to a health professional..."
      "category": "cluster:...",     # which reusable rule this is
      "binary_labels": [true, false],       # one bool per physician who rated THIS
      "anonymized_physician_ids": ["p1", "p2"],   # (unit, criterion) combination
    }

A UNIT = one (prompt_id, completion_id) pair -- one conversation plus one
specific model answer to it. A unit is usually rated against MORE THAN ONE
criterion, so the raw file has several rows per unit (one row per criterion
that was checked). A PAIR = one unit + one criterion + the physicians'
verdict on it -- the smallest thing anyone (physician, Jev, or us) ever
judges. 200 units in our sample produce 392 pairs because most units carry
2-3 criteria, not 1 (see SAMPLING below).

Ground truth per pair = majority vote over that criterion's `binary_labels`.
A tied vote (e.g. 1-of-2 physicians say yes) has no majority -- we mark it
CANNOT_ASSESS and drop it, because a tied pair has no real "correct answer"
to score Jev against. This matches autorubric's own convert_to_rubric_dataset.py.

------------------------------------------------------------------------
2. HOW THIS MAPS ONTO JEV'S INPUT FORMAT
------------------------------------------------------------------------
Jev takes, per call:
    state    = {"input": <the conversation>, "submission": <the completion>}
    question = "Determine whether this criterion is satisfied by the
                submission. Criterion: {criterion text}"

So: unit.prompt + unit.completion -> state (fixed per unit, never touched
by anything in this study). criterion.rubric -> the {criterion text} slot
of the question (the only thing paraphrasing ever changes). Everything a
pair needs to be sent to Jev -- prompt, submission, criterion text, ground
truth -- is written out flat, one row per pair, in build_jev_question()
and write_pairs() below.

------------------------------------------------------------------------
3. SAMPLING
------------------------------------------------------------------------
The clean pool (ties dropped) has 9,515 units. We take a FIXED-SEED RANDOM
sample of N_UNITS of them -- no stratification by theme or verdict class,
so the sample's MET/UNMET balance is whatever the pool's natural balance
is (checked empirically: ~85% MET / ~15% UNMET at N_UNITS=200).

------------------------------------------------------------------------
4. WHAT TO CHANGE FOR AN ABLATION
------------------------------------------------------------------------
  - SEED            : redraw a different random sample; check whether
                       findings (e.g. the agree/disagree accuracy gap)
                       hold across seeds, not just this one draw.
  - N_UNITS          : scale the sample up/down. Note: the number of
                       UNIQUE criterion texts will NOT scale with this --
                       it saturates near ~37 (the full consensus-criteria
                       catalog) well before 200 units, so paraphrasing
                       cost stays flat regardless of N_UNITS.
  - DROP_CANNOT_ASSESS: set False to keep tied-vote pairs as a third class
                       instead of dropping them -- turns this into a
                       3-way (MET/UNMET/CANNOT_ASSESS) study instead of
                       binary, closer to what Jev Choice (not Noul) would
                       need.
  - MIN_N_PHYSICIANS : raise from 1 to e.g. 3 to keep only pairs with
                       higher-confidence ground truth (fewer, cleaner
                       labels) -- trades sample size for label quality.
  - stratify_by_verdict(): swap sample_units() for this (not yet written
                       here) to force an even MET/UNMET split instead of
                       the pool's natural ~85/15 skew, the same fix
                       sample_healthbench_100.py in ../autorubric/health_bench
                       uses -- needed if the natural skew leaves too few
                       UNMET pairs to detect anything on.

Usage:
    python3 01_prepare_dataset.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).parent
ROOT = STUDY_DIR.parent
RAW_META_PATH = ROOT / "data" / "raw" / "healthbench_meta.jsonl"
OUT_DIR = ROOT / "data" / "healthbench"
PAIRS_PATH = OUT_DIR / "pairs.jsonl"
CRITERIA_PATH = OUT_DIR / "criteria.jsonl"

SEED = 20260929
N_UNITS = 200
DROP_CANNOT_ASSESS = True
MIN_N_PHYSICIANS = 1  # raise to e.g. 3 for a higher-confidence-labels ablation

JEV_QUESTION_TEMPLATE = (
    "Determine whether this criterion is satisfied by the submission. "
    "Criterion: {criterion_text}"
)


def render_conversation(messages: list[dict[str, Any]]) -> str:
    """Byte-for-byte match with simple-evals' / autorubric's rendering."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def majority_verdict(labels: list[bool]) -> str:
    n, t = len(labels), sum(labels)
    if t * 2 > n:
        return "MET"
    if t * 2 < n:
        return "UNMET"
    return "CANNOT_ASSESS"


def build_jev_question(criterion_text: str) -> str:
    """The exact string Jev receives for a given criterion text."""
    return JEV_QUESTION_TEMPLATE.format(criterion_text=criterion_text)


@dataclass(frozen=True)
class PairRecord:
    """One (unit, criterion) pair, Jev-ready: everything a call needs."""

    pair_id: str  # f"{prompt_id}:{completion_id}:{criterion_name}"
    prompt_id: str
    completion_id: str
    criterion_name: str
    state_input: str  # Jev state.input      -- the rendered conversation
    state_submission: str  # Jev state.submission -- the model's completion
    p0: str  # the criterion text -- the only thing paraphrasing changes
    category: str
    n_physicians: int
    ground_truth: str  # "MET" | "UNMET"


@dataclass(frozen=True)
class UniqueCriterion:
    """One distinct P0 text and every pair_id that shares it."""

    p0: str
    category: str
    pair_ids: tuple[str, ...]


def load_raw_meta(path: Path = RAW_META_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    drop_cannot_assess: bool = DROP_CANNOT_ASSESS,
    min_n_physicians: int = MIN_N_PHYSICIANS,
) -> list[PairRecord]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["prompt_id"], row["completion_id"])].append(row)

    pairs = []
    for (prompt_id, completion_id), unit_rows in grouped.items():
        first = unit_rows[0]
        state_input = render_conversation(first["prompt"])
        state_submission = first["completion"]
        unit_pairs = []
        for i, r in enumerate(unit_rows):
            if len(r["binary_labels"]) < min_n_physicians:
                continue
            verdict = majority_verdict(r["binary_labels"])
            if drop_cannot_assess and verdict == "CANNOT_ASSESS":
                unit_pairs = None  # ties contaminate the whole unit -- drop it
                break
            unit_pairs.append(
                PairRecord(
                    pair_id=f"{prompt_id}:{completion_id}:C{i + 1}",
                    prompt_id=prompt_id,
                    completion_id=completion_id,
                    criterion_name=f"C{i + 1}",
                    state_input=state_input,
                    state_submission=state_submission,
                    p0=r["rubric"],
                    category=r["category"],
                    n_physicians=len(r["binary_labels"]),
                    ground_truth=verdict,
                )
            )
        if unit_pairs:
            pairs.extend(unit_pairs)
    return pairs


def sample_units(pairs: list[PairRecord], n_units: int, seed: int) -> list[PairRecord]:
    """Fixed-seed random sample of N_UNITS units, keeping every pair of each
    sampled unit (not a per-pair sample -- units are the sampling frame)."""
    unit_ids = sorted({(p.prompt_id, p.completion_id) for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(unit_ids)
    chosen = set(unit_ids[:n_units])
    return [p for p in pairs if (p.prompt_id, p.completion_id) in chosen]


def dedupe_criteria(pairs: list[PairRecord]) -> list[UniqueCriterion]:
    by_p0: dict[str, list[PairRecord]] = defaultdict(list)
    for pair in pairs:
        by_p0[pair.p0].append(pair)
    return [
        UniqueCriterion(p0=p0, category=members[0].category, pair_ids=tuple(m.pair_id for m in members))
        for p0, members in by_p0.items()
    ]


def write_jsonl(records: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def summarize(pairs: list[PairRecord], criteria: list[UniqueCriterion]) -> None:
    n_units = len({(p.prompt_id, p.completion_id) for p in pairs})
    met = sum(1 for p in pairs if p.ground_truth == "MET")
    print(f"units={n_units} pairs={len(pairs)} (MET={met}, UNMET={len(pairs) - met})")
    print(f"unique criterion texts={len(criteria)} (mean reuse x{len(pairs) / len(criteria):.1f})")
    print("\nexample Jev question (from pair 0):")
    print(f"  {build_jev_question(criteria[0].p0)[:160]}...")


def main() -> None:
    if not RAW_META_PATH.exists():
        raise SystemExit(f"Missing {RAW_META_PATH} -- download it first (see README).")

    rows = load_raw_meta()
    all_pairs = build_pairs(rows)
    sample = sample_units(all_pairs, N_UNITS, SEED)
    criteria = dedupe_criteria(sample)

    write_jsonl(sample, PAIRS_PATH)
    write_jsonl(criteria, CRITERIA_PATH)

    print(f"wrote {PAIRS_PATH.relative_to(ROOT)}")
    print(f"wrote {CRITERIA_PATH.relative_to(ROOT)}")
    print()
    summarize(sample, criteria)


if __name__ == "__main__":
    main()
