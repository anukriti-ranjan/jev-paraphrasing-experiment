#!/usr/bin/env python3
"""Jev inference on atomic sub-requirements, for the decomposition experiment.

Reads:  ../data/healthbench/atomic_decomposition.jsonl  (frozen -- see 06's docstring)
        ../data/healthbench/pairs.jsonl
Writes: ../data/healthbench/atomic_judgments.jsonl        (one row per pair x atomic_idx x repeat)

Only criteria with decomposition_scope == "complete" and n_atomic >= 2 are
used (8 of 30 in this run) -- the ones a second model confirmed actually
decompose into non-overlapping, jointly-equivalent atomic pieces, not just
plausible-sounding ones. P0's own answer is NOT re-asked here: it already
exists in jev_judgments.jsonl from 03_jev_inference.py, and 08's analysis
joins against it. Same isolation rule as before: one atomic requirement
per call, same state, never co-visible with its siblings or with P0.

RELIABILITY: identical approach to 03_jev_inference.py (SDK retry with a
larger-than-default budget, fail-fast on configuration errors, a
consecutive-failure circuit breaker, inter-call pacing, a persistent error
log) -- see that script's docstring for the reasoning; not repeated here.

Credentials: TYPESAFE_API_KEY, read from a .env file in the project root.

Usage:
    .venv/bin/python3 07_atomic_jev_inference.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typesafe_sdk as ts
from dotenv import load_dotenv

STUDY_DIR = Path(__file__).parent
ROOT = STUDY_DIR.parent
DECOMPOSITION_PATH = ROOT / "data" / "healthbench" / "atomic_decomposition.jsonl"
PAIRS_PATH = ROOT / "data" / "healthbench" / "pairs.jsonl"
OUT_PATH = ROOT / "data" / "healthbench" / "atomic_judgments.jsonl"
ERRORS_PATH = ROOT / "data" / "healthbench" / "atomic_errors.jsonl"
ENV_PATH = ROOT / ".env"

MODEL = "jev-latest"
DECISION_THRESHOLD = 0.5
N_REPEATS = 3  # same reasoning as 03_jev_inference.py: separate signal from Jev's own noise

RETRY = ts.RetryPolicy(max_retries=4)
INTER_CALL_DELAY_SECONDS = 0.1
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_COOLDOWN = 30.0

FATAL_ERROR_TYPES = (
    ts.TypeSafeAuthenticationError,
    ts.TypeSafeBadRequestError,
    ts.TypeSafePermissionDeniedError,
    ts.TypeSafeNotFoundError,
)

JEV_QUESTION_TEMPLATE = (
    "Determine whether this criterion is satisfied by the submission. "
    "Criterion: {criterion_text}"
)


def build_jev_question(criterion_text: str) -> str:
    return JEV_QUESTION_TEMPLATE.format(criterion_text=criterion_text)


def verdict_and_confidence(p_met: float) -> tuple[str, float]:
    verdict = "MET" if p_met > DECISION_THRESHOLD else "UNMET"
    confidence = max(0.0, min(1.0, 2 * abs(p_met - DECISION_THRESHOLD)))
    return verdict, confidence


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def already_done(out_path: Path) -> set[tuple[str, int, int]]:
    if not out_path.exists():
        return set()
    rows = load_jsonl(out_path)
    return {(r["pair_id"], r["atomic_idx"], r["repeat_idx"]) for r in rows}


def main() -> None:
    load_dotenv(ENV_PATH)
    client = ts.TypeSafeClient(model=MODEL, retry=RETRY)

    decomposition = load_jsonl(DECOMPOSITION_PATH)
    usable = [r for r in decomposition if r["decomposition_scope"] == "complete" and r["n_atomic"] >= 2]
    print(f"{len(decomposition)} decomposed criteria; {len(usable)} usable (complete, n_atomic>=2)")

    pairs_by_id = {p["pair_id"]: p for p in load_jsonl(PAIRS_PATH)}

    work = [
        (pairs_by_id[pid], criterion, atomic_idx, repeat_idx)
        for criterion in usable
        for pid in criterion["pair_ids"]
        for atomic_idx in range(criterion["n_atomic"])
        for repeat_idx in range(N_REPEATS)
    ]
    done = already_done(OUT_PATH)
    work = [w for w in work if (w[0]["pair_id"], w[2], w[3]) not in done]
    total_pairs = sum(len(c["pair_ids"]) for c in usable)
    print(
        f"{total_pairs} pairs covered; "
        f"{len(done)} (pair, atomic_idx, repeat) rows already done; {len(work)} to run"
    )

    consecutive_failures = 0
    with OUT_PATH.open("a", encoding="utf-8") as out, ERRORS_PATH.open("a", encoding="utf-8") as errout:
        for i, (pair, criterion, atomic_idx, repeat_idx) in enumerate(work):
            atomic_text = criterion["atomic_requirements"][atomic_idx]
            question_text = build_jev_question(atomic_text)
            state = {"input": pair["state_input"], "submission": pair["state_submission"]}
            questions = {"c0": ts.Noul(instructions=question_text)}
            try:
                response = client.system_one(state, questions)
                p_met = response.answers["c0"].noul
                verdict, confidence = verdict_and_confidence(p_met)
                record = {
                    "pair_id": pair["pair_id"],
                    "p0": criterion["p0"],
                    "atomic_idx": atomic_idx,
                    "n_atomic": criterion["n_atomic"],
                    "repeat_idx": repeat_idx,
                    "question_text": question_text,
                    "p_met": p_met,
                    "verdict": verdict,
                    "confidence": confidence,
                    "jev_model": response.model,
                    "ground_truth": pair["ground_truth"],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                consecutive_failures = 0
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(work)}")
            except FATAL_ERROR_TYPES as e:
                sys.exit(
                    f"FATAL: {type(e).__name__} on {pair['pair_id']} atomic {atomic_idx} "
                    f"repeat {repeat_idx}: {e}\nStopping instead of retrying {len(work) - i} "
                    f"more times; rerun after fixing to resume."
                )
            except Exception as e:
                consecutive_failures += 1
                errout.write(
                    json.dumps(
                        {
                            "pair_id": pair["pair_id"],
                            "atomic_idx": atomic_idx,
                            "repeat_idx": repeat_idx,
                            "error_type": type(e).__name__,
                            "error": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                errout.flush()
                print(f"  ERROR on {pair['pair_id']} atomic {atomic_idx} repeat {repeat_idx}: {e}")
                if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                    print(f"  {consecutive_failures} consecutive failures -- cooling down {CIRCUIT_BREAKER_COOLDOWN}s")
                    time.sleep(CIRCUIT_BREAKER_COOLDOWN)
                    consecutive_failures = 0

            time.sleep(INTER_CALL_DELAY_SECONDS)

    print(f"\nWrote/updated {OUT_PATH.relative_to(ROOT)}")
    if ERRORS_PATH.exists() and ERRORS_PATH.stat().st_size > 0:
        print(f"Non-fatal failures logged to {ERRORS_PATH.relative_to(ROOT)} -- rerun to retry them")


if __name__ == "__main__":
    main()
