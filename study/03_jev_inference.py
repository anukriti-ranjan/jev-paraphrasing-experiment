#!/usr/bin/env python3
"""Jev inference: run P0 and all four probes through Jev, one isolated call each.

Reads:  ../data/healthbench/pairs.jsonl              (397 pairs)
        ../data/healthbench/criterion_probes.jsonl   (30 criteria x 4 probes)
Writes: ../data/healthbench/jev_judgments.jsonl       (one row per pair x formulation x repeat)

Each call sends ONE Noul question against the SAME state (pair's input +
submission, never altered) -- P0 or exactly one probe, never both, so no
call ever sees more than one phrasing of the same criterion (see
01_prepare_dataset.py section 2 and the earlier discussion of why
formulations must not be co-visible in one request).

    state     = {"input": pair.state_input, "submission": pair.state_submission}
    questions = {"c0": Noul(instructions=build_jev_question(formulation_text))}
    response  = client.system_one(state, questions)
    p_met     = response.answers["c0"].noul

verdict    = MET if p_met > 0.5, UNMET if p_met < 0.5, UNMET on an exact
             tie (all our criteria carry positive weight -- see
             convert_to_rubric_dataset.py's build_meta: weight=10 uniform --
             so UNMET is the correct conservative tie-break, matching the
             library's rule in autorubric's decision.py).
confidence = 2 * |p_met - 0.5|, clamped to [0, 1].

N_REPEATS -- WHY EACH (pair, formulation) IS CALLED MORE THAN ONCE:
autorubric's own docs state decision-model answers are "not bit-
deterministic: repeating an identical request can return slightly
different probabilities," and the paper's own stability check (5 repeated
passes over RiceChem) found ~3% of Jev's pairs flipped verdict at least
once. With only ONE call per formulation, a confidence difference between
P0 and a probe is indistinguishable from Jev's own run-to-run noise --
we'd have no way to tell "paraphrasing changed the answer" from "Jev is
just noisy." So every (pair, formulation) gets N_REPEATS independent
calls, each stored as its own row (never pre-averaged here -- Stage 4
computes the per-formulation mean and spread itself, so the raw repeats
stay inspectable). Compare the BETWEEN-FORMULATION difference in means
against the WITHIN-FORMULATION spread across repeats before calling
anything a real paraphrasing effect.

Credentials: TYPESAFE_API_KEY (+ optional TYPESAFE_BASE_URL,
TYPESAFE_DEFAULT_MODEL), read from a .env file in the project root (one
level above this script's directory):
    TYPESAFE_API_KEY=...

RELIABILITY, for an unattended ~6,000-call batch:
- Per-call retry is the SDK's own job, not hand-rolled here: TypeSafeClient
  retries 429/408/5xx/connection/timeout automatically with backoff and
  honours Retry-After (confirmed via typesafe_sdk.RetryPolicy's defaults).
  RETRY is passed explicitly below with a larger budget than the SDK's
  default (4 attempts, not 2) -- worth it for a long unattended run, where
  falling through to the slower script-level handling below is more
  costly than one extra retry.
- FAIL-FAST on configuration errors: TypeSafeAuthenticationError,
  TypeSafeBadRequestError, TypeSafePermissionDeniedError, and
  TypeSafeNotFoundError will recur identically on every remaining call
  (a bad key stays bad), so the first one aborts the whole run instead of
  burning through ~6,000 calls that would all fail the same way.
- CIRCUIT BREAKER for errors that survived the SDK's own retries
  (sustained rate-limiting, repeated timeouts): CIRCUIT_BREAKER_THRESHOLD
  consecutive failures triggers a CIRCUIT_BREAKER_COOLDOWN pause before
  continuing, since that pattern means the SDK's per-call backoff already
  wasn't enough -- hammering straight through it would not help.
- INTER_CALL_DELAY_SECONDS paces requests between calls, independent of
  retries, to make hitting a sustained rate limit less likely in the
  first place.
- Failures are appended to ERRORS_PATH (pair_id, formulation, repeat_idx,
  error type, message) -- not just printed -- since this is long enough
  to plausibly run detached from a terminal.

Resumable: already-written (pair_id, formulation, repeat_idx) rows are
skipped, so a rerun after a partial failure only fills in what's missing.

Usage:
    .venv/bin/python3 03_jev_inference.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Literal

import typesafe_sdk as ts
from dotenv import load_dotenv

STUDY_DIR = Path(__file__).parent
ROOT = STUDY_DIR.parent
PAIRS_PATH = ROOT / "data" / "healthbench" / "pairs.jsonl"
PROBES_PATH = ROOT / "data" / "healthbench" / "criterion_probes.jsonl"
OUT_PATH = ROOT / "data" / "healthbench" / "jev_judgments.jsonl"
ERRORS_PATH = ROOT / "data" / "healthbench" / "jev_errors.jsonl"
ENV_PATH = ROOT / ".env"

MODEL = "jev-latest"
DECISION_THRESHOLD = 0.5
N_REPEATS = 3  # independent calls per (pair, formulation), to separate signal from Jev's own noise
Formulation = Literal["P0", "A", "B", "C", "D"]

RETRY = ts.RetryPolicy(max_retries=4)  # SDK default is 2; worth more for a long unattended batch
INTER_CALL_DELAY_SECONDS = 0.1
CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive post-retry failures before pausing
CIRCUIT_BREAKER_COOLDOWN = 30.0

# Recur identically on every remaining call -- no point retrying 6,000 times.
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
    if p_met > DECISION_THRESHOLD:
        verdict = "MET"
    elif p_met < DECISION_THRESHOLD:
        verdict = "UNMET"
    else:
        verdict = "UNMET"  # exact tie, positive-weight criteria -> conservative worst case
    confidence = max(0.0, min(1.0, 2 * abs(p_met - DECISION_THRESHOLD)))
    return verdict, confidence


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_formulation_texts(probes_path: Path) -> dict[str, dict[Formulation, str]]:
    """p0 text -> {formulation: text}, with "P0" mapped to itself."""
    probe_rows = load_jsonl(probes_path)
    out = {}
    for row in probe_rows:
        probes = row["probes"]
        out[row["p0"]] = {
            "P0": row["p0"],
            "A": probes["probe_a"],
            "B": probes["probe_b"],
            "C": probes["probe_c"],
            "D": probes["probe_d"],
        }
    return out


def already_done(out_path: Path) -> set[tuple[str, str, int]]:
    if not out_path.exists():
        return set()
    rows = load_jsonl(out_path)
    return {(r["pair_id"], r["formulation"], r["repeat_idx"]) for r in rows}


def main() -> None:
    # load_dotenv only populates os.environ from the .env file -- it does the
    # file reading, nothing else. TypeSafeClient() below reads TYPESAFE_API_KEY
    # (required) and TYPESAFE_BASE_URL (optional) from os.environ itself, and
    # raises its own clear TypeSafeError at construction if the key is missing
    # -- confirmed directly against the SDK, so there's no need to duplicate
    # that check here.
    load_dotenv(ENV_PATH)
    client = ts.TypeSafeClient(model=MODEL, retry=RETRY)

    pairs = load_jsonl(PAIRS_PATH)
    formulations_by_p0 = build_formulation_texts(PROBES_PATH)
    missing_probes = {p["p0"] for p in pairs} - set(formulations_by_p0)
    if missing_probes:
        sys.exit(f"ERROR: {len(missing_probes)} criterion texts have no probes yet -- run 02_paraphrase.py first")

    done = already_done(OUT_PATH)
    work = [
        (pair, formulation, repeat_idx)
        for pair in pairs
        for formulation in ("P0", "A", "B", "C", "D")
        for repeat_idx in range(N_REPEATS)
        if (pair["pair_id"], formulation, repeat_idx) not in done
    ]
    total = len(pairs) * 5 * N_REPEATS
    print(
        f"{len(pairs)} pairs x 5 formulations x {N_REPEATS} repeats = {total} total; "
        f"{len(done)} done; {len(work)} to run"
    )

    consecutive_failures = 0
    with OUT_PATH.open("a", encoding="utf-8") as out, ERRORS_PATH.open("a", encoding="utf-8") as errout:
        for i, (pair, formulation, repeat_idx) in enumerate(work):
            question_text = build_jev_question(formulations_by_p0[pair["p0"]][formulation])
            state = {"input": pair["state_input"], "submission": pair["state_submission"]}
            questions = {"c0": ts.Noul(instructions=question_text)}
            try:
                response = client.system_one(state, questions)
                p_met = response.answers["c0"].noul
                verdict, confidence = verdict_and_confidence(p_met)
                record = {
                    "pair_id": pair["pair_id"],
                    "formulation": formulation,
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
                    f"FATAL: {type(e).__name__} on {pair['pair_id']} [{formulation}] "
                    f"repeat {repeat_idx}: {e}\nThis will recur on every remaining call "
                    f"(bad key, bad request, or similar) -- stopping instead of retrying "
                    f"{len(work) - i} more times. {i} rows already written are kept; rerun "
                    f"after fixing to resume."
                )
            except Exception as e:
                consecutive_failures += 1
                error_record = {
                    "pair_id": pair["pair_id"],
                    "formulation": formulation,
                    "repeat_idx": repeat_idx,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
                errout.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                errout.flush()
                print(f"  ERROR on {pair['pair_id']} [{formulation}] repeat {repeat_idx}: {e}")
                if consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                    print(
                        f"  {consecutive_failures} consecutive failures (post-retry) -- "
                        f"the SDK's own backoff wasn't enough; cooling down "
                        f"{CIRCUIT_BREAKER_COOLDOWN}s before continuing"
                    )
                    time.sleep(CIRCUIT_BREAKER_COOLDOWN)
                    consecutive_failures = 0

            time.sleep(INTER_CALL_DELAY_SECONDS)

    print(f"\nWrote/updated {OUT_PATH.relative_to(ROOT)}")
    if ERRORS_PATH.exists() and ERRORS_PATH.stat().st_size > 0:
        print(f"Non-fatal failures logged to {ERRORS_PATH.relative_to(ROOT)} -- rerun this script to retry them")


if __name__ == "__main__":
    main()
