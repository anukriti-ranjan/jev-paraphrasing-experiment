#!/usr/bin/env python3
"""Decompose each criterion into its atomic (AND-combined) requirements.

Reads:  ../data/healthbench/criteria.jsonl       (30 unique criterion texts)
Writes: ../data/healthbench/atomic_decomposition.jsonl

FROZEN BEFORE ANY JEV DATA: this script only ever reads criterion TEXT --
never jev_judgments.jsonl, never criterion_probes.jsonl's Jev-derived
flags. That ordering is deliberate, not incidental: if we decomposed
criteria after looking at which ones produced an interesting Probe-D
effect, any result from this stage would be contaminated by selecting on
the outcome we're trying to test. Run this, freeze its output, and only
THEN run 07_atomic_jev_inference.py.

Unlike 02_paraphrase.py's probe_d ("something P0 must entail" -- which an
audit found usually captures only one clause of a multi-part criterion,
not the whole thing), this stage asks for an EXPLICIT, EXHAUSTIVE
decomposition: list the distinct yes/no requirements such that satisfying
ALL of them (a logical AND) is equivalent to satisfying the whole
criterion. A criterion that is already atomic returns exactly one item.

Then a SECOND, different model (same pattern as Stage 2) checks the
decomposition is actually complete and non-overlapping -- not just
plausible-sounding -- and flags it otherwise.

Credentials: ANTHROPIC_AUTH_TOKEN, read from a .env file in the project
root (one level above this script's directory).

Usage:
    .venv/bin/python3 06_decompose_criteria.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

STUDY_DIR = Path(__file__).parent
ROOT = STUDY_DIR.parent
CRITERIA_PATH = ROOT / "data" / "healthbench" / "criteria.jsonl"
OUT_PATH = ROOT / "data" / "healthbench" / "atomic_decomposition.jsonl"
ENV_PATH = ROOT / ".env"

DECOMPOSE_MODEL = "claude-sonnet-5"
VALIDATE_MODEL = "claude-haiku-4-5-20251001"  # deliberately a different, smaller model
MAX_TOKENS = 1024
DELAY_SECONDS = 0.5

DECOMPOSE_SYSTEM_PROMPT = """\
You decompose evaluation-rubric criteria into their atomic, independently \
checkable requirements, for a calibration study. The requirements you list \
must be jointly equivalent to the original -- satisfying every one of them \
(a logical AND) must mean exactly the same thing as satisfying the whole \
original criterion, with nothing added, dropped, or left ambiguous. If the \
criterion already expresses a single atomic requirement that cannot be \
meaningfully split further, return exactly one item: that requirement, \
restated as a direct yes/no question.\
"""


def build_decompose_prompt(p0: str) -> str:
    return f"""\
Criterion:
\"\"\"{p0}\"\"\"

List this criterion's distinct, independently-checkable yes/no
requirements, each as a standalone question, such that answering "yes" to
ALL of them is equivalent to the whole criterion being satisfied. Do not
add requirements not implied by the text, and do not drop any. If the
criterion is already a single atomic requirement, return exactly one item.

Respond with ONLY this JSON, no markdown fences:
{{"atomic_requirements": ["...", "..."]}}"""


VALIDATE_PROMPT = """\
Original criterion:
\"\"\"{p0}\"\"\"

Proposed decomposition into atomic requirements (every one must hold for
the original to hold):
{items}

Classify the decomposition, choosing exactly one label:
- "complete": the AND of these items is equivalent to the original --
  nothing missing, nothing added, no overlap between items.
- "incomplete": the items miss something the original requires.
- "overlapping": two or more items overlap or restate each other.
- "trivial": there is only one item (the criterion wasn't decomposed).

Respond with ONLY the label, no other text."""


def extract_text(msg: anthropic.types.Message) -> str:
    for block in msg.content:
        if block.type == "text":
            return block.text
    raise ValueError(f"no text block in response: {[b.type for b in msg.content]}")


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def decompose(client: anthropic.Anthropic, p0: str) -> list[str]:
    msg = client.messages.create(
        model=DECOMPOSE_MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=DECOMPOSE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_decompose_prompt(p0)}],
    )
    return json.loads(strip_fences(extract_text(msg)))["atomic_requirements"]


def validate_decomposition(client: anthropic.Anthropic, p0: str, items: list[str]) -> str:
    items_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))
    msg = client.messages.create(
        model=VALIDATE_MODEL,
        max_tokens=16,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": VALIDATE_PROMPT.format(p0=p0, items=items_text)}],
    )
    return extract_text(msg).strip().strip('"').lower()


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with out_path.open(encoding="utf-8") as f:
        return {json.loads(line)["p0"] for line in f if line.strip()}


def main() -> None:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL") or None
    if not api_key:
        sys.exit(f"ERROR: no ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY found in {ENV_PATH}")
    client = anthropic.Anthropic(api_key=api_key, base_url=base_url) if base_url else anthropic.Anthropic(
        api_key=api_key
    )

    criteria = [json.loads(line) for line in CRITERIA_PATH.read_text(encoding="utf-8").splitlines()]
    done = already_done(OUT_PATH)
    work = [c for c in criteria if c["p0"] not in done]
    print(f"{len(criteria)} unique criteria, {len(done)} already done, {len(work)} to run")

    with OUT_PATH.open("a", encoding="utf-8") as out:
        for i, criterion in enumerate(work):
            p0 = criterion["p0"]
            label = f"[{i + 1}/{len(work)}] {criterion['category'][:50]}"
            print(label, end="  ", flush=True)
            try:
                items = decompose(client, p0)
                time.sleep(DELAY_SECONDS)
                scope = validate_decomposition(client, p0, items)
                record = {
                    "p0": p0,
                    "category": criterion["category"],
                    "pair_ids": criterion["pair_ids"],
                    "atomic_requirements": items,
                    "n_atomic": len(items),
                    "decomposition_scope": scope,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"n_atomic={len(items)} scope={scope}")
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(DELAY_SECONDS)

    print(f"\nWrote/updated {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
