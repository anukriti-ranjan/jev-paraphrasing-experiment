#!/usr/bin/env python3
"""Paraphrasing: generate 4 probes per unique criterion, then validate them.

Reads:  ../data/healthbench/criteria.jsonl   (30 unique criterion texts)
Writes: ../data/healthbench/criterion_probes.jsonl

For each unique criterion text (P0), ONE call asks an LLM for four
reformulations at once:
    probe_a  direct predicate      (bare yes/no, framing stripped)
    probe_b  semantic paraphrase   (reworded, same meaning)
    probe_c  logical negation      (true exactly when P0 is false)
    probe_d  consequence           (something P0-true must entail)

Only the criterion text is sent -- never the conversation or the model's
answer -- because this step rewrites the RULE, not an application of it to
a specific case (see 01_prepare_dataset.py section 2).

Then a SECOND, separate call -- a different, smaller model, so the check
isn't the same model grading its own homework -- classifies each probe's
actual relationship to P0 (paraphrase / negation / implication /
unrelated) and flags it if that doesn't match what it was asked to
produce. This is the Stage-3(a) text-level validation.

CAVEAT on probe_d (read this before trusting its label downstream): asking
for "something P0-true must entail" does not reliably produce a full-scope
consequence of the whole criterion -- a later audit (04_analysis.py's
audit_d_scope_and_filtered_rerun()) found 21/23 of the criteria behind its
disagreement cases are "partial_scope": D usually captures one clause of a
criterion with several implicit requirements, not the whole thing. The
Stage-3(a) check above only verifies "is this *some* implication", not
"is this a full-scope one" -- that's a different, stricter question this
stage doesn't ask. Downstream code calls it a subcriterion/implication-like
probe for this reason, not simply "the implication probe".

Credentials: ANTHROPIC_AUTH_TOKEN (+ optional ANTHROPIC_BASE_URL), read
from a .env file in the project root (one level above this script's
directory).

Usage:
    .venv/bin/python3 02_paraphrase.py
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
OUT_PATH = ROOT / "data" / "healthbench" / "criterion_probes.jsonl"
ENV_PATH = ROOT / ".env"

GENERATE_MODEL = "claude-sonnet-5"
VALIDATE_MODEL = "claude-haiku-4-5-20251001"  # deliberately a different, smaller model
MAX_TOKENS = 1024
DELAY_SECONDS = 0.5

EXPECTED_RELATIONSHIP = {
    "probe_a": "paraphrase",
    "probe_b": "paraphrase",
    "probe_c": "negation",
    "probe_d": "implication",
}

PARAPHRASE_SYSTEM_PROMPT = """\
You rewrite evaluation-rubric criteria into controlled variants for a \
calibration study. You never change what is being asked about the \
submission -- only how it is worded -- except for the negation and \
implication variants, which must change the logical relationship exactly \
as instructed. You never add facts not present in the criterion text.\
"""


def build_paraphrase_prompt(p0: str) -> str:
    return f"""\
Original criterion (P0):
\"\"\"{p0}\"\"\"

Produce four reformulations of this criterion as JSON:

1. "probe_a" (direct predicate): the bare yes/no predicate this criterion
   is really asking, stripped of narrative framing. Keep every substantive
   requirement.
2. "probe_b" (semantic paraphrase): reword with different vocabulary and
   sentence structure. Same meaning, same scope, nothing added or dropped.
3. "probe_c" (logical negation): state the logical negation of P0, such
   that answering "yes" to this statement should correspond to answering
   "no" to P0, and vice versa.
4. "probe_d" (consequence/implication): state something that must be true
   if P0 is true (a one-directional entailment of P0, not a restatement
   of it and not its converse).

Respond with ONLY this JSON, no markdown fences:
{{"probe_a": "...", "probe_b": "...", "probe_c": "...", "probe_d": "..."}}"""


RELATIONSHIP_PROMPT = """\
Statement 1 (P0):
\"\"\"{p0}\"\"\"

Statement 2:
\"\"\"{probe_text}\"\"\"

Classify the logical relationship of Statement 2 to Statement 1, choosing
exactly one label:
- "paraphrase": same meaning, same truth conditions, different wording.
- "negation": Statement 2 is true exactly when Statement 1 is false.
- "implication": Statement 1 being true requires Statement 2 to be true,
  but Statement 2 does not require Statement 1.
- "unrelated": none of the above.

Respond with ONLY the label, no other text."""


def extract_text(msg: anthropic.types.Message) -> str:
    """claude-sonnet-5 returns extended-thinking blocks before the text block,
    so content[0] is often a ThinkingBlock, not the answer -- find the text one."""
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


def generate_probes(client: anthropic.Anthropic, p0: str) -> dict[str, str]:
    msg = client.messages.create(
        model=GENERATE_MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=PARAPHRASE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_paraphrase_prompt(p0)}],
    )
    return json.loads(strip_fences(extract_text(msg)))


def classify_relationship(client: anthropic.Anthropic, p0: str, probe_text: str) -> str:
    msg = client.messages.create(
        model=VALIDATE_MODEL,
        max_tokens=16,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": RELATIONSHIP_PROMPT.format(p0=p0, probe_text=probe_text)}],
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
                probes = generate_probes(client, p0)
                validations = {}
                for key, probe_text in probes.items():
                    relationship = classify_relationship(client, p0, probe_text)
                    expected = EXPECTED_RELATIONSHIP[key]
                    validations[key] = {
                        "relationship": relationship,
                        "expected": expected,
                        "matches_expected": relationship == expected,
                    }
                    time.sleep(DELAY_SECONDS)

                record = {
                    "p0": p0,
                    "category": criterion["category"],
                    "pair_ids": criterion["pair_ids"],
                    "probes": probes,
                    "validation": validations,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                flags = [k for k, v in validations.items() if not v["matches_expected"]]
                status = "OK" if not flags else f"FLAGGED {flags}"
                print(status)
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(DELAY_SECONDS)

    print(f"\nWrote/updated {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
