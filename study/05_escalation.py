#!/usr/bin/env python3
"""Optional: does escalating disagreement-flagged pairs to a bigger LLM help?

This is NOT required to test H0/H1 (04_analysis.py already does that with
Jev + ground truth alone). This answers the separate, secondary question
the paper itself asks: if you build a cascade that defers uncertain pairs
to an LLM judge, does it actually recover accuracy -- and is "probe
disagreement" a better deferral signal than "Jev's own confidence" (the
paper's own cascade, which they found gains at most ~1.5-2.0 points)?

We don't have the paper's own LLM judges (Luna/Gemini/DeepSeek need
vendor keys we don't have). We substitute one Anthropic model as the
escalation judge -- same role, different vendor; the comparison below is
about WHICH DEFERRAL SIGNAL helps, not about matching the paper's exact
judge roster.

Reads:  ../data/healthbench/jev_judgments.jsonl
Writes: ../data/healthbench/escalation_judgments.jsonl (only for escalated pairs)
        prints a comparison table; no plot (small-n cascade curves are
        easy to overread -- a table is more honest here).

Three cascades compared, at a few escalation-rate operating points:
  (a) confidence-only  -- defer P0's lowest-confidence pairs (paper's method)
  (b) disagreement-only -- defer pairs where any probe disagrees with P0
  (c) combined          -- defer on EITHER signal

Usage:
    .venv/bin/python3 05_escalation.py
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
JUDGMENTS_PATH = ROOT / "data" / "healthbench" / "jev_judgments.jsonl"
PAIRS_PATH = ROOT / "data" / "healthbench" / "pairs.jsonl"
ESCALATION_PATH = ROOT / "data" / "healthbench" / "escalation_judgments.jsonl"
ENV_PATH = ROOT / ".env"

ESCALATION_MODEL = "claude-sonnet-5"
MAX_TOKENS = 256
DELAY_SECONDS = 0.3

ESCALATION_RATES = (0.1, 0.2, 0.3)  # fraction of pairs deferred, per cascade


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_pair_table():
    import numpy as np  # local import: only this script needs pandas+matplotlib free
    import pandas as pd

    df = pd.DataFrame(load_jsonl(JUDGMENTS_PATH))
    # 03_jev_inference.py writes N_REPEATS rows per (pair_id, formulation) --
    # collapse to one mean p_met each before pivoting (see 04_analysis.py's
    # aggregate_repeats() for why repeats exist and how spread is checked;
    # this script only needs the mean, not the spread).
    agg = df.groupby(["pair_id", "formulation"]).agg(
        p_met=("p_met", "mean"), ground_truth=("ground_truth", "first")
    ).reset_index()
    agg["verdict"] = np.where(agg["p_met"] > 0.5, "MET", "UNMET")

    wide = agg.pivot(index="pair_id", columns="formulation", values=["p_met", "verdict"])
    ground_truth = agg.drop_duplicates("pair_id").set_index("pair_id")["ground_truth"]
    out = pd.DataFrame(index=wide.index)
    out["ground_truth"] = ground_truth
    out["p0_verdict"] = wide["verdict"]["P0"]
    out["p0_p_met"] = wide["p_met"]["P0"]
    out["p0_confidence"] = (2 * (out["p0_p_met"] - 0.5).abs()).clip(0, 1)
    out["p0_correct"] = out["p0_verdict"] == out["ground_truth"]
    out["disagreement_count"] = sum(
        (wide["verdict"][probe] != out["p0_verdict"]).astype(int) for probe in ("A", "B", "C", "D")
    )
    return out.reset_index()


ESCALATION_PROMPT = """\
Conversation:
\"\"\"{state_input}\"\"\"

Response being evaluated:
\"\"\"{state_submission}\"\"\"

Criterion: {p0}

Does the response satisfy this criterion? Answer with exactly one word:
MET or UNMET."""


def call_escalation_judge(client: anthropic.Anthropic, pair_row: dict) -> str:
    msg = client.messages.create(
        model=ESCALATION_MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        messages=[
            {
                "role": "user",
                "content": ESCALATION_PROMPT.format(
                    state_input=pair_row["state_input"],
                    state_submission=pair_row["state_submission"],
                    p0=pair_row["p0"],
                ),
            }
        ],
    )
    text = next(b.text for b in msg.content if b.type == "text").strip().upper()
    return "MET" if "MET" in text and "UNMET" not in text else "UNMET"


def get_or_run_escalation_judge(client: anthropic.Anthropic, pair_ids: set[str], pairs_by_id: dict) -> dict[str, str]:
    done = {}
    if ESCALATION_PATH.exists():
        for row in load_jsonl(ESCALATION_PATH):
            done[row["pair_id"]] = row["verdict"]

    to_run = pair_ids - set(done)
    if to_run:
        print(f"escalating {len(to_run)} pairs to {ESCALATION_MODEL}...")
        with ESCALATION_PATH.open("a", encoding="utf-8") as out:
            for pair_id in to_run:
                verdict = call_escalation_judge(client, pairs_by_id[pair_id])
                done[pair_id] = verdict
                out.write(json.dumps({"pair_id": pair_id, "verdict": verdict}, ensure_ascii=False) + "\n")
                out.flush()
                time.sleep(DELAY_SECONDS)
    return done


def simulate_cascade(pairs, defer_mask, escalation_verdicts: dict[str, str]) -> tuple[float, int]:
    correct = 0
    for _, row in pairs.iterrows():
        if defer_mask.loc[row.name] and row["pair_id"] in escalation_verdicts:
            verdict = escalation_verdicts[row["pair_id"]]
        else:
            verdict = row["p0_verdict"]
        correct += verdict == row["ground_truth"]
    return correct / len(pairs), int(defer_mask.sum())


def main() -> None:
    if not JUDGMENTS_PATH.exists():
        raise SystemExit(f"Missing {JUDGMENTS_PATH} -- run 03_jev_inference.py first.")

    load_dotenv(ENV_PATH)
    api_key = os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: no Anthropic credentials found")
    client = anthropic.Anthropic(api_key=api_key)

    pairs = build_pair_table()
    pairs_by_id = {p["pair_id"]: p for p in load_jsonl(PAIRS_PATH)}

    baseline_acc = pairs["p0_correct"].mean()
    print(f"baseline (P0 only, no escalation): accuracy = {baseline_acc:.3f}\n")

    print(f"{'rate':>6} {'signal':>18} {'n_deferred':>10} {'accuracy':>10}")
    for rate in ESCALATION_RATES:
        n_defer = max(1, int(round(rate * len(pairs))))

        conf_order = pairs["p0_confidence"].rank(method="first")
        defer_conf = conf_order <= n_defer

        disagree_order = pairs["disagreement_count"].rank(method="first", ascending=False)
        defer_disagree = disagree_order <= n_defer

        defer_combined = defer_conf | defer_disagree

        all_defer_ids = set(pairs.loc[defer_conf | defer_disagree, "pair_id"])
        escalation_verdicts = get_or_run_escalation_judge(client, all_defer_ids, pairs_by_id)

        for name, mask in (
            ("confidence-only", defer_conf),
            ("disagreement-only", defer_disagree),
            ("combined (OR)", defer_combined),
        ):
            acc, n_deferred = simulate_cascade(pairs, mask, escalation_verdicts)
            print(f"{rate:>6.0%} {name:>18} {n_deferred:>10} {acc:>10.3f}")
        print()

    print(f"escalation verdicts cached in {ESCALATION_PATH.relative_to(ROOT)} (reused on rerun)")


if __name__ == "__main__":
    main()
