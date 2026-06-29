"""GSM8K loading.

By default we read a small bundled sample so the demo runs offline (conference
wifi is not to be trusted). Pass source="hf" to pull the real test split via the
`datasets` library if it is installed.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "..", "..", "data", "gsm8k_sample.jsonl")


@dataclass
class Problem:
    id: str
    question: str
    answer: str  # canonical numeric answer as a string


def _gold_from_hf(raw_answer: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    m = re.search(r"####\s*([-\d,\.]+)", raw_answer)
    num = (m.group(1) if m else raw_answer).replace(",", "").strip()
    return num


def load_problems(source: str = "sample", n: int | None = None) -> list[Problem]:
    if source == "hf":
        from datasets import load_dataset  # lazy import

        ds = load_dataset("gsm8k", "main", split="test")
        probs = [
            Problem(id=f"gsm8k-test-{i}", question=ex["question"],
                    answer=_gold_from_hf(ex["answer"]))
            for i, ex in enumerate(ds)
        ]
    else:
        probs = []
        with open(os.path.normpath(SAMPLE_PATH)) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    probs.append(Problem(id=d["id"], question=d["question"],
                                         answer=str(d["answer"])))
    return probs[:n] if n else probs
