"""Answer extraction, numeric normalization, and majority vote."""
from __future__ import annotations

import re
from collections import Counter

# Matches integers/decimals with optional $ , % decoration.
_NUM = re.compile(r"-?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?")


def normalize(num: str | None) -> str | None:
    if num is None:
        return None
    s = num.replace("$", "").replace(",", "").replace("%", "").strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    # Render 72.0 -> "72" so int/float gold answers compare equal.
    return str(int(f)) if f == int(f) else str(f)


def extract_final_number(text: str) -> str | None:
    """Prefer an explicit '#### x' marker, then 'answer is x', else last number."""
    if not text:
        return None
    m = re.search(r"####\s*(" + _NUM.pattern + r")", text)
    if m:
        return normalize(m.group(1))
    m = re.search(r"(?:final answer|answer)\s*(?:is|:)?\s*(" + _NUM.pattern + r")",
                  text, re.IGNORECASE)
    if m:
        return normalize(m.group(1))
    nums = _NUM.findall(text)
    return normalize(nums[-1]) if nums else None


def majority_vote(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def is_correct(pred: str | None, gold: str) -> bool:
    return pred is not None and normalize(pred) == normalize(gold)
