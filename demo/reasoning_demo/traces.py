"""Load and visualize real s1 reasoning traces shipped in the crosslingual repo.

Used by the ACL tutorial notebook for the two "mechanism" acts:
  - Act 3 (language mixing): highlight a trace by writing system to show how an
    English-centric model code-switches while reasoning in another language.
  - Act 4 (budget extrapolation): show where an injected "Wait" extends the
    reasoning, and find cases where it flips a wrong answer to right.

Language identification here is by Unicode **script** (Latin / Han / Kana /
Cyrillic / Thai / Bengali / Telugu / …), which is dependency-free, deterministic,
and exactly the signal we want: native-script vs. Latin (≈ English) reasoning.
The paper's own analysis uses lingua/fasttext for finer language ID.
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
from dataclasses import dataclass

from .mgsm import extract_answer, is_correct

HERE = os.path.dirname(os.path.abspath(__file__))
S1_ARTIFACTS_DIR = os.path.normpath(os.path.join(
    HERE, "..", "..", "repos", "crosslingual-test-time-scaling",
    "experiments", "crosslingual_mgsm", "artifacts", "s1"))

# Writing-system buckets by Unicode range, with a display colour for the HTML view.
SCRIPTS = [
    ("Han (Chinese)", "#e8743b", lambda c: "一" <= c <= "鿿" or "㐀" <= c <= "䶿"),
    ("Kana (Japanese)", "#d4a017", lambda c: "぀" <= c <= "ヿ"),
    ("Cyrillic", "#19a979", lambda c: "Ѐ" <= c <= "ӿ"),
    ("Thai", "#945ecf", lambda c: "฀" <= c <= "๿"),
    ("Bengali", "#13a4b4", lambda c: "ঀ" <= c <= "৿"),
    ("Telugu", "#6f9654", lambda c: "ఀ" <= c <= "౿"),
    ("Devanagari", "#bd2d28", lambda c: "ऀ" <= c <= "ॿ"),
    ("Arabic", "#885053", lambda c: "؀" <= c <= "ۿ"),
    ("Latin (≈ English)", "#3366cc", lambda c: ("a" <= c.lower() <= "z")),
]
NEUTRAL_COLOR = "#999999"  # digits / punctuation / whitespace / math symbols


def _script_of(ch: str) -> str | None:
    """Return a script name for a 'content' char, or None for neutral chars."""
    for name, _color, test in SCRIPTS:
        if test(ch):
            return name
    return None  # neutral (digit, space, punctuation, symbol)


@dataclass
class Trace:
    size: str
    lang: str
    budget: int
    doc_id: int
    question: str
    gold: str
    pred: str
    correct: bool
    cot: str          # the model's reasoning + answer, markers stripped
    raw: str          # original text incl. <|im_start|> markers


def _run_dir(size: str, lang: str, budget: int, wait: bool) -> str:
    name = (f"s1.1-{size}-wait-mgsm_direct_{lang}_think{budget}_wait1" if wait
            else f"s1.1-{size}-mgsm_direct_{lang}_{budget}")
    return os.path.join(S1_ARTIFACTS_DIR, name)


def _strip_markers(text: str) -> str:
    """Drop the s1 control tokens so the trace reads naturally."""
    text = text.replace("<|im_start|>think", "").replace("<|im_start|>answer", "\n\n")
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    return text.strip()


def load_trace(size: str, lang: str, budget: int = 8000, doc_id: int = 0,
               wait: bool = False) -> Trace:
    """Load one shipped s1 generation (the full chain of thought + answer)."""
    files = glob.glob(os.path.join(_run_dir(size, lang, budget, wait), "*", "samples_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No samples for size={size} lang={lang} budget={budget} wait={wait}")
    with open(files[0], encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["doc_id"] == doc_id:
                raw = d["resps"][0][0] if isinstance(d["resps"][0], list) else d["resps"][0]
                gold = str(d["target"])
                # Re-extract with our flexible-extract so correctness matches the
                # notebook's parsing (the sample-level field reflects a different filter).
                pred = extract_answer(raw)
                return Trace(
                    size=size, lang=lang, budget=budget, doc_id=doc_id,
                    question=d["doc"]["question"], gold=gold,
                    pred=pred or "", correct=is_correct(pred, gold),
                    cot=_strip_markers(raw), raw=raw)
    raise IndexError(f"doc_id {doc_id} not found")


def script_runs(text: str) -> list[tuple[str, str | None]]:
    """Segment text into (substring, script) runs; neutral chars extend the current run."""
    runs: list[list] = []
    current: str | None = None
    for ch in text:
        s = _script_of(ch)
        if s is None:                       # neutral: stay in the current run
            if runs:
                runs[-1][0] += ch
            else:
                runs.append([ch, None])
        elif s == current:
            runs[-1][0] += ch
        else:
            runs.append([ch, s])
            current = s
    return [(seg, scr) for seg, scr in runs]


def script_distribution(text: str) -> dict[str, float]:
    """Fraction of *content* characters (non-neutral) per script."""
    counts: dict[str, int] = {}
    for ch in text:
        s = _script_of(ch)
        if s is not None:
            counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def highlight_html(text: str, *, title: str = "") -> str:
    """Render the trace as HTML with each script coloured; includes a legend."""
    colors = {name: color for name, color, _ in SCRIPTS}
    spans = []
    for seg, scr in script_runs(text):
        color = colors.get(scr, NEUTRAL_COLOR) if scr else NEUTRAL_COLOR
        spans.append(f'<span style="color:{color}">{html.escape(seg)}</span>')
    legend = "  ".join(
        f'<span style="color:{c};font-weight:600">&#9632; {name}</span>'
        for name, c, _ in SCRIPTS) + f'  <span style="color:{NEUTRAL_COLOR}">&#9632; digits/punct</span>'
    dist = script_distribution(text)
    dist_str = ", ".join(f"{k.split(' ')[0]} {v:.0%}" for k, v in dist.items())
    body = "".join(spans).replace("\n", "<br>")
    return (f'<div style="font-family:monospace;font-size:13px;line-height:1.5">'
            f'{("<b>" + html.escape(title) + "</b><br>") if title else ""}'
            f'<div style="margin:6px 0">{legend}</div>'
            f'<div style="margin:6px 0;color:#555">script mix: {dist_str}</div>'
            f'<div style="border-left:3px solid #ddd;padding-left:10px">{body}</div></div>')


_SENT_SPLIT = re.compile(r"(?<=[.!?。！？\n])\s+")


def find_wait_moments(text: str, wait_token: str = "Wait", context: int = 220) -> list[str]:
    """Return snippets around each injected wait token, to show the model reconsidering."""
    snippets = []
    for m in re.finditer(re.escape(wait_token), text):
        start = max(0, m.start() - context)
        end = min(len(text), m.end() + context)
        snippets.append(("…" if start else "") + text[start:end] + ("…" if end < len(text) else ""))
    return snippets
