"""Reconstruct one model's repeated attempts at a single Game24 puzzle from the
shipped trace logs, and render the "one truth vs. a hydra of lies" view.

The point (for the ACL tutorial's instability act): the *same* model, on the
*same* puzzle, at the *same* settings, is sampled N times. Game24 is mechanically
checkable — a valid answer must use each input number exactly once and evaluate to
24 — so we can grade every attempt and watch the accuracy flip run to run.

The headline isn't just "it's ~50%". It's the *shape* of the failures: the correct
runs all rediscover the same single idea, while the wrong runs are each a different
confident-but-illegal expression that fabricates a number it wasn't given (usually
by slipping in n/n = 1 or n - n = 0 to absorb a leftover operand).

Trace logs live at logs/<model>/<benchmark>/io_test_<run>.log; each file is one run
over the same 50 puzzles, as `* USER *` / `* RESPONSE 1 *` blocks. No correctness is
stored in the logs — we re-derive it here.
"""
from __future__ import annotations

import ast
import glob
import html
import os
import re
from collections import Counter
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(HERE, "logs")


@dataclass
class Attempt:
    run: int
    expr: str | None       # the model's final answer expression, normalized
    value: float | None    # what it evaluates to
    used: tuple            # multiset of numbers it actually used
    valid: bool            # uses the exact puzzle numbers AND equals the target
    reason: str            # why it's wrong (empty if valid)
    leaked_prompt: bool    # echoed the few-shot scaffold into the answer
    raw: str               # the full response text


def _norm(s: str) -> str:
    for a, b in [("×", "*"), ("·", "*"), ("÷", "/"), ("\\times", "*"),
                 ("\\cdot", "*"), ("\\div", "/"), ("\\left", ""), ("\\right", ""),
                 ("$", ""), ("`", ""), ("\\", ""), ("−", "-")]:
        s = s.replace(a, b)
    return s.strip().strip("*").strip()


def _eval(expr: str):
    """Return (sorted_used_multiset, value) or (None, None) if not evaluable."""
    try:
        node = ast.parse(expr, mode="eval")
    except Exception:
        return None, None
    used: list[int] = []

    class V(ast.NodeVisitor):
        def visit_Constant(self, n):
            if isinstance(n.value, (int, float)):
                used.append(int(n.value))

    V().visit(node)
    try:
        val = eval(compile(node, "<e>", "eval"), {"__builtins__": {}})
    except Exception:
        return None, None
    if not isinstance(val, (int, float)):
        return None, None
    return tuple(sorted(used)), val


def _final_expr(resp: str) -> str | None:
    """Pull the model's final answer expression (prefer the last `Answer:` line)."""
    s = _norm(resp)
    cands: list[str] = []
    ans = re.findall(r"Answer:?\s*\**\s*([0-9()+\-*/.\s]+)", s)
    if ans:
        cands.append(re.split(r"=\s*24", _norm(ans[-1]))[0])
    boxed = re.findall(r"boxed\{([^}]*)\}", s)
    if boxed:
        cands.append(boxed[-1])
    eqs = re.findall(r"([0-9()+\-*/.\s]+?)\s*=\s*24\b", s)
    if eqs:
        cands.append(eqs[-1])
    for c in cands:
        c = _norm(c).rstrip(".").strip().strip("=").strip()
        if c.count("(") > c.count(")"):
            c += ")" * (c.count("(") - c.count(")"))
        if c and re.fullmatch(r"[0-9()+\-*/.\s]+", c) and re.search(r"\d", c):
            return c
    return None


def _why_wrong(puzzle, used, value, target) -> str:
    if used is None:
        return "no parseable expression"
    if abs(value - target) > 1e-6:
        v = int(value) if float(value).is_integer() else round(value, 3)
        return f"evaluates to {v}, not {target}"
    extra = Counter(used) - Counter(puzzle)
    missing = Counter(puzzle) - Counter(used)
    bits = []
    for n, k in sorted(extra.items()):
        bits.append(f"invents {'an extra ' if k == 1 else f'{k} extra '}{n}")
    for n, k in sorted(missing.items()):
        bits.append(f"never uses {n}")
    return "; ".join(bits) or "uses the wrong numbers"


# a self-cancelling device the model uses to fabricate a number: n/n=1 or n-n=0
_FABRICATE = re.compile(r"(\d+)\s*([/\-])\s*\1\b")


def load_attempts(model: str, puzzle, target: int = 24, benchmark: str = "game24") -> list[Attempt]:
    want = tuple(sorted(puzzle))
    files = sorted(
        glob.glob(os.path.join(LOGS_DIR, model, benchmark, "io_test_*.log")),
        key=lambda p: int(re.search(r"io_test_(\d+)", p).group(1)),
    )
    out: list[Attempt] = []
    for f in files:
        run = int(re.search(r"io_test_(\d+)", f).group(1))
        txt = open(f, errors="ignore").read()
        blocks = re.split(r"\*{8}\n\* USER \*\n\*{8}\n", txt)
        for b in blocks[1:]:
            user = b.split("* N:")[0]
            inputs = re.findall(r"Input:\s*([\d ]+?)\s*\n", user)
            if not inputs or tuple(sorted(int(x) for x in inputs[-1].split())) != want:
                continue
            m = re.search(r"\* RESPONSE 1 \*\n\*+\n(.*?)(?:\n={50,}|\Z)", b, re.S)
            resp = (m.group(1).strip() if m else "")
            expr = _final_expr(resp)
            used, value = _eval(expr) if expr else (None, None)
            valid = used == want and value is not None and abs(value - target) < 1e-6
            out.append(Attempt(
                run=run, expr=expr, value=value, used=used or (), valid=valid,
                reason="" if valid else _why_wrong(want, used, value, target),
                leaked_prompt="Input:" in resp, raw=resp,
            ))
            break
    return out


def render_html(model: str, puzzle, target: int = 24) -> str:
    attempts = load_attempts(model, puzzle, target)
    n = len(attempts)
    n_ok = sum(a.valid for a in attempts)
    nums = " ".join(str(x) for x in puzzle)

    # run-order strip
    chips = "".join(
        f'<span title="run {a.run}: {html.escape(a.expr or "—")}" '
        f'style="display:inline-block;width:1.35em;text-align:center">'
        f'{"✅" if a.valid else "❌"}</span>'
        for a in attempts
    )

    def expr_html(expr: str | None) -> str:
        if not expr:
            return '<em style="color:#999">no expression</em>'
        return _FABRICATE.sub(
            lambda m: f'<span style="background:#ffe1e1;border-bottom:2px solid #d62728">'
                      f'{html.escape(m.group(0))}</span>',
            html.escape(expr),
        )

    # left column: valid answers, grouped — the one repeated idea
    ok = Counter(a.expr for a in attempts if a.valid)
    left_rows = "".join(
        f'<tr><td style="font-family:monospace;color:#198038">{html.escape(e)}</td>'
        f'<td style="color:#999;padding-left:.8em">×{c}</td></tr>'
        for e, c in ok.most_common()
    )

    # right column: every failure, in run order — the hydra
    right_rows = "".join(
        f'<tr><td style="font-family:monospace">{expr_html(a.expr)}</td>'
        f'<td style="color:#a33;padding-left:.8em;white-space:nowrap">{html.escape(a.reason)}'
        f'{" · echoed prompt" if a.leaked_prompt else ""}</td></tr>'
        for a in attempts if not a.valid
    )

    return f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:880px;
            border:1px solid #e0e0e0;border-radius:10px;padding:18px 22px">
  <div style="font-size:15px">
    <b>{html.escape(model)}</b> · make <b>{target}</b> from
    <code style="background:#f4f4f4;padding:1px 6px;border-radius:4px">[{nums}]</code>
    · {n} identical calls, same settings
  </div>
  <div style="font-size:20px;margin:12px 0 4px;letter-spacing:1px">{chips}</div>
  <div style="font-size:14px;color:#555;margin-bottom:16px">
    <b style="color:#198038">{n_ok}</b> / {n} valid
    — a benchmark would report this as <b>{100*n_ok/n:.0f}% accuracy</b>,
    but each attempt is a coin toss.
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:13.5px">
    <tr style="vertical-align:top">
      <td style="width:42%;padding-right:18px;border-right:1px solid #eee">
        <div style="font-weight:600;color:#198038;margin-bottom:6px">
          ✅ SOLVED ({n_ok}) — one idea, rediscovered</div>
        <table>{left_rows}</table>
      </td>
      <td style="padding-left:18px">
        <div style="font-weight:600;color:#c0392b;margin-bottom:6px">
          ❌ NOT SOLVED ({n-n_ok}) — every failure a different confident lie</div>
        <table>{right_rows}</table>
      </td>
    </tr>
  </table>
  <div style="font-size:12.5px;color:#777;margin-top:14px;border-top:1px solid #eee;padding-top:10px">
    The wrong answers don't drift randomly — they <b>fabricate a number the model
    wasn't given</b>, usually by sneaking in <span style="background:#ffe1e1">n/n</span>
    or <span style="background:#ffe1e1">n−n</span> to absorb a leftover operand.
  </div>
</div>
"""


def show(model: str = "gpt-5-mini", puzzle=(6, 6, 7, 12), target: int = 24):
    """Display the instability view in a notebook."""
    from IPython.display import HTML, display
    display(HTML(render_html(model, puzzle, target)))


if __name__ == "__main__":
    # plain-text smoke test
    for a in load_attempts("gpt-5-mini", (6, 6, 7, 12)):
        print(f"run {a.run:>2} {'OK ' if a.valid else 'BAD'} "
              f"{a.expr!r:28} {a.reason}")
