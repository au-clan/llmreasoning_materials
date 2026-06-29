"""Seed-variance & "how you score it" views over the repeated-runs table
(`models.parquet`).

The companion to `instability.py`. Where that file zooms into one model on one
puzzle, this one steps back to the whole grid: every `(model, benchmark)` cell in
the parquet is ~30 repeated runs (seeds) of the *same* 50 problems at the *same*
settings (temp 0.7, identical decoding). One row of the parquet = one run; its
`scores` column is the length-50 per-sample correctness vector. Stack the seeds and
each cell becomes a **seeds x samples** 0/1 matrix.

Two things fall out of that matrix that a single accuracy number hides:

1. **Per-call instability (seed axis).** Sort the 50 problems by how often they're
   solved and three bands appear: an always-solved block, an always-failed block,
   and a wide *flippy* middle where the same problem is a coin toss across seeds.

2. **Methodology shapes perceived capability.** The same matrix reports wildly
   different "capability" depending on the scoring protocol you pick:
     - single-run    = mean over seeds of each run's accuracy (what one eval reports)
     - worst / best  = unluckiest / luckiest seed (the cherry-picking range)
     - maj@N         = per item, is it solved by the majority of seeds?
     - pass@N        = per item, is it solved by *at least one* seed?
   For several models these span 40+ points on the identical data.

Scores are treated as continuous in [0, 1] (humaneval is a unit-test pass fraction;
the rest are 0/1). "Solved" for the maj@N / pass@N protocols thresholds at >= 0.5.

Rendering is pure inline SVG/HTML (no matplotlib) so it embeds cleanly in the
tutorial notebook. Read with: `import consistency as C; C.show_heatmap(...)`.
"""
from __future__ import annotations

import html
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "models.parquet")

_DF: pd.DataFrame | None = None

# Display order / short names, roughly best-to-worst reasoning tier.
SHORT = {
    "Qwen/Qwen3-235B-A22B-Thinking-2507": "Qwen3-235B-Thinking",
    "deepseek-ai/DeepSeek-R1": "DeepSeek-R1",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick",
    "claude-haiku-4-5-20251001": "claude-haiku-4.5",
    "gemini-3-flash-preview": "gemini-3-flash",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
}


def load() -> pd.DataFrame:
    """Lazily read the parquet and tag each row with its seed index."""
    global _DF
    if _DF is None:
        df = pd.read_parquet(PARQUET)
        df["seed"] = df["log"].apply(
            lambda p: int(re.search(r"io_test_(\d+)", p).group(1))
        )
        _DF = df
    return _DF


def short(model: str) -> str:
    return SHORT.get(model, model.split("/")[-1])


def benchmarks() -> list[str]:
    return sorted(load().Benchmark.unique())


def models(benchmark: str | None = None) -> list[str]:
    df = load()
    if benchmark:
        df = df[df.Benchmark == benchmark]
    present = set(df.Model.unique())
    ordered = [m for m in SHORT if m in present]
    return ordered + sorted(present - set(ordered))


def resolve(model: str) -> str:
    """Accept either the full provider id or a friendly short name."""
    present = set(load().Model.unique())
    if model in present:
        return model
    for full, nick in SHORT.items():
        if model in (nick, full.split("/")[-1]) and full in present:
            return full
    hits = [m for m in present if model.lower() in m.lower()]
    return hits[0] if len(hits) == 1 else model


def matrix(model: str, benchmark: str) -> np.ndarray:
    """Return the seeds x samples score matrix for one cell (seeds in run order)."""
    model = resolve(model)
    c = load()
    c = c[(c.Model == model) & (c.Benchmark == benchmark)].sort_values("seed")
    if c.empty:
        return np.zeros((0, 0))
    return np.array([np.asarray(s, dtype=float) for s in c.scores])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def protocols(M: np.ndarray, thresh: float = 0.5) -> dict:
    """The same matrix scored four ways, plus the seed-luck range."""
    if M.size == 0:
        return {}
    seed_acc = M.mean(1)                  # one accuracy per seed
    item_rate = M.mean(0)                 # per-item pass-rate over seeds
    solved = (M >= thresh)
    return {
        "single": float(seed_acc.mean()),     # what a single eval reports (on avg)
        "worst": float(seed_acc.min()),       # unluckiest seed
        "best": float(seed_acc.max()),        # luckiest seed
        "maj": float((item_rate >= 0.5).mean()),   # solved by majority of seeds
        "pass": float(solved.any(0).mean()),       # solved by >= 1 seed
        "n_seeds": int(M.shape[0]),
        "n_items": int(M.shape[1]),
    }


def consistency(M: np.ndarray) -> dict:
    """Stability descriptors of a cell."""
    if M.size == 0:
        return {}
    p = M.mean(0)
    flippy = float(((p >= 0.2) & (p <= 0.8)).mean())   # genuine coin-toss items
    decided = float(((p == 0) | (p == 1)).mean())      # always same answer
    # mean per-item agreement: 1 = every seed agrees, 0.5 = maximal disagreement
    agreement = float(np.mean(np.maximum(p, 1 - p)))
    spread = float(M.mean(1).max() - M.mean(1).min())  # best - worst seed accuracy
    return {"flippy": flippy, "decided": decided,
            "agreement": agreement, "spread": spread}


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def _heat(v: float) -> str:
    """White-ish -> green colour for a score in [0,1]."""
    v = max(0.0, min(1.0, v))
    # interpolate #eef0ee (wrong) -> #1a9850 (right)
    a = (238, 240, 238)
    b = (26, 152, 80)
    c = tuple(round(a[i] + (b[i] - a[i]) * v) for i in range(3))
    return f"rgb({c[0]},{c[1]},{c[2]})"


# --------------------------------------------------------------------------- #
# view 1: the seed x sample heatmap for one cell
# --------------------------------------------------------------------------- #
def render_heatmap(model: str, benchmark: str) -> str:
    M = matrix(model, benchmark)
    if M.size == 0:
        return f"<div>no data for {html.escape(model)} / {html.escape(benchmark)}</div>"
    ns, ni = M.shape
    order = np.argsort(-M.mean(0), kind="stable")   # easiest items first
    Ms = M[:, order]
    rate = Ms.mean(0)
    pr = protocols(M)
    cs = consistency(M)

    cw, ch, gap = 11, 9, 1                # cell geometry
    x0, y0 = 8, 86                        # grid origin (body)
    strip_h = 14
    strip_y = y0 - strip_h - 8
    W = x0 + ni * (cw + gap) + 8
    H = y0 + ns * (ch + gap) + 28

    cells = []
    # top strip: per-item pass-rate
    for j in range(ni):
        x = x0 + j * (cw + gap)
        cells.append(
            f'<rect x="{x}" y="{strip_y}" width="{cw}" height="{strip_h}" '
            f'fill="{_heat(rate[j])}"><title>item {order[j]}: solved {rate[j]*100:.0f}% of seeds</title></rect>'
        )
    # body
    for i in range(ns):
        y = y0 + i * (ch + gap)
        for j in range(ni):
            x = x0 + j * (cw + gap)
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{_heat(Ms[i, j])}"/>'
            )

    title = (
        f"<tspan font-weight='700'>{html.escape(short(model))}</tspan>"
        f"<tspan font-weight='700'>  ·  {html.escape(short(benchmark))}  ·  "
        f"<tspan fill='#777'> {ns} seeds × {ni} problems, identical settings</tspan>"
    )
    sub = (
        f"single-run <tspan font-weight='700'>{pr['single']*100:.0f}%</tspan>"
        f"   ·   seed range {pr['worst']*100:.0f}–{pr['best']*100:.0f}%"
        f"   ·   {cs['flippy']*100:.0f}% of problems are coin-tosses"
    )

    return f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif">
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     style="max-width:100%">
  <text x="8" y="20" font-size="14">{title}</text>
  <text x="8" y="40" font-size="12.5" fill="#444">{sub}</text>
  <text x="8" y="{strip_y - 4}" font-size="10" fill="#999">per-problem solve-rate (sorted easy → hard) · each row below = one seed</text>
  {''.join(cells)}
  <text x="8" y="{H - 8}" font-size="11" fill="#666">
    green = solved · grey = failed · any mottled cells are where the same problem flips from seed to seed
  </text>
</svg>
</div>"""


# --------------------------------------------------------------------------- #
# view 2: "it depends how you score it" — protocol spread across models
# --------------------------------------------------------------------------- #
def render_protocols(benchmark: str, models_=None) -> str:
    ms = models_ or models(benchmark)
    rows = []
    for m in ms:
        pr = protocols(matrix(m, benchmark))
        if pr:
            rows.append((m, pr))
    rows.sort(key=lambda r: r[1]["single"])     # weakest at top

    rowh = 30
    x0, xw = 200, 520                            # axis geometry
    W = x0 + xw + 56
    H = 70 + rowh * len(rows) + 24

    def X(v):
        return x0 + v * xw

    bars = []
    # gridlines
    for t in range(0, 11, 2):
        x = X(t / 10)
        bars.append(f'<line x1="{x}" y1="56" x2="{x}" y2="{56 + rowh*len(rows)}" '
                    f'stroke="#eee"/>')
        bars.append(f'<text x="{x}" y="50" font-size="10" fill="#aaa" '
                    f'text-anchor="middle">{t*10}%</text>')

    for k, (m, pr) in enumerate(rows):
        y = 70 + k * rowh
        wy, by = X(pr["worst"]), X(pr["best"])
        bars.append(f'<text x="{x0 - 10}" y="{y + 4}" font-size="12" '
                    f'text-anchor="end">{html.escape(short(m))}</text>')
        # seed-luck range (worst..best single run)
        bars.append(f'<rect x="{wy}" y="{y - 5}" width="{max(by - wy, 1)}" height="10" '
                    f'rx="5" fill="#dfe7ef"><title>seed range {pr["worst"]*100:.0f}–{pr["best"]*100:.0f}%</title></rect>')
        # pass@N (ceiling) — open ring
        bars.append(f'<circle cx="{X(pr["pass"])}" cy="{y}" r="5" fill="none" '
                    f'stroke="#1a9850" stroke-width="2"><title>pass@N (≥1 seed): {pr["pass"]*100:.0f}%</title></circle>')
        # maj@N — diamond
        mx = X(pr["maj"])
        bars.append(f'<path d="M{mx-5},{y} L{mx},{y-5} L{mx+5},{y} L{mx},{y+5} Z" '
                    f'fill="#888"><title>maj@N: {pr["maj"]*100:.0f}%</title></path>')
        # single-run avg — solid dot (the number people report)
        bars.append(f'<circle cx="{X(pr["single"])}" cy="{y}" r="4.5" fill="#222">'
                    f'<title>single-run avg: {pr["single"]*100:.0f}%</title></circle>')
        # ceiling label
        bars.append(f'<text x="{by + 10}" y="{y + 4}" font-size="11" fill="#1a9850">'
                    f'{pr["pass"]*100:.0f}%</text>')

    legend = (
        '<circle cx="210" cy="0" r="4.5" fill="#222"/><text x="220" y="4" font-size="11">single-run</text>'
        '<path d="M315,0 L320,-5 L325,0 L320,5 Z" fill="#888"/><text x="332" y="4" font-size="11">majority@N</text>'
        '<circle cx="430" cy="0" r="5" fill="none" stroke="#1a9850" stroke-width="2"/><text x="440" y="4" font-size="11">pass@N (≥1 seed)</text>'
        '<rect x="565" y="-5" width="22" height="10" rx="5" fill="#dfe7ef"/><text x="592" y="4" font-size="11">seed range</text>'
    )

    return f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif">
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     style="max-width:100%">
  <text x="20" y="20" font-size="14" font-weight="700">Same runs, four scoring protocols — {html.escape(benchmark)}</text>
  <g transform="translate(0,32)">{legend}</g>
  {''.join(bars)}
</svg>
</div>"""


# --------------------------------------------------------------------------- #
# view 3: the leaderboard that reshuffles by protocol
# --------------------------------------------------------------------------- #
def render_table(benchmark: str) -> str:
    rows = []
    for m in models(benchmark):
        M = matrix(m, benchmark)
        pr, cs = protocols(M), consistency(M)
        if pr:
            rows.append((m, pr, cs))
    rows.sort(key=lambda r: r[1]["single"], reverse=True)

    def cell(v, hot=False):
        col = "#198028" if hot and v >= 0.5 else "#333"
        return f'<td style="text-align:right;padding:3px 12px;color:{col}">{v*100:.0f}%</td>'

    trs = []
    for m, pr, cs in rows:
        trs.append(
            f'<tr style="border-top:1px solid #f0f0f0">'
            f'<td style="padding:3px 12px">{html.escape(short(m))}</td>'
            f'<td style="text-align:right;padding:3px 12px"><b>{pr["single"]*100:.0f}%</b></td>'
            f'{cell(pr["worst"])}{cell(pr["best"])}{cell(pr["maj"])}'
            f'{cell(pr["pass"], hot=True)}'
            f'<td style="text-align:right;padding:3px 12px;color:#c0392b">{cs["flippy"]*100:.0f}%</td>'
            f'<td style="text-align:right;padding:3px 12px">{cs["spread"]*100:.0f}</td>'
            f'</tr>'
        )
    head = ("model", "single-run", "worst seed", "best seed",
            "maj@N", "pass@N", "flippy", "seed Δ")
    ths = "".join(f'<th style="text-align:right;padding:4px 12px;color:#666;'
                  f'font-weight:600">{h}</th>' for h in head[1:])
    return f"""
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:13px">
  <div style="font-weight:700;font-size:14px;margin-bottom:6px">
    {html.escape(benchmark)}: one model, eight "capabilities"</div>
  <table style="border-collapse:collapse">
    <tr><th style="text-align:left;padding:4px 12px;color:#666;font-weight:600">model</th>{ths}</tr>
    {''.join(trs)}
  </table>
  <div style="font-size:11.5px;color:#888;margin-top:6px">
    flippy = % of problems solved on 20–80% of seeds · seed Δ = best minus worst single run (points)
  </div>
</div>"""


# --------------------------------------------------------------------------- #
# notebook entry points
# --------------------------------------------------------------------------- #
def show_heatmap(model: str, benchmark: str):
    from IPython.display import HTML, display
    display(HTML(render_heatmap(model, benchmark)))


def show_protocols(benchmark: str):
    from IPython.display import HTML, display
    display(HTML(render_protocols(benchmark)))


def show_table(benchmark: str):
    from IPython.display import HTML, display
    display(HTML(render_table(benchmark)))


if __name__ == "__main__":
    df = load()
    print(f"{len(df)} runs · {df.Model.nunique()} models · {df.Benchmark.nunique()} benchmarks")
    for b in benchmarks():
        print(f"\n=== {b} ===")
        for m in models(b):
            pr, cs = protocols(matrix(m, b)), consistency(matrix(m, b))
            print(f"  {short(m):22} single {pr['single']*100:5.1f}%  "
                  f"seed[{pr['worst']*100:4.0f}-{pr['best']*100:4.0f}]  "
                  f"maj {pr['maj']*100:4.0f}  pass {pr['pass']*100:4.0f}  "
                  f"flippy {cs['flippy']*100:4.0f}%")
