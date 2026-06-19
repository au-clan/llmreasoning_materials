# Crosslingual Reasoning & Test-Time Scaling — ACL'26 Tutorial Demo

Live-demo materials in **two parts**, both driven through **OpenRouter** so they
run on a laptop with no local model or GPU:

1. **Crosslingual notebook** — *Does a model reason in the language you ask in?*
2. **Live reasoning-strategies demo** — *Which strategy wins under a latency budget?*

## Setup

```bash
uv sync                      # install dependencies (pyproject.toml / uv.lock)
```

Create a `.env` file with your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-...
MODEL=openai/gpt-4o-mini     # optional: the default model
```

---

## Part 1 — Crosslingual notebook

`crosslingual_demo.ipynb` chases one question: when you pose a math problem in
French, Russian, or Chinese, **what language does the model actually reason in?**

Open it in Jupyter or VS Code (with this project's `uv` venv selected as the
kernel) and run top to bottom. It walks through:

0. **Solve GSM8K** — one English problem, end to end (prompt → answer).
1. **Solve MGSM** — the *same* problem in other languages; watch accuracy degrade.
2. **A SOTA reasoning model** — load the paper's *real* shipped s1.1 generations
   and see it answer well across languages (best in English).
3. **Inside look** — highlight a real chain-of-thought by writing system: the
   model reasons in English even on a Russian problem, then translates the final
   answer back.
4. **A reasoning model live** — split a model's **thinking** vs **output** channels
   and compare them across languages.

It runs on the bundled MGSM data (`data/mgsm/`, 250 problems/language, offline)
and the precomputed s1.1 results in `repos/` — only the live cells call OpenRouter.

## Part 2 — Live reasoning-strategies demo

Three test-time reasoning strategies on **GSM8K**, compared under a latency budget:

| Strategy | What it does | Latency knob | Why it costs latency |
|---|---|---|---|
| **Iterative** (self-refine) | Answer, then critique-and-revise N times | `rounds` | Strictly **sequential** |
| **Self-consistency** | Sample K chains, majority vote | `k` | Chains run **in parallel** → latency ≈ slowest chain |
| **Agentic** (tool use) | Reason + call a calculator until confident | `max_steps` | Sequential tool-call turns |

The headline artifact is an **accuracy-vs-latency Pareto plot**: read it at any
latency budget to see which strategy wins there.

```bash
# one problem, all three strategies side by side:
uv run python run_demo.py demo --id gsm8k-3

# budget sweep (the numbers behind the plot) -> results/results.json:
uv run python run_demo.py bench            # bundled 12-problem sample, offline
uv run python run_demo.py bench --source hf --n 100   # real GSM8K test split (needs `datasets`)

# the plot:
uv run python run_demo.py plot             # writes results/pareto.png

# static trace explorer (self-contained results/traces.html):
uv run python run_demo.py trace --id gsm8k-3

# LIVE streaming trace server (streams every step over SSE as it runs):
uv run python run_demo.py live                       # http://127.0.0.1:8000
uv run python run_demo.py live --no-open --port 8011 # headless / custom port
```

**Knobs to turn live:**
- `--model` — swap the backing model (smaller models make self-consistency and
  tools matter more).
- `--strategies iterative,agentic` — restrict the sweep to save time on stage.
- Sweep ranges live in `reasoning_demo/strategies.py` (`STRATEGIES` registry).

The punchline: **there is no single best strategy** — the winner depends on
whether your budget is *latency*, *tokens*, or *dollars*.

---

## Layout

```
crosslingual_demo.ipynb    # Part 1 notebook
run_demo.py                # Part 2 CLI: demo | bench | plot | trace | live
reasoning_demo/
  client.py                # OpenRouter wrapper (sync + async), latency/token accounting
  mgsm.py                  # Part 1: faithful MGSM eval + solvers
  traces.py                # Part 1: load real s1.1 traces, highlight CoT by script
  strategies.py            # Part 2: the three strategies + sweep registry
  bench.py / plot.py       # Part 2: budget sweep -> results.json -> pareto.png
  extract.py / tools.py    # Part 2: answer parsing, majority vote, calculator tool
  tracevis.py / liveserver.py  # Part 2: static + live SSE trace explorers
  data.py                  # GSM8K loader (bundled sample or HF)
data/
  gsm8k_sample.jsonl       # 12 GSM8K problems (offline-safe)
  mgsm/                    # MGSM TSVs, 250 problems/language
repos/                     # vendored crosslingual-test-time-scaling (read-only data source)
```
