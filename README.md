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

Seven test-time reasoning strategies on **GSM8K**, compared under a latency budget:

| Strategy | What it does | Latency knob | Why it costs latency |
|---|---|---|---|
| **Input–output** (direct) | Send the question, read back the answer — one call, no reasoning scaffold | *(none)* | The baseline floor: a single point |
| **Self-consistency** | Sample K chains, majority vote | `k` | Chains run **in parallel** → latency ≈ slowest chain |
| **ReAct** (Yao et al. 2022) | Interleave Thought → Action → Observation in a text protocol; `Calculate[…]` / `Finish[…]` actions until done | `max_steps` | Sequential thought→action turns |
| **Agent** (tool-use) | Reason + call a calculator via the native function-calling API until confident | `max_steps` | Sequential tool-call turns |
| **Self-Refine** (Madaan et al. 2023) | Answer, then each round give explicit feedback on the draft and refine it, up to N rounds | `rounds` | Strictly **sequential** (feedback then refine each round) |
| **Tree of Thoughts** | Beam-search: expand several next-steps per level, score, prune to the best | `depth` | Branches fan out **in parallel** per level; `depth` levels run **sequentially** |
| **Fleet of Agents** (Klein & Potamitis) | Particle filter: a fleet of N agents each extend a trajectory, get value-scored, then **resampled** (good clones, bad die) every step | `n_agents` | Fleet steps **in parallel**; resampling concentrates compute on promising paths |

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
- `--strategies iterative,tree_of_thoughts,fleet_of_agents` — restrict the sweep to save time on stage.
- Sweep ranges live in `demo/reasoning_demo/strategies.py` (`STRATEGIES` registry).

The punchline: **there is no single best strategy** — the winner depends on
whether your budget is *latency*, *tokens*, or *dollars*.

---

## Layout

```
notebook.ipynb             # Part 1 stage notebook (crosslingual + instability)
run_demo.py                # Part 2 CLI: demo | bench | plot | trace | live
demo/                      # all live demo material, one launcher
  serve.py                 # unified launcher: `uv run python demo/serve.py`
  reasoning_demo/          # §2.1 — test-time reasoning strategies engine
    client.py              # OpenRouter wrapper (sync + async), latency/token accounting
    mgsm.py                # Part 1: faithful MGSM eval + solvers
    traces.py              # Part 1: load real s1.1 traces, highlight CoT by script
    strategies.py          # Part 2: the seven strategies + sweep registry
    bench.py / plot.py     # Part 2: budget sweep -> results.json -> pareto.png
    extract.py / tools.py  # Part 2: answer parsing, majority vote, calculator tool
    tracevis.py / liveserver.py  # Part 2: static + live SSE trace explorers
    data.py                # GSM8K loader (bundled sample or HF)
  instability/             # §1 — consistency/instability eval views (+ logs, models.parquet)
    attempts.py            # parse + re-grade repeated Game24 attempts
    consistency.py         # seed-variance grid over models.parquet
    game24_tree.py / protocols_app.py  # self-contained HTML visualizers
data/
  gsm8k_sample.jsonl       # 12 GSM8K problems (offline-safe)
  mgsm/                    # MGSM TSVs, 250 problems/language
repos/                     # vendored crosslingual-test-time-scaling (read-only data source)
```
