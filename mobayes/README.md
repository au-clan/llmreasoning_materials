# reasoning_ACL

Companion notebook for the MoBayes §6 ACL 2026 tutorial slot.

## Layout

    reasoning_ACL/
      reasoning_ACL.ipynb     # the notebook
      mobayes/                # lightweight inline Bayesian engine (~250 LoC)
        __init__.py
        engine.py
      data/
        ddxplus/              # DDxPlus train-split KB + 500 patient profiles
          diseases.csv
          features.csv
          priors.csv
          likelihood_counts.csv
          patients.jsonl
      assets/                 # MoBayes paper figures (PNG)
      .env                    # (not shipped) OPENAI_API_KEY=sk-...

## Run

1.  Put your OpenAI key in `./.env` (one line):

        OPENAI_API_KEY=sk-...

    The §6.2–§6.5 demos all call `gpt-5.4-nano` live; §6.8 (MoBayes audit
    trace) needs no key.

2.  Install dependencies:

    pip install openai python-dotenv numpy pandas

## §6 quick map

| Cell | What                                                               |
| ---- | ------------------------------------------------------------------ |
| 6.1  | Three failure modes of LLM-only clinical reasoning (md)            |
| 6.2  | Demo: probability inconsistency (live LLM)                         |
| 6.3  | Demo: coarse / unanchored rankings (live LLM)                      |
| 6.4  | Demo: single-shot vs multi-turn degradation (live LLM)             |
| 6.5  | Demo: opaque LLM doctor transcript (live LLM)                      |
| 6.6  | MoBayes architecture (md + figure)                                 |
| 6.7  | KB construction from DDxPlus train split (md)                      |
| 6.8  | MoBayes audit trace (deterministic; `mobayes` package, no API key) |
| 6.9  | What separation buys (md + figures)                                |

## Lightweight `mobayes/`

`mobayes/engine.py` is a teaching subset of the full MoBayes engine. It
implements:

- `load_kb(data_dir)` — read the four KB CSVs, build a Dirichlet
  likelihood `P(feature = v | disease)` from co-occurrence counts.
- `WorldModel` — closed-form Jeffrey-style soft update on the posterior,
  plus `predict_feature_distribution` and `simulate_update`
  counterfactuals for EIG.
- `eig_over(wm, candidates)` — one-step expected information gain sweep.
- `StoppingRules(tau, max_q, min_q)` — `should_stop(wm, num_q)` returns
  `(done, reason)`.

The full implementation (taxonomy, hierarchical EIG, top-k focused EIG,
deployment hooks) lives at the MoBayes repo referenced in the §6 Sources.
