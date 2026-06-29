"""Budget sweep: for each strategy and each knob value, measure mean latency
and accuracy over the dataset. Writes results/results.json for plotting.
"""
from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, dataclass

from .client import LLM
from .data import Problem, load_problems
from .extract import is_correct
from .strategies import STRATEGIES

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "results.json")


@dataclass
class ConfigResult:
    strategy: str
    knob: str
    knob_value: int
    accuracy: float
    mean_latency_s: float
    p50_latency_s: float
    p90_latency_s: float
    mean_total_tokens: float
    n: int


def run_config(llm: LLM, fn, knob: str, value: int,
               problems: list[Problem]) -> ConfigResult:
    latencies, tokens, correct = [], [], 0
    for p in problems:
        res = fn(llm, p.question, **{knob: value})
        latencies.append(res.latency_s)
        tokens.append(res.usage.total_tokens)
        correct += int(is_correct(res.answer, p.answer))
    n = len(problems)
    s = sorted(latencies)
    return ConfigResult(
        strategy=fn.__name__,
        knob=knob,
        knob_value=value,
        accuracy=correct / n,
        mean_latency_s=statistics.mean(latencies),
        p50_latency_s=s[int(0.50 * (n - 1))],
        p90_latency_s=s[int(0.90 * (n - 1))],
        mean_total_tokens=statistics.mean(tokens),
        n=n,
    )


def run_sweep(model: str | None = None, source: str = "sample",
              n: int | None = None,
              strategies: list[str] | None = None) -> list[ConfigResult]:
    llm = LLM(model=model)
    problems = load_problems(source=source, n=n)
    names = strategies or list(STRATEGIES)
    results: list[ConfigResult] = []

    print(f"Model: {llm.model}  |  {len(problems)} problems  |  "
          f"strategies: {', '.join(names)}\n", flush=True)
    for name in names:
        fn, knob, values = STRATEGIES[name]
        for v in values:
            cr = run_config(llm, fn, knob, v, problems)
            results.append(cr)
            print(f"  {name:<17} {knob}={v:<2}  "
                  f"acc={cr.accuracy:5.1%}  "
                  f"mean_lat={cr.mean_latency_s:6.2f}s  "
                  f"p90={cr.p90_latency_s:6.2f}s  "
                  f"tok={cr.mean_total_tokens:7.0f}", flush=True)

    os.makedirs(os.path.dirname(os.path.normpath(RESULTS_PATH)), exist_ok=True)
    with open(os.path.normpath(RESULTS_PATH), "w") as f:
        json.dump({"model": llm.model, "n": len(problems),
                   "results": [asdict(r) for r in results]}, f, indent=2)
    print(f"\nSaved -> {os.path.normpath(RESULTS_PATH)}")
    return results
