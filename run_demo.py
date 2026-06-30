#!/usr/bin/env python3
"""ACL'26 reasoning tutorial — live demo driver.

Usage:
  python run_demo.py demo  [--id gsm8k-3] [--model ...]      # one problem, all strategies side by side
  python run_demo.py bench [--n 12] [--source sample|hf]     # full budget sweep -> results.json
  python run_demo.py plot                                    # results.json -> pareto.png

Set OPENROUTER_API_KEY (and optionally MODEL) in your environment or a .env file.
"""
from __future__ import annotations

import argparse

from demo.reasoning_demo.client import LLM
from demo.reasoning_demo.data import load_problems
from demo.reasoning_demo.extract import is_correct
from demo.reasoning_demo.strategies import (agentic, fleet_of_agents, input_output,
                                       iterative, react, self_consistency,
                                       tree_of_thoughts)


def cmd_demo(args):
    llm = LLM(model=args.model)
    problems = load_problems(source=args.source)
    prob = next((p for p in problems if p.id == args.id), problems[0])

    print(f"\nModel: {llm.model}")
    print(f"Problem [{prob.id}]:\n  {prob.question}")
    print(f"Gold answer: {prob.answer}\n")
    print(f"{'strategy':<20}{'answer':>8}{'correct':>9}{'latency':>10}{'tokens':>9}")
    print("-" * 56)

    runs = [
        ("input-output", lambda: input_output(llm, prob.question)),
        ("self-consistency k=5", lambda: self_consistency(llm, prob.question, k=5)),
        ("react (≤6 steps)", lambda: react(llm, prob.question, max_steps=6)),
        ("agent tool-use", lambda: agentic(llm, prob.question, max_steps=6)),
        ("self-refine (2 rounds)", lambda: iterative(llm, prob.question, rounds=2)),
        ("tree-of-thoughts d=2", lambda: tree_of_thoughts(llm, prob.question, depth=2)),
        ("fleet-of-agents n=4", lambda: fleet_of_agents(llm, prob.question, n_agents=4)),
    ]
    for label, fn in runs:
        r = fn()
        ok = "yes" if is_correct(r.answer, prob.answer) else "NO"
        print(f"{label:<20}{str(r.answer):>8}{ok:>9}"
              f"{r.latency_s:>9.2f}s{r.usage.total_tokens:>9}")
    print()


def cmd_trace(args):
    import os
    import webbrowser

    from demo.reasoning_demo.tracevis import render

    llm = LLM(model=args.model)
    problems = load_problems(source=args.source)
    prob = next((p for p in problems if p.id == args.id), problems[0])

    print(f"Model: {llm.model}\nProblem [{prob.id}]: {prob.question[:70]}...")
    results = []
    for label, fn in [
        ("input_output", lambda: input_output(llm, prob.question)),
        ("self_consistency", lambda: self_consistency(llm, prob.question, k=args.k)),
        ("react", lambda: react(llm, prob.question, max_steps=args.max_steps)),
        ("agentic", lambda: agentic(llm, prob.question, max_steps=args.max_steps)),
        ("iterative", lambda: iterative(llm, prob.question, rounds=args.rounds)),
        ("tree_of_thoughts", lambda: tree_of_thoughts(llm, prob.question, depth=args.depth)),
        ("fleet_of_agents", lambda: fleet_of_agents(llm, prob.question, n_agents=args.n_agents)),
    ]:
        print(f"  running {label} ...", flush=True)
        results.append(fn())

    out = os.path.join(os.path.dirname(__file__), "results", "traces.html")
    path = render(results, prob, llm.model, out)
    print(f"\nTrace explorer -> {path}")
    if not args.no_open:
        webbrowser.open("file://" + path)


def cmd_live(args):
    import threading
    import webbrowser

    from demo.reasoning_demo.liveserver import serve

    url = f"http://{args.host}:{args.port}/"
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    serve(model=args.model, host=args.host, port=args.port)


def cmd_bench(args):
    from demo.reasoning_demo.bench import run_sweep

    strategies = args.strategies.split(",") if args.strategies else None
    run_sweep(model=args.model, source=args.source, n=args.n, strategies=strategies)


def cmd_plot(args):
    from demo.reasoning_demo.plot import plot

    plot()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=None, help="OpenRouter model id")
    common.add_argument("--source", default="sample", choices=["sample", "hf"])

    d = sub.add_parser("demo", parents=[common], help="one problem, all strategies")
    d.add_argument("--id", default="gsm8k-3", help="problem id to demo")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("bench", parents=[common], help="budget sweep")
    b.add_argument("--n", type=int, default=None, help="limit number of problems")
    b.add_argument("--strategies", default=None,
                   help="comma-separated subset, e.g. iterative,agentic")
    b.set_defaults(func=cmd_bench)

    t = sub.add_parser("trace", parents=[common],
                       help="run all strategies on one problem -> interactive HTML")
    t.add_argument("--id", default="gsm8k-3", help="problem id to trace")
    t.add_argument("--rounds", type=int, default=2, help="iterative refine rounds")
    t.add_argument("--k", type=int, default=5, help="self-consistency samples")
    t.add_argument("--max-steps", type=int, default=6, help="agentic step cap")
    t.add_argument("--depth", type=int, default=2, help="tree-of-thoughts beam depth")
    t.add_argument("--n-agents", type=int, default=4, help="fleet-of-agents fleet size")
    t.add_argument("--no-open", action="store_true", help="don't open the browser")
    t.set_defaults(func=cmd_trace)

    lv = sub.add_parser("live", help="start the live streaming trace server")
    lv.add_argument("--model", default=None, help="OpenRouter model id")
    lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=8000)
    lv.add_argument("--no-open", action="store_true", help="don't open the browser")
    lv.set_defaults(func=cmd_live)

    p = sub.add_parser("plot", help="plot results.json")
    p.set_defaults(func=cmd_plot)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
