"""The three inference strategies under comparison.

Each strategy takes (llm, question, **knob) and returns a StrategyResult. The
"knob" is the dial we sweep to trace out an accuracy-vs-latency curve:

    iterative          -> rounds       (sequential refine steps; latency adds up)
    self_consistency   -> k            (parallel samples; latency ~ slowest chain)
    agentic            -> max_steps    (tool-use loop; latency grows with steps)

Every strategy also records a step-level `steps` trace (text, [t0, t1] offsets
in seconds, tokens, and per-kind metadata) for the trace explorer. The trace is
free to ignore -- bench.py only ever reads answer/usage/latency.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .client import LLM, Usage
from .extract import extract_final_number, majority_vote
from .tools import CALCULATOR_TOOL, DISPATCH

SYSTEM = (
    "You are a careful math tutor. Solve the problem step by step, then state "
    "the final numeric answer on its own line in the form '#### <number>'."
)


@dataclass
class StrategyResult:
    strategy: str
    answer: str | None
    latency_s: float
    usage: Usage
    detail: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)  # trace events for visualization


def _step(kind: str, label: str, text: str, t0: float, t1: float,
          tokens: int, **meta) -> dict:
    """One trace event. `kind` drives colour/layout in the trace explorer."""
    return {"kind": kind, "label": label, "text": text or "",
            "t0": round(t0, 3), "t1": round(t1, 3), "tokens": tokens, "meta": meta}


def _emit(emit, ev: dict) -> None:
    """Push a live event (call_start / step) to a streaming sink, if any.

    `emit` is supplied by the live server; when None (bench, notebook, static
    trace) the strategies behave exactly as before and just collect `steps`.
    """
    if emit is not None:
        emit(ev)


# --------------------------------------------------------------------------- #
# 1. Iterative self-refine: answer, then critique-and-revise `rounds` times.
#    Latency is strictly sequential -> grows ~linearly with rounds.
# --------------------------------------------------------------------------- #
def iterative(llm: LLM, question: str, rounds: int = 1, emit=None) -> StrategyResult:
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    usage, steps = Usage(), []

    def call(sid, kind, label, msgs):
        nonlocal usage
        s = now()
        _emit(emit, {"type": "call_start", "sid": sid, "kind": kind,
                     "label": label, "t0": round(s, 3)})
        c = llm.chat(msgs, temperature=0.0)
        e = now()
        usage += c.usage
        st = _step(kind, label, c.text, s, e, c.usage.total_tokens,
                   answer=extract_final_number(c.text))
        steps.append(st)
        _emit(emit, {"type": "step", "sid": sid, **st})
        return c.text

    current = call(0, "solve", "initial solution",
                   [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question}])

    for i in range(rounds):
        current = call(i + 1, "refine", f"refine #{i + 1}", [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": current},
            {"role": "user", "content":
                "Review your solution for arithmetic or reasoning mistakes. "
                "If it is wrong, produce a corrected full solution. If it is "
                "already correct, repeat it. End with '#### <number>'."},
        ])

    return StrategyResult(
        strategy="iterative",
        answer=extract_final_number(current),
        latency_s=time.perf_counter() - t0,
        usage=usage,
        detail={"rounds": rounds},
        steps=steps,
    )


# --------------------------------------------------------------------------- #
# 2. Self-consistency: sample `k` independent chains IN PARALLEL, majority vote.
#    Wall-clock latency ~ slowest single chain, not the sum -> the whole point.
# --------------------------------------------------------------------------- #
def self_consistency(llm: LLM, question: str, k: int = 5,
                     temperature: float = 0.8, emit=None) -> StrategyResult:
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]

    def one_sample(i):
        # Each chain runs in its own thread; emit (a thread-safe queue.put on
        # the server side) is what makes the bars overlap live.
        s = now()
        _emit(emit, {"type": "call_start", "sid": i, "kind": "sample",
                     "label": f"chain {i + 1}", "t0": round(s, 3)})
        c = llm.chat(msgs, temperature=temperature)
        e = now()
        a = extract_final_number(c.text)
        st = _step("sample", f"chain {i + 1}", c.text, s, e,
                   c.usage.total_tokens, answer=a)
        _emit(emit, {"type": "step", "sid": i, **st})
        return i, st, c.usage, a

    with ThreadPoolExecutor(max_workers=k) as pool:
        out = list(pool.map(one_sample, range(k)))

    usage, steps, answers = Usage(), [], []
    for i, st, u, a in out:
        usage += u
        steps.append(st)
        answers.append(a)

    vote = majority_vote(answers)
    tally = dict(Counter(a for a in answers if a is not None))
    end = now()
    vote_step = _step("vote", "majority vote",
                      f"answers = {answers}\nvote = {vote}", end, end, 0,
                      answer=vote, tally=tally)
    steps.append(vote_step)
    _emit(emit, {"type": "step", "sid": "vote", **vote_step})

    return StrategyResult(
        strategy="self_consistency",
        answer=vote,
        latency_s=time.perf_counter() - t0,
        usage=usage,
        detail={"k": k, "samples": answers},
        steps=steps,
    )


# --------------------------------------------------------------------------- #
# 3. Agentic: a tool-use loop. The model may call the calculator until it is
#    ready to answer, up to `max_steps` turns. Sequential -> latency per step.
# --------------------------------------------------------------------------- #
AGENT_SYSTEM = (
    "You are a math problem solver with a calculator tool. Think step by step "
    "and call `calculate` for every arithmetic operation rather than computing "
    "in your head. When you have the result, reply WITHOUT a tool call and end "
    "with the final numeric answer on its own line: '#### <number>'."
)


def agentic(llm: LLM, question: str, max_steps: int = 6, emit=None) -> StrategyResult:
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    usage, steps = Usage(), []
    msgs = [{"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": question}]
    n_tool_calls = 0
    final_text = ""
    sid = 0

    def record(st):
        steps.append(st)
        _emit(emit, {"type": "step", "sid": sid, **st})

    for turn in range(max_steps):
        s = now()
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "think",
                     "label": f"turn {turn + 1}: reason", "t0": round(s, 3)})
        c = llm.chat(msgs, temperature=0.0, tools=[CALCULATOR_TOOL])
        e = now()
        usage += c.usage

        if not c.tool_calls:
            final_text = c.text
            record(_step("final", f"final answer (turn {turn + 1})", c.text, s, e,
                         c.usage.total_tokens, answer=extract_final_number(c.text)))
            break

        calls = "; ".join(f"{tc.function.name}({tc.function.arguments})"
                          for tc in c.tool_calls)
        body = (c.text + "\n\n" if c.text else "") + "→ calls: " + calls
        record(_step("think", f"turn {turn + 1}: reason + tool call",
                     body, s, e, c.usage.total_tokens))
        sid += 1

        # Echo the assistant's tool-call turn, then append each tool result.
        msgs.append({"role": "assistant", "content": c.text or None,
                     "tool_calls": [tc.model_dump() for tc in c.tool_calls]})
        for tc in c.tool_calls:
            n_tool_calls += 1
            expr = ""
            ts = now()
            try:
                args = json.loads(tc.function.arguments or "{}")
                expr = args.get("expression", "")
                result = DISPATCH[tc.function.name](args)
            except Exception as ex:
                result = f"ERROR: {ex}"
            te = now()
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            record(_step("tool", "calculate", f"{expr} = {result}",
                         ts, te, 0, expression=expr, result=result))
            sid += 1

    if not final_text:  # ran out of steps -> force a final answer
        msgs.append({"role": "user",
                     "content": "Stop and give the final answer now: '#### <number>'."})
        s = now()
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "final",
                     "label": "final answer (forced)", "t0": round(s, 3)})
        c = llm.chat(msgs, temperature=0.0)
        e = now()
        usage += c.usage
        final_text = c.text
        record(_step("final", "final answer (forced)", c.text, s, e,
                     c.usage.total_tokens, answer=extract_final_number(c.text)))

    return StrategyResult(
        strategy="agentic",
        answer=extract_final_number(final_text),
        latency_s=time.perf_counter() - t0,
        usage=usage,
        detail={"max_steps": max_steps, "tool_calls": n_tool_calls},
        steps=steps,
    )


# Registry: strategy name -> (callable, knob name, list of knob values to sweep).
STRATEGIES = {
    "iterative": (iterative, "rounds", [0, 1, 2, 3]),
    "self_consistency": (self_consistency, "k", [1, 3, 5, 9]),
    "agentic": (agentic, "max_steps", [2, 4, 6]),
}
