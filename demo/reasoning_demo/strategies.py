"""The three inference strategies under comparison.

Each strategy takes (llm, question, **knob) and returns a StrategyResult. The
"knob" is the dial we sweep to trace out an accuracy-vs-latency curve:

    input_output       -> (none)       (one call; the baseline, a single point)
    iterative          -> rounds       (Self-Refine: sequential feedback+refine
                                        rounds; latency adds up)
    react              -> max_steps    (ReAct: thought->action->observation loop;
                                        sequential, latency per turn)
    self_consistency   -> k            (parallel samples; latency ~ slowest chain)
    agentic            -> max_steps    (tool-use loop; latency grows with steps)
    tree_of_thoughts   -> depth        (parallel branches per level, sequential
                                        levels; latency ~ depth * (propose+score))

Every strategy also records a step-level `steps` trace (text, [t0, t1] offsets
in seconds, tokens, and per-kind metadata) for the trace explorer. The trace is
free to ignore -- bench.py only ever reads answer/usage/latency.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .client import LLM, Usage
from .extract import extract_final_number, majority_vote
from .tools import CALCULATOR_TOOL, DISPATCH, calculate

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
# 0. Input-output (direct): the zero-reasoning baseline. Send the question, read
#    back the answer -- ONE model call, no step-by-step, no refinement, no tools,
#    no search. It is the floor every other strategy is measured against, and a
#    single point on the latency curve (there is no knob to sweep). The `calls`
#    argument exists only so the bench sweep has one configuration; it is fixed
#    at 1 and ignored.
# --------------------------------------------------------------------------- #
IO_SYSTEM = (
    "You are a math problem solver. Read the problem and give the final numeric "
    "answer directly, on its own line in the form '#### <number>'."
)


def input_output(llm: LLM, question: str, calls: int = 1,
                 emit=None) -> StrategyResult:
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    s = now()
    # A single node hanging off the root: no search structure at all -- that
    # absence is the point when this sits next to the other strategies' trees.
    _emit(emit, {"type": "call_start", "sid": 0, "kind": "io",
                 "label": "direct answer", "t0": round(s, 3),
                 "node": "n0", "parent": "root", "depth": 0})
    c = llm.chat([{"role": "system", "content": IO_SYSTEM},
                  {"role": "user", "content": question}], temperature=0.0)
    e = now()
    answer = extract_final_number(c.text)
    st = _step("io", "direct answer", c.text, s, e, c.usage.total_tokens,
               answer=answer, node="n0", parent="root", depth=0)
    _emit(emit, {"type": "step", "sid": 0, **st})

    return StrategyResult(
        strategy="input_output",
        answer=answer,
        latency_s=time.perf_counter() - t0,
        usage=c.usage,
        detail={},
        steps=[st],
    )


# --------------------------------------------------------------------------- #
# 1. Self-Refine (Madaan et al. 2023, Algorithm 1): ONE model plays generator,
#    feedback-provider, and refiner. After an initial answer, each round runs two
#    DISTINCT calls -- FEEDBACK (the model critiques its own latest draft, Eqn 2)
#    then REFINE (it rewrites using that feedback, Eqn 3/4) -- for up to `rounds`
#    iterations. Per Eqn 4 the prompt RETAINS the full history of prior drafts and
#    feedback (y_0, fb_0, ..., y_t, fb_t), so the refiner can learn from earlier
#    mistakes. The stop condition stop(fb_t, t) ends early when the feedback judges
#    the draft already correct. Latency is strictly sequential (feedback then
#    refine, every round) -> grows ~linearly with rounds. Paper uses temperature
#    0.7; the knob `rounds` is the iteration cap (the paper used up to 4).
# --------------------------------------------------------------------------- #
FEEDBACK_PROMPT = (
    "Review your most recent solution above. Give specific, actionable feedback: "
    "check each arithmetic step and the reasoning, and point out concrete "
    "mistakes and exactly what to change. Do NOT write a new solution. If the "
    "solution is completely correct, reply with exactly 'CORRECT' and nothing else."
)
REFINE_PROMPT = (
    "Using the feedback above, write a corrected full solution. If the feedback "
    "says it is already correct, simply restate it. Show your work and end with "
    "the final numeric answer on its own line: '#### <number>'."
)


def iterative(llm: LLM, question: str, rounds: int = 1,
              temperature: float = 0.7, emit=None) -> StrategyResult:
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    usage, steps = Usage(), []
    sid = 0  # running id -> the trace is one chain: solve -> fb -> refine -> ...

    def call(kind, label, msgs, answerable=True):
        # Self-Refine reuses ONE model for every role; only the prompt changes.
        nonlocal usage, sid
        node = f"n{sid}"
        parent = "root" if sid == 0 else f"n{sid - 1}"
        s = now()
        _emit(emit, {"type": "call_start", "sid": sid, "kind": kind,
                     "label": label, "t0": round(s, 3),
                     "node": node, "parent": parent, "depth": sid})
        c = llm.chat(msgs, temperature=temperature)
        e = now()
        usage += c.usage
        # Only drafts carry an answer; feedback is a critique, so don't mine it
        # for a number (the last-number fallback would be spurious).
        st = _step(kind, label, c.text, s, e, c.usage.total_tokens,
                   answer=extract_final_number(c.text) if answerable else None,
                   node=node, parent=parent, depth=sid)
        steps.append(st)
        _emit(emit, {"type": "step", "sid": sid, **st})
        sid += 1
        return c.text

    # y_0 = M(p_gen || x): the initial generation, kept as the running history.
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": question}]
    current = call("solve", "initial solution", msgs)
    msgs.append({"role": "assistant", "content": current})

    rounds_run = 0
    for i in range(rounds):
        # FEEDBACK: fb_t = M(p_fb || x || y_0 || fb_0 || ... || y_t).
        fb = call("feedback", f"feedback #{i + 1}",
                  msgs + [{"role": "user", "content": FEEDBACK_PROMPT}],
                  answerable=False)
        msgs.append({"role": "user", "content": FEEDBACK_PROMPT})
        msgs.append({"role": "assistant", "content": fb})
        rounds_run += 1
        # stop(fb_t, t): the feedback itself is the stopping indicator.
        if fb.strip().upper().startswith("CORRECT"):
            break
        # REFINE: y_{t+1} = M(p_refine || x || y_0 || fb_0 || ... || y_t || fb_t).
        current = call("refine", f"refine #{i + 1}",
                       msgs + [{"role": "user", "content": REFINE_PROMPT}])
        msgs.append({"role": "user", "content": REFINE_PROMPT})
        msgs.append({"role": "assistant", "content": current})

    return StrategyResult(
        strategy="iterative",
        answer=extract_final_number(current),
        latency_s=time.perf_counter() - t0,
        usage=usage,
        detail={"rounds": rounds, "rounds_run": rounds_run},
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
        # the server side) is what makes the bars overlap live. In the tree these
        # are K leaves off the root that converge on the vote node.
        s = now()
        _emit(emit, {"type": "call_start", "sid": i, "kind": "sample",
                     "label": f"chain {i + 1}", "t0": round(s, 3),
                     "node": f"c{i}", "parent": "root", "depth": 0})
        c = llm.chat(msgs, temperature=temperature)
        e = now()
        a = extract_final_number(c.text)
        st = _step("sample", f"chain {i + 1}", c.text, s, e,
                   c.usage.total_tokens, answer=a,
                   node=f"c{i}", parent="root", depth=0)
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
                      answer=vote, tally=tally,
                      node="vote", parents=[f"c{i}" for i in range(k)], depth=1)
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
        # The trace is a chain: reason -> tool -> reason -> tool -> ... -> final.
        st["meta"].update(node=f"n{sid}",
                          parent=("root" if sid == 0 else f"n{sid - 1}"),
                          depth=sid)
        steps.append(st)
        _emit(emit, {"type": "step", "sid": sid, **st})

    for turn in range(max_steps):
        s = now()
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "think",
                     "label": f"turn {turn + 1}: reason", "t0": round(s, 3),
                     "node": f"n{sid}",
                     "parent": ("root" if sid == 0 else f"n{sid - 1}"),
                     "depth": sid})
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
                     "label": "final answer (forced)", "t0": round(s, 3),
                     "node": f"n{sid}",
                     "parent": ("root" if sid == 0 else f"n{sid - 1}"),
                     "depth": sid})
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


# --------------------------------------------------------------------------- #
# 3b. ReAct (Yao et al. 2022): interleave REASONING and ACTING in an explicit
#    text protocol. Each turn the model emits a `Thought` and an `Action`; the
#    environment returns an `Observation`; this repeats until the model decides
#    to `Finish`. Unlike `agentic` (which uses the native function-calling API),
#    ReAct drives everything through plain text -- the model writes the action,
#    we parse it, run it, and feed the observation back. For math the actions are
#    `Calculate[expr]` (the calculator is the environment) and `Finish[answer]`
#    (the terminating decision). Sequential -> latency per turn. Knob: `max_steps`.
# --------------------------------------------------------------------------- #
REACT_SYSTEM = (
    "Solve the math problem with the ReAct format: alternate a Thought and an "
    "Action, and you will be given an Observation after each Action.\n"
    "On every turn output EXACTLY these two lines:\n"
    "Thought: <your reasoning about what to do next>\n"
    "Action: <one action>\n"
    "An Action must be exactly one of:\n"
    "  Calculate[<arithmetic expression>]  — compute a value; the Observation is "
    "the result\n"
    "  Finish[<final numeric answer>]      — give the final numeric answer and stop\n"
    "Use Calculate for every arithmetic step instead of doing it in your head; "
    "when you know the answer, use Finish."
)

_REACT_ACTION = re.compile(r"Action:\s*(Calculate|Finish)\s*\[(.*?)\]",
                           re.IGNORECASE | re.DOTALL)


def _parse_react_action(text: str) -> tuple[str | None, str]:
    """Pull (action, argument) from a ReAct 'Action: Name[arg]' line."""
    m = _REACT_ACTION.search(text or "")
    if not m:
        return None, ""
    return m.group(1).lower(), m.group(2).strip()


def react(llm: LLM, question: str, max_steps: int = 6,
          temperature: float = 0.0, emit=None) -> StrategyResult:
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    usage, steps = Usage(), []
    msgs = [{"role": "system", "content": REACT_SYSTEM},
            {"role": "user", "content": question}]
    sid = 0
    answer = None

    def record(st):
        # The trace is a chain: act -> observe -> act -> observe -> ... -> finish.
        st["meta"].update(node=f"n{sid}",
                          parent=("root" if sid == 0 else f"n{sid - 1}"),
                          depth=sid)
        steps.append(st)
        _emit(emit, {"type": "step", "sid": sid, **st})

    for turn in range(max_steps):
        s = now()
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "act",
                     "label": f"thought + action {turn + 1}", "t0": round(s, 3),
                     "node": f"n{sid}",
                     "parent": ("root" if sid == 0 else f"n{sid - 1}"),
                     "depth": sid})
        c = llm.chat(msgs, temperature=temperature)
        e = now()
        usage += c.usage
        msgs.append({"role": "assistant", "content": c.text})
        act, arg = _parse_react_action(c.text)

        if act == "finish":  # the terminating decision
            answer = extract_final_number(arg) or extract_final_number(c.text)
            record(_step("final", f"finish (turn {turn + 1})", c.text, s, e,
                         c.usage.total_tokens, answer=answer))
            break

        record(_step("act", f"thought + action {turn + 1}", c.text, s, e,
                     c.usage.total_tokens))
        sid += 1

        # Run the action; its result becomes the next Observation.
        ts = now()
        if act == "calculate":
            result = calculate(arg)
        else:  # unparseable / unknown action -> nudge the model back on protocol
            result = ("No valid Action found. Respond with 'Action: "
                      "Calculate[<expr>]' or 'Action: Finish[<answer>]'.")
        te = now()
        msgs.append({"role": "user", "content": f"Observation: {result}"})
        record(_step("observe", f"observation {turn + 1}",
                     f"Observation: {result}", ts, te, 0,
                     expression=arg, result=result))
        sid += 1

    if answer is None:  # ran out of turns -> force a Finish
        msgs.append({"role": "user", "content":
                     "You are out of steps. Output 'Finish[<final numeric "
                     "answer>]' now."})
        s = now()
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "final",
                     "label": "finish (forced)", "t0": round(s, 3),
                     "node": f"n{sid}",
                     "parent": ("root" if sid == 0 else f"n{sid - 1}"),
                     "depth": sid})
        c = llm.chat(msgs, temperature=temperature)
        e = now()
        usage += c.usage
        act, arg = _parse_react_action(c.text)
        answer = (extract_final_number(arg) if act == "finish"
                  else extract_final_number(c.text))
        record(_step("final", "finish (forced)", c.text, s, e,
                     c.usage.total_tokens, answer=answer))

    return StrategyResult(
        strategy="react",
        answer=answer,
        latency_s=time.perf_counter() - t0,
        usage=usage,
        detail={"max_steps": max_steps},
        steps=steps,
    )


# --------------------------------------------------------------------------- #
# 4. Tree of Thoughts: beam search over partial solutions. At each level we
#    expand every surviving "thought" into several candidate next steps (IN
#    PARALLEL), score each one (also in parallel), and prune to the best
#    `breadth`. `depth` levels run sequentially, so latency grows ~linearly with
#    depth while each level fans out like self-consistency -> the timeline shows
#    both the parallel branching and the sequential deepening.
#    (Yao et al. 2023, "Tree of Thoughts"; the "value" evaluator variant.)
# --------------------------------------------------------------------------- #
TOT_PROPOSE = (
    "You are solving a math problem one reasoning step at a time. Given the work "
    "so far, write the SINGLE next step of the solution: one or two sentences of "
    "reasoning with any arithmetic. Do not jump ahead or restate earlier steps. "
    "If the problem is essentially finished, give the final numeric answer and "
    "end with '#### <number>'."
)
TOT_VALUE = (
    "You are grading a partial solution to a math problem. Rate how promising it "
    "is as a path to the correct final answer, from 0 (clearly wrong, stuck, or "
    "off-track) to 10 (on track or essentially solved). Reply with just the "
    "integer score, nothing else."
)


def _parse_score(text: str) -> float:
    """Pull a 0-10 value score out of the evaluator's reply; default 0."""
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    if not m:
        return 0.0
    return max(0.0, min(10.0, float(m.group())))


def _agent_name(i: int) -> str:
    """Spreadsheet-style agent names: 0->A, 25->Z, 26->AA, 27->AB, ..."""
    name = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def tree_of_thoughts(llm: LLM, question: str, depth: int = 2, breadth: int = 2,
                     n_expand: int = 2, temperature: float = 0.7,
                     emit=None) -> StrategyResult:
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    usage, steps = Usage(), []

    def record(sid, st):
        steps.append(st)  # list.append is GIL-atomic; threads may interleave
        _emit(emit, {"type": "step", "sid": sid, **st})

    # Each candidate is a tree node with a stable id and a parent id, so the
    # trace explorer can draw the branching (the `node`/`parent`/`depth` meta is
    # carried on both the call_start and the step events).
    # A node's name encodes its path through the tree: depth-1 branches are
    # lettered A, B, C, ...; each child appends its 1-based position under its
    # parent, so A's kids are A.1/A.2 and A.1's kids are A.1.1/A.1.2.
    def propose(idx: int, parent: dict, d: int) -> dict:
        node = f"n{d}-{idx}"
        path = (chr(ord("A") + idx) if d == 0
                else f"{parent['path']}.{idx % n_expand + 1}")
        sid = f"d{d}-p{idx}"
        s = now()
        label = f"branch {path}"
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "propose",
                     "label": label, "t0": round(s, 3),
                     "node": node, "parent": parent["node"], "depth": d})
        user = question if not parent["text"] else (
            question + "\n\nWork so far:\n" + parent["text"])
        c = llm.chat([{"role": "system", "content": TOT_PROPOSE},
                      {"role": "user", "content": user}], temperature=temperature)
        e = now()
        text = c.text if not parent["text"] else (parent["text"] + "\n" + c.text)
        record(sid, _step("propose", label, c.text, s, e, c.usage.total_tokens,
                          answer=extract_final_number(c.text),
                          node=node, parent=parent["node"], depth=d))
        return {"idx": idx, "node": node, "path": path,
                "text": text.strip(), "usage": c.usage}

    def value(cand: dict, d: int) -> dict:
        idx = cand["idx"]
        sid = f"d{d}-v{idx}"
        s = now()
        label = f"score {cand['path']}"
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "value",
                     "label": label, "t0": round(s, 3),
                     "node": cand["node"], "depth": d})
        c = llm.chat([{"role": "system", "content": TOT_VALUE},
                      {"role": "user", "content":
                       question + "\n\nPartial solution:\n" + cand["text"]}],
                     temperature=0.0)
        e = now()
        score = _parse_score(c.text)
        record(sid, _step("value", label,
                          c.text.strip() + f"\n→ score = {score}", s, e,
                          c.usage.total_tokens, score=score,
                          node=cand["node"], depth=d))
        return {**cand, "score": score, "usage_v": c.usage}

    beam = [{"text": "", "node": "root", "path": ""}]
    for d in range(depth):
        # Expand every beam state into n_expand candidate next-steps, in parallel.
        jobs = [(i, parent) for i, parent in enumerate(
            p for p in beam for _ in range(n_expand))]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            cands = list(pool.map(lambda a: propose(a[0], a[1], d), jobs))
        for c in cands:
            usage += c.pop("usage")
        # Score every candidate, also in parallel, then prune to the best `breadth`.
        with ThreadPoolExecutor(max_workers=len(cands)) as pool:
            scored = list(pool.map(lambda c: value(c, d), cands))
        for c in scored:
            usage += c.pop("usage_v")
        scored.sort(key=lambda c: c["score"], reverse=True)
        beam = scored[:breadth]
        # Mark which nodes survived this level (the explorer fades the rest).
        kept = [c["node"] for c in beam]
        tp = now()
        record(f"d{d}-prune", _step("prune", f"depth {d + 1}: keep top {breadth} "
                                    f"of {len(scored)}", "kept: " + ", ".join(kept),
                                    tp, tp, 0, kept=kept, depth=d))

    # Read the final answer off the best-scoring path (normalising to '#### x').
    best = beam[0]
    sid = "final"
    s = now()
    _emit(emit, {"type": "call_start", "sid": sid, "kind": "final",
                 "label": "best path → final answer", "t0": round(s, 3),
                 "node": "final", "parent": best["node"], "depth": depth})
    c = llm.chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content":
                   question + "\n\nReasoning so far:\n" + best["text"] +
                   "\n\nUsing this reasoning, state the final answer. "
                   "End with '#### <number>'."}], temperature=0.0)
    e = now()
    usage += c.usage
    answer = extract_final_number(c.text)
    record(sid, _step("final", "best path → final answer", c.text, s, e,
                      c.usage.total_tokens, answer=answer, score=best["score"],
                      node="final", parent=best["node"], depth=depth))

    return StrategyResult(
        strategy="tree_of_thoughts",
        answer=answer,
        latency_s=time.perf_counter() - t0,
        usage=usage,
        detail={"depth": depth, "breadth": breadth, "n_expand": n_expand,
                "best_score": best["score"]},
        steps=steps,
    )


# --------------------------------------------------------------------------- #
# 5. Fleet of Agents (Klein & Potamitis): a fixed-size fleet of N agents
#    searches the solution space like a particle filter. Each step every agent
#    extends its partial trajectory by one reasoning step (IN PARALLEL), a value
#    function scores each full trajectory, and the fleet is RESAMPLED in
#    proportion to value -- high-value trajectories get cloned, low-value ones
#    die -- keeping N agents but steering compute toward promising paths while
#    staying stochastic enough to keep exploring. Unlike ToT's deterministic
#    top-`breadth` prune, the population size is fixed and selection is random.
#    Swept knob: `n_agents` (the fleet size); `steps` is fixed.
# --------------------------------------------------------------------------- #
def fleet_of_agents(llm: LLM, question: str, n_agents: int = 4, steps: int = 3,
                    temperature: float = 0.7, emit=None) -> StrategyResult:
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0
    usage, trace = Usage(), []
    rng = random.Random()  # FoA is stochastic by design; left unseeded

    def record(sid, st):
        trace.append(st)
        _emit(emit, {"type": "step", "sid": sid, **st})

    # The tree interleaves two row kinds: a propose+score row at depth 2r (the
    # agents extend and are valued) and a resample row at depth 2r+1 (survivors,
    # each pointing back to the agent it was sampled from). An agent node id is
    # `r{step}-{slot}`; a resample node id is `s{step}-{j}`.
    def step_agent(slot: int, agent: dict, r: int) -> dict:
        node = f"r{r}-{slot}"
        sid = f"r{r}-a{slot}"
        s = now()
        name = agent["name"]
        label = f"agent {name}"
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "agent",
                     "label": label, "t0": round(s, 3),
                     "node": node, "parent": agent["node"], "depth": 2 * r})
        user = question if not agent["text"] else (
            question + "\n\nWork so far:\n" + agent["text"])
        c = llm.chat([{"role": "system", "content": TOT_PROPOSE},
                      {"role": "user", "content": user}], temperature=temperature)
        e = now()
        text = c.text if not agent["text"] else (agent["text"] + "\n" + c.text)
        record(sid, _step("agent", label, c.text, s, e, c.usage.total_tokens,
                          answer=extract_final_number(c.text),
                          node=node, parent=agent["node"], depth=2 * r))
        return {"slot": slot, "node": node, "name": name,
                "text": text.strip(), "usage": c.usage}

    def value(cand: dict, r: int) -> dict:
        slot = cand["slot"]
        sid = f"r{r}-v{slot}"
        s = now()
        label = f"score {cand['name']}"
        _emit(emit, {"type": "call_start", "sid": sid, "kind": "value",
                     "label": label, "t0": round(s, 3),
                     "node": cand["node"], "depth": 2 * r})
        c = llm.chat([{"role": "system", "content": TOT_VALUE},
                      {"role": "user", "content":
                       question + "\n\nPartial solution:\n" + cand["text"]}],
                     temperature=0.0)
        e = now()
        score = _parse_score(c.text)
        record(sid, _step("value", label,
                          c.text.strip() + f"\n→ score = {score}", s, e,
                          c.usage.total_tokens, score=score,
                          node=cand["node"], depth=2 * r))
        return {**cand, "score": score, "usage_v": c.usage}

    # Agents are named A, B, C, ...; on each resample a survivor's copies extend
    # the source's path with a 1-based index (A -> A.1, A.2; A.1 -> A.1.1, ...),
    # so a name encodes the agent's full lineage down the tree.
    fleet = [{"text": "", "node": "root", "name": _agent_name(i)}
             for i in range(n_agents)]
    for r in range(steps):
        # Every agent advances one step, in parallel.
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            cands = list(pool.map(lambda a: step_agent(a[0], a[1], r),
                                  list(enumerate(fleet))))
        for c in cands:
            usage += c.pop("usage")
        # Score each full trajectory, in parallel.
        with ThreadPoolExecutor(max_workers=n_agents) as pool:
            scored = list(pool.map(lambda c: value(c, r), cands))
        for c in scored:
            usage += c.pop("usage_v")
        # Resample N agents in proportion to value (with replacement): good
        # trajectories get cloned, poor ones die. Population size stays N. This
        # is emitted as its own row of survivor nodes (`s{r}-{j}`), each pointing
        # back to the agent it was sampled from.
        weights = [max(c["score"], 0.01) for c in scored]
        survivors = rng.choices(scored, weights=weights, k=n_agents)
        # Name each survivor by extending its source's path with a 1-based index,
        # so A's copies are A.1, A.2 and A.1's copies are A.1.1, A.1.2. A name
        # with multiple siblings (A.1, A.2) is exactly a cloned agent.
        seen: dict = {}
        new_fleet, res_nodes = [], []
        for j, srv in enumerate(survivors):
            src = srv["name"]
            seen[src] = seen.get(src, 0) + 1
            name = f"{src}.{seen[src]}"
            rnode = f"s{r}-{j}"
            res_nodes.append({"node": rnode, "name": name,
                              "parent": srv["node"], "score": srv["score"]})
            new_fleet.append({"text": srv["text"], "node": rnode,
                              "name": name, "score": srv["score"]})
        names = ", ".join(a["name"] for a in new_fleet)
        tp = now()
        record(f"r{r}-resample", _step("resample",
               f"step {r + 1}: resample {n_agents} agents by value",
               f"fleet → {names}", tp, tp, 0,
               nodes=res_nodes, depth=2 * r + 1))
        fleet = new_fleet

    # Take the answer from the highest-value agent in the final (resampled) fleet.
    best = max(fleet, key=lambda c: c["score"])
    final_label = f"agent {best['name']} → final answer"
    sid = "final"
    s = now()
    _emit(emit, {"type": "call_start", "sid": sid, "kind": "final",
                 "label": final_label, "t0": round(s, 3),
                 "node": "final", "parent": best["node"], "depth": 2 * steps})
    c = llm.chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content":
                   question + "\n\nReasoning so far:\n" + best["text"] +
                   "\n\nUsing this reasoning, state the final answer. "
                   "End with '#### <number>'."}], temperature=0.0)
    e = now()
    usage += c.usage
    answer = extract_final_number(c.text)
    record(sid, _step("final", final_label, c.text, s, e,
                      c.usage.total_tokens, answer=answer, score=best["score"],
                      node="final", parent=best["node"], depth=2 * steps))

    return StrategyResult(
        strategy="fleet_of_agents",
        answer=answer,
        latency_s=time.perf_counter() - t0,
        usage=usage,
        detail={"n_agents": n_agents, "steps": steps, "best_score": best["score"]},
        steps=trace,
    )


# Registry: strategy name -> (callable, knob name, list of knob values to sweep).
STRATEGIES = {
    "input_output": (input_output, "calls", [1]),
    "self_consistency": (self_consistency, "k", [1, 3, 5, 9]),
    "react": (react, "max_steps", [2, 4, 6]),
    "agentic": (agentic, "max_steps", [2, 4, 6]),
    "iterative": (iterative, "rounds", [0, 1, 2, 3]),
    "tree_of_thoughts": (tree_of_thoughts, "depth", [1, 2, 3]),
    "fleet_of_agents": (fleet_of_agents, "n_agents", [2, 4, 6]),
}
