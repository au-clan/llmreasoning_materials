"""Crosslingual MGSM eval — faithful to BatsResearch/crosslingual-test-time-scaling.

This reproduces the `mgsm_direct_*` task from their fork of lm-evaluation-harness
(repos/crosslingual-test-time-scaling), driven through the same OpenRouter `LLM`
client the rest of the tutorial uses so it runs on a laptop.

What is reproduced *exactly* (their setup):
  - Prompt (0-shot "direct"): localized question word + question + "\nAnswer:".
    e.g.  "Question: <q>\nAnswer:"  (en),  "问题: <q>\nAnswer:"  (zh).
    The per-language question word is hardcoded in their task YAMLs.
  - Generation: greedy (temperature = 0), stop on the localized question word,
    newlines, "</s>" and "<|im_end|>".
  - Parsing: the `flexible-extract` filter — regex (-?[$0-9.,]{2,})|(-?[0-9]+),
    take the LAST numeric match — then exact_match against the gold answer_number
    with ignore_case + ignore_punctuation (i.e. numeric normalization).

What is *adapted* for a chat API (the s1 reasoning-model bits):
  - Budget forcing. s1 controls raw thinking tokens inside vLLM. Over a chat API
    we approximate it: cap the reasoning turn at `think_budget` tokens
    (truncation), and inject the localized "Wait" token as `n_wait` follow-up
    turns to *extend* reasoning (extrapolation), then elicit the final answer.
  - Language forcing. Their `system` / `prefix` / `combined` knobs, ported to
    chat: a "think and answer only in <lang>" system prompt and/or an assistant
    reasoning prefill in the target language.

Sources:
  repos/.../lm-evaluation-harness/lm_eval/tasks/mgsm/{utils.py,direct/*}
  repos/.../experiments/language_forcing/eval_scripts/language_forcing.py
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import re
import statistics
import string
from collections import defaultdict
from dataclasses import dataclass, field

from .client import LLM, Usage

HERE = os.path.dirname(os.path.abspath(__file__))
MGSM_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "data", "mgsm"))
# Their pre-computed s1 generations (lm-eval results), shipped in the repo.
S1_ARTIFACTS_DIR = os.path.normpath(os.path.join(
    HERE, "..", "..", "repos", "crosslingual-test-time-scaling",
    "experiments", "crosslingual_mgsm", "artifacts", "s1"))

# Languages covered by the paper / MGSM. QUESTION + REGEX copied verbatim from
# their lm_eval/tasks/mgsm/utils.py (QUESTION drives the prompt + a stop string;
# REGEX is the optional language-specific "strict-match" pattern).
LANGUAGES: dict[str, dict[str, str]] = {
    "bn": {"name": "Bengali",  "QUESTION": "প্রশ্ন:",     "REGEX": r"The answer is (\-?[0-9\.\,]+)"},
    "de": {"name": "German",   "QUESTION": "Frage:",      "REGEX": r"Die Antwort lautet (\-?[0-9\.\,]+)"},
    "en": {"name": "English",  "QUESTION": "Question:",   "REGEX": r"The answer is (\-?[0-9\.\,]+)"},
    "es": {"name": "Spanish",  "QUESTION": "Pregunta:",   "REGEX": r"La respuesta es (\-?[0-9\.\,]+)"},
    "fr": {"name": "French",   "QUESTION": "Question :",  "REGEX": r"La réponse est (\-?[0-9\.\,]+)"},
    "ja": {"name": "Japanese", "QUESTION": "問題:",        "REGEX": r"答えは(\-?[0-9\.\,]+)です。"},
    "ru": {"name": "Russian",  "QUESTION": "Задача:",     "REGEX": r"Ответ — (\-?[0-9\.\,]+)"},
    "sw": {"name": "Swahili",  "QUESTION": "Swali:",      "REGEX": r"Jibu ni (\-?[0-9\.\,]+)"},
    "te": {"name": "Telugu",   "QUESTION": "ప్రశ్న:",      "REGEX": r"సమాధానం (\-?[0-9\.\,]+)"},
    "th": {"name": "Thai",     "QUESTION": "โจทย์:",       "REGEX": r"คำตอบคือ (\-?[0-9\.\,]+)"},
    "zh": {"name": "Chinese",  "QUESTION": "问题:",        "REGEX": r"答案是 (\-?[0-9\.\,]+)。"},
}

# Language-forcing tokens, ported from their language_forcing.py.
LANG_WAIT_TOKENS = {
    "bn": "অপেক্ষা করুন", "de": "Warte", "en": "Wait", "es": "Espera",
    "fr": "Attendez", "ja": "待って", "ru": "Подождите", "sw": "Subiri",
    "te": "వాహించు", "th": "รอ", "zh": "等待",
}
LANG_PREFIX_TOKENS = {
    "en": "Okay, let me try to figure this out.",
    "bn": "আচ্ছা, সমাধান করার চেষ্টা করি।",
    "de": "Okay, ich versuche, das herauszufinden.",
    "es": "Bien, déjame intentar resolver esto.",
    "fr": "D'accord, laissez-moi essayer de résoudre ça.",
    "ja": "よし、解いてみよう。",
    "ru": "Хорошо, давайте попробуем разобраться.",
    "sw": "Sawa, acha nijaribu kutatua hili.",
    "te": "సరే, దీన్ని పరిష్కరించడానికి ప్రయత్నిస్తాను.",
    "th": "โอเค เดี๋ยวฉันลองแก้ปัญหานี้ดู",
    "zh": "好的，让我来试着解答。",
}
LANG_SYSTEM_PROMPT = {
    "en": "You are a helpful assistant. You must think and answer only in English.",
    "bn": "তুমি একজন বুদ্ধিমান সহকারী, শুধুমাত্র বাংলায় চিন্তা ও উত্তর দাও।",
    "de": "Du bist ein hilfreicher Assistent. Du musst nur auf Deutsch denken und antworten.",
    "es": "Eres un asistente útil. Debes pensar y responder solo en español.",
    "fr": "Tu es un assistant utile. Tu dois réfléchir et répondre uniquement en français.",
    "ja": "あなたは有能なアシスタントです。日本語でのみ考え、回答してください。",
    "ru": "Вы полезный помощник. Вы должны мыслить и отвечать только на русском языке.",
    "sw": "Wewe ni msaidizi mwenye msaada. Unapaswa kufikiria na kujibu tu kwa Kiswahili.",
    "te": "నువ్వు సహాయకరమైన సహాయకుడు. నువ్వు తెలుగులోనే ఆలోచించి సమాధానాలు ఇవ్వాలి.",
    "th": "คุณเป็นผู้ช่วยที่เป็นประโยชน์ คุณต้องคิดและตอบเฉพาะเป็นภาษาไทยเท่านั้น",
    "zh": "你是一个有用的助手，你必须只用中文进行思考和回答。",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class MgsmProblem:
    id: str
    lang: str
    question: str
    answer: str  # gold answer_number, as a string


def load_mgsm(lang: str, n: int | None = None, source: str = "local") -> list[MgsmProblem]:
    """Load the MGSM test split for one language.

    source="local" (default) reads the bundled TSVs in data/mgsm/ (250 rows per
    language, "<question>\\t<answer_number>", no header) so the demo runs offline.
    source="hf" pulls juletxara/mgsm via the `datasets` library (the exact dataset
    their harness uses).
    """
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown language {lang!r}. Choose from {list(LANGUAGES)}.")

    probs: list[MgsmProblem] = []
    if source == "hf":
        from datasets import load_dataset  # lazy import

        ds = load_dataset("juletxara/mgsm", lang, split="test")
        for i, ex in enumerate(ds):
            probs.append(MgsmProblem(id=f"mgsm-{lang}-{i}", lang=lang,
                                     question=ex["question"],
                                     answer=str(ex["answer_number"])))
    else:
        path = os.path.join(MGSM_DIR, f"mgsm_{lang}.tsv")
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if not line:
                    continue
                question, answer = line.split("\t")
                probs.append(MgsmProblem(id=f"mgsm-{lang}-{i}", lang=lang,
                                         question=question, answer=answer.strip()))
    return probs[:n] if n else probs


# --------------------------------------------------------------------------- #
# Prompting (faithful to mgsm_direct_*)
# --------------------------------------------------------------------------- #
def build_prompt(question: str, lang: str) -> str:
    """The harness `doc_to_text` for a test doc: '<QUESTION-word> <q>\\nAnswer:'."""
    return f"{LANGUAGES[lang]['QUESTION']} {question}\nAnswer:"


def stop_sequences(lang: str, *, include_newlines: bool = True) -> list[str]:
    """The harness `until` list: newlines + localized question word + EOS markers.

    The harness stops on bare newlines (`["\\n\\n", "\\n"]`). That suits concise /
    reasoning models that emit the answer before a line break, but it truncates a
    verbose chat model's prose solution mid-stream. Pass include_newlines=False to
    keep only the "hard" stops (question word + EOS) and let the model finish.
    """
    stops = [LANGUAGES[lang]["QUESTION"], "</s>", "<|im_end|>"]
    if include_newlines:
        stops = ["\n\n", "\n"] + stops
    return stops


# --------------------------------------------------------------------------- #
# Parsing (faithful to the `flexible-extract` filter + exact_match)
# --------------------------------------------------------------------------- #
_FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def extract_answer(text: str) -> str | None:
    """`flexible-extract`: regex over the text, group_select=-1 (last match)."""
    if not text:
        return None
    matches = [m.group(0) for m in _FLEX.finditer(text)]
    return matches[-1] if matches else None


def _normalize(s: str) -> str:
    """exact_match with ignore_case + ignore_punctuation: lowercase, drop all
    punctuation (so '1,800' == '1800', '$5' == '5'), drop whitespace."""
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return "".join(s.split())


def is_correct(pred: str | None, gold: str) -> bool:
    return pred is not None and _normalize(pred) == _normalize(gold)


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
@dataclass
class SolveResult:
    answer: str | None
    text: str               # full model output used for extraction
    latency_s: float
    usage: Usage
    detail: dict = field(default_factory=dict)


# -- shared building blocks (sync + async solvers use the same message logic) --
def _direct_messages(question: str, lang: str, stop_on_newline: bool):
    prompt = build_prompt(question, lang)
    messages = [{"role": "user", "content": prompt}]
    stop = stop_sequences(lang, include_newlines=stop_on_newline)
    return prompt, messages, stop


def _direct_result(prompt: str, c) -> SolveResult:
    content = getattr(getattr(c, "raw", None), "content", None) or ""
    return SolveResult(answer=extract_answer(c.text), text=c.text,
                       latency_s=c.latency_s, usage=c.usage,
                       detail={"prompt": prompt, "reasoning": getattr(c, "reasoning", ""),
                               "content": content})


def _forced_messages(question: str, lang: str, force_language: bool,
                     force_mode: str, reason_lang: str | None):
    """Build the initial conversation + return (forced_lang, prompt, messages)."""
    flang = reason_lang or lang
    messages: list[dict] = []
    if force_language and force_mode in ("system", "combined"):
        messages.append({"role": "system", "content": LANG_SYSTEM_PROMPT[flang]})
    prompt = build_prompt(question, lang)
    messages.append({"role": "user", "content": prompt})
    # prefix forcing: seed the reasoning in the target language (assistant prefill).
    if force_language and force_mode in ("prefix", "combined"):
        messages.append({"role": "assistant", "content": LANG_PREFIX_TOKENS[flang]})
    return flang, prompt, messages


def _initial_reasoning(messages: list[dict], text: str) -> str:
    return (messages[-1]["content"] + " " + text
            if messages[-1]["role"] == "assistant" else text)


def _wait_conv(messages: list[dict], reasoning: str, wait: str) -> list[dict]:
    conv = messages[:-1] if messages[-1]["role"] == "assistant" else messages
    return conv + [{"role": "assistant", "content": reasoning},
                   {"role": "user", "content": wait}]


def _final_conv(messages: list[dict], prompt: str, reasoning: str) -> list[dict]:
    return ([m for m in messages if m["role"] == "system"]
            + [{"role": "user", "content": prompt},
               {"role": "assistant", "content": reasoning},
               {"role": "user", "content": "Final Answer:"}])


def _forced_result(prompt: str, reasoning: str, final_c, total: Usage,
                   latency: float) -> SolveResult:
    answer_text = final_c.text or reasoning
    return SolveResult(answer=extract_answer(answer_text),
                       text=reasoning + "\n" + final_c.text,
                       latency_s=latency, usage=total,
                       detail={"prompt": prompt, "reasoning": reasoning,
                               "final": final_c.text})


_DIRECT_DOC = """mgsm_direct baseline: single greedy turn over the faithful prompt.

    Prompt + greedy decoding + flexible-extract parsing are faithful to their
    harness. The one knob: stop_on_newline. Their `until` stops on bare newlines,
    which truncates a verbose chat model's solution — so we default it off and let
    flexible-extract read the last number from the full output. Set it True to
    reproduce the harness's exact stop behavior.
    """
_FORCED_DOC = """Reasoning-model-style solve with language forcing + budget forcing.

    force_language: turn on language forcing (their system/prefix/combined knobs).
    force_mode: "system" | "prefix" | "combined".
    reason_lang: language to force reasoning into (defaults to the problem `lang`).
    think_budget: max tokens for the reasoning turn (truncation budget forcing).
    n_wait: number of localized "Wait" follow-ups to extend reasoning (extrapolation).
    """


def solve_direct(llm: LLM, question: str, lang: str,
                 *, max_tokens: int = 768, stop_on_newline: bool = False) -> SolveResult:
    prompt, messages, stop = _direct_messages(question, lang, stop_on_newline)
    c = llm.chat(messages, temperature=0.0, max_tokens=max_tokens, stop=stop)
    return _direct_result(prompt, c)


async def asolve_direct(llm: LLM, question: str, lang: str,
                        *, max_tokens: int = 768, stop_on_newline: bool = False) -> SolveResult:
    prompt, messages, stop = _direct_messages(question, lang, stop_on_newline)
    c = await llm.achat(messages, temperature=0.0, max_tokens=max_tokens, stop=stop)
    return _direct_result(prompt, c)


def solve_forced(llm: LLM, question: str, lang: str, *,
                 force_language: bool = False, force_mode: str = "combined",
                 think_budget: int = 1024, n_wait: int = 0,
                 reason_lang: str | None = None) -> SolveResult:
    flang, prompt, messages = _forced_messages(question, lang, force_language,
                                               force_mode, reason_lang)
    total, latency = Usage(), 0.0
    c = llm.chat(messages, temperature=0.0, max_tokens=think_budget)
    total, latency = total + c.usage, latency + c.latency_s
    reasoning = _initial_reasoning(messages, c.text)

    wait = LANG_WAIT_TOKENS.get(flang, "Wait")
    for _ in range(n_wait):
        c = llm.chat(_wait_conv(messages, reasoning, wait), temperature=0.0,
                     max_tokens=think_budget)
        total, latency = total + c.usage, latency + c.latency_s
        reasoning += f"\n{wait} {c.text}"

    c = llm.chat(_final_conv(messages, prompt, reasoning), temperature=0.0, max_tokens=64)
    total, latency = total + c.usage, latency + c.latency_s
    return _forced_result(prompt, reasoning, c, total, latency)


async def asolve_forced(llm: LLM, question: str, lang: str, *,
                        force_language: bool = False, force_mode: str = "combined",
                        think_budget: int = 1024, n_wait: int = 0,
                        reason_lang: str | None = None) -> SolveResult:
    flang, prompt, messages = _forced_messages(question, lang, force_language,
                                               force_mode, reason_lang)
    total, latency = Usage(), 0.0
    c = await llm.achat(messages, temperature=0.0, max_tokens=think_budget)
    total, latency = total + c.usage, latency + c.latency_s
    reasoning = _initial_reasoning(messages, c.text)

    wait = LANG_WAIT_TOKENS.get(flang, "Wait")
    for _ in range(n_wait):
        c = await llm.achat(_wait_conv(messages, reasoning, wait), temperature=0.0,
                            max_tokens=think_budget)
        total, latency = total + c.usage, latency + c.latency_s
        reasoning += f"\n{wait} {c.text}"

    c = await llm.achat(_final_conv(messages, prompt, reasoning), temperature=0.0, max_tokens=64)
    total, latency = total + c.usage, latency + c.latency_s
    return _forced_result(prompt, reasoning, c, total, latency)


for _f in (solve_direct, asolve_direct):
    _f.__doc__ = _DIRECT_DOC
for _f in (solve_forced, asolve_forced):
    _f.__doc__ = _FORCED_DOC


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #
@dataclass
class MgsmResult:
    lang: str
    accuracy: float
    n: int
    mean_latency_s: float
    mean_total_tokens: float
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    records: list[dict] = field(default_factory=list)


DEFAULT_CONCURRENCY = 8  # max in-flight requests for the async runners


def _aggregate(lang: str, problems: list[MgsmProblem],
               solved: list[SolveResult]) -> MgsmResult:
    records, latencies, ptoks, ctoks, correct = [], [], [], [], 0
    for p, r in zip(problems, solved):
        ok = is_correct(r.answer, p.answer)
        correct += int(ok)
        latencies.append(r.latency_s)
        ptoks.append(r.usage.prompt_tokens)
        ctoks.append(r.usage.completion_tokens)
        records.append({"id": p.id, "question": p.question, "gold": p.answer,
                        "pred": r.answer, "correct": ok, "text": r.text})
    n_ = len(problems)
    mean_p = statistics.mean(ptoks) if ptoks else 0.0
    mean_c = statistics.mean(ctoks) if ctoks else 0.0
    return MgsmResult(
        lang=lang, accuracy=correct / n_ if n_ else 0.0, n=n_,
        mean_latency_s=statistics.mean(latencies) if latencies else 0.0,
        mean_total_tokens=mean_p + mean_c,
        mean_prompt_tokens=mean_p,
        mean_completion_tokens=mean_c,
        records=records,
    )


def run_mgsm(llm: LLM, lang: str, *, n: int = 20, source: str = "local",
             solver=solve_direct, **solver_kwargs) -> MgsmResult:
    """Run MGSM for one language (sequential). `solver` is solve_direct/solve_forced."""
    problems = load_mgsm(lang, n=n, source=source)
    solved = [solver(llm, p.question, lang, **solver_kwargs) for p in problems]
    return _aggregate(lang, problems, solved)


async def arun_mgsm(llm: LLM, lang: str, *, n: int = 20, source: str = "local",
                    solver=asolve_direct, sem: asyncio.Semaphore | None = None,
                    concurrency: int = DEFAULT_CONCURRENCY, **solver_kwargs) -> MgsmResult:
    """Async run_mgsm: solve all `n` problems concurrently (bounded by `sem`).

    `solver` must be an async solver (asolve_direct / asolve_forced). Pass a shared
    `sem` to bound total concurrency across several languages; otherwise one is made.
    """
    problems = load_mgsm(lang, n=n, source=source)
    sem = sem or asyncio.Semaphore(concurrency)

    async def one(p: MgsmProblem) -> SolveResult:
        async with sem:
            return await solver(llm, p.question, lang, **solver_kwargs)

    solved = await asyncio.gather(*(one(p) for p in problems))
    return _aggregate(lang, problems, solved)


async def arun_languages(llm: LLM, langs: list[str], *, n: int = 20, source: str = "local",
                         solver=asolve_direct, concurrency: int = DEFAULT_CONCURRENCY,
                         **solver_kwargs) -> dict[str, MgsmResult]:
    """Run several languages concurrently; returns {lang: MgsmResult}.

    One shared semaphore bounds total in-flight requests across all languages, so
    this stays within rate limits no matter how many languages you pass.
    """
    sem = asyncio.Semaphore(concurrency)

    async def run_lang(lang: str):
        return lang, await arun_mgsm(llm, lang, n=n, source=source, solver=solver,
                                     sem=sem, **solver_kwargs)

    pairs = await asyncio.gather(*(run_lang(l) for l in langs))
    return dict(pairs)


# --------------------------------------------------------------------------- #
# Reproducing the paper's figure (a): "Crosslingual test-time scaling"
# --------------------------------------------------------------------------- #
# Their run dirs are named  s1.1-<size>-mgsm_direct_<lang>_<budget>  (truncation
# budget forcing). The "-wait-" dirs are the extrapolation runs and are skipped.
_RUN_DIR = re.compile(r"^s1\.1-([0-9.]+B)-mgsm_direct_([a-z]+)_([0-9]+)$")
# Model sizes / thinking-token budgets, in the order the figure plots them.
S1_SIZES = ["1.5B", "3B", "7B", "14B", "32B"]
S1_BUDGETS = [500, 1000, 2000, 4000, 8000]


def load_s1_scores(artifacts_dir: str | None = None,
                   metric: str = "flexible-extract") -> dict[tuple[str, int], dict[str, float]]:
    """Parse their shipped lm-eval results into {(size, budget): {lang: accuracy}}.

    Reads `exact_match,<metric>` from each run's results_*.json — the same
    `flexible-extract` number the paper reports. No model calls; pure file parsing.
    """
    root = artifacts_dir or S1_ARTIFACTS_DIR
    scores: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for d in glob.glob(os.path.join(root, "s1.1-*-mgsm_direct_*")):
        m = _RUN_DIR.match(os.path.basename(d))
        if not m:  # skips "-wait-" extrapolation dirs and anything off-pattern
            continue
        size, lang, budget = m.group(1), m.group(2), int(m.group(3))
        results_files = glob.glob(os.path.join(d, "*", "results_*.json"))
        if not results_files:
            continue
        with open(results_files[0]) as f:
            res = json.load(f)["results"][f"mgsm_direct_{lang}"]
        scores[(size, budget)][lang] = res[f"exact_match,{metric}"]
    return dict(scores)


def average_accuracy(scores: dict[tuple[str, int], dict[str, float]],
                     size: str, budget: int, exclude=("en",)) -> float | None:
    """Mean accuracy (%) over languages for one (size, budget), excluding `exclude`."""
    per_lang = scores.get((size, budget), {})
    vals = [v for k, v in per_lang.items() if k not in exclude]
    return 100 * statistics.mean(vals) if vals else None


def run_thinking_sweep(llm: LLM, *, langs: list[str], budgets: list[int] | None = None,
                       n: int = 1, exclude=("en",), source: str = "local",
                       progress: bool = True):
    """Live version of the figure-(a) sweep: truncation budget forcing on MGSM.

    For each thinking-token budget, run `n` problems per language through
    `solve_forced(..., n_wait=0)` — i.e. cap the reasoning at the budget, then
    elicit "Final Answer:" — the same truncation procedure the paper used. Returns
    (avg, detail): `avg` maps budget -> mean accuracy (%) across languages
    EXCLUDING `exclude` (mirroring the figure's "excluding English"); `detail` maps
    budget -> {lang: accuracy}.

    Cost: len(langs) * len(budgets) * n problems, each ~2 model calls.
    """
    budgets = budgets or S1_BUDGETS
    detail: dict[int, dict[str, float]] = {}
    for b in budgets:
        per_lang: dict[str, float] = {}
        for lang in langs:
            res = run_mgsm(llm, lang, n=n, source=source,
                           solver=solve_forced, think_budget=b, n_wait=0)
            per_lang[lang] = res.accuracy
        detail[b] = per_lang
        if progress:
            _print_sweep_row(b, per_lang, exclude)
    return _sweep_avg(detail, budgets, exclude), detail


def _sweep_avg(detail, budgets, exclude):
    return {b: (100 * statistics.mean([v for k, v in detail[b].items() if k not in exclude])
                if any(k not in exclude for k in detail[b]) else float("nan"))
            for b in budgets}


def _print_sweep_row(budget, per_lang, exclude):
    scored = [v for k, v in per_lang.items() if k not in exclude]
    avg_b = 100 * statistics.mean(scored) if scored else float("nan")
    print(f"  budget={budget:<5} avg(excl {','.join(exclude)})={avg_b:5.1f}%   "
          + " ".join(f"{k}:{int(100 * v)}" for k, v in per_lang.items()))


async def arun_thinking_sweep(llm: LLM, *, langs: list[str], budgets: list[int] | None = None,
                              n: int = 1, exclude=("en",), source: str = "local",
                              concurrency: int = DEFAULT_CONCURRENCY, progress: bool = True):
    """Async figure-(a) sweep: fires every (budget × language × problem) concurrently.

    Same result as run_thinking_sweep — (avg, detail) — but all budgets and languages
    run at once under one shared semaphore, so the whole sweep takes about as long as
    its slowest single problem rather than the sum.
    """
    budgets = budgets or S1_BUDGETS
    sem = asyncio.Semaphore(concurrency)

    async def cell(budget: int, lang: str):
        res = await arun_mgsm(llm, lang, n=n, source=source, solver=asolve_forced,
                              sem=sem, think_budget=budget, n_wait=0)
        return budget, lang, res.accuracy

    triples = await asyncio.gather(*(cell(b, l) for b in budgets for l in langs))
    detail: dict[int, dict[str, float]] = {b: {} for b in budgets}
    for b, lang, acc in triples:
        detail[b][lang] = acc
    if progress:
        for b in budgets:
            _print_sweep_row(b, detail[b], exclude)
    return _sweep_avg(detail, budgets, exclude), detail
