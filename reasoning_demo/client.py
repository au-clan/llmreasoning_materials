"""Thin OpenRouter wrapper (OpenAI-compatible) with latency + token accounting.

A single LLM instance is shared across strategies so that token/latency
bookkeeping is comparable. Every call is timed with a monotonic clock.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI, OpenAI

try:  # optional: load OPENROUTER_API_KEY / MODEL from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv not installed -> rely on real env vars
    pass

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("MODEL", "openai/gpt-4o-mini")


@dataclass
class Usage:
    """Running totals so a strategy can report its compute cost."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0      # all generated tokens (incl. reasoning)
    reasoning_tokens: int = 0       # "thinking" tokens, when the model reports them

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def output_tokens(self) -> int:
        """Visible answer tokens = generated minus thinking."""
        return max(0, self.completion_tokens - self.reasoning_tokens)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class Completion:
    text: str
    latency_s: float
    usage: Usage
    raw: object = field(repr=False, default=None)
    tool_calls: list = field(default_factory=list)
    reasoning: str = ""   # the model's thinking text, when exposed separately


class LLM:
    """One model, reused everywhere. Thread-safe for concurrent .chat calls
    (the OpenAI client uses a connection pool under the hood)."""

    def __init__(self, model: str | None = None, temperature: float = 0.7):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and "
                "fill it in, or `export OPENROUTER_API_KEY=...`."
            )
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self._api_key = api_key
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self._aclient: AsyncOpenAI | None = None  # created lazily on first achat

    @property
    def aclient(self) -> AsyncOpenAI:
        """Async OpenAI client, created on first use (for concurrent .achat calls)."""
        if self._aclient is None:
            self._aclient = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=self._api_key)
        return self._aclient

    def _request_kwargs(self, messages, temperature, tools, max_tokens, stop) -> dict:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    @staticmethod
    def _to_completion(resp, latency: float) -> Completion:
        choices = getattr(resp, "choices", None)
        if not choices:
            # OpenRouter returns some failures in-band (HTTP 200 with an `error`
            # field and no choices). Surface a clear message instead of crashing.
            err = getattr(resp, "error", None)
            if err is None and getattr(resp, "model_extra", None):
                err = resp.model_extra.get("error")
            detail = err.get("message") if isinstance(err, dict) else err
            raise RuntimeError(
                f"No choices returned (model={getattr(resp, 'model', None)!r}): "
                f"{detail or 'empty response'}")
        msg = choices[0].message
        reasoning = getattr(msg, "reasoning", None) or ""
        # Reasoning models put their text in `reasoning` when `content` is empty.
        text = msg.content or reasoning or ""
        u = resp.usage
        details = getattr(u, "completion_tokens_details", None)
        usage = Usage(
            calls=1,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            reasoning_tokens=(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        )
        return Completion(
            text=text,
            reasoning=reasoning,
            latency_s=latency,
            usage=usage,
            raw=msg,
            tool_calls=list(msg.tool_calls or []),
        )

    def chat(self, messages: list[dict], *, temperature: float | None = None,
             tools: list | None = None, max_tokens: int = 1024,
             stop: list[str] | None = None) -> Completion:
        kwargs = self._request_kwargs(messages, temperature, tools, max_tokens, stop)
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        return self._to_completion(resp, time.perf_counter() - t0)

    async def achat(self, messages: list[dict], *, temperature: float | None = None,
                    tools: list | None = None, max_tokens: int = 1024,
                    stop: list[str] | None = None) -> Completion:
        """Async twin of .chat — lets callers fire many requests concurrently."""
        kwargs = self._request_kwargs(messages, temperature, tools, max_tokens, stop)
        t0 = time.perf_counter()
        resp = await self.aclient.chat.completions.create(**kwargs)
        return self._to_completion(resp, time.perf_counter() - t0)
