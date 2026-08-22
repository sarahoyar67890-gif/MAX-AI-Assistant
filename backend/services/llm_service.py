"""
Centralized LLM access.

Two models are exposed on purpose:
    - fast_llm  -> cheap/small model, used for routing & classification
                   (this is the "don't use a cannon to kill a fly" optimization)
    - chat_llm  -> full model, used for actual response generation

Every call goes through `invoke_with_retry`, which adds timeout + retry
handling so a transient Groq API hiccup doesn't crash the whole graph.
"""

import time
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage

from backend.config.settings import settings


class LLMError(Exception):
    """Raised when the LLM cannot be reached after all retries."""


@lru_cache(maxsize=4)
def _get_model(model_name: str, temperature: float) -> ChatGroq:
    if not settings.GROQ_API_KEY:
        raise LLMError(
            "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
        )
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=settings.GROQ_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


def fast_llm(temperature: float = 0.0) -> ChatGroq:
    """Small/cheap model — use for routing, classification, simple checks."""
    return _get_model(settings.FAST_MODEL, temperature)


def chat_llm(temperature: float = 0.4) -> ChatGroq:
    """Full model — use for actual user-facing response generation."""
    return _get_model(settings.CHAT_MODEL, temperature)


def invoke_with_retry(llm, messages: list[BaseMessage], max_retries: int = None):
    """
    Wraps llm.invoke() with retry + backoff. Returns the AIMessage on
    success, raises LLMError after exhausting retries so callers can
    handle failure gracefully instead of the whole request 500-ing.
    """
    max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))  # simple linear backoff
            continue

    raise LLMError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4) — good enough for cost/perf display
    without needing a real tokenizer dependency."""
    return max(1, len(text) // 4)
