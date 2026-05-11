"""Shared LLM rate limiting helpers.

The pipeline mainly needs this for Gemini free-tier runs.  The limiter is
deliberately provider-agnostic, but by default it only turns itself on for
Gemini so paid/OpenAI workflows are not slowed down unless explicitly enabled.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger(__name__)


class LLMRateLimitExceeded(RuntimeError):
    """Raised when the local daily request budget has been exhausted."""


@dataclass(frozen=True)
class LLMRateLimitConfig:
    enabled: bool
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: int
    min_interval_seconds: float
    state_path: Path


class LLMRateLimiter:
    """Simple rolling-window limiter with a persistent daily request counter."""

    def __init__(self, config: LLMRateLimitConfig) -> None:
        self.config = config
        self._request_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()
        self._last_request_at = 0.0

    def wait_for_capacity(self, *, estimated_tokens: int, operation: str) -> None:
        if not self.config.enabled:
            return

        if estimated_tokens > self.config.tokens_per_minute:
            raise LLMRateLimitExceeded(
                f"{operation} estimates {estimated_tokens} tokens, which exceeds "
                f"the configured per-minute token budget of {self.config.tokens_per_minute}."
            )

        daily_count = self._load_daily_count()
        if daily_count >= self.config.requests_per_day:
            raise LLMRateLimitExceeded(
                f"Local LLM daily request budget exhausted "
                f"({daily_count}/{self.config.requests_per_day})."
            )

        while True:
            now = time.monotonic()
            self._drop_old_events(now)

            request_wait = self._seconds_until_request_capacity(now)
            token_wait = self._seconds_until_token_capacity(now, estimated_tokens)
            interval_wait = max(0.0, self.config.min_interval_seconds - (now - self._last_request_at))
            wait_seconds = max(request_wait, token_wait, interval_wait)

            if wait_seconds <= 0:
                break

            LOGGER.info(
                "Waiting %.1fs before %s to respect configured LLM free-tier limits",
                wait_seconds,
                operation,
            )
            time.sleep(wait_seconds)

        now = time.monotonic()
        self._request_times.append(now)
        self._token_events.append((now, estimated_tokens))
        self._last_request_at = now
        self._increment_daily_count()

    def _drop_old_events(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def _seconds_until_request_capacity(self, now: float) -> float:
        if len(self._request_times) < self.config.requests_per_minute:
            return 0.0
        return max(0.0, 60.0 - (now - self._request_times[0]) + 0.05)

    def _seconds_until_token_capacity(self, now: float, estimated_tokens: int) -> float:
        used_tokens = sum(tokens for _, tokens in self._token_events)
        if used_tokens + estimated_tokens <= self.config.tokens_per_minute:
            return 0.0
        if not self._token_events:
            return 0.0
        return max(0.0, 60.0 - (now - self._token_events[0][0]) + 0.05)

    def _today_key(self) -> str:
        return datetime.now().date().isoformat()

    def _read_state(self) -> dict:
        path = self.config.state_path
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_state(self, state: dict) -> None:
        path = self.config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_daily_count(self) -> int:
        state = self._read_state()
        today = self._today_key()
        if state.get("date") != today:
            return 0
        try:
            return int(state.get("request_count", 0))
        except (TypeError, ValueError):
            return 0

    def _increment_daily_count(self) -> None:
        state = self._read_state()
        today = self._today_key()
        if state.get("date") != today:
            state = {"date": today, "request_count": 0}
        state["request_count"] = int(state.get("request_count", 0)) + 1
        self._write_state(state)


_LIMITERS: dict[str, LLMRateLimiter] = {}


def estimate_llm_tokens(*texts: str) -> int:
    """Conservatively estimate request tokens without a provider tokenizer.

    For mixed Japanese/SNS text, character count is a safer upper-ish bound than
    English-style four-characters-per-token heuristics.  The fixed overhead
    covers JSON schema, system prompts, and provider request framing.
    """
    char_count = sum(len(text or "") for text in texts)
    return max(1, char_count + 1000)


def limited_llm_call(
    provider: str,
    model: str,
    operation: str,
    payload_text: str,
    request_fn: Callable[[], object],
):
    """Run an LLM request after waiting for configured capacity."""
    limiter = get_llm_rate_limiter(provider, model)
    limiter.wait_for_capacity(
        estimated_tokens=estimate_llm_tokens(payload_text),
        operation=operation,
    )
    return request_fn()


def get_llm_rate_limiter(provider: str, model: str) -> LLMRateLimiter:
    provider_key = provider.strip().lower()
    model_key = model.strip().lower()
    key = f"{provider_key}:{model_key}"
    limiter = _LIMITERS.get(key)
    if limiter is None:
        limiter = LLMRateLimiter(_load_rate_limit_config(provider_key, model_key))
        _LIMITERS[key] = limiter
    return limiter


def _load_rate_limit_config(provider: str, model: str) -> LLMRateLimitConfig:
    del model
    default_enabled = provider == "gemini"
    enabled = _env_bool("LLM_FREE_TIER_LIMITING", default_enabled)
    if provider == "gemini":
        enabled = _env_bool("GEMINI_FREE_TIER_LIMITING", enabled)

    rpm = _env_int("LLM_REQUESTS_PER_MINUTE", _env_int("GEMINI_REQUESTS_PER_MINUTE", 10))
    tpm = _env_int("LLM_TOKENS_PER_MINUTE", _env_int("GEMINI_TOKENS_PER_MINUTE", 250_000))
    rpd = _env_int("LLM_REQUESTS_PER_DAY", _env_int("GEMINI_REQUESTS_PER_DAY", 250))

    default_min_interval = 60.0 / max(rpm, 1)
    min_interval = _env_float(
        "LLM_MIN_SECONDS_BETWEEN_REQUESTS",
        _env_float("GEMINI_MIN_SECONDS_BETWEEN_REQUESTS", default_min_interval),
    )
    state_path = Path(os.getenv("LLM_RATE_LIMIT_STATE_PATH", "data/reports/llm_rate_limit_state.json"))

    return LLMRateLimitConfig(
        enabled=enabled,
        requests_per_minute=max(1, rpm),
        tokens_per_minute=max(1, tpm),
        requests_per_day=max(1, rpd),
        min_interval_seconds=max(0.0, min_interval),
        state_path=state_path,
    )


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default
