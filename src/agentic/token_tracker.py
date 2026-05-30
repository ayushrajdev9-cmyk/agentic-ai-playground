"""Token usage tracking and rate limiting for LLM calls."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def add(self, prompt: int, completion: int, cost_per_1k: float = 0.0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.cost += ((prompt + completion) / 1000) * cost_per_1k


@dataclass
class RateLimit:
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    _request_times: list[float] = field(default_factory=list)
    _token_counts: list[tuple[float, int]] = field(default_factory=list)

    def check(self, estimated_tokens: int = 0) -> bool:
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        self._token_counts = [(t, c) for t, c in self._token_counts if now - t < 60]

        if len(self._request_times) >= self.requests_per_minute:
            return False

        recent_tokens = sum(c for _, c in self._token_counts)
        if recent_tokens + estimated_tokens > self.tokens_per_minute:
            return False

        return True

    def record(self, tokens: int = 0) -> None:
        now = time.time()
        self._request_times.append(now)
        self._token_counts.append((now, tokens))


class TokenTracker:
    def __init__(self):
        self._sessions: dict[str, TokenUsage] = defaultdict(TokenUsage)
        self._rate_limiter = RateLimit()

    def get_session(self, session_id: str = "default") -> TokenUsage:
        return self._sessions[session_id]

    def record_usage(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o",
    ) -> None:
        cost_per_1k = {"gpt-4o": 0.01, "gpt-4o-mini": 0.002, "gpt-3.5-turbo": 0.002}.get(model, 0.01)
        self._sessions[session_id].add(prompt_tokens, completion_tokens, cost_per_1k)
        self._rate_limiter.record(prompt_tokens + completion_tokens)

    def summary(self) -> dict[str, Any]:
        total = TokenUsage()
        for usage in self._sessions.values():
            total.prompt_tokens += usage.prompt_tokens
            total.completion_tokens += usage.completion_tokens
            total.total_tokens += usage.total_tokens
            total.cost += usage.cost

        return {
            "total_tokens": total.total_tokens,
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
            "estimated_cost_usd": round(total.cost, 4),
            "sessions": len(self._sessions),
        }

    def reset(self, session_id: str | None = None) -> None:
        if session_id:
            self._sessions.pop(session_id, None)
        else:
            self._sessions.clear()


tracker = TokenTracker()
