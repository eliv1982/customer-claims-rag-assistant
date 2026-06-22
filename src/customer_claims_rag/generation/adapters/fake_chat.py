"""Deterministic fake chat model for grounded generation tests."""

from __future__ import annotations

from collections.abc import Sequence

from customer_claims_rag.generation.ports import ChatMessage


class FakeChatModel:
    """In-memory chat model with deterministic behavior."""

    def __init__(
        self,
        *,
        response: str = "",
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.call_count = 0
        self.last_messages: list[ChatMessage] = []

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.call_count += 1
        self.last_messages = list(messages)
        if self._error is not None:
            raise self._error
        return self._response
