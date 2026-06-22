"""Generation port interfaces."""

from __future__ import annotations

from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    """Single chat message for grounded generation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatModel(Protocol):
    """Framework-independent chat completion contract."""

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Return raw model output for the provided messages."""
        ...
