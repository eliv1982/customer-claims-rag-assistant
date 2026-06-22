"""Unit tests for FakeChatModel."""

from __future__ import annotations

import pytest

from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
from customer_claims_rag.generation.ports import ChatMessage


def test_deterministic_response() -> None:
    model = FakeChatModel(response='{"response_mode":"insufficient_context","answer":""}')
    output = model.complete([ChatMessage(role="user", content="hello")])
    assert "insufficient_context" in output


def test_call_count() -> None:
    model = FakeChatModel(response="ok")
    model.complete([ChatMessage(role="user", content="one")])
    model.complete([ChatMessage(role="user", content="two")])
    assert model.call_count == 2


def test_captured_request() -> None:
    model = FakeChatModel(response="ok")
    messages = [
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="user"),
    ]
    model.complete(messages)
    assert model.last_messages == messages


def test_configured_exception() -> None:
    model = FakeChatModel(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        model.complete([ChatMessage(role="user", content="x")])
