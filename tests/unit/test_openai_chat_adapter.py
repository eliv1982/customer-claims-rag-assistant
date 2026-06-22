"""Unit tests for OpenAI chat adapter."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from customer_claims_rag.exceptions import LLMCallError
from customer_claims_rag.generation.adapters.openai_chat import OpenAIChatAdapter
from customer_claims_rag.generation.ports import ChatMessage


class StubChatClient:
    def __init__(
        self,
        *,
        response: BaseMessage | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.invoke_count = 0
        self.last_messages: list[BaseMessage] = []

    def invoke(self, messages: Sequence[BaseMessage], /) -> BaseMessage:
        self.invoke_count += 1
        self.last_messages = list(messages)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def test_maps_system_role() -> None:
    client = StubChatClient(response=AIMessage(content='{"ok":true}'))
    adapter = OpenAIChatAdapter(client=client)
    adapter.complete([ChatMessage(role="system", content="system text")])

    assert isinstance(client.last_messages[0], SystemMessage)
    assert client.last_messages[0].content == "system text"


def test_maps_user_role() -> None:
    client = StubChatClient(response=AIMessage(content='{"ok":true}'))
    adapter = OpenAIChatAdapter(client=client)
    adapter.complete([ChatMessage(role="user", content="user text")])

    assert isinstance(client.last_messages[0], HumanMessage)


def test_maps_assistant_role() -> None:
    client = StubChatClient(response=AIMessage(content='{"ok":true}'))
    adapter = OpenAIChatAdapter(client=client)
    adapter.complete([ChatMessage(role="assistant", content="assistant text")])

    assert isinstance(client.last_messages[0], AIMessage)


def test_input_messages_not_mutated() -> None:
    client = StubChatClient(response=AIMessage(content='{"ok":true}'))
    adapter = OpenAIChatAdapter(client=client)
    messages = [
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="user"),
    ]
    original = list(messages)
    adapter.complete(messages)
    assert messages == original


def test_client_invoke_called_once() -> None:
    client = StubChatClient(response=AIMessage(content='{"ok":true}'))
    adapter = OpenAIChatAdapter(client=client)
    adapter.complete([ChatMessage(role="user", content="hello")])
    assert client.invoke_count == 1


def test_string_content_returned() -> None:
    client = StubChatClient(
        response=AIMessage(content='{"response_mode":"insufficient_context","answer":""}')
    )
    adapter = OpenAIChatAdapter(client=client)
    output = adapter.complete([ChatMessage(role="user", content="hello")])
    assert "insufficient_context" in output


def test_text_block_content_extracted() -> None:
    client = StubChatClient(
        response=AIMessage(content=[{"type": "text", "text": '{"response_mode":"grounded_answer","answer":"x"}'}])
    )
    adapter = OpenAIChatAdapter(client=client)
    output = adapter.complete([ChatMessage(role="user", content="hello")])
    assert "grounded_answer" in output


def test_empty_string_rejected() -> None:
    client = StubChatClient(response=AIMessage(content=""))
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="empty"):
        adapter.complete([ChatMessage(role="user", content="hello")])


def test_whitespace_only_rejected() -> None:
    client = StubChatClient(response=AIMessage(content="   "))
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="empty"):
        adapter.complete([ChatMessage(role="user", content="hello")])


def test_non_text_content_rejected() -> None:
    client = StubChatClient(response=AIMessage(content=[{"type": "image_url", "image_url": "x"}]))
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="empty"):
        adapter.complete([ChatMessage(role="user", content="hello")])


def test_client_exception_wrapped() -> None:
    client = StubChatClient(error=RuntimeError("network down"))
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="request failed") as exc_info:
        adapter.complete([ChatMessage(role="user", content="hello")])
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_adapter_does_not_parse_json() -> None:
    client = StubChatClient(response=AIMessage(content="not-json"))
    adapter = OpenAIChatAdapter(client=client)
    output = adapter.complete([ChatMessage(role="user", content="hello")])
    assert output == "not-json"


def test_existing_llm_call_error_re_raised_unchanged() -> None:
    inner = LLMCallError("inner")
    client = StubChatClient(error=inner)
    adapter = OpenAIChatAdapter(client=client)

    with pytest.raises(LLMCallError) as exc_info:
        adapter.complete([ChatMessage(role="user", content="hello")])

    assert exc_info.value is inner
    assert str(exc_info.value) == "inner"


class TextOnlyResponse:
    def text(self) -> str:
        return '{"response_mode":"grounded_answer","answer":"x"}'


def test_response_without_content_uses_text_method() -> None:
    client = StubChatClient(response=TextOnlyResponse())  # type: ignore[arg-type]
    adapter = OpenAIChatAdapter(client=client)
    output = adapter.complete([ChatMessage(role="user", content="hello")])
    assert "grounded_answer" in output


class NoContentNoTextResponse:
    pass


def test_response_without_content_or_text_rejected() -> None:
    client = StubChatClient(response=NoContentNoTextResponse())  # type: ignore[arg-type]
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="not textual"):
        adapter.complete([ChatMessage(role="user", content="hello")])


class NonStringTextResponse:
    def text(self) -> int:
        return 123


def test_text_method_non_string_rejected() -> None:
    client = StubChatClient(response=NonStringTextResponse())  # type: ignore[arg-type]
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="not textual"):
        adapter.complete([ChatMessage(role="user", content="hello")])


class WhitespaceTextResponse:
    def text(self) -> str:
        return "   "


def test_text_method_whitespace_only_rejected() -> None:
    client = StubChatClient(response=WhitespaceTextResponse())  # type: ignore[arg-type]
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="empty"):
        adapter.complete([ChatMessage(role="user", content="hello")])


class FailingTextResponse:
    def text(self) -> str:
        raise RuntimeError("text extraction failed")


def test_text_method_exception_wrapped_with_cause() -> None:
    client = StubChatClient(response=FailingTextResponse())  # type: ignore[arg-type]
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="Failed to extract text") as exc_info:
        adapter.complete([ChatMessage(role="user", content="hello")])
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_extraction_llm_call_error_not_double_wrapped() -> None:
    inner = LLMCallError("extraction failed")

    class RaisingInnerLLMCallErrorTextResponse:
        def text(self) -> str:
            raise inner

    client = StubChatClient(response=RaisingInnerLLMCallErrorTextResponse())  # type: ignore[arg-type]
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError) as exc_info:
        adapter.complete([ChatMessage(role="user", content="hello")])
    assert exc_info.value is inner


def test_exception_messages_do_not_include_prompt_payload() -> None:
    secret_prompt = "SECRET_PROMPT_PAYLOAD_12345"
    client = StubChatClient(error=RuntimeError(secret_prompt))
    adapter = OpenAIChatAdapter(client=client)
    with pytest.raises(LLMCallError, match="request failed") as exc_info:
        adapter.complete([ChatMessage(role="user", content=secret_prompt)])
    assert secret_prompt not in str(exc_info.value)
