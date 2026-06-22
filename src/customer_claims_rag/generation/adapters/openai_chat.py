"""OpenAI chat adapter for grounded generation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from customer_claims_rag.exceptions import LLMCallError
from customer_claims_rag.generation.ports import ChatMessage

_MISSING_CONTENT = object()


@runtime_checkable
class InvokableChatClient(Protocol):
    """Minimal chat client contract for adapter tests."""

    def invoke(self, messages: Sequence[BaseMessage], /) -> BaseMessage:
        """Invoke the chat model with LangChain messages."""
        ...


class OpenAIChatAdapter:
    """LangChain-backed OpenAI chat adapter."""

    def __init__(self, *, client: InvokableChatClient) -> None:
        self._client = client

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        langchain_messages = [_to_langchain_message(message) for message in messages]
        try:
            response = self._client.invoke(langchain_messages)
        except LLMCallError:
            raise
        except Exception as exc:
            raise LLMCallError("OpenAI chat request failed") from exc

        return _extract_response_text(response)


def _to_langchain_message(message: ChatMessage) -> BaseMessage:
    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "user":
        return HumanMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(content=message.content)
    raise LLMCallError(f"unsupported chat message role: {message.role}")


def _extract_response_text(response: object) -> str:
    content = getattr(response, "content", _MISSING_CONTENT)
    if isinstance(content, str):
        text = content
    else:
        text_method = getattr(response, "text", None)
        if text_method is None or not callable(text_method):
            raise LLMCallError("OpenAI chat response is not textual")
        try:
            text = text_method()
        except LLMCallError:
            raise
        except Exception as exc:
            raise LLMCallError("Failed to extract text from OpenAI chat response") from exc
        if not isinstance(text, str):
            raise LLMCallError("OpenAI chat response is not textual")

    if not text.strip():
        raise LLMCallError("OpenAI chat response is empty")

    return text
