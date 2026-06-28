"""Unit tests for GroundedGenerator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.exceptions import GenerationParseError, GenerationValidationError, LLMCallError
from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
from customer_claims_rag.generation.fallback import INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.models import (
    ContextItem,
    ContextPackage,
    GroundedGenerationRequest,
)
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation.ports import ChatMessage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"


def _package() -> ContextPackage:
    return ContextPackage(
        items=[
            ContextItem(
                citation_key="S1",
                rank=1,
                document_id="doc-a",
                chunk_id="doc-a::1",
                heading="Heading A",
                source_path="data/02_clean_markdown/doc-a.md",
                content="Policy text",
            )
        ]
    )


def _request(*, items: list[ContextItem] | None = None) -> GroundedGenerationRequest:
    package = ContextPackage(items=items if items is not None else _package().items)
    return GroundedGenerationRequest(
        customer_query="Где мой заказ?",
        context_package=package,
    )


def _generator(
    *,
    chat_model: FakeChatModel | None = None,
    prompt_builder: PromptBuilder | MagicMock | None = None,
) -> GroundedGenerator:
    return GroundedGenerator(
        chat_model=chat_model or FakeChatModel(),
        prompt_builder=prompt_builder or PromptBuilder(prompt_path=PROMPT_PATH),
    )


def test_valid_grounded_answer() -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"Подтвержденный ответ [S1]."}'
    )
    result = _generator(chat_model=model).generate(_request())

    assert result.response_mode == "grounded_answer"
    assert result.customer_response == "Подтвержденный ответ [S1]."
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "doc-a"


def test_citation_metadata_integrity() -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"Ответ [S1]."}'
    )
    result = _generator(chat_model=model).generate(_request())
    citation = result.citations[0]
    assert citation.chunk_id == "doc-a::1"
    assert citation.heading == "Heading A"


def test_literal_citation_marker_preserved() -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"Текст [S1] далее."}'
    )
    result = _generator(chat_model=model).generate(_request())
    assert "[S1]" in result.customer_response


def test_valid_insufficient_context_from_llm() -> None:
    model = FakeChatModel(
        response='{"response_mode":"insufficient_context","answer":""}'
    )
    result = _generator(chat_model=model).generate(_request())
    assert result.response_mode == "insufficient_context"
    assert result.customer_response == INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
    assert result.citations == []


def test_empty_context_short_circuit() -> None:
    model = FakeChatModel(response="should-not-be-used")
    result = _generator(chat_model=model).generate(_request(items=[]))
    assert result.response_mode == "insufficient_context"
    assert result.customer_response == INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE


def test_empty_context_does_not_call_chat_model() -> None:
    model = FakeChatModel(response="unused")
    _generator(chat_model=model).generate(_request(items=[]))
    assert model.call_count == 0


def test_empty_context_does_not_call_prompt_builder() -> None:
    model = FakeChatModel(response="unused")
    prompt_builder = MagicMock(spec=PromptBuilder)
    _generator(chat_model=model, prompt_builder=prompt_builder).generate(_request(items=[]))
    prompt_builder.build_messages.assert_not_called()


def test_malformed_json_raises_parse_error() -> None:
    model = FakeChatModel(response="{bad-json")
    with pytest.raises(GenerationParseError):
        _generator(chat_model=model).generate(_request())


def test_schema_invalid_json_raises_parse_error() -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"x","extra":1}'
    )
    with pytest.raises(GenerationParseError):
        _generator(chat_model=model).generate(_request())


def test_valid_parsed_draft_with_invalid_marker_raises_validation_error() -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"Без ссылок"}'
    )
    with pytest.raises(GenerationValidationError):
        _generator(chat_model=model).generate(_request())


def test_chat_model_error_propagates() -> None:
    model = FakeChatModel(error=LLMCallError("boom"))
    with pytest.raises(LLMCallError, match="boom"):
        _generator(chat_model=model).generate(_request())


def test_raw_response_not_in_result() -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"Ответ [S1]."}'
    )
    result = _generator(chat_model=model).generate(_request())
    dumped = result.model_dump()
    assert "raw" not in dumped
    assert "raw_response" not in dumped


def test_repeated_calls_are_stateless() -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"Ответ [S1]."}'
    )
    generator = _generator(chat_model=model)
    first = generator.generate(_request())
    second = generator.generate(_request())
    assert first == second
    assert model.call_count == 2


def test_prompt_messages_order() -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"Ответ [S1]."}'
    )
    _generator(chat_model=model).generate(_request())
    assert len(model.last_messages) == 2
    assert model.last_messages[0].role == "system"
    assert model.last_messages[1].role == "user"
    assert model.last_messages[0].content
    assert "Где мой заказ?" in model.last_messages[1].content


def test_prompt_builder_exception_propagates() -> None:
    prompt_builder = MagicMock(spec=PromptBuilder)
    prompt_builder.build_messages.side_effect = RuntimeError("prompt failed")
    model = FakeChatModel(response='{"response_mode":"grounded_answer","answer":"x [S1]"}')

    with pytest.raises(RuntimeError, match="prompt failed"):
        _generator(chat_model=model, prompt_builder=prompt_builder).generate(_request())


def test_empty_context_does_not_call_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    import customer_claims_rag.generation.generator as generator_module

    parser_spy = MagicMock(side_effect=AssertionError("parser should not be called"))
    monkeypatch.setattr(generator_module, "parse_generation_draft", parser_spy)

    model = FakeChatModel(response="unused")
    _generator(chat_model=model).generate(_request(items=[]))

    parser_spy.assert_not_called()
