"""Unit tests for strict generation JSON parser."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from customer_claims_rag.exceptions import GenerationParseError
from customer_claims_rag.generation.models import RawGenerationDraft
from customer_claims_rag.generation.parser import parse_generation_draft


def test_valid_grounded_json() -> None:
    draft = parse_generation_draft(
        '{"response_mode":"grounded_answer","answer":"Ответ [S1]"}'
    )
    assert draft == RawGenerationDraft(
        response_mode="grounded_answer",
        answer="Ответ [S1]",
    )


def test_valid_insufficient_context_json() -> None:
    draft = parse_generation_draft(
        '{"response_mode":"insufficient_context","answer":""}'
    )
    assert draft.response_mode == "insufficient_context"
    assert draft.answer == ""


def test_leading_trailing_whitespace() -> None:
    draft = parse_generation_draft(
        '  {"response_mode":"grounded_answer","answer":"x"}  '
    )
    assert draft.answer == "x"


def test_unicode_answer() -> None:
    draft = parse_generation_draft(
        '{"response_mode":"grounded_answer","answer":"Жалоба на доставку [S1]"}'
    )
    assert "Жалоба" in draft.answer


def test_empty_raw() -> None:
    with pytest.raises(GenerationParseError, match="empty"):
        parse_generation_draft("")


def test_whitespace_only_raw() -> None:
    with pytest.raises(GenerationParseError, match="empty"):
        parse_generation_draft("   \n\t  ")


def test_invalid_json() -> None:
    with pytest.raises(GenerationParseError, match="not valid JSON") as exc_info:
        parse_generation_draft("{not-json")
    assert isinstance(exc_info.value.__cause__, Exception)


def test_fenced_json() -> None:
    payload = '```json\n{"response_mode":"grounded_answer","answer":"x"}\n```'
    with pytest.raises(GenerationParseError, match="not valid JSON"):
        parse_generation_draft(payload)


def test_prefix_text() -> None:
    with pytest.raises(GenerationParseError, match="not valid JSON"):
        parse_generation_draft('prefix {"response_mode":"grounded_answer","answer":"x"}')


def test_suffix_text() -> None:
    with pytest.raises(GenerationParseError, match="not valid JSON"):
        parse_generation_draft('{"response_mode":"grounded_answer","answer":"x"} suffix')


def test_array_instead_of_object() -> None:
    with pytest.raises(GenerationParseError, match="JSON object"):
        parse_generation_draft('["grounded_answer","x"]')


def test_scalar_instead_of_object() -> None:
    with pytest.raises(GenerationParseError, match="JSON object"):
        parse_generation_draft('"grounded_answer"')


def test_missing_field() -> None:
    with pytest.raises(GenerationParseError, match="generation schema") as exc_info:
        parse_generation_draft('{"response_mode":"grounded_answer"}')
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_extra_field() -> None:
    with pytest.raises(GenerationParseError, match="generation schema") as exc_info:
        parse_generation_draft(
            '{"response_mode":"grounded_answer","answer":"x","extra":1}'
        )
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_invalid_enum() -> None:
    with pytest.raises(GenerationParseError, match="generation schema"):
        parse_generation_draft('{"response_mode":"bad","answer":"x"}')


def test_wrong_answer_type() -> None:
    with pytest.raises(GenerationParseError, match="generation schema"):
        parse_generation_draft('{"response_mode":"grounded_answer","answer":1}')


def test_duplicate_top_level_key() -> None:
    with pytest.raises(GenerationParseError, match="duplicate JSON key"):
        parse_generation_draft(
            '{"response_mode":"grounded_answer","response_mode":"grounded_answer","answer":"x"}'
        )


def test_duplicate_nested_key() -> None:
    payload = (
        '{"response_mode":"grounded_answer","answer":"x",'
        '"meta":{"flag":true,"flag":false}}'
    )
    with pytest.raises(GenerationParseError, match="duplicate JSON key"):
        parse_generation_draft(payload)


def test_non_string_raw_rejected() -> None:
    with pytest.raises(GenerationParseError, match="must be a string"):
        parse_generation_draft(123)  # type: ignore[arg-type]


def test_exception_message_does_not_include_full_raw_response() -> None:
    long_answer = "A" * 500
    raw = f'{{"response_mode":"grounded_answer","answer":{long_answer!r},"extra":1}}'
    with pytest.raises(GenerationParseError) as exc_info:
        parse_generation_draft(raw)
    assert long_answer not in str(exc_info.value)


def test_answer_with_literal_triple_backticks_allowed() -> None:
    draft = parse_generation_draft(
        '{"response_mode":"grounded_answer","answer":"use ```json``` markers [S1]"}'
    )
    assert "```" in draft.answer


def test_response_mode_wrong_type_rejected() -> None:
    with pytest.raises(GenerationParseError, match="generation schema"):
        parse_generation_draft('{"response_mode":1,"answer":"x"}')


@pytest.mark.parametrize(
    "raw_input",
    [None, b'{"response_mode":"grounded_answer","answer":"x"}', ["json"]],
)
def test_non_string_raw_types_rejected(raw_input: object) -> None:
    with pytest.raises(GenerationParseError, match="must be a string"):
        parse_generation_draft(raw_input)  # type: ignore[arg-type]
