"""Unit tests for grounded generation structural validator."""

from __future__ import annotations

import pytest

from customer_claims_rag.exceptions import GenerationValidationError
from customer_claims_rag.generation.fallback import INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
from customer_claims_rag.generation.models import (
    ContextItem,
    ContextPackage,
    RawGenerationDraft,
)
from customer_claims_rag.generation.validator import (
    insufficient_context_for_empty_package,
    validate_generation_draft,
)


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
            ),
            ContextItem(
                citation_key="S2",
                rank=2,
                document_id="doc-b",
                chunk_id="doc-b::2",
                heading="Heading B",
                source_path="data/02_clean_markdown/doc-b.md",
                content="More text",
            ),
        ]
    )


def test_valid_grounded_answer() -> None:
    result = validate_generation_draft(
        RawGenerationDraft(
            response_mode="grounded_answer",
            answer="Подтвержденный ответ [S1].",
        ),
        _package(),
    )
    assert result.response_mode == "grounded_answer"
    assert result.customer_response == "Подтвержденный ответ [S1]."
    assert len(result.citations) == 1
    assert result.citations[0].document_id == "doc-a"


def test_missing_marker_rejected() -> None:
    with pytest.raises(GenerationValidationError, match="at least one citation marker"):
        validate_generation_draft(
            RawGenerationDraft(response_mode="grounded_answer", answer="Без ссылок"),
            _package(),
        )


def test_unknown_marker_rejected() -> None:
    with pytest.raises(GenerationValidationError, match="unknown citation marker"):
        validate_generation_draft(
            RawGenerationDraft(response_mode="grounded_answer", answer="Ответ [S9]."),
            _package(),
        )


def test_repeated_marker_single_citation() -> None:
    result = validate_generation_draft(
        RawGenerationDraft(
            response_mode="grounded_answer",
            answer="Часть [S1] и снова [S1].",
        ),
        _package(),
    )
    assert len(result.citations) == 1
    assert result.citations[0].citation_key == "S1"


def test_citation_metadata_integrity() -> None:
    result = validate_generation_draft(
        RawGenerationDraft(
            response_mode="grounded_answer",
            answer="Ответ [S2].",
        ),
        _package(),
    )
    citation = result.citations[0]
    assert citation.chunk_id == "doc-b::2"
    assert citation.heading == "Heading B"
    assert citation.source_path == "data/02_clean_markdown/doc-b.md"


def test_grounded_empty_text_rejected() -> None:
    with pytest.raises(GenerationValidationError, match="non-empty answer"):
        validate_generation_draft(
            RawGenerationDraft(response_mode="grounded_answer", answer="   "),
            _package(),
        )


def test_valid_insufficient_context() -> None:
    result = validate_generation_draft(
        RawGenerationDraft(response_mode="insufficient_context", answer=""),
        _package(),
    )
    assert result.response_mode == "insufficient_context"
    assert result.customer_response == INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
    assert result.citations == []


def test_insufficient_context_with_nonempty_answer_rejected() -> None:
    with pytest.raises(GenerationValidationError, match="must be empty"):
        validate_generation_draft(
            RawGenerationDraft(response_mode="insufficient_context", answer="text"),
            _package(),
        )


def test_empty_context_short_circuit() -> None:
    result = insufficient_context_for_empty_package()
    assert result.response_mode == "insufficient_context"
    assert result.citations == []
    assert result.customer_response == INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE


def test_malformed_marker_in_grounded_answer_rejected() -> None:
    with pytest.raises(GenerationValidationError, match="at least one citation marker"):
        validate_generation_draft(
            RawGenerationDraft(response_mode="grounded_answer", answer="[[S1]]"),
            _package(),
        )
