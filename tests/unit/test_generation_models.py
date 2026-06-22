"""Unit tests for grounded generation models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from customer_claims_rag.generation.models import (
    Citation,
    ContextItem,
    ContextPackage,
    GroundedGenerationRequest,
    GroundedGenerationResult,
    RawGenerationDraft,
)


def test_context_package_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        ContextPackage(items=[], unexpected=True)  # type: ignore[call-arg]


def test_empty_customer_query_rejected() -> None:
    with pytest.raises(ValidationError):
        GroundedGenerationRequest(customer_query="   ", context_package=ContextPackage())


def test_invalid_citation_key_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextItem(
            citation_key="S0",
            rank=1,
            document_id="doc",
            chunk_id="doc::1",
            heading="Heading",
            source_path="path",
            content="body",
        )


def test_invalid_rank_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextItem(
            citation_key="S1",
            rank=0,
            document_id="doc",
            chunk_id="doc::1",
            heading="Heading",
            source_path="path",
            content="body",
        )


def test_empty_context_package_allowed() -> None:
    package = ContextPackage(items=[])
    assert package.items == []


def test_grounded_result_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        GroundedGenerationResult(
            response_mode="grounded_answer",
            customer_response="ok",
            extra=True,  # type: ignore[call-arg]
        )


def test_raw_draft_modes() -> None:
    draft = RawGenerationDraft(response_mode="insufficient_context", answer="")
    assert draft.answer == ""


def test_citation_metadata_fields() -> None:
    citation = Citation(
        citation_key="S1",
        document_id="doc",
        chunk_id="doc::1",
        heading="Heading",
        source_path="path",
    )
    assert citation.document_id == "doc"
