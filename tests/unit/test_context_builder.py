"""Unit tests for ContextPackageBuilder."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from customer_claims_rag.exceptions import GenerationValidationError
from customer_claims_rag.generation.context_builder import build_context_package
from customer_claims_rag.retrieval.models import SearchResult


def _result(
    *,
    rank: int,
    chunk_id: str,
    document_id: str = "doc",
    content: str = "body",
) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        source_path=f"data/02_clean_markdown/{document_id}.md",
        chunk_type="policy",
        topic="topic",
        risk_level="low",
        heading="Heading",
        heading_path=["Heading"],
        section="Section",
        subsection="Sub",
        similarity=0.9,
        distance=0.1,
    )


def test_maps_required_fields_only() -> None:
    package = build_context_package([_result(rank=2, chunk_id="doc::2")])
    item = package.items[0]
    assert item.citation_key == "S1"
    assert item.rank == 2
    assert item.document_id == "doc"
    assert item.chunk_id == "doc::2"
    assert item.heading == "Heading"
    assert item.source_path == "data/02_clean_markdown/doc.md"
    assert item.content == "body"
    dumped = item.model_dump()
    assert "similarity" not in dumped
    assert "risk_level" not in dumped
    assert "topic" not in dumped
    assert "chunk_type" not in dumped


def test_sorts_by_rank_before_assigning_keys() -> None:
    package = build_context_package(
        [
            _result(rank=3, chunk_id="doc::3"),
            _result(rank=1, chunk_id="doc::1"),
            _result(rank=2, chunk_id="doc::2"),
        ]
    )
    assert [item.citation_key for item in package.items] == ["S1", "S2", "S3"]
    assert [item.chunk_id for item in package.items] == ["doc::1", "doc::2", "doc::3"]


def test_input_objects_not_mutated() -> None:
    results = [
        _result(rank=2, chunk_id="doc::2"),
        _result(rank=1, chunk_id="doc::1"),
    ]
    original = deepcopy(results)
    build_context_package(results)
    assert results == original


def test_empty_input() -> None:
    package = build_context_package([])
    assert package.items == []


def test_duplicate_rank_rejected() -> None:
    with pytest.raises(GenerationValidationError, match="duplicate rank"):
        build_context_package(
            [
                _result(rank=1, chunk_id="doc::1"),
                _result(rank=1, chunk_id="doc::2"),
            ]
        )


def test_duplicate_chunk_id_rejected() -> None:
    with pytest.raises(GenerationValidationError, match="duplicate chunk_id"):
        build_context_package(
            [
                _result(rank=1, chunk_id="doc::1"),
                _result(rank=2, chunk_id="doc::1"),
            ]
        )


# ---------------------------------------------------------------------------
# Stage 2B.2: invalid ContextItem payloads surface as GenerationValidationError
# ---------------------------------------------------------------------------

_RETRIEVED_TEXT = "SENSITIVE-RETRIEVED-TEXT"

_INVALID_PAYLOADS = [
    pytest.param({"rank": 0}, "rank", id="rank-zero"),
    pytest.param({"rank": -3}, "rank", id="rank-negative"),
    pytest.param({"content": ""}, "content", id="blank-content"),
    pytest.param({"content": "  \n\t "}, "content", id="whitespace-content"),
    pytest.param({"heading": ""}, "heading", id="blank-heading"),
    pytest.param({"heading": "   "}, "heading", id="whitespace-heading"),
    pytest.param({"chunk_id": ""}, "chunk_id", id="blank-chunk-id"),
    pytest.param({"document_id": " "}, "document_id", id="blank-document-id"),
    pytest.param({"source_path": ""}, "source_path", id="blank-source-path"),
]


@pytest.mark.parametrize(("update", "field"), _INVALID_PAYLOADS)
def test_invalid_context_item_payload_is_translated(update: dict, field: str) -> None:
    invalid = _result(rank=2, chunk_id="doc::2").model_copy(update=update)

    with pytest.raises(GenerationValidationError, match="invalid context item") as excinfo:
        build_context_package([_result(rank=1, chunk_id="doc::1"), invalid])

    assert field in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_invalid_context_item_message_does_not_expose_retrieved_text() -> None:
    # A mistyped content value is echoed by pydantic's own error text; ours must not echo it.
    invalid = _result(rank=1, chunk_id="doc::1").model_copy(update={"content": [_RETRIEVED_TEXT]})

    with pytest.raises(GenerationValidationError) as excinfo:
        build_context_package([invalid])

    assert _RETRIEVED_TEXT in str(excinfo.value.__cause__)
    assert _RETRIEVED_TEXT not in str(excinfo.value)
    assert "content" in str(excinfo.value)


def test_duplicate_checks_still_run_before_item_validation() -> None:
    results = [
        _result(rank=0, chunk_id="doc::1"),
        _result(rank=0, chunk_id="doc::2"),
    ]

    with pytest.raises(GenerationValidationError, match="duplicate rank in context input: 0"):
        build_context_package(results)


@pytest.mark.parametrize(
    "error",
    [RuntimeError("programmer bug"), ValueError("programmer bug"), KeyError("programmer bug")],
    ids=lambda error: type(error).__name__,
)
def test_non_validation_errors_during_item_construction_propagate(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def explode(**kwargs: object) -> None:
        raise error

    monkeypatch.setattr("customer_claims_rag.generation.context_builder.ContextItem", explode)

    with pytest.raises(type(error), match="programmer bug") as excinfo:
        build_context_package([_result(rank=1, chunk_id="doc::1")])

    assert not isinstance(excinfo.value, GenerationValidationError)
