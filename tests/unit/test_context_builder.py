"""Unit tests for ContextPackageBuilder."""

from __future__ import annotations

from copy import deepcopy

import pytest

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
