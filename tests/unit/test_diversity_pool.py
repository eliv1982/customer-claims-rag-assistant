"""Unit tests for per-document cap pool shaping."""

from __future__ import annotations

import pytest

from customer_claims_rag.evaluation.diversity_pool import (
    CapPoolValidationError,
    apply_per_document_cap,
    validate_cap_params,
)
from customer_claims_rag.retrieval.models import SearchResult


def _result(rank: int, chunk_id: str, document_id: str, similarity: float = 0.9) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {document_id}",
        source_path=f"data/02_clean_markdown/{document_id}.md",
        chunk_type="policy",
        topic=None,
        risk_level=None,
        heading="Heading",
        heading_path=["Heading"],
        section=None,
        subsection=None,
        similarity=similarity,
        distance=1.0 - similarity,
    )


def test_cap_preserves_vector_order_and_limits_per_document() -> None:
    results = [
        _result(1, "a::1", "doc_a", 0.99),
        _result(2, "a::2", "doc_a", 0.98),
        _result(3, "a::3", "doc_a", 0.97),
        _result(4, "a::4", "doc_a", 0.96),
        _result(5, "a::5", "doc_a", 0.95),
        _result(6, "b::1", "doc_b", 0.94),
    ]
    pool, diagnostics = apply_per_document_cap(
        results,
        fetch_k=48,
        pool_k=36,
        per_document_cap=4,
    )
    assert [item.chunk_id for item in pool] == ["a::1", "a::2", "a::3", "a::4", "b::1"]
    assert diagnostics.removed_by_cap_count == 1
    assert diagnostics.document_counts == {"doc_a": 4, "doc_b": 1}


def test_cap_stops_at_pool_size_36() -> None:
    results = [_result(i, f"d{i % 10}::c", f"doc_{i % 10}", 1.0 - i * 0.001) for i in range(1, 49)]
    pool, diagnostics = apply_per_document_cap(
        results,
        fetch_k=48,
        pool_k=36,
        per_document_cap=4,
    )
    assert len(pool) == 36
    assert diagnostics.pool_size == 36
    assert all(count <= 4 for count in diagnostics.document_counts.values())


def test_cap_with_fewer_than_target_results_available() -> None:
    results = [_result(1, "a::1", "doc_a"), _result(2, "b::1", "doc_b")]
    pool, diagnostics = apply_per_document_cap(
        results,
        fetch_k=48,
        pool_k=36,
        per_document_cap=4,
    )
    assert len(pool) == 2
    assert diagnostics.pool_shorter_than_target is True


def test_cap_reassigns_pool_ranks_from_one() -> None:
    results = [_result(1, "a::1", "doc_a"), _result(2, "b::1", "doc_b")]
    pool, diagnostics = apply_per_document_cap(
        results,
        fetch_k=48,
        pool_k=36,
        per_document_cap=4,
    )
    assert [item.rank for item in pool] == [1, 2]
    assert [entry.vector_rank for entry in diagnostics.entries] == [1, 2]


def test_cap_does_not_mutate_input_results() -> None:
    original = _result(1, "a::1", "doc_a")
    original_rank = original.rank
    apply_per_document_cap([original], fetch_k=48, pool_k=36, per_document_cap=4)
    assert original.rank == original_rank


def test_cap_is_deterministic() -> None:
    results = [_result(i, f"d::c{i}", f"doc_{i % 3}") for i in range(1, 20)]
    first = apply_per_document_cap(results, fetch_k=48, pool_k=36, per_document_cap=4)
    second = apply_per_document_cap(results, fetch_k=48, pool_k=36, per_document_cap=4)
    assert [item.chunk_id for item in first[0]] == [item.chunk_id for item in second[0]]


def test_cap_handles_empty_input() -> None:
    pool, diagnostics = apply_per_document_cap([], fetch_k=48, pool_k=36, per_document_cap=4)
    assert pool == []
    assert diagnostics.pool_size == 0


def test_validate_cap_params_rejects_invalid_values() -> None:
    with pytest.raises(CapPoolValidationError):
        validate_cap_params(fetch_k=0, pool_k=36, per_document_cap=4)
    with pytest.raises(CapPoolValidationError):
        validate_cap_params(fetch_k=48, pool_k=0, per_document_cap=4)
    with pytest.raises(CapPoolValidationError):
        validate_cap_params(fetch_k=24, pool_k=36, per_document_cap=4)
    with pytest.raises(CapPoolValidationError):
        validate_cap_params(fetch_k=48, pool_k=36, per_document_cap=0)


def test_baseline_path_without_cap_uses_prefix_pool() -> None:
    results = [_result(i, f"d::c{i}", f"doc_{i}") for i in range(1, 30)]
    pool, diagnostics = apply_per_document_cap(
        results,
        fetch_k=24,
        pool_k=24,
        per_document_cap=None,
    )
    assert len(pool) == 24
    assert diagnostics.cap_applied is False
    assert diagnostics.removed_by_cap_count == 0
