"""Unit tests for baseline identity classification."""

from __future__ import annotations

from customer_claims_rag.evaluation.ab_metrics import (
    FROZEN_SIMILARITY_TOLERANCE,
    classify_case_baseline_identity,
)
from customer_claims_rag.evaluation.models import CaseResult, RetrievedChunkResult


def _chunk(
    *,
    rank: int,
    chunk_id: str,
    document_id: str = "doc",
    similarity: float = 0.8,
    distance: float = 0.2,
) -> RetrievedChunkResult:
    return RetrievedChunkResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        source_path="data/02_clean_markdown/doc.md",
        similarity=similarity,
        distance=distance,
        heading="Heading",
    )


def _case(chunks: list[RetrievedChunkResult], **kwargs) -> CaseResult:
    defaults = {
        "test_id": "T001",
        "query": "query",
        "expected_risk": "low",
        "expected_primary_documents": ["doc"],
        "expected_supporting_documents": [],
        "fallback_expected": False,
        "category": "refund",
        "status": "success",
        "hit_at_1": True,
        "hit_at_4": True,
        "hit_at_12": True,
        "document_recall_at_1": 1.0,
        "document_recall_at_4": 1.0,
        "document_recall_at_12": 1.0,
        "reciprocal_rank": 1.0,
        "primary_hit_at_1": True,
        "primary_hit_at_4": True,
        "supporting_hit_at_4": False,
        "top1_similarity": chunks[0].similarity if chunks else None,
        "retrieved_document_ids": [chunk.document_id for chunk in chunks],
        "retrieved_document_ids_at_4": [chunk.document_id for chunk in chunks[:4]],
    }
    defaults.update(kwargs)
    return CaseResult(retrieved_chunks=chunks, **defaults)


def test_exact_classification() -> None:
    chunks = [_chunk(rank=1, chunk_id="a"), _chunk(rank=2, chunk_id="b")]
    frozen = _case(chunks)
    computed = _case(chunks)
    classification, _, _ = classify_case_baseline_identity(frozen, computed)
    assert classification == "exact"


def test_same_order_float_drift() -> None:
    chunks = [_chunk(rank=1, chunk_id="a", similarity=0.81, distance=0.19)]
    drifted = [_chunk(rank=1, chunk_id="a", similarity=0.812, distance=0.188)]
    frozen = _case(chunks)
    computed = _case(drifted, top1_similarity=0.812)
    classification, _, _ = classify_case_baseline_identity(frozen, computed)
    assert classification == "same_order_float_drift"


def test_same_set_different_order() -> None:
    frozen = _case([_chunk(rank=1, chunk_id="a"), _chunk(rank=2, chunk_id="b")])
    computed = _case([_chunk(rank=1, chunk_id="b"), _chunk(rank=2, chunk_id="a")])
    classification, _, _ = classify_case_baseline_identity(frozen, computed)
    assert classification == "same_set_different_order"


def test_substantive_set_mismatch() -> None:
    frozen = _case([_chunk(rank=1, chunk_id="a")])
    computed = _case([_chunk(rank=1, chunk_id="b")])
    classification, _, _ = classify_case_baseline_identity(frozen, computed)
    assert classification == "substantive_mismatch"


def test_status_mismatch() -> None:
    frozen = _case([_chunk(rank=1, chunk_id="a")])
    computed = _case([_chunk(rank=1, chunk_id="a")], status="technical_error")
    classification, _, _ = classify_case_baseline_identity(frozen, computed)
    assert classification == "substantive_mismatch"


def test_metrics_mismatch() -> None:
    frozen = _case([_chunk(rank=1, chunk_id="a")])
    computed = _case([_chunk(rank=1, chunk_id="a")], hit_at_4=False)
    classification, _, _ = classify_case_baseline_identity(frozen, computed)
    assert classification == "substantive_mismatch"


def test_float_drift_above_tolerance_is_substantive_mismatch() -> None:
    chunks = [_chunk(rank=1, chunk_id="a", similarity=0.80, distance=0.20)]
    drifted = [
        _chunk(
            rank=1,
            chunk_id="a",
            similarity=0.80 + FROZEN_SIMILARITY_TOLERANCE + 0.001,
            distance=0.20,
        )
    ]
    frozen = _case(chunks)
    computed = _case(drifted)
    classification, _, _ = classify_case_baseline_identity(frozen, computed)
    assert classification == "substantive_mismatch"


def test_tolerance_does_not_hide_order_difference() -> None:
    frozen = _case(
        [_chunk(rank=1, chunk_id="a", similarity=0.80), _chunk(rank=2, chunk_id="b", similarity=0.70)],
        top1_similarity=0.80,
    )
    computed = _case(
        [_chunk(rank=1, chunk_id="b", similarity=0.7001), _chunk(rank=2, chunk_id="a", similarity=0.8001)],
        top1_similarity=0.8001,
    )
    classification, _, _ = classify_case_baseline_identity(frozen, computed)
    assert classification == "same_set_different_order"
