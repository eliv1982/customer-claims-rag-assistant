"""Unit tests for candidate-generation diagnostics."""

from __future__ import annotations

from customer_claims_rag.evaluation.ab_metrics import (
    build_case_comparison,
    compute_candidate_generation_flags,
)
from customer_claims_rag.evaluation.ab_models import ArmMetrics, RankedCandidateAudit


def _audit(chunk_id: str, document_id: str, rank: int) -> RankedCandidateAudit:
    return RankedCandidateAudit(
        baseline_rank=rank,
        candidate_rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_type="policy",
        similarity=0.5,
        distance=0.5,
    )


def _metrics(**kwargs) -> ArmMetrics:
    return ArmMetrics(**kwargs)


def test_two_primary_alternatives_one_present_primary_failure_false() -> None:
    flags = compute_candidate_generation_flags(
        expected_primary_documents=["doc_a", "doc_b"],
        expected_supporting_documents=[],
        baseline_document_ids=["doc_a", "other"],
        fallback_expected=False,
    )
    assert flags["primary_candidate_generation_failure"] is False
    assert flags["reranker_not_applicable"] is False


def test_no_primary_in_pool_primary_failure_true() -> None:
    flags = compute_candidate_generation_flags(
        expected_primary_documents=["doc_a"],
        expected_supporting_documents=[],
        baseline_document_ids=["other"],
        fallback_expected=False,
    )
    assert flags["primary_candidate_generation_failure"] is True
    assert flags["reranker_not_applicable"] is True


def test_primary_available_supporting_missing_reranker_applicable() -> None:
    flags = compute_candidate_generation_flags(
        expected_primary_documents=["doc_a"],
        expected_supporting_documents=["doc_support"],
        baseline_document_ids=["doc_a"],
        fallback_expected=False,
    )
    assert flags["primary_candidate_generation_failure"] is False
    assert flags["supporting_candidate_generation_failure"] is True
    assert flags["reranker_not_applicable"] is False


def test_supporting_available_primary_missing_reranker_applicable() -> None:
    flags = compute_candidate_generation_flags(
        expected_primary_documents=["doc_primary"],
        expected_supporting_documents=["doc_support"],
        baseline_document_ids=["doc_support"],
        fallback_expected=False,
    )
    assert flags["primary_candidate_generation_failure"] is True
    assert flags["supporting_candidate_generation_failure"] is False
    assert flags["reranker_not_applicable"] is False


def test_neither_primary_nor_supporting_reachable_reranker_not_applicable() -> None:
    flags = compute_candidate_generation_flags(
        expected_primary_documents=["doc_primary"],
        expected_supporting_documents=["doc_support"],
        baseline_document_ids=["other"],
        fallback_expected=False,
    )
    assert flags["reranker_not_applicable"] is True
    assert flags["candidate_generation_gap"] is True


def test_fallback_case_not_marked_candidate_generation_failure() -> None:
    flags = compute_candidate_generation_flags(
        expected_primary_documents=[],
        expected_supporting_documents=[],
        baseline_document_ids=[],
        fallback_expected=True,
    )
    assert flags["primary_candidate_generation_failure"] is False
    assert flags["supporting_candidate_generation_failure"] is False
    assert flags["candidate_generation_gap"] is False
    assert flags["reranker_not_applicable"] is False


def test_t023_like_case_not_reranker_not_applicable() -> None:
    comparison = build_case_comparison(
        baseline_metrics=_metrics(primary_hit_at_4=False, reciprocal_rank=0.25),
        candidate_metrics=_metrics(primary_hit_at_4=True, reciprocal_rank=0.5),
        baseline_audits=[
            _audit("chunk_support", "07_complaint_handling_procedure", 1),
            _audit("chunk_primary", "03_refund_policy", 4),
        ],
        candidate_audits=[
            _audit("chunk_support", "07_complaint_handling_procedure", 2),
            _audit("chunk_primary", "03_refund_policy", 1),
        ],
        expected_primary_documents=["03_refund_policy"],
        expected_supporting_documents=["07_complaint_handling_procedure"],
        fallback_expected=False,
    )
    assert comparison.primary_hit_at_4_delta == 1
    assert comparison.reranker_not_applicable is False
