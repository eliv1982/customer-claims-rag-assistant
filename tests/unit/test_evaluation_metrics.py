"""Evaluation metrics tests."""

from __future__ import annotations

from customer_claims_rag.evaluation.metrics import (
    aggregate_case_metrics,
    compute_case_metrics,
    document_ids_in_chunk_window,
    unique_document_ids,
)
from customer_claims_rag.evaluation.models import CaseResult
from customer_claims_rag.evaluation.threshold_analysis import analyze_thresholds
from tests.evaluation_helpers import make_retrieved_chunk


def test_unique_document_ids_preserve_order() -> None:
    chunks = [
        make_retrieved_chunk(rank=1, document_id="01_service_overview"),
        make_retrieved_chunk(rank=2, document_id="01_service_overview"),
        make_retrieved_chunk(rank=3, document_id="02_delivery_rules"),
    ]
    assert unique_document_ids(chunks) == ["01_service_overview", "02_delivery_rules"]


def test_at_k_uses_raw_chunk_window_not_global_dedup() -> None:
    chunks = [
        make_retrieved_chunk(rank=1, document_id="10_customer_faq", similarity=0.9),
        make_retrieved_chunk(rank=2, document_id="10_customer_faq", similarity=0.85),
        make_retrieved_chunk(rank=3, document_id="10_customer_faq", similarity=0.8),
        make_retrieved_chunk(rank=4, document_id="01_service_overview", similarity=0.7),
        make_retrieved_chunk(rank=5, document_id="02_delivery_rules", similarity=0.6),
    ]
    result = compute_case_metrics(
        test_id="T001",
        query="q",
        expected_risk="low",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=[],
        fallback_expected=False,
        category="general",
        retrieved_chunks=chunks,
    )
    assert document_ids_in_chunk_window(chunks, 4) == ["10_customer_faq", "01_service_overview"]
    assert result.primary_hit_at_4 is True
    assert result.primary_hit_at_1 is False
    assert result.hit_at_4 is True


def test_primary_miss_at_4_but_found_in_top12() -> None:
    chunks = [
        make_retrieved_chunk(rank=i, document_id="10_customer_faq", similarity=0.9 - i * 0.01)
        for i in range(1, 5)
    ] + [
        make_retrieved_chunk(rank=5, document_id="01_service_overview", similarity=0.5),
    ]
    result = compute_case_metrics(
        test_id="T002",
        query="q",
        expected_risk="low",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=[],
        fallback_expected=False,
        category="general",
        retrieved_chunks=chunks,
    )
    assert result.primary_hit_at_4 is False
    assert result.hit_at_12 is True
    assert result.expected_in_top12_outside_top4 is True


def test_hit_at_k_and_document_recall() -> None:
    chunks = [
        make_retrieved_chunk(rank=1, document_id="10_customer_faq", similarity=0.9),
        make_retrieved_chunk(rank=2, document_id="01_service_overview", similarity=0.8),
        make_retrieved_chunk(rank=3, document_id="02_delivery_rules", similarity=0.7),
    ]
    result = compute_case_metrics(
        test_id="T001",
        query="q",
        expected_risk="low",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=["02_delivery_rules"],
        fallback_expected=False,
        category="general",
        retrieved_chunks=chunks,
    )
    assert result.hit_at_1 is False
    assert result.hit_at_4 is True
    assert result.document_recall_at_4 == 1.0
    assert result.primary_hit_at_4 is True
    assert result.supporting_hit_at_4 is True
    assert result.reciprocal_rank == 0.5


def test_supporting_denominator_only_counts_cases_with_supporting() -> None:
    with_supporting = compute_case_metrics(
        test_id="T010",
        query="q",
        expected_risk="low",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=["02_delivery_rules"],
        fallback_expected=False,
        category="general",
        retrieved_chunks=[
            make_retrieved_chunk(rank=1, document_id="01_service_overview"),
            make_retrieved_chunk(rank=2, document_id="02_delivery_rules"),
        ],
    )
    without_supporting = compute_case_metrics(
        test_id="T011",
        query="q",
        expected_risk="low",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=[],
        fallback_expected=False,
        category="general",
        retrieved_chunks=[make_retrieved_chunk(rank=1, document_id="01_service_overview")],
    )
    aggregate = aggregate_case_metrics([with_supporting, without_supporting])
    assert aggregate.cases_with_supporting_documents == 1
    assert aggregate.supporting_source_hits_at_4 == 1
    assert aggregate.supporting_source_hit_rate_at_4 == 1.0


def test_fallback_case_excluded_from_source_metrics() -> None:
    chunks = [make_retrieved_chunk(rank=1, document_id="01_service_overview")]
    result = compute_case_metrics(
        test_id="T006",
        query="q",
        expected_risk="low",
        expected_primary_documents=[],
        expected_supporting_documents=["01_service_overview"],
        fallback_expected=True,
        category="general",
        retrieved_chunks=chunks,
    )
    assert result.status == "no_grounding"
    aggregate = aggregate_case_metrics([result])
    assert aggregate.source_recall_case_count == 0
    assert aggregate.fallback_case_count == 1


def test_aggregate_metrics_and_technical_error() -> None:
    success = compute_case_metrics(
        test_id="T001",
        query="q",
        expected_risk="low",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=[],
        fallback_expected=False,
        category="general",
        retrieved_chunks=[make_retrieved_chunk(rank=1, document_id="01_service_overview")],
    )
    error = CaseResult(
        test_id="T002",
        query="q2",
        expected_risk="low",
        expected_primary_documents=["02_delivery_rules"],
        expected_supporting_documents=[],
        fallback_expected=False,
        status="technical_error",
        error_message="boom",
    )
    aggregate = aggregate_case_metrics([success, error])
    assert aggregate.total_cases == 2
    assert aggregate.technical_error_count == 1
    assert aggregate.hit_rate_at_1 == 1.0


def test_threshold_sweep_is_deterministic_without_new_search() -> None:
    case = compute_case_metrics(
        test_id="T010",
        query="q",
        expected_risk="high",
        expected_primary_documents=["04_refund_policy"],
        expected_supporting_documents=[],
        fallback_expected=False,
        category="refund",
        retrieved_chunks=[
            make_retrieved_chunk(rank=1, document_id="10_customer_faq", similarity=0.55),
            make_retrieved_chunk(rank=2, document_id="04_refund_policy", similarity=0.45),
        ],
    )
    first = analyze_thresholds([case])
    second = analyze_thresholds([case])
    assert first == second
    low = next(item for item in first if item.threshold == 0.0)
    high = next(item for item in first if item.threshold == 0.60)
    assert low.hit_rate_at_4 == 1.0
    assert high.hit_rate_at_4 == 0.0
