"""Retrieval evaluation metrics."""

from __future__ import annotations

import statistics
from typing import Iterable

from customer_claims_rag.evaluation.models import (
    AggregateMetrics,
    CaseResult,
    CategorySliceMetrics,
    RetrievedChunkResult,
    RiskLevel,
    RiskSliceMetrics,
)

RISK_LEVELS: tuple[RiskLevel, ...] = ("low", "medium", "high", "critical")


def unique_document_ids(chunks: Iterable[RetrievedChunkResult]) -> list[str]:
    """Return document IDs in first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        if chunk.document_id not in seen:
            seen.add(chunk.document_id)
            ordered.append(chunk.document_id)
    return ordered


def document_ids_in_chunk_window(
    chunks: list[RetrievedChunkResult],
    k: int,
) -> list[str]:
    """Return deduplicated document IDs from the first k raw chunks only."""
    return unique_document_ids(chunks[:k])


def document_hit_in_chunk_window(
    chunks: list[RetrievedChunkResult],
    expected_document_ids: Iterable[str],
    k: int,
) -> bool:
    """Return True if any expected document appears in the first k raw chunks."""
    return _document_hit_at_k(chunks, set(expected_document_ids), k)


def primary_found_in_top12_outside_top4(
    chunks: list[RetrievedChunkResult],
    expected_primary_documents: list[str],
) -> bool:
    """True when primary is absent from top-4 chunks but present in raw chunks 5-12."""
    if not expected_primary_documents:
        return False
    primary = set(expected_primary_documents)
    if document_hit_in_chunk_window(chunks, primary, 4):
        return False
    for chunk in chunks[4:12]:
        if chunk.document_id in primary:
            return True
    return False


def compute_case_metrics(
    *,
    test_id: str,
    query: str,
    expected_risk: RiskLevel,
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
    fallback_expected: bool,
    category: str | None,
    retrieved_chunks: list[RetrievedChunkResult],
) -> CaseResult:
    """Compute per-case retrieval metrics from ranked chunk results."""
    expected_any = set(expected_primary_documents) | set(expected_supporting_documents)
    document_ids = unique_document_ids(retrieved_chunks)
    document_ids_at_4 = document_ids_in_chunk_window(retrieved_chunks, 4)
    top1_similarity = retrieved_chunks[0].similarity if retrieved_chunks else None

    if fallback_expected:
        return CaseResult(
            test_id=test_id,
            query=query,
            expected_risk=expected_risk,
            expected_primary_documents=expected_primary_documents,
            expected_supporting_documents=expected_supporting_documents,
            fallback_expected=fallback_expected,
            category=category,
            retrieved_chunks=retrieved_chunks,
            retrieved_document_ids=document_ids,
            retrieved_document_ids_at_4=document_ids_at_4,
            top1_similarity=top1_similarity,
            status="no_grounding",
        )

    reciprocal_rank = _reciprocal_rank(retrieved_chunks, expected_any, k=12)
    missing_expected = sorted(expected_any - set(document_ids_in_chunk_window(retrieved_chunks, 12)))
    unexpected_top = _unexpected_top_sources(document_ids_at_4, expected_any)
    hit_at_12 = _any_expected_in_top_k(retrieved_chunks, expected_any, 12)
    hit_at_4 = _any_expected_in_top_k(retrieved_chunks, expected_any, 4)
    expected_in_top12_outside_top4 = hit_at_12 and not hit_at_4

    return CaseResult(
        test_id=test_id,
        query=query,
        expected_risk=expected_risk,
        expected_primary_documents=expected_primary_documents,
        expected_supporting_documents=expected_supporting_documents,
        fallback_expected=fallback_expected,
        category=category,
        retrieved_chunks=retrieved_chunks,
        retrieved_document_ids=document_ids,
        retrieved_document_ids_at_4=document_ids_at_4,
        top1_similarity=top1_similarity,
        hit_at_1=_any_expected_in_top_k(retrieved_chunks, expected_any, 1),
        hit_at_4=hit_at_4,
        hit_at_12=hit_at_12,
        document_recall_at_1=_document_recall_fraction(retrieved_chunks, expected_any, 1),
        document_recall_at_4=_document_recall_fraction(retrieved_chunks, expected_any, 4),
        document_recall_at_12=_document_recall_fraction(retrieved_chunks, expected_any, 12),
        reciprocal_rank=reciprocal_rank,
        primary_hit_at_1=_document_hit_at_k(
            retrieved_chunks,
            set(expected_primary_documents),
            1,
        ),
        primary_hit_at_4=_document_hit_at_k(
            retrieved_chunks,
            set(expected_primary_documents),
            4,
        ),
        supporting_hit_at_4=_document_hit_at_k(
            retrieved_chunks,
            set(expected_supporting_documents),
            4,
        ),
        expected_in_top12_outside_top4=expected_in_top12_outside_top4,
        missing_expected_sources=missing_expected,
        unexpected_top_sources=unexpected_top,
        status="success",
    )


def aggregate_case_metrics(
    case_results: list[CaseResult],
) -> AggregateMetrics:
    """Aggregate metrics across source-recall cases."""
    source_cases = [
        case
        for case in case_results
        if case.status == "success" and not case.fallback_expected
    ]
    supporting_cases = [
        case
        for case in source_cases
        if case.expected_supporting_documents
    ]
    supporting_hits = sum(1 for case in supporting_cases if case.supporting_hit_at_4)
    fallback_cases = [case for case in case_results if case.fallback_expected]
    technical_errors = [case for case in case_results if case.status == "technical_error"]
    successful = [case for case in case_results if case.status != "technical_error"]

    top1_values = [
        case.top1_similarity
        for case in successful
        if case.top1_similarity is not None
    ]
    no_result_count = sum(1 for case in successful if not case.retrieved_chunks)
    supporting_rate = (
        supporting_hits / len(supporting_cases) if supporting_cases else None
    )

    return AggregateMetrics(
        total_cases=len(case_results),
        successfully_evaluated_cases=len(successful),
        technical_error_count=len(technical_errors),
        source_recall_case_count=len(source_cases),
        fallback_case_count=len(fallback_cases),
        hit_rate_at_1=_rate(source_cases, lambda case: case.hit_at_1),
        hit_rate_at_4=_rate(source_cases, lambda case: case.hit_at_4),
        hit_rate_at_12=_rate(source_cases, lambda case: case.hit_at_12),
        document_recall_at_1=_mean([case.document_recall_at_1 for case in source_cases]),
        document_recall_at_4=_mean([case.document_recall_at_4 for case in source_cases]),
        document_recall_at_12=_mean([case.document_recall_at_12 for case in source_cases]),
        mrr=_mean([case.reciprocal_rank for case in source_cases]),
        primary_source_hit_rate_at_1=_rate(source_cases, lambda case: case.primary_hit_at_1),
        primary_source_hit_rate_at_4=_rate(source_cases, lambda case: case.primary_hit_at_4),
        cases_with_supporting_documents=len(supporting_cases),
        supporting_source_hits_at_4=supporting_hits,
        supporting_source_hit_rate_at_4=supporting_rate,
        no_result_rate=no_result_count / len(successful) if successful else 0.0,
        mean_top1_similarity=_mean(top1_values) if top1_values else None,
        median_top1_similarity=_median(top1_values) if top1_values else None,
        min_top1_similarity=min(top1_values) if top1_values else None,
        max_top1_similarity=max(top1_values) if top1_values else None,
    )


def aggregate_risk_metrics(case_results: list[CaseResult]) -> list[RiskSliceMetrics]:
    slices: list[RiskSliceMetrics] = []
    source_cases = [
        case
        for case in case_results
        if case.status == "success" and not case.fallback_expected
    ]
    for risk_level in RISK_LEVELS:
        group = [case for case in source_cases if case.expected_risk == risk_level]
        if not group:
            continue
        top1_values = [
            case.top1_similarity for case in group if case.top1_similarity is not None
        ]
        slices.append(
            RiskSliceMetrics(
                risk_level=risk_level,
                case_count=len(group),
                hit_rate_at_1=_rate(group, lambda case: case.hit_at_1),
                hit_rate_at_4=_rate(group, lambda case: case.hit_at_4),
                hit_rate_at_12=_rate(group, lambda case: case.hit_at_12),
                mrr=_mean([case.reciprocal_rank for case in group]),
                mean_top1_similarity=_mean(top1_values) if top1_values else None,
            )
        )
    return slices


def aggregate_category_metrics(case_results: list[CaseResult]) -> list[CategorySliceMetrics]:
    source_cases = [
        case
        for case in case_results
        if case.status == "success" and not case.fallback_expected and case.category
    ]
    categories = sorted({case.category for case in source_cases if case.category})
    slices: list[CategorySliceMetrics] = []
    for category in categories:
        group = [case for case in source_cases if case.category == category]
        top1_values = [
            case.top1_similarity for case in group if case.top1_similarity is not None
        ]
        slices.append(
            CategorySliceMetrics(
                category=category,
                case_count=len(group),
                hit_rate_at_1=_rate(group, lambda case: case.hit_at_1),
                hit_rate_at_4=_rate(group, lambda case: case.hit_at_4),
                hit_rate_at_12=_rate(group, lambda case: case.hit_at_12),
                mrr=_mean([case.reciprocal_rank for case in group]),
                mean_top1_similarity=_mean(top1_values) if top1_values else None,
            )
        )
    return slices


def summarize_fallback_cases(case_results: list[CaseResult]) -> dict[str, float | int | None]:
    fallback_cases = [case for case in case_results if case.fallback_expected]
    similarities = [
        case.top1_similarity
        for case in fallback_cases
        if case.top1_similarity is not None
    ]
    return {
        "case_count": len(fallback_cases),
        "mean_top1_similarity": _mean(similarities) if similarities else None,
        "min_top1_similarity": min(similarities) if similarities else None,
        "max_top1_similarity": max(similarities) if similarities else None,
    }


def _any_expected_in_top_k(
    chunks: list[RetrievedChunkResult],
    expected: set[str],
    k: int,
) -> bool:
    if not expected:
        return False
    for chunk in chunks[:k]:
        if chunk.document_id in expected:
            return True
    return False


def _document_hit_at_k(
    chunks: list[RetrievedChunkResult],
    expected: set[str],
    k: int,
) -> bool:
    if not expected:
        return False
    docs = set(document_ids_in_chunk_window(chunks, k))
    return bool(expected & docs)


def _document_recall_fraction(
    chunks: list[RetrievedChunkResult],
    expected: set[str],
    k: int,
) -> float:
    if not expected:
        return 0.0
    docs = set(document_ids_in_chunk_window(chunks, k))
    return len(expected & docs) / len(expected)


def _reciprocal_rank(
    chunks: list[RetrievedChunkResult],
    expected: set[str],
    *,
    k: int,
) -> float:
    if not expected:
        return 0.0
    for index, document_id in enumerate(document_ids_in_chunk_window(chunks, k), start=1):
        if document_id in expected:
            return 1.0 / index
    return 0.0


def _unexpected_top_sources(document_ids: list[str], expected: set[str]) -> list[str]:
    if not document_ids:
        return []
    top_document = document_ids[0]
    if top_document in expected:
        return []
    return [top_document]


def _rate(cases: list[CaseResult], predicate) -> float:
    if not cases:
        return 0.0
    return sum(1 for case in cases if predicate(case)) / len(cases)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0
