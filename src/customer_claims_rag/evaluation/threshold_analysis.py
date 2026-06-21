"""Post-hoc threshold analysis over stored retrieval candidates."""

from __future__ import annotations

from customer_claims_rag.evaluation.metrics import (
    _any_expected_in_top_k,
    _document_hit_at_k,
    _mean,
    _reciprocal_rank,
    document_ids_in_chunk_window,
)
from customer_claims_rag.evaluation.models import CaseResult, ThresholdSliceMetrics

DEFAULT_THRESHOLD_SWEEP: tuple[float, ...] = (
    0.00,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)


def analyze_thresholds(
    case_results: list[CaseResult],
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_SWEEP,
) -> list[ThresholdSliceMetrics]:
    """Recompute metrics by filtering stored candidates without new searches."""
    source_cases = [
        case
        for case in case_results
        if case.status == "success" and not case.fallback_expected
    ]
    slices: list[ThresholdSliceMetrics] = []
    for threshold in thresholds:
        filtered_groups = [
            _filter_case(case, threshold=threshold)
            for case in source_cases
        ]
        successful = [group for group in filtered_groups if group["has_result"]]
        slices.append(
            ThresholdSliceMetrics(
                threshold=threshold,
                cases_with_at_least_one_result=len(successful),
                no_result_rate=1.0 - (len(successful) / len(source_cases))
                if source_cases
                else 0.0,
                hit_rate_at_1=_rate_bool([group["hit_at_1"] for group in filtered_groups]),
                hit_rate_at_4=_rate_bool([group["hit_at_4"] for group in filtered_groups]),
                primary_source_hit_rate_at_4=_rate_bool(
                    [group["primary_hit_at_4"] for group in filtered_groups]
                ),
                mrr=_mean([group["reciprocal_rank"] for group in filtered_groups]),
                critical_case_hit_rate_at_4=_risk_hit_at_4(filtered_groups, "critical"),
                high_case_hit_rate_at_4=_risk_hit_at_4(filtered_groups, "high"),
            )
        )
    return slices


def _filter_case(case: CaseResult, *, threshold: float) -> dict[str, object]:
    filtered_chunks = [
        chunk for chunk in case.retrieved_chunks if chunk.similarity >= threshold
    ]
    expected_any = set(case.expected_primary_documents) | set(
        case.expected_supporting_documents
    )
    expected_primary = set(case.expected_primary_documents)
    has_result = bool(filtered_chunks)
    return {
        "has_result": has_result,
        "hit_at_1": _any_expected_in_top_k(filtered_chunks, expected_any, 1),
        "hit_at_4": _any_expected_in_top_k(filtered_chunks, expected_any, 4),
        "primary_hit_at_4": _document_hit_at_k(filtered_chunks, expected_primary, 4),
        "reciprocal_rank": _reciprocal_rank(filtered_chunks, expected_any, k=12),
        "expected_risk": case.expected_risk,
    }


def _rate_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def _risk_hit_at_4(groups: list[dict[str, object]], risk_level: str) -> float:
    risk_groups = [
        group for group in groups if group["expected_risk"] == risk_level
    ]
    if not risk_groups:
        return 0.0
    return _rate_bool([bool(group["hit_at_4"]) for group in risk_groups])
