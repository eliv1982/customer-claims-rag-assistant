"""Footprint and case diagnostics for doc08 atomic experiment."""

from __future__ import annotations

from customer_claims_rag.evaluation.doc08_atomic_contract import (
    NEGATIVE_EXTENSION_IDS,
    PRIVACY_EXTENSION_IDS,
    THREAT_EXTENSION_IDS,
    assert_baseline_reproduction,
    build_doc08_reachability_comparison,
    build_extension_case_diagnostic,
    build_frozen_reachability_snapshot,
    classify_threat_case_delta,
    compute_extension_metrics,
    compute_primary_hit_at_12,
    count_faq_in_top4,
    evaluate_extension_acceptance,
    evaluate_frozen_acceptance,
    fully_unreachable_case_ids,
    load_baseline_reference_oracle,
    primary_unreachable_case_ids,
    validate_baseline_reproduction,
)
from customer_claims_rag.evaluation.doc08_atomic_models import (
    Doc08AtomicCaseResult,
    Doc08CaseDiagnostic,
    Doc08Footprint,
)
from customer_claims_rag.ingestion.corpus_overlay import DOC08_DOCUMENT_ID

DOC12_DOCUMENT_ID = "12_staff_safety_and_threat_handling"

REQUIRED_CASE_IDS = (
    "T011",
    "T035",
    "T039",
    "T040",
    "T044",
    "T046",
    "T047",
    "T051",
)


def build_doc08_footprint(
    case_results: list[Doc08AtomicCaseResult],
    *,
    arm: str,
) -> Doc08Footprint:
    pool_cases: list[str] = []
    top4_cases: list[str] = []
    pool_count = top12_count = top4_count = top1_count = 0

    for item in case_results:
        pool_docs = (
            item.baseline_pool_doc_ids if arm == "baseline" else item.candidate_pool_doc_ids
        )
        final_docs = (
            item.baseline_final_doc_ids if arm == "baseline" else item.candidate_final_doc_ids
        )
        if DOC08_DOCUMENT_ID in pool_docs:
            pool_count += 1
            pool_cases.append(item.case_id)
        if DOC08_DOCUMENT_ID in final_docs[:12]:
            top12_count += 1
        if DOC08_DOCUMENT_ID in final_docs[:4]:
            top4_count += 1
            top4_cases.append(item.case_id)
        if final_docs and final_docs[0] == DOC08_DOCUMENT_ID:
            top1_count += 1

    return Doc08Footprint(
        pool_appearances=pool_count,
        top12_appearances=top12_count,
        top4_appearances=top4_count,
        top1_appearances=top1_count,
        cases_in_pool=sorted(pool_cases),
        cases_in_top4=sorted(top4_cases),
    )


def _ranks_for_doc(results, document_id: str) -> list[int]:
    return [result.rank for result in results if result.document_id == document_id]


def build_case_diagnostic(
    item: Doc08AtomicCaseResult,
    *,
    baseline_vector,
    candidate_vector,
    query: str,
    expected_primary: list[str],
) -> Doc08CaseDiagnostic:
    def pool_rank(doc_ids: list[str]) -> int | None:
        for index, doc_id in enumerate(doc_ids, start=1):
            if doc_id == DOC08_DOCUMENT_ID:
                return index
        return None

    def final_rank(doc_ids: list[str], doc_id: str) -> int | None:
        for index, did in enumerate(doc_ids, start=1):
            if did == doc_id:
                return index
        return None

    best08 = next((result for result in candidate_vector if result.document_id == DOC08_DOCUMENT_ID), None)
    top_competing = [
        result.document_id
        for result in candidate_vector[:5]
        if result.document_id != DOC08_DOCUMENT_ID
    ]

    heading = best08.heading if best08 else None
    sim = float(best08.similarity) if best08 else None
    content_lower = (best08.content if best08 else "").lower()
    grounding = None
    if item.case_id == "T044" and heading:
        grounding = any(
            token in content_lower
            for token in ("персональн", "утечк", "чуж", "наклейк", "адрес", "possible leak")
        )
    elif item.case_id == "T047" and heading:
        grounding = "угроз" in content_lower or "threat" in heading.lower()

    return Doc08CaseDiagnostic(
        case_id=item.case_id,
        query_preview=query[:120],
        expected_primary=expected_primary,
        baseline_vector_ranks_doc08=_ranks_for_doc(baseline_vector, DOC08_DOCUMENT_ID),
        candidate_vector_ranks_doc08=_ranks_for_doc(candidate_vector, DOC08_DOCUMENT_ID),
        baseline_vector_ranks_doc12=_ranks_for_doc(baseline_vector, DOC12_DOCUMENT_ID),
        candidate_vector_ranks_doc12=_ranks_for_doc(candidate_vector, DOC12_DOCUMENT_ID),
        baseline_pool_rank_doc08=pool_rank(item.baseline_pool_doc_ids),
        candidate_pool_rank_doc08=pool_rank(item.candidate_pool_doc_ids),
        baseline_final_rank_doc08=final_rank(item.baseline_final_doc_ids, DOC08_DOCUMENT_ID),
        candidate_final_rank_doc08=final_rank(item.candidate_final_doc_ids, DOC08_DOCUMENT_ID),
        baseline_final_rank_doc12=final_rank(item.baseline_final_doc_ids, DOC12_DOCUMENT_ID),
        candidate_final_rank_doc12=final_rank(item.candidate_final_doc_ids, DOC12_DOCUMENT_ID),
        candidate_best_doc08_heading=heading,
        candidate_best_doc08_similarity=sim,
        top_competing_docs_candidate=top_competing,
        doc08_grounding_relevant=grounding,
        baseline_primary_reachable=item.baseline_pool.primary_reachable,
        candidate_primary_reachable=item.candidate_pool.primary_reachable,
        baseline_primary_hit_at_4=item.baseline_case.primary_hit_at_4,
        candidate_primary_hit_at_4=item.candidate_case.primary_hit_at_4,
        baseline_primary_hit_at_12=item.baseline_ranking.metrics.hit_at_12,
        candidate_primary_hit_at_12=item.candidate_ranking.metrics.hit_at_12,
    )


__all__ = [
    "DOC12_DOCUMENT_ID",
    "NEGATIVE_EXTENSION_IDS",
    "PRIVACY_EXTENSION_IDS",
    "REQUIRED_CASE_IDS",
    "THREAT_EXTENSION_IDS",
    "assert_baseline_reproduction",
    "build_case_diagnostic",
    "build_doc08_footprint",
    "build_doc08_reachability_comparison",
    "build_extension_case_diagnostic",
    "build_frozen_reachability_snapshot",
    "classify_threat_case_delta",
    "compute_extension_metrics",
    "compute_primary_hit_at_12",
    "count_faq_in_top4",
    "evaluate_extension_acceptance",
    "evaluate_frozen_acceptance",
    "fully_unreachable_case_ids",
    "load_baseline_reference_oracle",
    "primary_unreachable_case_ids",
    "validate_baseline_reproduction",
]
