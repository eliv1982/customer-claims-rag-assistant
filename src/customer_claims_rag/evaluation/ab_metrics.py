"""Metrics and acceptance criteria for reranking A/B evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from customer_claims_rag.evaluation.ab_models import (
    AbCaseResult,
    AbComparison,
    AcceptanceCriteriaResult,
    AggregateComparison,
    ArmMetrics,
    BaselineIdentityReport,
    BaselineIdentityStatus,
    CaseArmResult,
    CaseComparison,
    CaseIdentityClassification,
    CaseIdentityClass,
    CategoryComparison,
    DocumentMovementSummary,
    FaqPolicyMovement,
    RankedCandidateAudit,
    RiskComparison,
)
from customer_claims_rag.evaluation.metrics import (
    aggregate_case_metrics,
    aggregate_category_metrics,
    aggregate_risk_metrics,
)
from customer_claims_rag.evaluation.models import CaseResult, EvaluationRun, RetrievedChunkResult, RunMetadata
from customer_claims_rag.evaluation.result_id import compute_evaluation_result_id
from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.retrieval.reranker import RerankedCandidate


def compute_evaluation_dataset_fingerprint(
    *,
    questions_path: Path,
    expected_path: Path,
) -> str:
    """Hash evaluation corpus source files."""
    payload = {
        "questions_sha256": _file_sha256(questions_path),
        "expected_sha256": _file_sha256(expected_path),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def case_result_to_arm_metrics(case: CaseResult) -> ArmMetrics:
    """Convert a CaseResult into arm metrics."""
    return ArmMetrics(
        hit_at_1=case.hit_at_1,
        hit_at_4=case.hit_at_4,
        hit_at_12=case.hit_at_12,
        document_recall_at_1=case.document_recall_at_1,
        document_recall_at_4=case.document_recall_at_4,
        document_recall_at_12=case.document_recall_at_12,
        reciprocal_rank=case.reciprocal_rank,
        primary_hit_at_1=case.primary_hit_at_1,
        primary_hit_at_4=case.primary_hit_at_4,
        supporting_hit_at_4=case.supporting_hit_at_4,
        status=case.status,
        error_message=case.error_message,
    )


def search_results_to_audits(
    results: list[SearchResult],
    *,
    reranked: list[RerankedCandidate] | None = None,
) -> list[RankedCandidateAudit]:
    """Build audit records from baseline or reranked results."""
    if reranked is not None:
        return [
            RankedCandidateAudit(
                baseline_rank=item.baseline_rank,
                candidate_rank=item.candidate_rank,
                chunk_id=item.result.chunk_id,
                document_id=item.result.document_id,
                chunk_type=item.result.chunk_type,
                similarity=item.result.similarity,
                distance=item.result.distance,
                source_authority_bonus=item.source_authority_bonus,
                rerank_score=item.rerank_score,
            )
            for item in reranked
        ]

    return [
        RankedCandidateAudit(
            baseline_rank=result.rank,
            candidate_rank=None,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            chunk_type=result.chunk_type,
            similarity=result.similarity,
            distance=result.distance,
            source_authority_bonus=None,
            rerank_score=None,
        )
        for result in results
    ]


def compute_candidate_generation_flags(
    *,
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
    baseline_document_ids: list[str],
    fallback_expected: bool,
) -> dict[str, bool]:
    """Derive candidate-generation diagnostics for one case."""
    if fallback_expected:
        return {
            "primary_candidate_generation_failure": False,
            "supporting_candidate_generation_failure": False,
            "candidate_generation_gap": False,
            "reranker_not_applicable": False,
        }

    baseline_set = set(baseline_document_ids)
    primary_failure = bool(expected_primary_documents) and not (
        set(expected_primary_documents) & baseline_set
    )
    supporting_failure = bool(expected_supporting_documents) and not (
        set(expected_supporting_documents) & baseline_set
    )
    has_expected_sources = bool(expected_primary_documents or expected_supporting_documents)
    any_expected_source_reachable = bool(
        (set(expected_primary_documents) | set(expected_supporting_documents)) & baseline_set
    )
    reranker_not_applicable = has_expected_sources and not any_expected_source_reachable

    return {
        "primary_candidate_generation_failure": primary_failure,
        "supporting_candidate_generation_failure": supporting_failure,
        "candidate_generation_gap": primary_failure or supporting_failure,
        "reranker_not_applicable": reranker_not_applicable,
    }


def build_case_comparison(
    *,
    baseline_metrics: ArmMetrics,
    candidate_metrics: ArmMetrics,
    baseline_audits: list[RankedCandidateAudit],
    candidate_audits: list[RankedCandidateAudit],
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
    fallback_expected: bool = False,
) -> CaseComparison:
    """Compare baseline and candidate arms for one case."""
    baseline_ranks = {item.chunk_id: item.baseline_rank for item in baseline_audits}
    candidate_ranks = {
        item.chunk_id: item.candidate_rank
        for item in candidate_audits
        if item.candidate_rank is not None
    }

    promotions: list[str] = []
    demotions: list[str] = []
    rank_changes: dict[str, int] = {}
    for chunk_id, baseline_rank in baseline_ranks.items():
        candidate_rank = candidate_ranks.get(chunk_id)
        if candidate_rank is None:
            continue
        delta = baseline_rank - candidate_rank
        rank_changes[chunk_id] = delta
        if delta > 0:
            promotions.append(chunk_id)
        elif delta < 0:
            demotions.append(chunk_id)

    baseline_doc_ids = _unique_doc_ids(baseline_audits)
    generation_flags = compute_candidate_generation_flags(
        expected_primary_documents=expected_primary_documents,
        expected_supporting_documents=expected_supporting_documents,
        baseline_document_ids=baseline_doc_ids,
        fallback_expected=fallback_expected,
    )

    return CaseComparison(
        promotions=promotions,
        demotions=demotions,
        rank_changes=rank_changes,
        primary_hit_at_4_delta=int(candidate_metrics.primary_hit_at_4)
        - int(baseline_metrics.primary_hit_at_4),
        supporting_hit_at_4_delta=int(candidate_metrics.supporting_hit_at_4)
        - int(baseline_metrics.supporting_hit_at_4),
        mrr_delta=candidate_metrics.reciprocal_rank - baseline_metrics.reciprocal_rank,
        shared_pool_match=sorted(baseline_ranks) == sorted(candidate_ranks),
        **generation_flags,
    )


def build_ab_comparison(
    *,
    baseline_run: EvaluationRun,
    candidate_run: EvaluationRun,
    case_results: list[AbCaseResult],
    baseline_identity_pass: bool,
    shared_pool_pass: bool,
) -> AbComparison:
    """Build aggregate comparison and acceptance criteria."""
    baseline_agg = baseline_run.aggregate_metrics
    candidate_agg = candidate_run.aggregate_metrics
    aggregate = AggregateComparison(
        hit_rate_at_1_delta=candidate_agg.hit_rate_at_1 - baseline_agg.hit_rate_at_1,
        hit_rate_at_4_delta=candidate_agg.hit_rate_at_4 - baseline_agg.hit_rate_at_4,
        hit_rate_at_12_delta=candidate_agg.hit_rate_at_12 - baseline_agg.hit_rate_at_12,
        document_recall_at_1_delta=candidate_agg.document_recall_at_1
        - baseline_agg.document_recall_at_1,
        document_recall_at_4_delta=candidate_agg.document_recall_at_4
        - baseline_agg.document_recall_at_4,
        document_recall_at_12_delta=candidate_agg.document_recall_at_12
        - baseline_agg.document_recall_at_12,
        mrr_delta=candidate_agg.mrr - baseline_agg.mrr,
        primary_source_hit_rate_at_1_delta=candidate_agg.primary_source_hit_rate_at_1
        - baseline_agg.primary_source_hit_rate_at_1,
        primary_source_hit_rate_at_4_delta=candidate_agg.primary_source_hit_rate_at_4
        - baseline_agg.primary_source_hit_rate_at_4,
        supporting_source_hit_rate_at_4_delta=_supporting_delta(
            baseline_agg.supporting_source_hit_rate_at_4,
            candidate_agg.supporting_source_hit_rate_at_4,
        ),
    )

    risk_slices = _build_risk_comparisons(baseline_run, candidate_run)
    category_slices = _build_category_comparisons(baseline_run, candidate_run)
    document_movements = _build_document_movements(case_results)
    faq_policy_movement = _build_faq_policy_movement(case_results)
    acceptance = evaluate_acceptance_criteria(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        case_results=case_results,
        aggregate=aggregate,
        risk_slices=risk_slices,
        baseline_identity_pass=baseline_identity_pass,
        shared_pool_pass=shared_pool_pass,
    )

    return AbComparison(
        aggregate=aggregate,
        risk_slices=risk_slices,
        category_slices=category_slices,
        document_movements=document_movements,
        faq_policy_movement=faq_policy_movement,
        acceptance=acceptance,
    )


# Tolerance applies only to float comparison between separate live embedding runs.
# It is not used for chunk IDs, order, status, or case metrics. Observed max drift
# is stored in the identity artifact. This value is not a production retrieval
# threshold and does not make different rankings an exact match.
FROZEN_SIMILARITY_TOLERANCE = 0.004


def compare_to_frozen_baseline(
    *,
    computed_run: EvaluationRun,
    frozen_run: EvaluationRun,
    case_results: list[AbCaseResult] | None = None,
) -> BaselineIdentityReport:
    """Compare computed baseline arm against frozen baseline artifact."""
    return build_baseline_identity_report(
        computed_run=computed_run,
        frozen_run=frozen_run,
        case_results=case_results or [],
    )


def build_baseline_identity_report(
    *,
    computed_run: EvaluationRun,
    frozen_run: EvaluationRun,
    case_results: list[AbCaseResult],
) -> BaselineIdentityReport:
    """Build full baseline identity report with per-case classification."""
    frozen_cases = {case.test_id: case for case in frozen_run.case_results}
    computed_cases = {case.test_id: case for case in computed_run.case_results}
    all_ids = sorted(set(frozen_cases) | set(computed_cases))

    classifications: list[CaseIdentityClassification] = []
    max_similarity_drift = 0.0
    max_distance_drift = 0.0
    counts = {
        "exact": 0,
        "same_order_float_drift": 0,
        "same_set_different_order": 0,
        "substantive_mismatch": 0,
    }

    for test_id in all_ids:
        frozen = frozen_cases.get(test_id)
        computed = computed_cases.get(test_id)
        if frozen is None or computed is None:
            classification: CaseIdentityClass = "substantive_mismatch"
        else:
            classification, sim_drift, dist_drift = classify_case_baseline_identity(
                frozen,
                computed,
            )
            max_similarity_drift = max(max_similarity_drift, sim_drift)
            max_distance_drift = max(max_distance_drift, dist_drift)
        counts[classification] += 1
        classifications.append(
            CaseIdentityClassification(case_id=test_id, classification=classification)
        )

    normalized_computed = _normalize_git_metadata(computed_run, frozen_run.run_metadata)
    normalized_frozen = _normalize_git_metadata(frozen_run, frozen_run.run_metadata)
    computed_id = compute_evaluation_result_id(
        normalized_computed.model_copy(
            update={
                "run_metadata": normalized_computed.run_metadata.model_copy(
                    update={"evaluation_result_id": "pending"},
                )
            }
        )
    )
    frozen_id = compute_evaluation_result_id(
        normalized_frozen.model_copy(
            update={
                "run_metadata": normalized_frozen.run_metadata.model_copy(
                    update={"evaluation_result_id": "pending"},
                )
            }
        )
    )

    shared_pool_exact_match = (
        not case_results or all(case.comparison.shared_pool_match for case in case_results)
    )
    frozen_metrics_match = _frozen_aggregate_metrics_match(computed_run, frozen_run)
    exact_count = counts["exact"]
    substantive_mismatch_count = counts["substantive_mismatch"]
    case_count = len(all_ids)
    frozen_exact_order_match = exact_count == case_count
    identity_status = _resolve_identity_status(
        exact_count=exact_count,
        case_count=case_count,
        substantive_mismatch_count=substantive_mismatch_count,
        frozen_metrics_match=frozen_metrics_match,
        shared_pool_exact_match=shared_pool_exact_match,
    )
    mismatched_cases = [
        item.case_id
        for item in classifications
        if item.classification == "substantive_mismatch"
    ]

    return BaselineIdentityReport(
        frozen_evaluation_result_id=frozen_run.run_metadata.evaluation_result_id,
        computed_evaluation_result_id=computed_run.run_metadata.evaluation_result_id,
        frozen_index_fingerprint=frozen_run.run_metadata.index_fingerprint,
        evaluation_result_id_match=computed_id == frozen_id,
        case_count=case_count,
        exact_count=exact_count,
        same_order_float_drift_count=counts["same_order_float_drift"],
        same_set_different_order_count=counts["same_set_different_order"],
        substantive_mismatch_count=substantive_mismatch_count,
        case_classifications=classifications,
        max_similarity_drift=max_similarity_drift,
        max_distance_drift=max_distance_drift,
        similarity_tolerance=FROZEN_SIMILARITY_TOLERANCE,
        shared_pool_exact_match=shared_pool_exact_match,
        frozen_metrics_match=frozen_metrics_match,
        frozen_exact_order_match=frozen_exact_order_match,
        identity_status=identity_status,
        per_case_match=substantive_mismatch_count == 0,
        mismatched_cases=mismatched_cases,
    )


def classify_case_baseline_identity(
    frozen: CaseResult,
    computed: CaseResult,
) -> tuple[CaseIdentityClass, float, float]:
    """Classify one baseline case against frozen 2B output."""
    max_sim_drift = 0.0
    max_dist_drift = 0.0

    if frozen.status != computed.status:
        return "substantive_mismatch", max_sim_drift, max_dist_drift
    if frozen.error_message != computed.error_message:
        return "substantive_mismatch", max_sim_drift, max_dist_drift
    if not _case_scalar_fields_equal(frozen, computed):
        return "substantive_mismatch", max_sim_drift, max_dist_drift
    if len(frozen.retrieved_chunks) != len(computed.retrieved_chunks):
        return "substantive_mismatch", max_sim_drift, max_dist_drift

    frozen_chunks = frozen.retrieved_chunks
    computed_chunks = computed.retrieved_chunks
    frozen_ids_ordered = [chunk.chunk_id for chunk in frozen_chunks]
    computed_ids_ordered = [chunk.chunk_id for chunk in computed_chunks]

    if sorted(frozen_ids_ordered) != sorted(computed_ids_ordered):
        return "substantive_mismatch", max_sim_drift, max_dist_drift

    frozen_by_id = {chunk.chunk_id: chunk for chunk in frozen_chunks}
    computed_by_id = {chunk.chunk_id: chunk for chunk in computed_chunks}
    for chunk_id in frozen_by_id:
        frozen_chunk = frozen_by_id[chunk_id]
        computed_chunk = computed_by_id[chunk_id]
        if (
            frozen_chunk.document_id != computed_chunk.document_id
            or frozen_chunk.source_path != computed_chunk.source_path
            or frozen_chunk.heading != computed_chunk.heading
        ):
            return "substantive_mismatch", max_sim_drift, max_dist_drift
        sim_drift = abs(frozen_chunk.similarity - computed_chunk.similarity)
        dist_drift = abs(frozen_chunk.distance - computed_chunk.distance)
        max_sim_drift = max(max_sim_drift, sim_drift)
        max_dist_drift = max(max_dist_drift, dist_drift)
        if sim_drift > FROZEN_SIMILARITY_TOLERANCE or dist_drift > FROZEN_SIMILARITY_TOLERANCE:
            return "substantive_mismatch", max_sim_drift, max_dist_drift

    if frozen_ids_ordered != computed_ids_ordered:
        return "same_set_different_order", max_sim_drift, max_dist_drift

    all_exact = True
    for frozen_chunk, computed_chunk in zip(frozen_chunks, computed_chunks, strict=True):
        if (
            frozen_chunk.similarity != computed_chunk.similarity
            or frozen_chunk.distance != computed_chunk.distance
        ):
            all_exact = False
            break

    if all_exact:
        return "exact", max_sim_drift, max_dist_drift
    return "same_order_float_drift", max_sim_drift, max_dist_drift


def baseline_identity_pass(identity_report: BaselineIdentityReport) -> bool:
    """Return True when baseline identity is acceptable for acceptance criteria."""
    return identity_report.identity_status != "substantive_mismatch"


def build_evaluation_run_from_cases(
    *,
    case_results: list[CaseResult],
    run_metadata_template: EvaluationRun,
) -> EvaluationRun:
    """Build an EvaluationRun from case results using shared metadata template."""
    aggregate = aggregate_case_metrics(case_results)
    risk_metrics = aggregate_risk_metrics(case_results)
    category_metrics = aggregate_category_metrics(case_results)
    technical_errors = [
        f"{case.test_id}: {case.error_message}"
        for case in case_results
        if case.status == "technical_error" and case.error_message
    ]
    return EvaluationRun(
        run_metadata=run_metadata_template.run_metadata,
        aggregate_metrics=aggregate,
        risk_metrics=risk_metrics,
        category_metrics=category_metrics,
        threshold_analysis=run_metadata_template.threshold_analysis,
        case_results=case_results,
        technical_errors=technical_errors,
        fallback_analysis=run_metadata_template.fallback_analysis,
    )


def chunks_from_search_results(results: list[SearchResult]) -> list[RetrievedChunkResult]:
    """Convert SearchResult list to RetrievedChunkResult list."""
    return [
        RetrievedChunkResult(
            rank=result.rank,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            source_path=result.source_path.replace("\\", "/"),
            similarity=result.similarity,
            distance=result.distance,
            heading=result.heading,
        )
        for result in results
    ]


def build_case_arm_result(
    *,
    results: list[SearchResult],
    reranked: list[RerankedCandidate] | None,
    case: CaseResult,
) -> CaseArmResult:
    """Build one arm result with ordered audit records."""
    audits = search_results_to_audits(results, reranked=reranked)
    return CaseArmResult(
        ordered_candidates=audits,
        metrics=case_result_to_arm_metrics(case),
    )


def evaluate_acceptance_criteria(
    *,
    baseline_run: EvaluationRun,
    candidate_run: EvaluationRun,
    case_results: list[AbCaseResult],
    aggregate: AggregateComparison,
    risk_slices: list[RiskComparison],
    baseline_identity_pass: bool,
    shared_pool_pass: bool,
) -> AcceptanceCriteriaResult:
    """Evaluate hard invariants and candidate success criteria."""
    failure_reasons: list[str] = []
    technical_errors_zero = candidate_run.aggregate_metrics.technical_error_count == 0
    if not technical_errors_zero:
        failure_reasons.append("technical_errors > 0")

    if not baseline_identity_pass:
        failure_reasons.append("baseline identity mismatch")
    if not shared_pool_pass:
        failure_reasons.append("shared candidate pool mismatch")

    high_critical_net = _net_primary_promotions(case_results, {"high", "critical"})
    high_critical_pass = high_critical_net >= 1
    if not high_critical_pass:
        failure_reasons.append("high+critical net primary_hit@4 promotions < 1")

    critical_regressions = _primary_regressions(case_results, "critical")
    critical_pass = critical_regressions == 0
    if not critical_pass:
        failure_reasons.append("critical primary_hit@4 regressions > 0")

    overall_primary_delta = aggregate.primary_source_hit_rate_at_4_delta
    overall_primary_pass = overall_primary_delta >= 0.0
    if not overall_primary_pass:
        failure_reasons.append("overall primary_hit@4 delta < 0")

    overall_mrr_pass = aggregate.mrr_delta >= -0.01
    if not overall_mrr_pass:
        failure_reasons.append("overall MRR delta < -0.01")

    low_hit_regressions = _hit_at_4_regressions(case_results, "low")
    low_mrr_delta = _risk_mrr_delta(risk_slices, "low")
    low_hit_pass = low_hit_regressions <= 1
    low_mrr_pass = low_mrr_delta >= -0.02
    if not low_hit_pass:
        failure_reasons.append("low Hit@4 regressions > 1")
    if not low_mrr_pass:
        failure_reasons.append("low MRR delta < -0.02")

    medium_hit_regressions = _hit_at_4_regressions(case_results, "medium")
    medium_mrr_delta = _risk_mrr_delta(risk_slices, "medium")
    medium_hit_pass = medium_hit_regressions <= 1
    medium_mrr_pass = medium_mrr_delta >= -0.02
    if not medium_hit_pass:
        failure_reasons.append("medium Hit@4 regressions > 1")
    if not medium_mrr_pass:
        failure_reasons.append("medium MRR delta < -0.02")

    fallback_status_preserved = _fallback_status_preserved(
        baseline_run.case_results,
        candidate_run.case_results,
    )
    if not fallback_status_preserved:
        failure_reasons.append("fallback status changed")

    hard_invariants_pass = (
        baseline_identity_pass
        and shared_pool_pass
        and technical_errors_zero
    )
    candidate_accepted = hard_invariants_pass and not failure_reasons

    return AcceptanceCriteriaResult(
        hard_invariants_pass=hard_invariants_pass,
        baseline_identity_pass=baseline_identity_pass,
        shared_pool_pass=shared_pool_pass,
        technical_errors_zero=technical_errors_zero,
        high_critical_net_primary_promotions=high_critical_net,
        high_critical_net_primary_promotions_pass=high_critical_pass,
        critical_primary_regressions=critical_regressions,
        critical_primary_regressions_pass=critical_pass,
        overall_primary_hit_at_4_delta=overall_primary_delta,
        overall_primary_hit_at_4_pass=overall_primary_pass,
        overall_mrr_delta=aggregate.mrr_delta,
        overall_mrr_pass=overall_mrr_pass,
        low_hit_at_4_regressions=low_hit_regressions,
        low_hit_at_4_regressions_pass=low_hit_pass,
        low_mrr_delta=low_mrr_delta,
        low_mrr_pass=low_mrr_pass,
        medium_hit_at_4_regressions=medium_hit_regressions,
        medium_hit_at_4_regressions_pass=medium_hit_pass,
        medium_mrr_delta=medium_mrr_delta,
        medium_mrr_pass=medium_mrr_pass,
        fallback_status_preserved=fallback_status_preserved,
        candidate_accepted=candidate_accepted,
        failure_reasons=failure_reasons,
    )


def _build_risk_comparisons(
    baseline_run: EvaluationRun,
    candidate_run: EvaluationRun,
) -> list[RiskComparison]:
    baseline_map = {item.risk_level: item for item in baseline_run.risk_metrics}
    candidate_map = {item.risk_level: item for item in candidate_run.risk_metrics}
    slices: list[RiskComparison] = []
    for risk_level in sorted(set(baseline_map) | set(candidate_map)):
        baseline = baseline_map[risk_level]
        candidate = candidate_map[risk_level]
        baseline_primary = _primary_hit_rate_for_risk(baseline_run, risk_level)
        candidate_primary = _primary_hit_rate_for_risk(candidate_run, risk_level)
        slices.append(
            RiskComparison(
                risk_level=risk_level,
                case_count=baseline.case_count,
                baseline_hit_rate_at_4=baseline.hit_rate_at_4,
                candidate_hit_rate_at_4=candidate.hit_rate_at_4,
                hit_rate_at_4_delta=candidate.hit_rate_at_4 - baseline.hit_rate_at_4,
                baseline_mrr=baseline.mrr,
                candidate_mrr=candidate.mrr,
                mrr_delta=candidate.mrr - baseline.mrr,
                baseline_primary_hit_rate_at_4=baseline_primary,
                candidate_primary_hit_rate_at_4=candidate_primary,
                primary_hit_rate_at_4_delta=candidate_primary - baseline_primary,
            )
        )
    return slices


def _build_category_comparisons(
    baseline_run: EvaluationRun,
    candidate_run: EvaluationRun,
) -> list[CategoryComparison]:
    baseline_map = {item.category: item for item in baseline_run.category_metrics}
    candidate_map = {item.category: item for item in candidate_run.category_metrics}
    slices: list[CategoryComparison] = []
    for category in sorted(set(baseline_map) | set(candidate_map)):
        baseline = baseline_map[category]
        candidate = candidate_map[category]
        baseline_primary = _primary_hit_rate_for_category(baseline_run, category)
        candidate_primary = _primary_hit_rate_for_category(candidate_run, category)
        slices.append(
            CategoryComparison(
                category=category,
                case_count=baseline.case_count,
                baseline_hit_rate_at_4=baseline.hit_rate_at_4,
                candidate_hit_rate_at_4=candidate.hit_rate_at_4,
                hit_rate_at_4_delta=candidate.hit_rate_at_4 - baseline.hit_rate_at_4,
                baseline_mrr=baseline.mrr,
                candidate_mrr=candidate.mrr,
                mrr_delta=candidate.mrr - baseline.mrr,
                baseline_primary_hit_rate_at_4=baseline_primary,
                candidate_primary_hit_rate_at_4=candidate_primary,
                primary_hit_rate_at_4_delta=candidate_primary - baseline_primary,
            )
        )
    return slices


def _build_document_movements(case_results: list[AbCaseResult]) -> list[DocumentMovementSummary]:
    counts: dict[str, DocumentMovementSummary] = {}
    for case in case_results:
        baseline_docs = _unique_doc_ids(case.baseline.ordered_candidates)
        candidate_docs = _unique_doc_ids(case.candidate.ordered_candidates)
        baseline_positions = {doc: index + 1 for index, doc in enumerate(baseline_docs)}
        candidate_positions = {doc: index + 1 for index, doc in enumerate(candidate_docs)}
        all_docs = set(baseline_positions) | set(candidate_positions)
        for document_id in all_docs:
            summary = counts.setdefault(
                document_id,
                DocumentMovementSummary(document_id=document_id),
            )
            baseline_pos = baseline_positions.get(document_id)
            candidate_pos = candidate_positions.get(document_id)
            if baseline_pos is None or candidate_pos is None:
                continue
            delta = baseline_pos - candidate_pos
            summary.net_rank_change += delta
            if delta > 0:
                summary.promotions += 1
            elif delta < 0:
                summary.demotions += 1
    return sorted(counts.values(), key=lambda item: item.document_id)


def _build_faq_policy_movement(case_results: list[AbCaseResult]) -> FaqPolicyMovement:
    movement = FaqPolicyMovement()
    for case in case_results:
        for chunk_id, delta in case.comparison.rank_changes.items():
            chunk_type = _chunk_type_for_id(case.baseline.ordered_candidates, chunk_id)
            if chunk_type == "faq":
                if delta > 0:
                    movement.faq_chunks_promoted += 1
                elif delta < 0:
                    movement.faq_chunks_demoted += 1
            elif chunk_type == "policy":
                if delta > 0:
                    movement.policy_chunks_promoted += 1
                elif delta < 0:
                    movement.policy_chunks_demoted += 1
            elif chunk_type == "escalation":
                if delta > 0:
                    movement.escalation_chunks_promoted += 1
                elif delta < 0:
                    movement.escalation_chunks_demoted += 1
    return movement


def _case_scalar_fields_equal(frozen: CaseResult, computed: CaseResult) -> bool:
    frozen_data = frozen.model_dump(
        exclude={
            "retrieved_chunks",
            "query",
            "top1_similarity",
            "retrieved_document_ids",
            "retrieved_document_ids_at_4",
        }
    )
    computed_data = computed.model_dump(
        exclude={
            "retrieved_chunks",
            "query",
            "top1_similarity",
            "retrieved_document_ids",
            "retrieved_document_ids_at_4",
        }
    )
    if frozen_data != computed_data:
        return False
    if frozen.top1_similarity is None and computed.top1_similarity is None:
        return True
    if frozen.top1_similarity is None or computed.top1_similarity is None:
        return False
    return math.isclose(
        frozen.top1_similarity,
        computed.top1_similarity,
        abs_tol=FROZEN_SIMILARITY_TOLERANCE,
    )


def _frozen_aggregate_metrics_match(
    computed_run: EvaluationRun,
    frozen_run: EvaluationRun,
) -> bool:
    computed = computed_run.aggregate_metrics.model_dump()
    frozen = frozen_run.aggregate_metrics.model_dump()
    float_stat_fields = {
        "mean_top1_similarity",
        "median_top1_similarity",
        "min_top1_similarity",
        "max_top1_similarity",
    }
    for key, frozen_value in frozen.items():
        computed_value = computed.get(key)
        if key in float_stat_fields:
            if computed_value is None or frozen_value is None:
                if computed_value != frozen_value:
                    return False
                continue
            if not math.isclose(
                float(computed_value),
                float(frozen_value),
                abs_tol=FROZEN_SIMILARITY_TOLERANCE,
            ):
                return False
            continue
        if computed_value != frozen_value:
            return False
    return True


def _resolve_identity_status(
    *,
    exact_count: int,
    case_count: int,
    substantive_mismatch_count: int,
    frozen_metrics_match: bool,
    shared_pool_exact_match: bool,
) -> BaselineIdentityStatus:
    if substantive_mismatch_count > 0 or not frozen_metrics_match or not shared_pool_exact_match:
        return "substantive_mismatch"
    if exact_count == case_count:
        return "exact"
    return "equivalent_with_embedding_drift"


def _file_sha256(path: Path) -> str:
    import hashlib as _hashlib

    digest = _hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _unique_doc_ids(audits: list[RankedCandidateAudit]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for audit in audits:
        if audit.document_id not in seen:
            seen.add(audit.document_id)
            ordered.append(audit.document_id)
    return ordered


def _supporting_delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


def _primary_hit_rate_for_risk(run: EvaluationRun, risk_level: str) -> float:
    cases = [
        case
        for case in run.case_results
        if case.status == "success"
        and not case.fallback_expected
        and case.expected_risk == risk_level
        and case.expected_primary_documents
    ]
    if not cases:
        return 0.0
    hits = sum(1 for case in cases if case.primary_hit_at_4)
    return hits / len(cases)


def _primary_hit_rate_for_category(run: EvaluationRun, category: str) -> float:
    cases = [
        case
        for case in run.case_results
        if case.status == "success"
        and not case.fallback_expected
        and case.category == category
        and case.expected_primary_documents
    ]
    if not cases:
        return 0.0
    hits = sum(1 for case in cases if case.primary_hit_at_4)
    return hits / len(cases)


def _net_primary_promotions(case_results: list[AbCaseResult], risks: set[str]) -> int:
    promotions = 0
    regressions = 0
    for case in case_results:
        if case.risk not in risks or case.fallback_expected:
            continue
        if case.comparison.primary_hit_at_4_delta > 0:
            promotions += 1
        elif case.comparison.primary_hit_at_4_delta < 0:
            regressions += 1
    return promotions - regressions


def _primary_regressions(case_results: list[AbCaseResult], risk: str) -> int:
    return sum(
        1
        for case in case_results
        if case.risk == risk
        and not case.fallback_expected
        and case.comparison.primary_hit_at_4_delta < 0
    )


def _hit_at_4_regressions(case_results: list[AbCaseResult], risk: str) -> int:
    return sum(
        1
        for case in case_results
        if case.risk == risk
        and not case.fallback_expected
        and case.baseline.metrics.hit_at_4
        and not case.candidate.metrics.hit_at_4
    )


def _risk_mrr_delta(risk_slices: list[RiskComparison], risk_level: str) -> float:
    for item in risk_slices:
        if item.risk_level == risk_level:
            return item.mrr_delta
    return 0.0


def _fallback_status_preserved(
    baseline_cases: list[CaseResult],
    candidate_cases: list[CaseResult],
) -> bool:
    baseline_map = {case.test_id: case.status for case in baseline_cases}
    candidate_map = {case.test_id: case.status for case in candidate_cases}
    fallback_ids = {
        case.test_id for case in baseline_cases if case.fallback_expected
    } | {
        case.test_id for case in candidate_cases if case.fallback_expected
    }
    for test_id in fallback_ids:
        if baseline_map.get(test_id) != candidate_map.get(test_id):
            return False
    return True


def _normalize_git_metadata(run: EvaluationRun, template: RunMetadata) -> EvaluationRun:
    metadata = run.run_metadata.model_copy(
        update={
            "git_commit": template.git_commit,
            "git_dirty": template.git_dirty,
            "git_status_summary": template.git_status_summary,
        }
    )
    return run.model_copy(update={"run_metadata": metadata})


def _chunk_type_for_id(audits: list[RankedCandidateAudit], chunk_id: str) -> str | None:
    for audit in audits:
        if audit.chunk_id == chunk_id:
            return audit.chunk_type
    return None
