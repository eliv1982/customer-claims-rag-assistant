"""Paired frozen-benchmark regression for expanded corpus evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from customer_claims_rag.evaluation.expanded_corpus_regression_models import (
    NEW_DOCUMENT_IDS,
    SEMANTIC_OVERLAY_BY_CASE,
    ComparativeVerdict,
    ExpandedCorpusRegressionRun,
    FaqDominanceComparison,
    IndexArmSummary,
    MetricDelta,
    NewDocumentFootprint,
    PairedCaseResult,
    PairedChangeClass,
    SemanticOverlayReview,
)
from customer_claims_rag.evaluation.pool_expansion_models import (
    PoolExpansionCaseResult,
    PoolExpansionEvaluationRun,
)

EVALUATION_ID = "expanded_corpus_frozen_regression_v1"
FROZEN_QUESTION_SET_ID = "foodflow-60-case-frozen-v1"
PRODUCTION_RETRIEVAL_CHAIN = "vector-top-24 -> source-authority-v1 -> final-top-12"
FAQ_DOCUMENT_ID = "10_customer_faq"
ARM_A_ID = "arm_a_historical_10doc_215chunk"
ARM_B_ID = "arm_b_production_15doc_333chunk"


def build_expanded_corpus_regression(
    *,
    arm_a_run: PoolExpansionEvaluationRun,
    arm_b_run: PoolExpansionEvaluationRun,
    arm_a_path: str,
    arm_b_path: str,
) -> ExpandedCorpusRegressionRun:
    """Build paired regression artifact from two production-chain pool expansion runs."""
    _validate_paired_runs(arm_a_run, arm_b_run)
    arm_a = _summarize_arm(
        run=arm_a_run,
        arm_id=ARM_A_ID,
        arm_label="Historical 10-document / 215-chunk index",
        index_path=arm_a_path,
    )
    arm_b = _summarize_arm(
        run=arm_b_run,
        arm_id=ARM_B_ID,
        arm_label="Production 15-document / 333-chunk index",
        index_path=arm_b_path,
    )
    paired_cases = _build_paired_cases(arm_a_run, arm_b_run)
    semantic_overlay = _build_semantic_overlay(paired_cases)
    new_document_footprints = _build_new_document_footprints(paired_cases)
    faq_dominance = _build_faq_dominance(paired_cases)
    harmful_regressions = sorted(
        case.case_id for case in paired_cases if case.harmful_regression
    )
    promotions = sorted(
        case.case_id for case in paired_cases if case.classification == "promotion"
    )
    metric_deltas = _compute_metric_deltas(arm_a, arm_b)
    verdict, rationale = _compute_verdict(
        paired_cases=paired_cases,
        harmful_regressions=harmful_regressions,
        metric_deltas=metric_deltas,
        arm_a=arm_a,
        arm_b=arm_b,
    )
    return ExpandedCorpusRegressionRun(
        evaluation_id=EVALUATION_ID,
        timestamp=datetime.now(timezone.utc),
        evaluation_dataset_fingerprint=arm_a_run.shared_context.evaluation_dataset_fingerprint,
        frozen_question_set_id=FROZEN_QUESTION_SET_ID,
        retrieval_chain=PRODUCTION_RETRIEVAL_CHAIN,
        arm_a=arm_a,
        arm_b=arm_b,
        metric_deltas=metric_deltas,
        paired_cases=paired_cases,
        semantic_overlay=semantic_overlay,
        new_document_footprints=new_document_footprints,
        faq_dominance=faq_dominance,
        harmful_regressions=harmful_regressions,
        promotions=promotions,
        verdict=verdict,
        verdict_rationale=rationale,
        arm_a_source_run=arm_a_run,
        arm_b_source_run=arm_b_run,
    )


def compute_evaluation_result_id(run: ExpandedCorpusRegressionRun) -> str:
    """Return deterministic hash for the paired regression artifact."""
    payload = {
        "evaluation_id": run.evaluation_id,
        "evaluation_dataset_fingerprint": run.evaluation_dataset_fingerprint,
        "arm_a_fingerprint": run.arm_a.index_fingerprint,
        "arm_b_fingerprint": run.arm_b.index_fingerprint,
        "paired_case_ids": [case.case_id for case in run.paired_cases],
        "verdict": run.verdict,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_paired_runs(
    arm_a_run: PoolExpansionEvaluationRun,
    arm_b_run: PoolExpansionEvaluationRun,
) -> None:
    if arm_a_run.experiment.config_hash != arm_b_run.experiment.config_hash:
        raise ValueError("paired arms must use identical frozen retrieval config hash")
    if (
        arm_a_run.shared_context.evaluation_dataset_fingerprint
        != arm_b_run.shared_context.evaluation_dataset_fingerprint
    ):
        raise ValueError("paired arms must use identical evaluation dataset fingerprint")
    if len(arm_a_run.case_results) != len(arm_b_run.case_results):
        raise ValueError("paired arms must contain the same number of cases")
    arm_a_ids = [case.case_id for case in arm_a_run.case_results]
    arm_b_ids = [case.case_id for case in arm_b_run.case_results]
    if arm_a_ids != arm_b_ids:
        raise ValueError("paired arms must contain identical case IDs in the same order")
    for case_a, case_b in zip(arm_a_run.case_results, arm_b_run.case_results, strict=True):
        if case_a.expected_primary_documents != case_b.expected_primary_documents:
            raise ValueError(f"{case_a.case_id}: expected primary documents differ between arms")
        if case_a.expected_supporting_documents != case_b.expected_supporting_documents:
            raise ValueError(
                f"{case_a.case_id}: expected supporting documents differ between arms",
            )


def _summarize_arm(
    *,
    run: PoolExpansionEvaluationRun,
    arm_id: str,
    arm_label: str,
    index_path: str,
) -> IndexArmSummary:
    reach = run.reachability_comparison
    return IndexArmSummary(
        arm_id=arm_id,
        arm_label=arm_label,
        index_path=index_path,
        index_fingerprint=run.shared_context.index_fingerprint,
        chunk_count=run.shared_context.chunk_count or 0,
        document_count=run.shared_context.document_count or 0,
        retrieval_chain=PRODUCTION_RETRIEVAL_CHAIN,
        frozen_retrieval_config_id=run.experiment.experiment_id,
        frozen_retrieval_config_hash=run.experiment.config_hash,
        reranker_id=run.experiment.reranker_id,
        candidate_pool_k=run.shared_context.candidate_pool_k,
        final_top_k=run.shared_context.final_top_k,
        threshold=run.shared_context.threshold,
        aggregate_metrics=run.candidate_ranking.aggregate_metrics,
        risk_metrics=run.candidate_ranking.risk_metrics,
        pool24_primary_reachable=reach.candidate_primary_reachable,
        pool24_primary_total=reach.candidate_primary_total,
        pool24_supporting_reachable=reach.candidate_supporting_reachable,
        pool24_supporting_total=reach.candidate_supporting_total,
        pool24_fully_unreachable_cases=list(reach.candidate_fully_unreachable_cases),
        high_primary_reachable_pool24=reach.high_candidate_primary_reachable,
        critical_primary_reachable_pool24=reach.critical_candidate_primary_reachable,
        high_primary_total=reach.high_primary_total,
        critical_primary_total=reach.critical_primary_total,
    )


def _build_paired_cases(
    arm_a_run: PoolExpansionEvaluationRun,
    arm_b_run: PoolExpansionEvaluationRun,
) -> list[PairedCaseResult]:
    paired: list[PairedCaseResult] = []
    for case_a, case_b in zip(arm_a_run.case_results, arm_b_run.case_results, strict=True):
        paired.append(_compare_case(case_a, case_b))
    return paired


def _compare_case(
    case_a: PoolExpansionCaseResult,
    case_b: PoolExpansionCaseResult,
) -> PairedCaseResult:
    top4_a = _top_document_ids(case_a, 4)
    top4_b = _top_document_ids(case_b, 4)
    top12_a = _top_document_ids(case_a, 12)
    top12_b = _top_document_ids(case_b, 12)
    pool24_b = _pool_document_ids(case_b)
    new_top4 = [doc for doc in top4_b if doc in NEW_DOCUMENT_IDS]
    new_top12 = [doc for doc in top12_b if doc in NEW_DOCUMENT_IDS]
    new_pool24 = [doc for doc in pool24_b if doc in NEW_DOCUMENT_IDS]
    overlay_doc = SEMANTIC_OVERLAY_BY_CASE.get(case_a.case_id)
    frozen_displaced = bool(
        overlay_doc
        and overlay_doc in top4_b
        and not any(doc in top4_b for doc in case_a.expected_primary_documents)
    )
    harmful = _is_harmful_regression(case_a, case_b, top4_b=top4_b, top12_b=top12_b)
    classification, explanation = _classify_case(
        case_a=case_a,
        case_b=case_b,
        top4_a=top4_a,
        top4_b=top4_b,
        harmful=harmful,
        frozen_displaced=frozen_displaced,
        overlay_doc=overlay_doc,
    )
    return PairedCaseResult(
        case_id=case_a.case_id,
        risk=case_a.risk,
        category=case_a.category,
        expected_primary_documents=list(case_a.expected_primary_documents),
        expected_supporting_documents=list(case_a.expected_supporting_documents),
        fallback_expected=case_a.fallback_expected,
        arm_a_primary_rank_final=_primary_rank_final(case_a),
        arm_b_primary_rank_final=_primary_rank_final(case_b),
        arm_a_top4_document_ids=top4_a,
        arm_b_top4_document_ids=top4_b,
        arm_a_top12_document_ids=top12_a,
        arm_b_top12_document_ids=top12_b,
        arm_a_primary_in_pool24=case_a.candidate_pool.reachability.primary_reachable,
        arm_b_primary_in_pool24=case_b.candidate_pool.reachability.primary_reachable,
        arm_a_primary_in_final_top12=case_a.candidate_ranking.flags.primary_in_final_top12,
        arm_b_primary_in_final_top12=case_b.candidate_ranking.flags.primary_in_final_top12,
        arm_a_primary_hit_at_4=case_a.candidate_ranking.metrics.primary_hit_at_4,
        arm_b_primary_hit_at_4=case_b.candidate_ranking.metrics.primary_hit_at_4,
        faq_in_arm_a_top4=FAQ_DOCUMENT_ID in top4_a,
        faq_in_arm_b_top4=FAQ_DOCUMENT_ID in top4_b,
        new_documents_in_arm_b_top4=new_top4,
        new_documents_in_arm_b_top12=new_top12,
        new_documents_in_arm_b_pool24=new_pool24,
        frozen_primary_displaced_by_new_doc=frozen_displaced,
        harmful_regression=harmful,
        classification=classification,
        explanation=explanation,
    )


def _classify_case(
    *,
    case_a: PoolExpansionCaseResult,
    case_b: PoolExpansionCaseResult,
    top4_a: list[str],
    top4_b: list[str],
    harmful: bool,
    frozen_displaced: bool,
    overlay_doc: str | None,
) -> tuple[PairedChangeClass, str]:
    if case_a.fallback_expected or case_b.fallback_expected:
        return "stable", "fallback case excluded from strict source scoring"

    rank_a = _primary_rank_final(case_a)
    rank_b = _primary_rank_final(case_b)
    hit4_a = case_a.candidate_ranking.metrics.primary_hit_at_4
    hit4_b = case_b.candidate_ranking.metrics.primary_hit_at_4
    pool_a = case_a.candidate_pool.reachability.primary_reachable
    pool_b = case_b.candidate_pool.reachability.primary_reachable
    final_a = case_a.candidate_ranking.flags.primary_in_final_top12
    final_b = case_b.candidate_ranking.flags.primary_in_final_top12

    improved = (
        (not hit4_a and hit4_b)
        or (rank_a is None and rank_b is not None)
        or (rank_a is not None and rank_b is not None and rank_b < rank_a)
        or (not final_a and final_b)
        or (not pool_a and pool_b)
    )
    worsened = (
        (hit4_a and not hit4_b)
        or (rank_a is not None and rank_b is None)
        or (rank_a is not None and rank_b is not None and rank_b > rank_a)
        or (final_a and not final_b)
        or (pool_a and not pool_b)
    )

    if harmful:
        if (
            overlay_doc
            and overlay_doc in top4_b
            and worsened
            and not hit4_b
            and case_a.case_id in SEMANTIC_OVERLAY_BY_CASE
        ):
            return (
                "strict_frozen_regression_with_semantically_stronger_new_source",
                "frozen primary strict score declined while a more specific new policy "
                f"({overlay_doc}) surfaced in top-4",
            )
        return "regression", "harmful retrieval regression under frozen benchmark scoring"

    if improved and worsened:
        return "mixed", "primary rank and supporting evidence changed in opposite directions"

    if improved:
        return "promotion", "frozen expected primary retrieval improved in final top-12 or pool@24"

    if worsened:
        if frozen_displaced and overlay_doc:
            return (
                "strict_frozen_regression_with_semantically_stronger_new_source",
                f"new dedicated policy {overlay_doc} outranks frozen primary in top-4",
            )
        return "regression", "frozen expected primary retrieval declined without harmful threshold"

    if frozen_displaced and overlay_doc:
        return (
            "strict_frozen_regression_with_semantically_stronger_new_source",
            f"new dedicated policy {overlay_doc} visible in top-4 while strict score unchanged",
        )

    return "stable", "no material frozen-metric change between index arms"


def _is_harmful_regression(
    case_a: PoolExpansionCaseResult,
    case_b: PoolExpansionCaseResult,
    *,
    top4_b: list[str],
    top12_b: list[str],
) -> bool:
    if case_a.fallback_expected or case_b.fallback_expected:
        return False

    pool_a = case_a.candidate_pool.reachability.primary_reachable
    pool_b = case_b.candidate_pool.reachability.primary_reachable
    if pool_a and not pool_b:
        return True

    final_a = case_a.candidate_ranking.flags.primary_in_final_top12
    final_b = case_b.candidate_ranking.flags.primary_in_final_top12
    overlay_doc = SEMANTIC_OVERLAY_BY_CASE.get(case_a.case_id)
    overlay_compensates = bool(overlay_doc and overlay_doc in top12_b)

    if case_a.risk in {"high", "critical"} and final_a and not final_b and not overlay_compensates:
        return True

    if case_a.risk in {"high", "critical"}:
        faq_outranks_primary = (
            FAQ_DOCUMENT_ID in top4_b
            and not any(doc in top4_b for doc in case_a.expected_primary_documents)
        )
        if faq_outranks_primary and not overlay_compensates:
            return True

    irrelevant_new_top4 = [
        doc
        for doc in top4_b
        if doc in NEW_DOCUMENT_IDS and doc != overlay_doc
    ]
    if (
        irrelevant_new_top4
        and case_a.candidate_ranking.metrics.primary_hit_at_4
        and not case_b.candidate_ranking.metrics.primary_hit_at_4
        and case_a.case_id not in SEMANTIC_OVERLAY_BY_CASE
    ):
        return True

    if (
        "15_conflicting_rules_and_remedy_priority" in top4_b
        and case_a.case_id not in {"T023", "T027", "T053"}
        and not case_b.candidate_ranking.metrics.primary_hit_at_4
    ):
        return True

    return False


def _build_semantic_overlay(paired_cases: list[PairedCaseResult]) -> list[SemanticOverlayReview]:
    reviews: list[SemanticOverlayReview] = []
    by_id = {case.case_id: case for case in paired_cases}
    for case_id, overlay_doc in SEMANTIC_OVERLAY_BY_CASE.items():
        case = by_id.get(case_id)
        if case is None:
            continue
        frozen_in_top12 = any(
            doc in case.arm_b_top12_document_ids for doc in case.expected_primary_documents
        )
        frozen_in_pool24 = case.arm_b_primary_in_pool24
        overlay_top4 = overlay_doc in case.arm_b_top4_document_ids
        overlay_top12 = overlay_doc in case.arm_b_top12_document_ids
        overlay_pool24 = overlay_doc in case.new_documents_in_arm_b_pool24
        more_specific = overlay_top4 or overlay_top12
        reviews.append(
            SemanticOverlayReview(
                case_id=case_id,
                frozen_expected_primary=list(case.expected_primary_documents),
                overlay_document_id=overlay_doc,
                overlay_in_arm_b_top4=overlay_top4,
                overlay_in_arm_b_top12=overlay_top12,
                overlay_in_arm_b_pool24=overlay_pool24,
                frozen_primary_in_arm_b_top12=frozen_in_top12,
                frozen_primary_in_arm_b_pool24=frozen_in_pool24,
                more_specific_than_frozen=more_specific,
                frozen_should_remain_supporting=more_specific and frozen_in_pool24,
                recommend_extension_set=more_specific,
                recommend_benchmark_modification=False,
                notes=(
                    "do not modify frozen benchmark; add extension coverage"
                    if more_specific
                    else "overlay document not yet dominant"
                ),
            )
        )
    return reviews


def _build_new_document_footprints(
    paired_cases: list[PairedCaseResult],
) -> list[NewDocumentFootprint]:
    footprints: dict[str, NewDocumentFootprint] = {
        doc_id: NewDocumentFootprint(document_id=doc_id) for doc_id in NEW_DOCUMENT_IDS
    }
    for case in paired_cases:
        if case.fallback_expected:
            continue
        for doc_id in NEW_DOCUMENT_IDS:
            footprint = footprints[doc_id]
            if _rank_of_document(case, doc_id, case.arm_b_top4_document_ids) == 1:
                footprint.top1_questions.append(case.case_id)
            if doc_id in case.arm_b_top4_document_ids:
                footprint.top4_questions.append(case.case_id)
            if doc_id in case.arm_b_top12_document_ids:
                footprint.top12_questions.append(case.case_id)
            if doc_id in case.new_documents_in_arm_b_pool24:
                footprint.pool24_questions.append(case.case_id)
            if case.case_id in SEMANTIC_OVERLAY_BY_CASE and SEMANTIC_OVERLAY_BY_CASE[case.case_id] == doc_id:
                footprint.likely_helpful_questions.append(case.case_id)
            elif doc_id in case.arm_b_top4_document_ids and case.case_id not in SEMANTIC_OVERLAY_BY_CASE:
                footprint.irrelevant_noise_questions.append(case.case_id)
            if (
                case.frozen_primary_displaced_by_new_doc
                and doc_id in case.arm_b_top4_document_ids
            ):
                footprint.displaces_frozen_primary_questions.append(case.case_id)
    return [footprints[doc_id] for doc_id in NEW_DOCUMENT_IDS]


def _build_faq_dominance(paired_cases: list[PairedCaseResult]) -> FaqDominanceComparison:
    by_risk: dict[str, dict[str, int]] = {}
    arm_a_top1 = arm_b_top1 = 0
    arm_a_top4_slots = arm_b_top4_slots = 0
    arm_a_questions = arm_b_questions = 0
    arm_a_outrank = arm_b_outrank = 0

    for case in paired_cases:
        if case.fallback_expected:
            continue
        risk_bucket = by_risk.setdefault(
            case.risk,
            {
                "arm_a_faq_top4_questions": 0,
                "arm_b_faq_top4_questions": 0,
            },
        )
        if case.arm_a_top4_document_ids and case.arm_a_top4_document_ids[0] == FAQ_DOCUMENT_ID:
            arm_a_top1 += 1
        if case.arm_b_top4_document_ids and case.arm_b_top4_document_ids[0] == FAQ_DOCUMENT_ID:
            arm_b_top1 += 1
        if case.faq_in_arm_a_top4:
            arm_a_top4_slots += sum(1 for doc in case.arm_a_top4_document_ids if doc == FAQ_DOCUMENT_ID)
            arm_a_questions += 1
            risk_bucket["arm_a_faq_top4_questions"] += 1
        if case.faq_in_arm_b_top4:
            arm_b_top4_slots += sum(1 for doc in case.arm_b_top4_document_ids if doc == FAQ_DOCUMENT_ID)
            arm_b_questions += 1
            risk_bucket["arm_b_faq_top4_questions"] += 1
        if _faq_outranks_all_primaries(case.arm_a_top4_document_ids, case.expected_primary_documents):
            arm_a_outrank += 1
        if _faq_outranks_all_primaries(case.arm_b_top4_document_ids, case.expected_primary_documents):
            arm_b_outrank += 1

    return FaqDominanceComparison(
        arm_a_faq_top1_count=arm_a_top1,
        arm_b_faq_top1_count=arm_b_top1,
        arm_a_faq_top4_chunk_slots=arm_a_top4_slots,
        arm_b_faq_top4_chunk_slots=arm_b_top4_slots,
        arm_a_questions_with_faq_in_top4=arm_a_questions,
        arm_b_questions_with_faq_in_top4=arm_b_questions,
        arm_a_faq_outranks_all_primaries=arm_a_outrank,
        arm_b_faq_outranks_all_primaries=arm_b_outrank,
        by_risk=by_risk,
    )


def _compute_metric_deltas(arm_a: IndexArmSummary, arm_b: IndexArmSummary) -> MetricDelta:
    a = arm_a.aggregate_metrics
    b = arm_b.aggregate_metrics
    supporting_delta = None
    if (
        a.supporting_source_hit_rate_at_4 is not None
        and b.supporting_source_hit_rate_at_4 is not None
    ):
        supporting_delta = b.supporting_source_hit_rate_at_4 - a.supporting_source_hit_rate_at_4
    return MetricDelta(
        hit_rate_at_1=b.hit_rate_at_1 - a.hit_rate_at_1,
        hit_rate_at_4=b.hit_rate_at_4 - a.hit_rate_at_4,
        hit_rate_at_12=b.hit_rate_at_12 - a.hit_rate_at_12,
        document_recall_at_1=b.document_recall_at_1 - a.document_recall_at_1,
        document_recall_at_4=b.document_recall_at_4 - a.document_recall_at_4,
        document_recall_at_12=b.document_recall_at_12 - a.document_recall_at_12,
        mrr=b.mrr - a.mrr,
        primary_source_hit_rate_at_1=b.primary_source_hit_rate_at_1 - a.primary_source_hit_rate_at_1,
        primary_source_hit_rate_at_4=b.primary_source_hit_rate_at_4 - a.primary_source_hit_rate_at_4,
        supporting_source_hit_rate_at_4=supporting_delta,
        pool24_primary_reachable_delta=arm_b.pool24_primary_reachable - arm_a.pool24_primary_reachable,
        pool24_fully_unreachable_delta=len(arm_b.pool24_fully_unreachable_cases)
        - len(arm_a.pool24_fully_unreachable_cases),
        high_primary_reachable_pool24_delta=arm_b.high_primary_reachable_pool24
        - arm_a.high_primary_reachable_pool24,
        critical_primary_reachable_pool24_delta=arm_b.critical_primary_reachable_pool24
        - arm_a.critical_primary_reachable_pool24,
    )


def _compute_verdict(
    *,
    paired_cases: list[PairedCaseResult],
    harmful_regressions: list[str],
    metric_deltas: MetricDelta,
    arm_a: IndexArmSummary,
    arm_b: IndexArmSummary,
) -> tuple[ComparativeVerdict, str]:
    high_critical_harmful = [
        case.case_id
        for case in paired_cases
        if case.harmful_regression and case.risk in {"high", "critical"}
    ]
    if high_critical_harmful:
        return (
            "REJECT / REPAIR REQUIRED",
            "high/critical harmful regressions detected: "
            + ", ".join(high_critical_harmful),
        )
    if harmful_regressions:
        return (
            "REJECT / REPAIR REQUIRED",
            "harmful regressions detected: " + ", ".join(harmful_regressions),
        )
    if (
        metric_deltas.critical_primary_reachable_pool24_delta < 0
        or metric_deltas.high_primary_reachable_pool24_delta < 0
    ):
        return (
            "REJECT / REPAIR REQUIRED",
            "high/critical primary reachability in pool@24 declined with expanded corpus",
        )

    semantic_displacements = [
        case.case_id
        for case in paired_cases
        if case.classification
        == "strict_frozen_regression_with_semantically_stronger_new_source"
    ]
    if semantic_displacements or metric_deltas.primary_source_hit_rate_at_4 < 0:
        return (
            "ACCEPT_WITH_TARGETED_FOLLOW-UP",
            "no safety-critical harmful regression; strict frozen metrics show explainable "
            "new-policy displacement requiring extension-set coverage before release validation",
        )

    if (
        metric_deltas.mrr >= 0
        and metric_deltas.primary_source_hit_rate_at_4 >= 0
        and arm_b.pool24_primary_reachable >= arm_a.pool24_primary_reachable
    ):
        return (
            "ACCEPT",
            "expanded corpus preserves or improves frozen retrieval with no harmful regressions",
        )

    return (
        "ACCEPT_WITH_TARGETED_FOLLOW-UP",
        "mixed aggregate movement without safety-critical failure; extension-set review recommended",
    )


def _top_document_ids(case: PoolExpansionCaseResult, k: int) -> list[str]:
    ordered = [candidate.document_id for candidate in case.candidate_ranking.ordered_candidates[:k]]
    return unique_document_ids_from_ids(ordered)


def _pool_document_ids(case: PoolExpansionCaseResult) -> list[str]:
    doc_ids = [chunk_id.split("::", 1)[0] for chunk_id in case.candidate_pool.candidate_ids]
    return unique_document_ids_from_ids(doc_ids)


def unique_document_ids_from_ids(document_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for document_id in document_ids:
        if document_id not in seen:
            seen.add(document_id)
            ordered.append(document_id)
    return ordered


def _primary_rank_final(case: PoolExpansionCaseResult) -> int | None:
    if case.fallback_expected:
        return None
    primary = set(case.expected_primary_documents)
    for index, candidate in enumerate(case.candidate_ranking.ordered_candidates, start=1):
        if candidate.document_id in primary:
            return index
    return None


def _rank_of_document(case: PairedCaseResult, document_id: str, top_docs: list[str]) -> int | None:
    if document_id not in top_docs:
        return None
    for index, doc in enumerate(top_docs, start=1):
        if doc == document_id:
            return index
    return None


def _faq_outranks_all_primaries(top_docs: list[str], expected_primary: list[str]) -> bool:
    if not top_docs or FAQ_DOCUMENT_ID not in top_docs:
        return False
    faq_rank = top_docs.index(FAQ_DOCUMENT_ID)
    primary_ranks = [
        top_docs.index(doc) for doc in expected_primary if doc in top_docs
    ]
    if not primary_ranks:
        return True
    return faq_rank < min(primary_ranks)
