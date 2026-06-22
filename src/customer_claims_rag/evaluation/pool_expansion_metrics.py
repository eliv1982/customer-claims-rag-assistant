"""Metrics and acceptance criteria for pool expansion evaluation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from customer_claims_rag.evaluation.ab_metrics import (
    build_case_arm_result,
    case_result_to_arm_metrics,
    chunks_from_search_results,
    compute_evaluation_dataset_fingerprint,
)
from customer_claims_rag.evaluation.metrics import (
    aggregate_case_metrics,
    aggregate_risk_metrics,
)
from customer_claims_rag.evaluation.models import CaseResult
from customer_claims_rag.evaluation.pool_expansion_models import (
    AcceptanceCriteriaResult,
    CaseComparisonDetail,
    FailureClassification,
    FinalRankingFlags,
    PoolExpansionCaseResult,
    PoolExpansionExperimentConfig,
    PoolReachability,
    RankingAggregateComparison,
    RankingArmResult,
    RankingComparison,
    RankingVerdict,
    ReachabilityComparison,
    ReachabilitySliceComparison,
    ReachabilityVerdict,
    RiskRankingComparison,
)
from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.retrieval.reranker import RerankedCandidate, SourceAuthorityV1Reranker


FROZEN_RERANKER_CONFIG_HASH = (
    "c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957"
)


def load_pool_expansion_config(path: Path) -> PoolExpansionExperimentConfig:
    """Load pool expansion experiment config from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PoolExpansionExperimentConfig.model_validate(
        {**payload, "config_path": str(path.resolve())},
    )


def compute_pool_expansion_config_hash(config: PoolExpansionExperimentConfig) -> str:
    """Return SHA-256 hash of canonical experiment config payload."""
    payload = {
        "experiment_id": config.experiment_id,
        "version": config.version,
        "experiment_mode": config.experiment_mode,
        "baseline_pool_k": config.baseline_pool_k,
        "candidate_pool_k": config.candidate_pool_k,
        "final_top_k": config.final_top_k,
        "threshold": config.threshold,
        "reranker_id": config.reranker_id,
        "reranker_config_hash": config.reranker_config_hash,
        "tie_breaking": config.tie_breaking,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_pool(deep_results: list[SearchResult], pool_k: int) -> list[SearchResult]:
    """Return a ranked pool prefix with baseline ranks assigned."""
    pool = deepcopy(deep_results[:pool_k])
    for index, item in enumerate(pool, start=1):
        item.rank = index
    return pool


def rerank_to_final_top_k(
    reranker: SourceAuthorityV1Reranker,
    query: str,
    pool: list[SearchResult],
    *,
    final_top_k: int,
) -> tuple[list[SearchResult], list[RerankedCandidate]]:
    """Rerank a pool and return final top-k search results plus audit records."""
    reranked = reranker.rerank(query, pool)
    final = [item.result for item in reranked[:final_top_k]]
    return final, reranked[:final_top_k]


def compute_pool_reachability(
    pool: list[SearchResult],
    *,
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
    fallback_expected: bool,
) -> PoolReachability:
    """Compute reachability for one pre-rerank pool prefix."""
    if fallback_expected:
        return PoolReachability()

    primary_rank = _best_document_pool_rank(pool, expected_primary_documents)
    supporting_rank = _best_document_pool_rank(pool, expected_supporting_documents)
    primary_reachable = primary_rank is not None if expected_primary_documents else False
    supporting_reachable = supporting_rank is not None if expected_supporting_documents else False
    has_expected = bool(expected_primary_documents or expected_supporting_documents)
    any_reachable = primary_reachable or supporting_reachable
    fully_unreachable = has_expected and not any_reachable

    return PoolReachability(
        primary_reachable=primary_reachable,
        supporting_reachable=supporting_reachable,
        any_expected_source_reachable=any_reachable,
        primary_best_pool_rank=primary_rank,
        supporting_best_pool_rank=supporting_rank,
        fully_unreachable=fully_unreachable,
    )


def compute_final_ranking_flags(
    *,
    pool_reachability: PoolReachability,
    final_results: list[SearchResult],
    case_metrics: CaseResult,
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
) -> FinalRankingFlags:
    """Compute post-rerank outcome flags."""
    primary_docs = set(expected_primary_documents)
    supporting_docs = set(expected_supporting_documents)
    final_doc_ids = {item.document_id for item in final_results}
    primary_in_top12 = bool(primary_docs & final_doc_ids)
    supporting_in_top12 = bool(supporting_docs & final_doc_ids)
    pool_any_reachable = pool_reachability.any_expected_source_reachable

    return FinalRankingFlags(
        primary_in_final_top12=primary_in_top12,
        supporting_in_final_top12=supporting_in_top12,
        pool_reachable_but_not_in_final_top12=pool_any_reachable and not primary_in_top12,
        final_top12_but_not_top4=primary_in_top12 and not case_metrics.primary_hit_at_4,
    )


def classify_failure(
    *,
    fallback_expected: bool,
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
    baseline_pool: PoolReachability,
    candidate_pool: PoolReachability,
    baseline_flags: FinalRankingFlags,
    candidate_flags: FinalRankingFlags,
    baseline_metrics: CaseResult,
    candidate_metrics: CaseResult,
) -> FailureClassification:
    """Derive diagnostic failure classification without hardcoded case IDs."""
    if fallback_expected:
        return "fallback_case"

    has_expected = bool(expected_primary_documents or expected_supporting_documents)
    if not has_expected:
        return "none"

    if baseline_pool.any_expected_source_reachable and baseline_metrics.primary_hit_at_4:
        if candidate_metrics.primary_hit_at_4 or not candidate_pool.fully_unreachable:
            return "already_reachable_at_baseline"

    if candidate_pool.fully_unreachable:
        if baseline_pool.fully_unreachable:
            return "reachable_only_beyond_selected_pool"
        return "fully_unreachable_in_candidate_pool"

    if (
        not baseline_pool.primary_reachable
        and candidate_pool.primary_reachable
        and candidate_flags.primary_in_final_top12
        and not candidate_metrics.primary_hit_at_4
    ):
        return "candidate_generation_fixed_ranking_limited"

    if (
        not baseline_pool.primary_reachable
        and candidate_pool.primary_reachable
        and not candidate_flags.primary_in_final_top12
    ):
        return "candidate_generation_fixed_ranking_limited"

    if baseline_pool.fully_unreachable and not candidate_pool.fully_unreachable:
        return "candidate_generation_fixed_ranking_limited"

    return "none"


def build_ranking_arm_result(
    *,
    pool: list[SearchResult],
    final_results: list[SearchResult],
    reranked: list[RerankedCandidate],
    case: CaseResult,
    pool_reachability: PoolReachability,
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
) -> RankingArmResult:
    """Build one final ranking arm result."""
    arm = build_case_arm_result(results=pool, reranked=reranked, case=case)
    flags = compute_final_ranking_flags(
        pool_reachability=pool_reachability,
        final_results=final_results,
        case_metrics=case,
        expected_primary_documents=expected_primary_documents,
        expected_supporting_documents=expected_supporting_documents,
    )
    return RankingArmResult(
        ordered_candidates=arm.ordered_candidates,
        metrics=arm.metrics,
        flags=flags,
    )


def build_case_comparison_detail(
    *,
    baseline_metrics: CaseResult,
    candidate_metrics: CaseResult,
    baseline_pool_ids: list[str],
    candidate_pool_ids: list[str],
    baseline_ranking_ids: list[str],
    candidate_ranking_ids: list[str],
) -> CaseComparisonDetail:
    """Compare pool and ranking arms for one case."""
    pool_promotions, pool_regressions = _diff_id_sets(baseline_pool_ids, candidate_pool_ids)
    ranking_promotions, ranking_regressions = _diff_id_sets(
        baseline_ranking_ids,
        candidate_ranking_ids,
    )
    return CaseComparisonDetail(
        pool_promotions=pool_promotions,
        pool_regressions=pool_regressions,
        ranking_promotions=ranking_promotions,
        ranking_regressions=ranking_regressions,
        primary_hit_at_4_delta=int(candidate_metrics.primary_hit_at_4)
        - int(baseline_metrics.primary_hit_at_4),
        supporting_hit_at_4_delta=int(candidate_metrics.supporting_hit_at_4)
        - int(baseline_metrics.supporting_hit_at_4),
        mrr_delta=candidate_metrics.reciprocal_rank - baseline_metrics.reciprocal_rank,
        baseline_pool_is_prefix_of_candidate_pool=candidate_pool_ids[: len(baseline_pool_ids)]
        == baseline_pool_ids,
    )


def build_reachability_comparison(
    case_results: list[PoolExpansionCaseResult],
) -> ReachabilityComparison:
    """Aggregate reachability metrics across cases."""
    baseline_primary = 0
    candidate_primary = 0
    primary_total = 0
    baseline_supporting = 0
    candidate_supporting = 0
    supporting_total = 0
    baseline_unreachable: list[str] = []
    candidate_unreachable: list[str] = []
    high_baseline = 0
    high_candidate = 0
    high_total = 0
    critical_baseline = 0
    critical_candidate = 0
    critical_total = 0
    risk_map: dict[str, ReachabilitySliceComparison] = {}

    for case in case_results:
        if case.fallback_expected:
            continue
        if case.expected_primary_documents:
            primary_total += 1
            if case.baseline_pool.reachability.primary_reachable:
                baseline_primary += 1
            if case.candidate_pool.reachability.primary_reachable:
                candidate_primary += 1
        if case.expected_supporting_documents:
            supporting_total += 1
            if case.baseline_pool.reachability.supporting_reachable:
                baseline_supporting += 1
            if case.candidate_pool.reachability.supporting_reachable:
                candidate_supporting += 1
        if case.baseline_pool.reachability.fully_unreachable:
            baseline_unreachable.append(case.case_id)
        if case.candidate_pool.reachability.fully_unreachable:
            candidate_unreachable.append(case.case_id)

        if case.expected_primary_documents and case.risk == "high":
            high_total += 1
            if case.baseline_pool.reachability.primary_reachable:
                high_baseline += 1
            if case.candidate_pool.reachability.primary_reachable:
                high_candidate += 1
        if case.expected_primary_documents and case.risk == "critical":
            critical_total += 1
            if case.baseline_pool.reachability.primary_reachable:
                critical_baseline += 1
            if case.candidate_pool.reachability.primary_reachable:
                critical_candidate += 1

        if case.expected_primary_documents:
            slice_item = risk_map.setdefault(
                case.risk,
                ReachabilitySliceComparison(
                    risk_level=case.risk,
                    case_count=0,
                    baseline_primary_reachable=0,
                    candidate_primary_reachable=0,
                    baseline_primary_total=0,
                    candidate_primary_total=0,
                ),
            )
            slice_item.case_count += 1
            slice_item.baseline_primary_total += 1
            slice_item.candidate_primary_total += 1
            if case.baseline_pool.reachability.primary_reachable:
                slice_item.baseline_primary_reachable += 1
            if case.candidate_pool.reachability.primary_reachable:
                slice_item.candidate_primary_reachable += 1

    return ReachabilityComparison(
        baseline_primary_reachable=baseline_primary,
        baseline_primary_total=primary_total,
        candidate_primary_reachable=candidate_primary,
        candidate_primary_total=primary_total,
        baseline_supporting_reachable=baseline_supporting,
        baseline_supporting_total=supporting_total,
        candidate_supporting_reachable=candidate_supporting,
        candidate_supporting_total=supporting_total,
        baseline_fully_unreachable_cases=sorted(baseline_unreachable),
        candidate_fully_unreachable_cases=sorted(candidate_unreachable),
        high_baseline_primary_reachable=high_baseline,
        high_candidate_primary_reachable=high_candidate,
        high_primary_total=high_total,
        critical_baseline_primary_reachable=critical_baseline,
        critical_candidate_primary_reachable=critical_candidate,
        critical_primary_total=critical_total,
        risk_slices=sorted(risk_map.values(), key=lambda item: item.risk_level),
    )


def build_ranking_comparison(
    *,
    baseline_cases: list[CaseResult],
    candidate_cases: list[CaseResult],
    case_results: list[PoolExpansionCaseResult],
) -> RankingComparison:
    """Build aggregate ranking comparison."""
    baseline_agg = aggregate_case_metrics(baseline_cases)
    candidate_agg = aggregate_case_metrics(candidate_cases)
    aggregate = RankingAggregateComparison(
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

    baseline_risk = {item.risk_level: item for item in aggregate_risk_metrics(baseline_cases)}
    candidate_risk = {item.risk_level: item for item in aggregate_risk_metrics(candidate_cases)}
    risk_slices: list[RiskRankingComparison] = []
    for risk_level in sorted(set(baseline_risk) | set(candidate_risk)):
        baseline = baseline_risk[risk_level]
        candidate = candidate_risk[risk_level]
        baseline_primary = _primary_hit_rate_for_risk(baseline_cases, risk_level)
        candidate_primary = _primary_hit_rate_for_risk(candidate_cases, risk_level)
        risk_slices.append(
            RiskRankingComparison(
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

    promotions = [
        case.case_id
        for case in case_results
        if case.comparison.primary_hit_at_4_delta > 0 and not case.fallback_expected
    ]
    regressions = [
        case.case_id
        for case in case_results
        if case.comparison.primary_hit_at_4_delta < 0 and not case.fallback_expected
    ]

    return RankingComparison(
        aggregate=aggregate,
        risk_slices=risk_slices,
        primary_hit_at_4_promotions=sorted(promotions),
        primary_hit_at_4_regressions=sorted(regressions),
    )


def evaluate_acceptance_criteria(
    *,
    config: PoolExpansionExperimentConfig,
    reranker_config_hash: str,
    reachability: ReachabilityComparison,
    ranking: RankingComparison,
    case_results: list[PoolExpansionCaseResult],
    shared_prefix_pass: bool,
    single_retrieval_pass: bool,
    technical_errors_zero: bool,
) -> AcceptanceCriteriaResult:
    """Evaluate hard invariants and dual verdicts."""
    failure_reasons: list[str] = []
    hard_invariants_pass = True

    if config.threshold != 0.0:
        hard_invariants_pass = False
        failure_reasons.append("threshold != 0.0")
    if config.final_top_k != 12:
        hard_invariants_pass = False
        failure_reasons.append("final_top_k != 12")
    if not shared_prefix_pass:
        hard_invariants_pass = False
        failure_reasons.append("baseline pool is not prefix of candidate pool")
    if not single_retrieval_pass:
        hard_invariants_pass = False
        failure_reasons.append("multiple retrieval calls per case")
    if not technical_errors_zero:
        hard_invariants_pass = False
        failure_reasons.append("technical_errors > 0")

    reranker_hash_match = reranker_config_hash == FROZEN_RERANKER_CONFIG_HASH
    if not reranker_hash_match:
        hard_invariants_pass = False
        failure_reasons.append("reranker config hash mismatch")

    critical_improved = (
        reachability.critical_candidate_primary_reachable
        > reachability.critical_baseline_primary_reachable
    )
    high_non_regressed = (
        reachability.high_candidate_primary_reachable
        >= reachability.high_baseline_primary_reachable
    )
    unreachable_decreased = len(reachability.candidate_fully_unreachable_cases) < len(
        reachability.baseline_fully_unreachable_cases
    )
    reachability_regressions = _reachability_regressions(case_results)
    reachability_pass = (
        critical_improved
        and high_non_regressed
        and unreachable_decreased
        and reachability_regressions == 0
        and technical_errors_zero
    )
    if not critical_improved:
        failure_reasons.append("critical primary reachability did not improve")
    if not high_non_regressed:
        failure_reasons.append("high primary reachability regressed")
    if not unreachable_decreased:
        failure_reasons.append("fully unreachable count did not decrease")
    if reachability_regressions > 0:
        failure_reasons.append("high/critical primary reachability regressions > 0")

    primary_delta = ranking.aggregate.primary_source_hit_rate_at_4_delta
    mrr_delta = ranking.aggregate.mrr_delta
    critical_regressions = len(
        [
            case
            for case in case_results
            if case.risk == "critical"
            and not case.fallback_expected
            and case.comparison.primary_hit_at_4_delta < 0
        ]
    )
    primary_regressions = len(ranking.primary_hit_at_4_regressions)

    if (
        primary_delta > 0
        and critical_regressions == 0
        and mrr_delta >= -0.01
        and hard_invariants_pass
    ):
        ranking_verdict: RankingVerdict = "improved"
        ranking_pass = True
    elif primary_delta == 0 and primary_regressions == 0 and mrr_delta >= -0.01:
        ranking_verdict = "neutral"
        ranking_pass = True
    else:
        ranking_verdict = "rejected"
        ranking_pass = False
        if primary_delta < 0:
            failure_reasons.append("overall primary_hit@4 decreased")
        if critical_regressions > 0:
            failure_reasons.append("critical primary_hit@4 regressions > 0")
        if primary_regressions > 0:
            failure_reasons.append("primary_hit@4 regressions > 0")
        if mrr_delta < -0.01:
            failure_reasons.append("MRR delta < -0.01")

    return AcceptanceCriteriaResult(
        hard_invariants_pass=hard_invariants_pass,
        shared_prefix_pass=shared_prefix_pass,
        single_retrieval_pass=single_retrieval_pass,
        technical_errors_zero=technical_errors_zero,
        reranker_config_hash_match=reranker_hash_match,
        reachability_verdict="accepted" if reachability_pass else "rejected",
        reachability_pass=reachability_pass,
        critical_primary_reachability_improved=critical_improved,
        high_primary_reachability_non_regressed=high_non_regressed,
        fully_unreachable_decreased=unreachable_decreased,
        high_critical_reachability_regressions=reachability_regressions,
        ranking_verdict=ranking_verdict,
        ranking_pass=ranking_pass,
        overall_primary_hit_at_4_delta=primary_delta,
        overall_mrr_delta=mrr_delta,
        critical_primary_regressions=critical_regressions,
        failure_reasons=failure_reasons,
    )


def _best_document_pool_rank(
    pool: list[SearchResult],
    expected_documents: list[str],
) -> int | None:
    if not expected_documents:
        return None
    expected = set(expected_documents)
    best: int | None = None
    for item in pool:
        if item.document_id in expected:
            best = item.rank if best is None else min(best, item.rank)
    return best


def _diff_id_sets(baseline_ids: list[str], candidate_ids: list[str]) -> tuple[list[str], list[str]]:
    baseline_set = set(baseline_ids)
    candidate_set = set(candidate_ids)
    promotions = sorted(candidate_set - baseline_set)
    regressions = sorted(baseline_set - candidate_set)
    return promotions, regressions


def _supporting_delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


def _primary_hit_rate_for_risk(cases: list[CaseResult], risk_level: str) -> float:
    filtered = [
        case
        for case in cases
        if case.status == "success"
        and not case.fallback_expected
        and case.expected_risk == risk_level
        and case.expected_primary_documents
    ]
    if not filtered:
        return 0.0
    hits = sum(1 for case in filtered if case.primary_hit_at_4)
    return hits / len(filtered)


def _reachability_regressions(case_results: list[PoolExpansionCaseResult]) -> int:
    regressions = 0
    for case in case_results:
        if case.fallback_expected or not case.expected_primary_documents:
            continue
        if case.risk not in {"high", "critical"}:
            continue
        if (
            case.baseline_pool.reachability.primary_reachable
            and not case.candidate_pool.reachability.primary_reachable
        ):
            regressions += 1
    return regressions


__all__ = [
    "FROZEN_RERANKER_CONFIG_HASH",
    "build_case_comparison_detail",
    "build_ranking_arm_result",
    "build_ranking_comparison",
    "build_reachability_comparison",
    "classify_failure",
    "compute_evaluation_dataset_fingerprint",
    "compute_final_ranking_flags",
    "compute_pool_expansion_config_hash",
    "compute_pool_reachability",
    "evaluate_acceptance_criteria",
    "load_pool_expansion_config",
    "prepare_pool",
    "rerank_to_final_top_k",
]
