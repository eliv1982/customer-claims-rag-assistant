"""Metrics and acceptance criteria for hybrid lexical + vector evaluation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from customer_claims_rag.evaluation.ab_metrics import (
    case_result_to_arm_metrics,
    compute_evaluation_dataset_fingerprint,
)
from customer_claims_rag.evaluation.metrics import aggregate_case_metrics, aggregate_risk_metrics
from customer_claims_rag.evaluation.models import CaseResult
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    compute_final_ranking_flags,
    compute_pool_reachability,
)
from customer_claims_rag.evaluation.hybrid_models import (
    AcceptanceCriteriaResult,
    BaselineCandidateAudit,
    ChannelReachabilityComparison,
    FusionPoolSnapshot,
    HybridCandidateAudit,
    HybridCaseResult,
    HybridEvaluationRun,
    HybridExperimentConfig,
    HybridRankingArmResult,
    LexicalPoolCandidate,
    LexicalPoolSnapshot,
    PrimaryRetrievalChannel,
    RankingVerdict,
    ReachabilityVerdict,
    VectorPoolSnapshot,
)
from customer_claims_rag.evaluation.pool_expansion_models import (
    CaseComparisonDetail,
    RankingAggregateComparison,
    RankingComparison,
    RiskRankingComparison,
)
from customer_claims_rag.retrieval.lexical.bm25 import LexicalHit
from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.retrieval.reranker import RerankedCandidate, SourceAuthorityV1Reranker
from customer_claims_rag.retrieval.rrf_adapter import HybridRerankAudit as AdapterAudit

FROZEN_RERANKER_CONFIG_HASH = (
    "c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957"
)
FROZEN_HYBRID_CONFIG_HASH = (
    "00476d68eef021da58180b8b97c5e5c4a04f74cb1d1b8c8229d2871c70385b3e"
)


def load_hybrid_config(path: Path) -> HybridExperimentConfig:
    """Load hybrid experiment config from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return HybridExperimentConfig.model_validate(
        {**payload, "config_path": str(path.resolve())},
    )


def compute_hybrid_config_hash(config: HybridExperimentConfig) -> str:
    """Return SHA-256 hash of canonical experiment config payload."""
    payload = {
        "experiment_id": config.experiment_id,
        "version": config.version,
        "experiment_mode": config.experiment_mode,
        "vector_k": config.vector_k,
        "lexical_k": config.lexical_k,
        "fusion_k": config.fusion_k,
        "rrf_k": config.rrf_k,
        "vector_weight": config.vector_weight,
        "lexical_weight": config.lexical_weight,
        "final_top_k": config.final_top_k,
        "threshold": config.threshold,
        "lexical_algorithm": config.lexical_algorithm,
        "bm25_k1": config.bm25_k1,
        "bm25_b": config.bm25_b,
        "reranker_id": config.reranker_id,
        "reranker_config_hash": config.reranker_config_hash,
        "score_adapter_id": config.score_adapter_id,
        "tie_breaking": config.tie_breaking,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_vector_pool(deep_results: list[SearchResult], pool_k: int) -> list[SearchResult]:
    """Return a ranked vector pool prefix with ranks assigned."""
    pool = deepcopy(deep_results[:pool_k])
    for index, item in enumerate(pool, start=1):
        item.rank = index
    return pool


def rerank_baseline_to_final_top_k(
    reranker: SourceAuthorityV1Reranker,
    query: str,
    pool: list[SearchResult],
    *,
    final_top_k: int,
) -> tuple[list[SearchResult], list[RerankedCandidate]]:
    """Rerank baseline pool using original vector similarity."""
    reranked = reranker.rerank(query, pool)
    final = [item.result for item in reranked[:final_top_k]]
    return final, reranked[:final_top_k]


def compute_channel_reachability(
    *,
    vector_pool: list[SearchResult],
    fusion_pool: list[SearchResult],
    lexical_pool: list[LexicalHit] | list[str],
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
    fallback_expected: bool,
) -> tuple[PoolReachability, PoolReachability, PrimaryRetrievalChannel | None, bool, bool, bool]:
    """Compute baseline/fusion reachability and primary channel diagnostics."""
    baseline_reach = compute_pool_reachability(
        vector_pool,
        expected_primary_documents=expected_primary_documents,
        expected_supporting_documents=expected_supporting_documents,
        fallback_expected=fallback_expected,
    )
    fusion_reach = compute_pool_reachability(
        fusion_pool,
        expected_primary_documents=expected_primary_documents,
        expected_supporting_documents=expected_supporting_documents,
        fallback_expected=fallback_expected,
    )
    channel = compute_primary_retrieval_channel(
        vector_pool_chunk_ids=[item.chunk_id for item in vector_pool],
        lexical_pool_chunk_ids=_lexical_pool_chunk_ids(lexical_pool),
        expected_primary_documents=expected_primary_documents,
        fallback_expected=fallback_expected,
    )
    vector_only_flag = channel == "vector_only"
    lexical_only_flag = channel == "lexical_only"
    both_flag = channel == "both"
    return baseline_reach, fusion_reach, channel, vector_only_flag, lexical_only_flag, both_flag


def compute_primary_retrieval_channel(
    *,
    vector_pool_chunk_ids: list[str],
    lexical_pool_chunk_ids: list[str],
    expected_primary_documents: list[str],
    fallback_expected: bool,
) -> PrimaryRetrievalChannel | None:
    """Classify how expected primary documents are reachable across retrieval channels."""
    if fallback_expected or not expected_primary_documents:
        return None

    primary_docs = set(expected_primary_documents)
    vector_reachable = _primary_doc_reachable_in_chunk_ids(vector_pool_chunk_ids, primary_docs)
    lexical_reachable = _primary_doc_reachable_in_chunk_ids(lexical_pool_chunk_ids, primary_docs)
    return classify_primary_retrieval_channel(vector_reachable, lexical_reachable)


def classify_primary_retrieval_channel(
    vector_reachable: bool,
    lexical_reachable: bool,
) -> PrimaryRetrievalChannel:
    """Map per-channel reachability flags to a single channel label."""
    if vector_reachable and lexical_reachable:
        return "both"
    if vector_reachable:
        return "vector_only"
    if lexical_reachable:
        return "lexical_only"
    return "neither"


def channel_flags_from_label(
    channel: PrimaryRetrievalChannel | None,
) -> tuple[bool, bool, bool, bool, bool]:
    """Return vector/lexical reachability and legacy boolean flags for one channel label."""
    if channel is None:
        return False, False, False, False, False
    vector_reachable = channel in {"both", "vector_only"}
    lexical_reachable = channel in {"both", "lexical_only"}
    return (
        vector_reachable,
        lexical_reachable,
        channel == "vector_only",
        channel == "lexical_only",
        channel == "both",
    )


def lexical_pool_snapshot_from_hits(
    hits: list[LexicalHit],
    *,
    pool_k: int,
    exact: bool,
) -> LexicalPoolSnapshot:
    """Build a lexical pool snapshot from ordered BM25 hits."""
    candidates = [
        LexicalPoolCandidate(
            lexical_rank=hit.lexical_rank,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            bm25_score=hit.bm25_score,
            matched_tokens=list(hit.matched_tokens),
        )
        for hit in hits
    ]
    return LexicalPoolSnapshot(
        pool_k=pool_k,
        candidate_ids=[candidate.chunk_id for candidate in candidates],
        candidates=candidates,
        exact=exact,
    )


def derive_lexical_pool_candidate_ids(case: HybridCaseResult) -> list[str]:
    """Return lexical pool chunk IDs from an exact self-contained case record."""
    if case.lexical_pool is None:
        raise ValueError(f"{case.case_id}: lexical_pool is missing from artifact")

    if case.lexical_pool.candidates:
        return [candidate.chunk_id for candidate in case.lexical_pool.candidates]
    if case.lexical_pool.candidate_ids:
        return list(case.lexical_pool.candidate_ids)
    return []


def recompute_case_channel_diagnostics(case: HybridCaseResult) -> HybridCaseResult:
    """Recompute derived channel diagnostics for one frozen case."""
    lexical_pool_ids = derive_lexical_pool_candidate_ids(case)
    channel = compute_primary_retrieval_channel(
        vector_pool_chunk_ids=case.baseline_pool.candidate_ids,
        lexical_pool_chunk_ids=lexical_pool_ids,
        expected_primary_documents=case.expected_primary_documents,
        fallback_expected=case.fallback_expected,
    )
    vector_reachable, lexical_reachable, vector_only, lexical_only, both = channel_flags_from_label(
        channel
    )
    updated_fusion = case.candidate_fusion_pool.model_copy(
        update={
            "vector_reachable": vector_reachable,
            "lexical_reachable": lexical_reachable,
            "retrieval_channel": channel,
            "vector_only_reachable": vector_only,
            "lexical_only_reachable": lexical_only,
            "both_reachable": both,
        }
    )
    lexical_pool = case.lexical_pool
    if lexical_pool is None:
        raise ValueError(f"{case.case_id}: lexical_pool required for channel diagnostics")
    return case.model_copy(
        update={
            "candidate_fusion_pool": updated_fusion,
        }
    )


def rebuild_hybrid_diagnostics(run: HybridEvaluationRun) -> HybridEvaluationRun:
    """Recompute channel diagnostics from self-contained JSON without BM25 or retrieval."""
    if not _artifact_has_exact_lexical_pools(run):
        raise ValueError("artifact lacks exact lexical pools; run lexical replay first")
    updated_cases = [recompute_case_channel_diagnostics(case) for case in run.case_results]
    reachability = build_channel_reachability_comparison(updated_cases)
    return run.model_copy(
        update={
            "case_results": updated_cases,
            "candidate_generation_comparison": reachability,
        }
    )


def _artifact_has_exact_lexical_pools(run: HybridEvaluationRun) -> bool:
    return all(
        case.lexical_pool is not None
        and case.lexical_pool.exact
        and case.lexical_pool.candidates
        for case in run.case_results
    )


def _experiment_config_from_run(run: HybridEvaluationRun) -> HybridExperimentConfig:
    payload = {
        **run.experiment.config,
        "experiment_id": run.experiment.experiment_id,
        "version": run.experiment.version,
        "experiment_mode": run.experiment.experiment_mode,
        "reranker_id": run.experiment.reranker_id,
        "reranker_config_hash": run.experiment.reranker_config_hash,
        "score_adapter_id": run.experiment.score_adapter_id,
        "tie_breaking": run.experiment.config.get("tie_breaking", ["rerank_score_desc"]),
    }
    return HybridExperimentConfig.model_validate(payload)


def extract_immutability_snapshot(run: HybridEvaluationRun) -> dict[str, object]:
    """Extract frozen experimental fields for before/after repair comparison."""
    case_snapshots: list[dict[str, object]] = []
    for case in run.case_results:
        candidate_audits = []
        for audit in case.candidate_ranking.ordered_candidates:
            if isinstance(audit, HybridCandidateAudit):
                candidate_audits.append(
                    {
                        "chunk_id": audit.chunk_id,
                        "vector_rank": audit.vector_rank,
                        "vector_similarity": audit.vector_similarity,
                        "vector_distance": audit.vector_distance,
                        "lexical_rank": audit.lexical_rank,
                        "bm25_score": audit.bm25_score,
                        "raw_rrf_score": audit.raw_rrf_score,
                        "normalized_rrf_score": audit.normalized_rrf_score,
                        "source_authority_bonus": audit.source_authority_bonus,
                        "final_rerank_score": audit.final_rerank_score,
                        "final_rank": audit.final_rank,
                    }
                )
            else:
                candidate_audits.append(audit.model_dump(mode="json"))
        case_snapshots.append(
            {
                "case_id": case.case_id,
                "vector_pool_ids": list(case.baseline_pool.candidate_ids),
                "fusion_pool_ids": list(case.candidate_fusion_pool.candidate_ids),
                "baseline_final_ids": [
                    item.chunk_id for item in case.baseline_ranking.ordered_candidates
                ],
                "candidate_final_ids": [
                    item.chunk_id for item in case.candidate_ranking.ordered_candidates
                ],
                "candidate_audits": candidate_audits,
                "comparison": case.comparison.model_dump(mode="json"),
            }
        )
    return {
        "config_hash": run.experiment.config_hash,
        "reranker_config_hash": run.experiment.reranker_config_hash,
        "baseline_aggregate": run.baseline.aggregate_metrics.model_dump(mode="json"),
        "candidate_aggregate": run.candidate.aggregate_metrics.model_dump(mode="json"),
        "ranking_comparison": run.ranking_comparison.model_dump(mode="json"),
        "acceptance": run.acceptance.model_dump(mode="json"),
        "cases": case_snapshots,
    }


def build_baseline_ranking_arm_result(
    *,
    pool: list[SearchResult],
    final_results: list[SearchResult],
    reranked: list[RerankedCandidate],
    case: CaseResult,
    pool_reachability,
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
) -> HybridRankingArmResult:
    """Build baseline ranking arm result with vector audit fields."""
    audits: list[BaselineCandidateAudit] = []
    for item in reranked:
        audits.append(
            BaselineCandidateAudit(
                chunk_id=item.result.chunk_id,
                document_id=item.result.document_id,
                chunk_type=item.result.chunk_type,
                vector_rank=item.baseline_rank,
                vector_similarity=item.result.similarity,
                vector_distance=item.result.distance,
                source_authority_bonus=item.source_authority_bonus,
                final_rerank_score=item.rerank_score,
                final_rank=item.candidate_rank,
            )
        )
    metrics = case_result_to_arm_metrics(case)
    flags = compute_final_ranking_flags(
        pool_reachability=pool_reachability,
        final_results=final_results,
        case_metrics=case,
        expected_primary_documents=expected_primary_documents,
        expected_supporting_documents=expected_supporting_documents,
    )
    return HybridRankingArmResult(
        ordered_candidates=audits[: len(final_results)],
        metrics=metrics,
        flags=flags,
    )


def build_candidate_ranking_arm_result(
    *,
    final_results: list[SearchResult],
    audits: list[AdapterAudit],
    case: CaseResult,
    pool_reachability,
    expected_primary_documents: list[str],
    expected_supporting_documents: list[str],
) -> HybridRankingArmResult:
    """Build candidate ranking arm result with hybrid audit fields."""
    ordered: list[HybridCandidateAudit] = []
    for audit in audits:
        chunk_type = next(
            (item.chunk_type for item in final_results if item.chunk_id == audit.chunk_id),
            "unknown",
        )
        ordered.append(
            HybridCandidateAudit(
                chunk_id=audit.chunk_id,
                document_id=audit.document_id,
                chunk_type=chunk_type,
                vector_rank=audit.vector_rank,
                vector_similarity=audit.vector_similarity,
                vector_distance=audit.vector_distance,
                lexical_rank=audit.lexical_rank,
                bm25_score=audit.bm25_score,
                matched_tokens=list(audit.matched_tokens),
                retrieval_channel=audit.retrieval_channel,
                raw_rrf_score=audit.raw_rrf_score,
                normalized_rrf_score=audit.normalized_rrf_score,
                fusion_rank=audit.fusion_rank,
                source_authority_bonus=audit.source_authority_bonus,
                final_rerank_score=audit.final_rerank_score,
                final_rank=audit.final_rank,
            )
        )
    metrics = case_result_to_arm_metrics(case)
    flags = compute_final_ranking_flags(
        pool_reachability=pool_reachability,
        final_results=final_results,
        case_metrics=case,
        expected_primary_documents=expected_primary_documents,
        expected_supporting_documents=expected_supporting_documents,
    )
    return HybridRankingArmResult(
        ordered_candidates=ordered,
        metrics=metrics,
        flags=flags,
    )


def build_case_comparison_detail(
    *,
    baseline_metrics: CaseResult,
    candidate_metrics: CaseResult,
    baseline_ranking_ids: list[str],
    candidate_ranking_ids: list[str],
) -> CaseComparisonDetail:
    """Compare ranking arms for one case."""
    ranking_promotions, ranking_regressions = _diff_id_sets(
        baseline_ranking_ids,
        candidate_ranking_ids,
    )
    return CaseComparisonDetail(
        ranking_promotions=ranking_promotions,
        ranking_regressions=ranking_regressions,
        primary_hit_at_4_delta=int(candidate_metrics.primary_hit_at_4)
        - int(baseline_metrics.primary_hit_at_4),
        supporting_hit_at_4_delta=int(candidate_metrics.supporting_hit_at_4)
        - int(baseline_metrics.supporting_hit_at_4),
        mrr_delta=candidate_metrics.reciprocal_rank - baseline_metrics.reciprocal_rank,
        baseline_pool_is_prefix_of_candidate_pool=False,
    )


def build_channel_reachability_comparison(
    case_results: list[HybridCaseResult],
) -> ChannelReachabilityComparison:
    """Aggregate candidate-generation reachability across cases."""
    baseline_primary = 0
    candidate_primary = 0
    primary_total = 0
    baseline_supporting = 0
    candidate_supporting = 0
    supporting_total = 0
    baseline_unreachable: list[str] = []
    candidate_unreachable: list[str] = []
    vector_only_cases: list[str] = []
    lexical_only_cases: list[str] = []
    both_cases: list[str] = []
    neither_cases: list[str] = []
    high_baseline = 0
    high_candidate = 0
    high_total = 0
    critical_baseline = 0
    critical_candidate = 0
    critical_total = 0

    for case in case_results:
        if case.fallback_expected:
            continue
        if case.expected_primary_documents:
            primary_total += 1
            if case.baseline_pool.reachability.primary_reachable:
                baseline_primary += 1
            if case.candidate_fusion_pool.reachability.primary_reachable:
                candidate_primary += 1
        if case.expected_supporting_documents:
            supporting_total += 1
            if case.baseline_pool.reachability.supporting_reachable:
                baseline_supporting += 1
            if case.candidate_fusion_pool.reachability.supporting_reachable:
                candidate_supporting += 1
        if case.baseline_pool.reachability.fully_unreachable:
            baseline_unreachable.append(case.case_id)
        if case.candidate_fusion_pool.reachability.fully_unreachable:
            candidate_unreachable.append(case.case_id)
        if case.candidate_fusion_pool.retrieval_channel == "vector_only":
            vector_only_cases.append(case.case_id)
        elif case.candidate_fusion_pool.retrieval_channel == "lexical_only":
            lexical_only_cases.append(case.case_id)
        elif case.candidate_fusion_pool.retrieval_channel == "both":
            both_cases.append(case.case_id)
        elif case.candidate_fusion_pool.retrieval_channel == "neither":
            neither_cases.append(case.case_id)
        if case.expected_primary_documents and case.risk == "high":
            high_total += 1
            if case.baseline_pool.reachability.primary_reachable:
                high_baseline += 1
            if case.candidate_fusion_pool.reachability.primary_reachable:
                high_candidate += 1
        if case.expected_primary_documents and case.risk == "critical":
            critical_total += 1
            if case.baseline_pool.reachability.primary_reachable:
                critical_baseline += 1
            if case.candidate_fusion_pool.reachability.primary_reachable:
                critical_candidate += 1

    return ChannelReachabilityComparison(
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
        vector_only_reachable_cases=sorted(vector_only_cases),
        lexical_only_reachable_cases=sorted(lexical_only_cases),
        both_reachable_cases=sorted(both_cases),
        neither_reachable_cases=sorted(neither_cases),
        high_baseline_primary_reachable=high_baseline,
        high_candidate_primary_reachable=high_candidate,
        high_primary_total=high_total,
        critical_baseline_primary_reachable=critical_baseline,
        critical_candidate_primary_reachable=critical_candidate,
        critical_primary_total=critical_total,
    )


def build_ranking_comparison(
    *,
    baseline_cases: list[CaseResult],
    candidate_cases: list[CaseResult],
    case_results: list[HybridCaseResult],
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
    config: HybridExperimentConfig,
    reranker_config_hash: str,
    reachability: ChannelReachabilityComparison,
    ranking: RankingComparison,
    case_results: list[HybridCaseResult],
    single_retrieval_pass: bool,
    technical_errors_zero: bool,
    lexical_index_valid: bool,
) -> AcceptanceCriteriaResult:
    """Evaluate hard invariants and dual verdicts for hybrid experiment."""
    failure_reasons: list[str] = []
    hard_invariants_pass = True

    if config.threshold != 0.0:
        hard_invariants_pass = False
        failure_reasons.append("threshold != 0.0")
    if config.final_top_k != 12:
        hard_invariants_pass = False
        failure_reasons.append("final_top_k != 12")
    if config.vector_k != 24 or config.lexical_k != 24 or config.fusion_k != 24:
        hard_invariants_pass = False
        failure_reasons.append("vector/lexical/fusion k mismatch")
    if config.vector_weight != 1.0 or config.lexical_weight != 1.0:
        hard_invariants_pass = False
        failure_reasons.append("weighted RRF is not allowed in v1")
    if not single_retrieval_pass:
        hard_invariants_pass = False
        failure_reasons.append("multiple vector retrieval calls per case")
    if not technical_errors_zero:
        hard_invariants_pass = False
        failure_reasons.append("technical_errors > 0")
    if not lexical_index_valid:
        hard_invariants_pass = False
        failure_reasons.append("lexical index validation failed")

    reranker_hash_match = reranker_config_hash == FROZEN_RERANKER_CONFIG_HASH
    if not reranker_hash_match:
        hard_invariants_pass = False
        failure_reasons.append("reranker config hash mismatch")

    high_non_regressed = (
        reachability.high_candidate_primary_reachable
        >= reachability.high_baseline_primary_reachable
    )
    critical_non_regressed = (
        reachability.critical_candidate_primary_reachable
        >= reachability.critical_baseline_primary_reachable
    )
    unreachable_non_increased = len(reachability.candidate_fully_unreachable_cases) <= len(
        reachability.baseline_fully_unreachable_cases
    )
    reachability_regressions = _reachability_regressions(case_results)
    reachability_pass = (
        high_non_regressed
        and critical_non_regressed
        and unreachable_non_increased
        and reachability_regressions == 0
        and technical_errors_zero
    )
    if not high_non_regressed:
        failure_reasons.append("high primary reachability regressed")
    if not critical_non_regressed:
        failure_reasons.append("critical primary reachability regressed")
    if not unreachable_non_increased:
        failure_reasons.append("fully unreachable count increased")
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
    low_medium_guardrails = _low_medium_guardrails_pass(ranking, case_results)

    if (
        primary_delta > 0
        and critical_regressions == 0
        and mrr_delta >= -0.01
        and hard_invariants_pass
        and low_medium_guardrails
    ):
        ranking_verdict: RankingVerdict = "improved"
        ranking_pass = True
    elif (
        primary_delta == 0
        and primary_regressions == 0
        and mrr_delta >= -0.01
        and hard_invariants_pass
        and low_medium_guardrails
    ):
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
        if not low_medium_guardrails:
            failure_reasons.append("low/medium slice guardrails failed")

    return AcceptanceCriteriaResult(
        hard_invariants_pass=hard_invariants_pass,
        single_retrieval_pass=single_retrieval_pass,
        technical_errors_zero=technical_errors_zero,
        reranker_config_hash_match=reranker_hash_match,
        lexical_index_valid=lexical_index_valid,
        reachability_verdict="accepted" if reachability_pass else "rejected",
        reachability_pass=reachability_pass,
        high_primary_reachability_non_regressed=high_non_regressed,
        critical_primary_reachability_non_regressed=critical_non_regressed,
        fully_unreachable_non_increased=unreachable_non_increased,
        high_critical_reachability_regressions=reachability_regressions,
        ranking_verdict=ranking_verdict,
        ranking_pass=ranking_pass,
        overall_primary_hit_at_4_delta=primary_delta,
        overall_mrr_delta=mrr_delta,
        critical_primary_regressions=critical_regressions,
        low_medium_slice_guardrails_pass=low_medium_guardrails,
        failure_reasons=failure_reasons,
    )


def _lexical_pool_chunk_ids(lexical_pool: list[LexicalHit] | list[str]) -> list[str]:
    if not lexical_pool:
        return []
    if isinstance(lexical_pool[0], str):
        return list(lexical_pool)
    return [item.chunk_id for item in lexical_pool]


def _chunk_id_to_document_id(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def _primary_doc_reachable_in_chunk_ids(
    chunk_ids: list[str],
    primary_docs: set[str],
) -> bool:
    """Return True when any expected primary document appears in the pool."""
    for chunk_id in chunk_ids:
        if _chunk_id_to_document_id(chunk_id) in primary_docs:
            return True
    return False


def _diff_id_sets(baseline_ids: list[str], candidate_ids: list[str]) -> tuple[list[str], list[str]]:
    baseline_set = set(baseline_ids)
    candidate_set = set(candidate_ids)
    return sorted(candidate_set - baseline_set), sorted(baseline_set - candidate_set)


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
    return sum(1 for case in filtered if case.primary_hit_at_4) / len(filtered)


def _reachability_regressions(case_results: list[HybridCaseResult]) -> int:
    regressions = 0
    for case in case_results:
        if case.fallback_expected or not case.expected_primary_documents:
            continue
        if case.risk not in {"high", "critical"}:
            continue
        if (
            case.baseline_pool.reachability.primary_reachable
            and not case.candidate_fusion_pool.reachability.primary_reachable
        ):
            regressions += 1
    return regressions


def _low_medium_guardrails_pass(
    ranking: RankingComparison,
    case_results: list[HybridCaseResult],
) -> bool:
    for risk_level in ("low", "medium"):
        regressions = [
            case
            for case in case_results
            if case.risk == risk_level
            and not case.fallback_expected
            and case.comparison.primary_hit_at_4_delta < 0
        ]
        if len(regressions) > 1:
            return False
        slice_item = next(
            (item for item in ranking.risk_slices if item.risk_level == risk_level),
            None,
        )
        if slice_item is not None and slice_item.mrr_delta < -0.02:
            return False
    return True


__all__ = [
    "FROZEN_HYBRID_CONFIG_HASH",
    "FROZEN_RERANKER_CONFIG_HASH",
    "build_baseline_ranking_arm_result",
    "build_candidate_ranking_arm_result",
    "build_case_comparison_detail",
    "build_channel_reachability_comparison",
    "build_ranking_comparison",
    "channel_flags_from_label",
    "classify_primary_retrieval_channel",
    "compute_channel_reachability",
    "compute_evaluation_dataset_fingerprint",
    "compute_hybrid_config_hash",
    "compute_primary_retrieval_channel",
    "derive_lexical_pool_candidate_ids",
    "evaluate_acceptance_criteria",
    "extract_immutability_snapshot",
    "lexical_pool_snapshot_from_hits",
    "load_hybrid_config",
    "prepare_vector_pool",
    "rebuild_hybrid_diagnostics",
    "recompute_case_channel_diagnostics",
    "rerank_baseline_to_final_top_k",
]
