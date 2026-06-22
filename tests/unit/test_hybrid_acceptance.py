"""Unit tests for hybrid acceptance criteria."""

from __future__ import annotations

import pytest

from customer_claims_rag.evaluation.ab_models import ArmMetrics
from customer_claims_rag.evaluation.hybrid_metrics import (
    FROZEN_RERANKER_CONFIG_HASH,
    evaluate_acceptance_criteria,
)
from customer_claims_rag.evaluation.hybrid_models import (
    CaseComparisonDetail,
    ChannelReachabilityComparison,
    FusionPoolSnapshot,
    HybridCaseResult,
    HybridExperimentConfig,
    HybridRankingArmResult,
    VectorPoolSnapshot,
)
from customer_claims_rag.evaluation.pool_expansion_models import PoolReachability, RankingAggregateComparison, RankingComparison


DEFAULT_CONFIG = HybridExperimentConfig(
    experiment_id="hybrid-lexical-vector-v1",
    version="1.0.0",
    experiment_mode="production-like",
    vector_k=24,
    lexical_k=24,
    fusion_k=24,
    rrf_k=60,
    vector_weight=1.0,
    lexical_weight=1.0,
    final_top_k=12,
    threshold=0.0,
    lexical_algorithm="BM25",
    bm25_k1=1.5,
    bm25_b=0.75,
    reranker_id="source-authority-v1",
    reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
    score_adapter_id="rrf-base-score-adapter-v1",
    tie_breaking=["rerank_score_desc"],
)


def _reachability(**kwargs) -> ChannelReachabilityComparison:
    defaults = dict(
        baseline_primary_reachable=56,
        baseline_primary_total=57,
        candidate_primary_reachable=57,
        candidate_primary_total=57,
        baseline_supporting_reachable=35,
        baseline_supporting_total=38,
        candidate_supporting_reachable=36,
        candidate_supporting_total=38,
        baseline_fully_unreachable_cases=["T004"],
        candidate_fully_unreachable_cases=[],
        high_baseline_primary_reachable=15,
        high_candidate_primary_reachable=15,
        high_primary_total=15,
        critical_baseline_primary_reachable=8,
        critical_candidate_primary_reachable=8,
        critical_primary_total=8,
    )
    defaults.update(kwargs)
    return ChannelReachabilityComparison(**defaults)


def _ranking(*, primary_delta: float = 0.01, mrr_delta: float = 0.0, regressions: list[str] | None = None) -> RankingComparison:
    return RankingComparison(
        aggregate=RankingAggregateComparison(
            primary_source_hit_rate_at_4_delta=primary_delta,
            mrr_delta=mrr_delta,
        ),
        primary_hit_at_4_regressions=regressions or [],
        risk_slices=[],
    )


def _case(case_id: str = "T001", risk: str = "low", primary_delta: int = 0) -> HybridCaseResult:
    reach = PoolReachability(primary_reachable=True, any_expected_source_reachable=True)
    return HybridCaseResult(
        case_id=case_id,
        risk=risk,
        category="general",
        query="q",
        expected_primary_documents=["doc"],
        baseline_pool=VectorPoolSnapshot(pool_k=24, reachability=reach),
        candidate_fusion_pool=FusionPoolSnapshot(pool_k=24, reachability=reach),
        baseline_ranking=HybridRankingArmResult(metrics=ArmMetrics()),
        candidate_ranking=HybridRankingArmResult(metrics=ArmMetrics()),
        comparison=CaseComparisonDetail(primary_hit_at_4_delta=primary_delta),
    )


def test_improved_verdict() -> None:
    result = evaluate_acceptance_criteria(
        config=DEFAULT_CONFIG,
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        reachability=_reachability(),
        ranking=_ranking(primary_delta=0.02),
        case_results=[_case()],
        single_retrieval_pass=True,
        technical_errors_zero=True,
        lexical_index_valid=True,
    )
    assert result.ranking_verdict == "improved"


def test_neutral_verdict_mrr_boundary() -> None:
    result = evaluate_acceptance_criteria(
        config=DEFAULT_CONFIG,
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        reachability=_reachability(),
        ranking=_ranking(primary_delta=0.0, mrr_delta=-0.01),
        case_results=[_case()],
        single_retrieval_pass=True,
        technical_errors_zero=True,
        lexical_index_valid=True,
    )
    assert result.ranking_verdict == "neutral"


def test_rejected_on_critical_regression() -> None:
    result = evaluate_acceptance_criteria(
        config=DEFAULT_CONFIG,
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        reachability=_reachability(),
        ranking=_ranking(primary_delta=0.0, regressions=["T040"]),
        case_results=[_case(case_id="T040", risk="critical", primary_delta=-1)],
        single_retrieval_pass=True,
        technical_errors_zero=True,
        lexical_index_valid=True,
    )
    assert result.ranking_verdict == "rejected"


def test_hard_invariant_weighted_rrf_rejected() -> None:
    config = DEFAULT_CONFIG.model_copy(update={"lexical_weight": 2.0})
    result = evaluate_acceptance_criteria(
        config=config,
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        reachability=_reachability(),
        ranking=_ranking(),
        case_results=[_case()],
        single_retrieval_pass=True,
        technical_errors_zero=True,
        lexical_index_valid=True,
    )
    assert result.hard_invariants_pass is False


def test_unreachable_increase_rejects_candidate_generation() -> None:
    result = evaluate_acceptance_criteria(
        config=DEFAULT_CONFIG,
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        reachability=_reachability(
            candidate_fully_unreachable_cases=["T004", "T099"],
            baseline_fully_unreachable_cases=["T004"],
        ),
        ranking=_ranking(primary_delta=0.0),
        case_results=[_case()],
        single_retrieval_pass=True,
        technical_errors_zero=True,
        lexical_index_valid=True,
    )
    assert result.reachability_verdict == "rejected"
