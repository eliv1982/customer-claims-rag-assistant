"""Direct unit tests for pool expansion acceptance criteria."""

from __future__ import annotations

import pytest

from customer_claims_rag.evaluation.ab_models import ArmMetrics
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    FROZEN_RERANKER_CONFIG_HASH,
    evaluate_acceptance_criteria,
)
from customer_claims_rag.evaluation.pool_expansion_models import (
    CaseComparisonDetail,
    PoolArmSnapshot,
    PoolExpansionCaseResult,
    PoolExpansionExperimentConfig,
    PoolReachability,
    RankingAggregateComparison,
    RankingArmResult,
    RankingComparison,
    ReachabilityComparison,
)

DEFAULT_CONFIG = PoolExpansionExperimentConfig(
    experiment_id="vector-pool-expansion-v1",
    version="1.0.0",
    experiment_mode="production-like",
    baseline_pool_k=12,
    candidate_pool_k=24,
    final_top_k=12,
    threshold=0.0,
    reranker_id="source-authority-v1",
    reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
    tie_breaking=["rerank_score_desc"],
)


def _reachability(
    *,
    critical_baseline: int = 6,
    critical_candidate: int = 8,
    high_baseline: int = 13,
    high_candidate: int = 15,
    baseline_unreachable: list[str] | None = None,
    candidate_unreachable: list[str] | None = None,
) -> ReachabilityComparison:
    return ReachabilityComparison(
        baseline_primary_reachable=51,
        baseline_primary_total=57,
        candidate_primary_reachable=56,
        candidate_primary_total=57,
        baseline_supporting_reachable=29,
        baseline_supporting_total=38,
        candidate_supporting_reachable=35,
        candidate_supporting_total=38,
        baseline_fully_unreachable_cases=baseline_unreachable or ["T004", "T040", "T047"],
        candidate_fully_unreachable_cases=candidate_unreachable or ["T004"],
        high_baseline_primary_reachable=high_baseline,
        high_candidate_primary_reachable=high_candidate,
        high_primary_total=15,
        critical_baseline_primary_reachable=critical_baseline,
        critical_candidate_primary_reachable=critical_candidate,
        critical_primary_total=8,
    )


def _ranking(
    *,
    primary_delta: float = 0.0,
    mrr_delta: float = 0.009,
    promotions: list[str] | None = None,
    regressions: list[str] | None = None,
) -> RankingComparison:
    return RankingComparison(
        aggregate=RankingAggregateComparison(
            primary_source_hit_rate_at_4_delta=primary_delta,
            mrr_delta=mrr_delta,
        ),
        primary_hit_at_4_promotions=promotions or [],
        primary_hit_at_4_regressions=regressions or [],
    )


def _case_result(
    *,
    case_id: str = "T001",
    risk: str = "high",
    baseline_primary_reachable: bool = True,
    candidate_primary_reachable: bool = True,
    primary_hit_at_4_delta: int = 0,
) -> PoolExpansionCaseResult:
    baseline_reach = PoolReachability(
        primary_reachable=baseline_primary_reachable,
        any_expected_source_reachable=baseline_primary_reachable,
        fully_unreachable=not baseline_primary_reachable,
    )
    candidate_reach = PoolReachability(
        primary_reachable=candidate_primary_reachable,
        any_expected_source_reachable=candidate_primary_reachable,
        fully_unreachable=not candidate_primary_reachable,
    )
    return PoolExpansionCaseResult(
        case_id=case_id,
        risk=risk,
        category="quality",
        query="query",
        expected_primary_documents=["doc_a"],
        baseline_pool=PoolArmSnapshot(pool_k=12, reachability=baseline_reach),
        candidate_pool=PoolArmSnapshot(pool_k=24, reachability=candidate_reach),
        baseline_ranking=RankingArmResult(metrics=ArmMetrics()),
        candidate_ranking=RankingArmResult(metrics=ArmMetrics()),
        comparison=CaseComparisonDetail(primary_hit_at_4_delta=primary_hit_at_4_delta),
    )


def _evaluate(
    *,
    reachability: ReachabilityComparison | None = None,
    ranking: RankingComparison | None = None,
    case_results: list[PoolExpansionCaseResult] | None = None,
    single_retrieval_pass: bool = True,
    technical_errors_zero: bool = True,
) -> object:
    return evaluate_acceptance_criteria(
        config=DEFAULT_CONFIG,
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        reachability=reachability or _reachability(),
        ranking=ranking or _ranking(),
        case_results=case_results or [],
        shared_prefix_pass=True,
        single_retrieval_pass=single_retrieval_pass,
        technical_errors_zero=technical_errors_zero,
    )


def test_reachability_accepted() -> None:
    result = _evaluate()
    assert result.reachability_verdict == "accepted"
    assert result.reachability_pass is True
    assert result.critical_primary_reachability_improved is True
    assert result.high_primary_reachability_non_regressed is True
    assert result.fully_unreachable_decreased is True
    assert result.high_critical_reachability_regressions == 0


def test_reachability_rejected_when_critical_unchanged() -> None:
    result = _evaluate(reachability=_reachability(critical_baseline=8, critical_candidate=8))
    assert result.reachability_verdict == "rejected"
    assert result.reachability_pass is False


def test_reachability_rejected_when_high_regresses() -> None:
    result = _evaluate(reachability=_reachability(high_baseline=15, high_candidate=14))
    assert result.reachability_verdict == "rejected"
    assert result.reachability_pass is False


def test_reachability_rejected_when_unreachable_count_unchanged() -> None:
    result = _evaluate(
        reachability=_reachability(
            baseline_unreachable=["T004"],
            candidate_unreachable=["T004"],
        )
    )
    assert result.reachability_verdict == "rejected"
    assert result.fully_unreachable_decreased is False


def test_reachability_rejected_when_high_critical_pool_regression() -> None:
    case = _case_result(
        case_id="T099",
        risk="high",
        baseline_primary_reachable=True,
        candidate_primary_reachable=False,
    )
    result = _evaluate(case_results=[case])
    assert result.reachability_verdict == "rejected"
    assert result.high_critical_reachability_regressions == 1


def test_ranking_improved() -> None:
    result = _evaluate(ranking=_ranking(primary_delta=0.05, mrr_delta=0.02))
    assert result.ranking_verdict == "improved"
    assert result.ranking_pass is True


def test_ranking_improved_rejected_when_mrr_guardrail_breached() -> None:
    result = _evaluate(ranking=_ranking(primary_delta=0.01, mrr_delta=-0.011))
    assert result.ranking_verdict == "rejected"
    assert result.ranking_pass is False


def test_ranking_improved_accepts_exact_mrr_boundary() -> None:
    result = _evaluate(ranking=_ranking(primary_delta=0.01, mrr_delta=-0.01))
    assert result.ranking_verdict == "improved"
    assert result.ranking_pass is True


def test_ranking_neutral_accepts_exact_mrr_boundary() -> None:
    result = _evaluate(ranking=_ranking(primary_delta=0.0, mrr_delta=-0.01))
    assert result.ranking_verdict == "neutral"
    assert result.ranking_pass is True


def test_ranking_neutral() -> None:
    result = _evaluate(ranking=_ranking(primary_delta=0.0, mrr_delta=0.0))
    assert result.ranking_verdict == "neutral"
    assert result.ranking_pass is True


def test_ranking_rejected_when_primary_hit_at_4_decreases() -> None:
    result = _evaluate(ranking=_ranking(primary_delta=-0.05))
    assert result.ranking_verdict == "rejected"
    assert result.ranking_pass is False


def test_ranking_rejected_when_critical_primary_regression() -> None:
    case = _case_result(case_id="T040", risk="critical", primary_hit_at_4_delta=-1)
    result = _evaluate(
        ranking=_ranking(primary_delta=0.0, regressions=["T040"]),
        case_results=[case],
    )
    assert result.ranking_verdict == "rejected"
    assert result.critical_primary_regressions == 1


def test_ranking_improved_rejected_when_critical_primary_regression() -> None:
    case = _case_result(case_id="T040", risk="critical", primary_hit_at_4_delta=-1)
    result = _evaluate(
        ranking=_ranking(primary_delta=0.01, mrr_delta=0.0),
        case_results=[case],
    )
    assert result.ranking_verdict == "rejected"
    assert result.critical_primary_regressions == 1


def test_ranking_rejected_when_mrr_guardrail_breached() -> None:
    result = _evaluate(ranking=_ranking(primary_delta=0.0, mrr_delta=-0.02))
    assert result.ranking_verdict == "rejected"
    assert result.ranking_pass is False


def test_hard_invariant_failure_single_retrieval() -> None:
    result = _evaluate(single_retrieval_pass=False)
    assert result.hard_invariants_pass is False
    assert result.single_retrieval_pass is False
    assert "multiple retrieval calls per case" in result.failure_reasons
