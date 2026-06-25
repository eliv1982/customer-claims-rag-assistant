"""Unit tests for diversity experiment acceptance and evaluator behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.evaluation.diversity_evaluator import VectorPoolCapAbEvaluator
from customer_claims_rag.evaluation.diversity_metrics import (
    ALLOWED_UNREACHABLE_CASES,
    evaluate_diversity_acceptance,
    load_vector_pool_cap_config,
)
from customer_claims_rag.evaluation.diversity_models import (
    ArmReachabilityConsistency,
    CapPoolDiagnostics,
    DiversityCaseResult,
    PoolArmSnapshot,
    ReachabilityConsistencySummary,
)
from customer_claims_rag.evaluation.diversity_pool import apply_per_document_cap
from customer_claims_rag.evaluation.models import AggregateMetrics
from customer_claims_rag.evaluation.pool_expansion_metrics import FROZEN_RERANKER_CONFIG_HASH
from customer_claims_rag.evaluation.pool_expansion_models import (
    CaseComparisonDetail,
    FinalRankingFlags,
    PoolReachability,
    RankingArmResult,
    ReachabilityComparison,
)
from customer_claims_rag.evaluation.ab_models import ArmMetrics
from customer_claims_rag.retrieval.models import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_36_cap4_v1.json"


def _aggregate(primary_hit4: float = 0.621, mrr: float = 0.643) -> AggregateMetrics:
    return AggregateMetrics(
        total_cases=60,
        successfully_evaluated_cases=60,
        technical_error_count=0,
        source_recall_case_count=57,
        fallback_case_count=3,
        hit_rate_at_1=0.5,
        hit_rate_at_4=0.75,
        hit_rate_at_12=0.9,
        document_recall_at_1=0.4,
        document_recall_at_4=0.5,
        document_recall_at_12=0.7,
        mrr=mrr,
        primary_source_hit_rate_at_1=0.5,
        primary_source_hit_rate_at_4=primary_hit4,
        cases_with_supporting_documents=10,
        supporting_source_hits_at_4=5,
        supporting_source_hit_rate_at_4=0.5,
        no_result_rate=0.0,
        mean_top1_similarity=0.8,
        median_top1_similarity=0.8,
        min_top1_similarity=0.5,
        max_top1_similarity=0.95,
    )


def _ranking(primary_hit4: bool = False, primary_top12: bool = False) -> RankingArmResult:
    return RankingArmResult(
        ordered_candidates=[],
        metrics=ArmMetrics(
            hit_at_1=False,
            hit_at_4=False,
            hit_at_12=primary_top12,
            document_recall_at_1=0.0,
            document_recall_at_4=0.0,
            document_recall_at_12=0.0,
            reciprocal_rank=0.0,
            primary_hit_at_1=False,
            primary_hit_at_4=primary_hit4,
            supporting_hit_at_4=False,
            status="success",
        ),
        flags=FinalRankingFlags(primary_in_final_top12=primary_top12),
    )


def _case(
    case_id: str,
    *,
    baseline_reachable: bool = True,
    candidate_reachable: bool = True,
    overlay_doc: str | None = None,
) -> DiversityCaseResult:
    cap_counts = {overlay_doc: 1} if overlay_doc else {}
    return DiversityCaseResult(
        case_id=case_id,
        risk="high",
        query="q",
        expected_primary_documents=["01_policy"],
        baseline_pool=PoolArmSnapshot(
            fetch_k=24,
            pool_k=24,
            candidate_ids=["c1"],
            reachability=PoolReachability(primary_reachable=baseline_reachable),
            cap_diagnostics=CapPoolDiagnostics(
                pool_size=24,
                removed_by_cap_count=0,
                cap_applied=False,
                unique_document_count=10,
                max_chunks_from_one_document=4,
            ),
        ),
        candidate_pool=PoolArmSnapshot(
            fetch_k=48,
            pool_k=36,
            per_document_cap=4,
            candidate_ids=["c1"],
            reachability=PoolReachability(primary_reachable=candidate_reachable),
            cap_diagnostics=CapPoolDiagnostics(
                pool_size=36,
                removed_by_cap_count=1,
                cap_applied=True,
                unique_document_count=12,
                max_chunks_from_one_document=4,
                document_counts=cap_counts,
            ),
        ),
        baseline_ranking=_ranking(),
        candidate_ranking=_ranking(primary_top12=True),
        comparison=CaseComparisonDetail(),
    )


def _reachability(
    *,
    baseline_primary: int = 51,
    candidate_primary: int = 55,
    baseline_unreachable: list[str] | None = None,
    candidate_unreachable: list[str] | None = None,
    critical_candidate: int = 7,
) -> ReachabilityComparison:
    return ReachabilityComparison(
        baseline_primary_reachable=baseline_primary,
        baseline_primary_total=57,
        candidate_primary_reachable=candidate_primary,
        candidate_primary_total=57,
        baseline_supporting_reachable=10,
        baseline_supporting_total=10,
        candidate_supporting_reachable=10,
        candidate_supporting_total=10,
        baseline_fully_unreachable_cases=baseline_unreachable or ["T004"],
        candidate_fully_unreachable_cases=candidate_unreachable or ["T044", "T047"],
        high_baseline_primary_reachable=14,
        high_candidate_primary_reachable=14,
        high_primary_total=15,
        critical_baseline_primary_reachable=5,
        critical_candidate_primary_reachable=critical_candidate,
        critical_primary_total=8,
    )


def _consistency_summary(
    reachability: ReachabilityComparison,
    *,
    baseline_unreachable: list[str] | None = None,
    candidate_unreachable: list[str] | None = None,
) -> ReachabilityConsistencySummary:
    baseline_unreachable = baseline_unreachable or []
    candidate_unreachable = candidate_unreachable or []
    return ReachabilityConsistencySummary(
        baseline=ArmReachabilityConsistency(
            primary_reachable_count=reachability.baseline_primary_reachable,
            primary_denominator=reachability.baseline_primary_total,
            primary_unreachable_case_ids=sorted(baseline_unreachable),
            fully_unreachable_case_ids=reachability.baseline_fully_unreachable_cases,
        ),
        candidate=ArmReachabilityConsistency(
            primary_reachable_count=reachability.candidate_primary_reachable,
            primary_denominator=reachability.candidate_primary_total,
            primary_unreachable_case_ids=sorted(candidate_unreachable),
            fully_unreachable_case_ids=reachability.candidate_fully_unreachable_cases,
        ),
    )


def test_config_loads_and_reranker_hash_matches() -> None:
    config = load_vector_pool_cap_config(CONFIG)
    assert config.experiment_id == "vector-pool-36-cap4-v1"
    assert config.reranker_config_hash == FROZEN_RERANKER_CONFIG_HASH
    assert config.baseline_per_document_cap is None
    assert config.candidate_per_document_cap == 4


def test_acceptance_rejected_when_any_criterion_fails() -> None:
    reachability = _reachability()
    checks, hard, verdict, _ = evaluate_diversity_acceptance(
        baseline_metrics=_aggregate(),
        candidate_metrics=_aggregate(primary_hit4=0.60),
        reachability=reachability,
        reachability_consistency=_consistency_summary(
            reachability,
            candidate_unreachable=["T044", "T047"],
        ),
        faq_comparison=MagicMock(candidate_questions_with_faq_in_top4=36),
        case_results=[],
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        technical_errors_zero=True,
    )
    assert verdict == "REJECTED"
    assert any(not check.passed for check in checks)


def test_unreachable_guardrail_checked_explicitly() -> None:
    reachability = _reachability(
        candidate_primary=54,
        candidate_unreachable=["T044", "T047", "T099"],
    )
    checks, hard, _, _ = evaluate_diversity_acceptance(
        baseline_metrics=_aggregate(),
        candidate_metrics=_aggregate(primary_hit4=0.63, mrr=0.65),
        reachability=reachability,
        reachability_consistency=_consistency_summary(
            reachability,
            candidate_unreachable=["T044", "T047", "T099"],
        ),
        faq_comparison=MagicMock(candidate_questions_with_faq_in_top4=30),
        case_results=[],
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        technical_errors_zero=True,
    )
    unreachable_check = next(item for item in checks if item.criterion_id == "unreachable_subset")
    assert unreachable_check.passed is False
    assert hard.new_unreachable_primary_outside_allowed is True


def test_faq_hard_regression_flagged() -> None:
    reachability = _reachability()
    _, hard, _, _ = evaluate_diversity_acceptance(
        baseline_metrics=_aggregate(),
        candidate_metrics=_aggregate(primary_hit4=0.63, mrr=0.65),
        reachability=reachability,
        reachability_consistency=_consistency_summary(
            reachability,
            candidate_unreachable=["T044", "T047"],
        ),
        faq_comparison=MagicMock(candidate_questions_with_faq_in_top4=41),
        case_results=[],
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        technical_errors_zero=True,
    )
    assert hard.faq_top4_above_ceiling is True


def test_baseline_arm_does_not_apply_cap() -> None:
    results = [_make_result(i, f"doc_{i % 2}") for i in range(1, 30)]
    pool, diagnostics = apply_per_document_cap(
        results,
        fetch_k=24,
        pool_k=24,
        per_document_cap=None,
    )
    assert diagnostics.cap_applied is False
    assert len(pool) == 24


def test_candidate_arm_applies_cap_before_reranker() -> None:
    config = load_vector_pool_cap_config(CONFIG)
    assert config.candidate_fetch_k == 48
    assert config.candidate_per_document_cap == 4
    assert config.candidate_candidate_pool_k == 36


def _make_result(rank: int, document_id: str) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=f"{document_id}::chunk-{rank:03d}",
        document_id=document_id,
        content="x",
        source_path=f"data/02_clean_markdown/{document_id}.md",
        chunk_type="policy",
        topic=None,
        risk_level=None,
        heading="H",
        heading_path=["H"],
        section=None,
        subsection=None,
        similarity=0.9,
        distance=0.1,
    )


def test_evaluator_uses_same_reranker_for_both_arms() -> None:
    config = load_vector_pool_cap_config(CONFIG)
    reranker = MagicMock()
    reranker.config = MagicMock()
    evaluator = VectorPoolCapAbEvaluator(
        baseline_retriever=MagicMock(),
        candidate_retriever=MagicMock(),
        reranker=reranker,
        config=config,
        index_dir=PROJECT_ROOT / "data" / "04_index",
        cases=[],
        questions_path=PROJECT_ROOT / "tests" / "01_test_questions.md",
        expected_path=PROJECT_ROOT / "tests" / "02_expected_answers.md",
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
    )
    assert evaluator.reranker is reranker
    assert evaluator.config.reranker_id == "source-authority-v1"
