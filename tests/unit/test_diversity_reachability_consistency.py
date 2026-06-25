"""Tests for diversity experiment reachability reporting consistency."""

from __future__ import annotations

import json
from pathlib import Path

from unittest.mock import MagicMock

import pytest

from customer_claims_rag.evaluation.diversity_metrics import (
    ALLOWED_UNREACHABLE_CASES,
    ReachabilityConsistencyError,
    build_reachability_consistency_summary,
    evaluate_diversity_acceptance,
    validate_reachability_consistency,
)
from customer_claims_rag.evaluation.diversity_models import (
    ArmReachabilityConsistency,
    CapPoolDiagnostics,
    DiversityCaseResult,
    DiversityEvaluationRun,
    PoolArmSnapshot,
    ReachabilityConsistencySummary,
)
from customer_claims_rag.evaluation.diversity_reporting import render_diversity_markdown
from customer_claims_rag.evaluation.pool_expansion_metrics import FROZEN_RERANKER_CONFIG_HASH
from customer_claims_rag.evaluation.pool_expansion_models import (
    CaseComparisonDetail,
    PoolReachability,
    RankingArmResult,
    ReachabilityComparison,
)
from customer_claims_rag.evaluation.ab_models import ArmMetrics
from tests.unit.test_diversity_evaluator import _aggregate, _case, _reachability

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = PROJECT_ROOT / "data" / "05_evaluation" / "vector_pool_36_cap4_v1.json"


def _load_artifact() -> DiversityEvaluationRun:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    payload["reachability_consistency"] = build_reachability_consistency_summary(
        [DiversityCaseResult.model_validate(item) for item in payload["case_results"]],
    ).model_dump()
    return DiversityEvaluationRun.model_validate(payload)


def test_branch_a_case_level_unreachable_matches_55_of_57() -> None:
    run = _load_artifact()
    summary = build_reachability_consistency_summary(run.case_results)
    candidate = summary.candidate
    assert candidate.primary_reachable_count == 55
    assert candidate.primary_denominator == 57
    assert candidate.primary_unreachable_case_ids == ["T044", "T047"]
    assert candidate.primary_reachable_count + len(candidate.primary_unreachable_case_ids) == 57


def test_unreachable_subset_guardrail_passes_for_t044_and_t047() -> None:
    run = _load_artifact()
    summary = build_reachability_consistency_summary(run.case_results)
    checks, _, verdict, _ = evaluate_diversity_acceptance(
        baseline_metrics=run.baseline_ranking.aggregate_metrics,
        candidate_metrics=run.candidate_ranking.aggregate_metrics,
        reachability=run.reachability_comparison,
        reachability_consistency=summary,
        faq_comparison=run.faq_comparison,
        case_results=run.case_results,
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        technical_errors_zero=True,
    )
    unreachable_check = next(item for item in checks if item.criterion_id == "unreachable_subset")
    assert unreachable_check.passed is True
    assert unreachable_check.candidate_value == "['T044', 'T047']"
    assert verdict == "ACCEPTED AS PARTIAL CANDIDATE-GENERATION REPAIR"


def test_inconsistent_aggregate_raises_validation_error() -> None:
    run = _load_artifact()
    summary = build_reachability_consistency_summary(run.case_results)
    bad = summary.model_copy(
        update={
            "candidate": summary.candidate.model_copy(
                update={"primary_reachable_count": 56},
            )
        }
    )
    with pytest.raises(ReachabilityConsistencyError, match="arithmetic failed"):
        validate_reachability_consistency(
            bad,
            run.case_results,
            reachability_comparison=run.reachability_comparison,
        )


def test_markdown_and_json_use_same_unreachable_set() -> None:
    run = _load_artifact()
    run = run.model_copy(
        update={
            "reachability_consistency": build_reachability_consistency_summary(run.case_results),
        }
    )
    markdown = render_diversity_markdown(run)
    unreachable = run.reachability_consistency.candidate.primary_unreachable_case_ids
    assert "T044" in markdown
    assert "T047" in markdown
    assert "['T047']" not in markdown or "primary-unreachable" in markdown
    assert str(unreachable) in markdown or "T044, T047" in markdown


def test_limitations_do_not_claim_single_unreachable_when_two_exist() -> None:
    run = _load_artifact()
    run = run.model_copy(
        update={
            "reachability_consistency": build_reachability_consistency_summary(run.case_results),
        }
    )
    markdown = render_diversity_markdown(run)
    assert "primary-unreachable cases" in markdown
    assert "T044" in markdown and "T047" in markdown
    assert "single unreachable" not in markdown.lower()


def test_fallback_case_excluded_from_denominator() -> None:
    case = DiversityCaseResult(
        case_id="T999",
        risk="low",
        query="q",
        fallback_expected=True,
        expected_primary_documents=["01_policy"],
        baseline_pool=PoolArmSnapshot(
            fetch_k=24,
            pool_k=24,
            candidate_ids=[],
            reachability=PoolReachability(),
            cap_diagnostics=CapPoolDiagnostics(
                pool_size=0,
                removed_by_cap_count=0,
                cap_applied=False,
                unique_document_count=0,
                max_chunks_from_one_document=0,
            ),
        ),
        candidate_pool=PoolArmSnapshot(
            fetch_k=48,
            pool_k=36,
            per_document_cap=4,
            candidate_ids=[],
            reachability=PoolReachability(),
            cap_diagnostics=CapPoolDiagnostics(
                pool_size=0,
                removed_by_cap_count=0,
                cap_applied=True,
                unique_document_count=0,
                max_chunks_from_one_document=0,
            ),
        ),
        baseline_ranking=RankingArmResult(
            ordered_candidates=[],
            metrics=ArmMetrics(status="success"),
        ),
        candidate_ranking=RankingArmResult(
            ordered_candidates=[],
            metrics=ArmMetrics(status="success"),
        ),
        comparison=CaseComparisonDetail(),
    )
    summary = build_reachability_consistency_summary([case])
    assert summary.candidate.primary_denominator == 0
    assert summary.candidate.primary_unreachable_case_ids == []


def test_acceptance_verdict_unchanged_for_allowed_unreachable_set() -> None:
    cases = [
        _case("T004", candidate_reachable=True),
        _case("T044", candidate_reachable=False),
        _case("T047", candidate_reachable=False, overlay_doc="12_staff_safety_and_threat_handling"),
    ]
    summary = build_reachability_consistency_summary(cases)
    reachability = ReachabilityComparison(
        baseline_primary_reachable=summary.baseline.primary_reachable_count,
        baseline_primary_total=summary.baseline.primary_denominator,
        candidate_primary_reachable=summary.candidate.primary_reachable_count,
        candidate_primary_total=summary.candidate.primary_denominator,
        baseline_supporting_reachable=0,
        baseline_supporting_total=0,
        candidate_supporting_reachable=0,
        candidate_supporting_total=0,
        baseline_fully_unreachable_cases=summary.baseline.fully_unreachable_case_ids,
        candidate_fully_unreachable_cases=summary.candidate.fully_unreachable_case_ids,
        high_baseline_primary_reachable=0,
        high_candidate_primary_reachable=0,
        high_primary_total=0,
        critical_baseline_primary_reachable=0,
        critical_candidate_primary_reachable=0,
        critical_primary_total=0,
    )
    faq = MagicMock(
        baseline_questions_with_faq_in_top4=36,
        candidate_questions_with_faq_in_top4=36,
    )
    _, _, verdict, boundary = evaluate_diversity_acceptance(
        baseline_metrics=_aggregate(primary_hit4=0.621, mrr=0.643),
        candidate_metrics=_aggregate(primary_hit4=0.621, mrr=0.646),
        reachability=reachability,
        reachability_consistency=summary,
        faq_comparison=faq,
        case_results=cases,
        reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        technical_errors_zero=True,
    )
    unreachable_check_criterion = summary.candidate.primary_unreachable_case_ids
    assert verdict == "REJECTED"
    assert boundary.primary_unreachable_cases == ["T044", "T047"]
    assert set(unreachable_check_criterion).issubset(ALLOWED_UNREACHABLE_CASES)
