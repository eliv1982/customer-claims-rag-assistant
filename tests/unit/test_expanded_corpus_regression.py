"""Unit tests for expanded corpus frozen regression comparison logic."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from customer_claims_rag.evaluation.ab_models import ArmMetrics, RankedCandidateAudit
from customer_claims_rag.evaluation.expanded_corpus_regression import (
    build_expanded_corpus_regression,
    compute_evaluation_result_id,
)
from customer_claims_rag.evaluation.expanded_corpus_regression_models import (
    SEMANTIC_OVERLAY_BY_CASE,
)
from customer_claims_rag.evaluation.expanded_corpus_regression_reporting import (
    DEFAULT_REGRESSION_JSON,
    PROTECTED_ARTIFACT_PATHS,
    write_expanded_corpus_regression_outputs,
)
from customer_claims_rag.evaluation.models import AggregateMetrics, RiskSliceMetrics
from customer_claims_rag.evaluation.pool_expansion_metrics import FROZEN_RERANKER_CONFIG_HASH
from customer_claims_rag.evaluation.pool_expansion_models import (
    AcceptanceCriteriaResult,
    CaseComparisonDetail,
    ExperimentMetadata,
    FinalRankingFlags,
    PoolArmSnapshot,
    PoolExpansionCaseResult,
    PoolExpansionEvaluationRun,
    PoolExpansionExperimentConfig,
    PoolReachability,
    RankingAggregateComparison,
    RankingArmResult,
    RankingComparison,
    ReachabilityComparison,
    SharedContext,
)
from customer_claims_rag.exceptions import EvaluationOutputError
from tests.evaluation_helpers import make_fake_evaluation_run

CONFIG_HASH = "ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048"
DATASET_FINGERPRINT = "339ee42b744e3f17df5c963b01895147c29d64c55163582293332baf11a36861"


def _aggregate(primary_hit4: float = 0.6, mrr: float = 0.7) -> AggregateMetrics:
    return make_fake_evaluation_run().aggregate_metrics.model_copy(
        update={
            "primary_source_hit_rate_at_4": primary_hit4,
            "mrr": mrr,
        }
    )


def _risk_metrics() -> list[RiskSliceMetrics]:
    return [
        RiskSliceMetrics(
            risk_level="low",
            case_count=20,
            hit_rate_at_1=0.5,
            hit_rate_at_4=0.6,
            hit_rate_at_12=0.7,
            mrr=0.6,
        )
    ]


def _candidate_audit(document_id: str, rank: int) -> RankedCandidateAudit:
    return RankedCandidateAudit(
        baseline_rank=rank,
        candidate_rank=rank,
        chunk_id=f"{document_id}::chunk-{rank:03d}",
        document_id=document_id,
        chunk_type="policy",
        similarity=0.8,
        distance=0.2,
        rerank_score=1.0,
    )


def _ranking_arm(
    *,
    ordered_docs: list[str],
    primary_hit4: bool,
    primary_in_final: bool,
) -> RankingArmResult:
    return RankingArmResult(
        ordered_candidates=[
            _candidate_audit(document_id, index + 1) for index, document_id in enumerate(ordered_docs)
        ],
        metrics=ArmMetrics(primary_hit_at_4=primary_hit4, reciprocal_rank=0.5),
        flags=FinalRankingFlags(primary_in_final_top12=primary_in_final),
    )


def _case(
    case_id: str,
    *,
    risk: str = "low",
    expected_primary: list[str] | None = None,
    ordered_docs: list[str] | None = None,
    primary_reachable: bool = True,
    primary_hit4: bool = True,
    primary_in_final: bool = True,
) -> PoolExpansionCaseResult:
    primary = expected_primary or ["07_complaint_handling_procedure"]
    docs = ordered_docs or primary + ["10_customer_faq"]
    return PoolExpansionCaseResult(
        case_id=case_id,
        risk=risk,
        query=f"query for {case_id}",
        expected_primary_documents=primary,
        expected_supporting_documents=[],
        shared_deep_candidate_ids=[f"{doc}::chunk-001" for doc in docs],
        baseline_pool=PoolArmSnapshot(pool_k=12, candidate_ids=[], reachability=PoolReachability()),
        candidate_pool=PoolArmSnapshot(
            pool_k=24,
            candidate_ids=[f"{doc}::chunk-001" for doc in docs],
            reachability=PoolReachability(primary_reachable=primary_reachable),
        ),
        baseline_ranking=_ranking_arm(
            ordered_docs=docs,
            primary_hit4=primary_hit4,
            primary_in_final=primary_in_final,
        ),
        candidate_ranking=_ranking_arm(
            ordered_docs=docs,
            primary_hit4=primary_hit4,
            primary_in_final=primary_in_final,
        ),
        comparison=CaseComparisonDetail(),
    )


def _pool_run(
    *,
    fingerprint: str,
    chunk_count: int,
    document_count: int,
    case_results: list[PoolExpansionCaseResult],
    primary_hit4: float = 0.6,
) -> PoolExpansionEvaluationRun:
    eval_run = make_fake_evaluation_run().model_copy(
        update={
            "aggregate_metrics": _aggregate(primary_hit4=primary_hit4),
            "risk_metrics": _risk_metrics(),
        }
    )
    return PoolExpansionEvaluationRun(
        timestamp=datetime.now(timezone.utc),
        experiment=ExperimentMetadata(
            experiment_id="vector-pool-expansion-v1",
            version="1.0.0",
            experiment_mode="production-like",
            config={},
            config_hash=CONFIG_HASH,
            reranker_id="source-authority-v1",
            reranker_config_hash=FROZEN_RERANKER_CONFIG_HASH,
        ),
        shared_context=SharedContext(
            index_fingerprint=fingerprint,
            embedding_model="text-embedding-3-small",
            evaluation_dataset_fingerprint=DATASET_FINGERPRINT,
            case_count=len(case_results),
            shared_fetch_k=24,
            baseline_pool_k=12,
            candidate_pool_k=24,
            final_top_k=12,
            threshold=0.0,
            collection="customer_claims",
            chunk_count=chunk_count,
            document_count=document_count,
        ),
        baseline_pool=eval_run,
        candidate_pool=eval_run,
        baseline_ranking=eval_run,
        candidate_ranking=eval_run,
        reachability_comparison=ReachabilityComparison(
            baseline_primary_reachable=1,
            baseline_primary_total=1,
            candidate_primary_reachable=1,
            candidate_primary_total=1,
            baseline_supporting_reachable=0,
            baseline_supporting_total=0,
            candidate_supporting_reachable=0,
            candidate_supporting_total=0,
            high_baseline_primary_reachable=1,
            high_candidate_primary_reachable=1,
            high_primary_total=1,
            critical_baseline_primary_reachable=1,
            critical_candidate_primary_reachable=1,
            critical_primary_total=1,
        ),
        ranking_comparison=RankingComparison(
            aggregate=RankingAggregateComparison(primary_source_hit_rate_at_4_delta=0.0, mrr_delta=0.0),
        ),
        case_results=case_results,
        acceptance=AcceptanceCriteriaResult(
            hard_invariants_pass=True,
            shared_prefix_pass=True,
            single_retrieval_pass=True,
            technical_errors_zero=True,
            reranker_config_hash_match=True,
            reachability_verdict="accepted",
            reachability_pass=True,
            critical_primary_reachability_improved=False,
            high_primary_reachability_non_regressed=True,
            fully_unreachable_decreased=False,
            high_critical_reachability_regressions=0,
            ranking_verdict="neutral",
            ranking_pass=True,
            overall_primary_hit_at_4_delta=0.0,
            overall_mrr_delta=0.0,
            critical_primary_regressions=0,
        ),
    )


def test_build_paired_regression_requires_identical_config_hash() -> None:
    arm_a = _pool_run(
        fingerprint="fp-a",
        chunk_count=215,
        document_count=10,
        case_results=[_case("T001")],
    )
    arm_b = arm_a.model_copy(
        update={
            "experiment": arm_a.experiment.model_copy(update={"config_hash": "other"}),
            "shared_context": arm_a.shared_context.model_copy(
                update={"index_fingerprint": "fp-b", "chunk_count": 333, "document_count": 15}
            ),
        }
    )
    with pytest.raises(ValueError, match="identical frozen retrieval config hash"):
        build_expanded_corpus_regression(
            arm_a_run=arm_a,
            arm_b_run=arm_b,
            arm_a_path="data/a",
            arm_b_path="data/b",
        )


def test_paired_diff_contains_all_cases_exactly_once() -> None:
    cases = [_case(f"T{index:03d}") for index in range(1, 4)]
    arm_a = _pool_run(fingerprint="fp-a", chunk_count=215, document_count=10, case_results=cases)
    arm_b = _pool_run(fingerprint="fp-b", chunk_count=333, document_count=15, case_results=cases)
    regression = build_expanded_corpus_regression(
        arm_a_run=arm_a,
        arm_b_run=arm_b,
        arm_a_path="data/a",
        arm_b_path="data/b",
    )
    assert [case.case_id for case in regression.paired_cases] == ["T001", "T002", "T003"]
    assert regression.arm_a.frozen_retrieval_config_hash == regression.arm_b.frozen_retrieval_config_hash
    assert regression.arm_a.index_fingerprint != regression.arm_b.index_fingerprint


def test_metric_delta_calculation() -> None:
    cases = [_case("T001")]
    arm_a = _pool_run(
        fingerprint="fp-a",
        chunk_count=215,
        document_count=10,
        case_results=cases,
        primary_hit4=0.5,
    )
    arm_b = _pool_run(
        fingerprint="fp-b",
        chunk_count=333,
        document_count=15,
        case_results=cases,
        primary_hit4=0.7,
    )
    regression = build_expanded_corpus_regression(
        arm_a_run=arm_a,
        arm_b_run=arm_b,
        arm_a_path="data/a",
        arm_b_path="data/b",
    )
    assert regression.metric_deltas.primary_source_hit_rate_at_4 == pytest.approx(0.2)


def test_semantic_overlay_does_not_change_strict_metrics_fields() -> None:
    case = _case(
        "T004",
        expected_primary=["07_complaint_handling_procedure"],
        ordered_docs=[
            "11_payment_security_and_dispute_handling",
            "07_complaint_handling_procedure",
        ],
        primary_hit4=False,
        primary_in_final=False,
    )
    arm_a = _pool_run(fingerprint="fp-a", chunk_count=215, document_count=10, case_results=[case])
    arm_b = _pool_run(fingerprint="fp-b", chunk_count=333, document_count=15, case_results=[case])
    regression = build_expanded_corpus_regression(
        arm_a_run=arm_a,
        arm_b_run=arm_b,
        arm_a_path="data/a",
        arm_b_path="data/b",
    )
    overlay = next(item for item in regression.semantic_overlay if item.case_id == "T004")
    assert overlay.overlay_document_id == SEMANTIC_OVERLAY_BY_CASE["T004"]
    assert overlay.recommend_benchmark_modification is False


def test_regression_classification_is_deterministic_for_pool_loss() -> None:
    case_a = _case(
        "T099",
        risk="high",
        expected_primary=["08_escalation_and_risk_rules"],
        ordered_docs=["08_escalation_and_risk_rules"],
        primary_reachable=True,
    )
    case_b = case_a.model_copy(
        update={
            "candidate_pool": PoolArmSnapshot(
                pool_k=24,
                candidate_ids=[],
                reachability=PoolReachability(primary_reachable=False, fully_unreachable=True),
            ),
            "candidate_ranking": _ranking_arm(
                ordered_docs=["10_customer_faq"],
                primary_hit4=False,
                primary_in_final=False,
            ),
        }
    )
    arm_a = _pool_run(fingerprint="fp-a", chunk_count=215, document_count=10, case_results=[case_a])
    arm_b = _pool_run(fingerprint="fp-b", chunk_count=333, document_count=15, case_results=[case_b])
    regression = build_expanded_corpus_regression(
        arm_a_run=arm_a,
        arm_b_run=arm_b,
        arm_a_path="data/a",
        arm_b_path="data/b",
    )
    paired = regression.paired_cases[0]
    assert paired.harmful_regression is True
    assert paired.classification == "regression"
    assert regression.verdict == "REJECT / REPAIR REQUIRED"


def test_write_refuses_protected_artifacts(tmp_path: Path) -> None:
    cases = [_case("T001")]
    arm_a = _pool_run(fingerprint="fp-a", chunk_count=215, document_count=10, case_results=cases)
    arm_b = _pool_run(fingerprint="fp-b", chunk_count=333, document_count=15, case_results=cases)
    regression = build_expanded_corpus_regression(
        arm_a_run=arm_a,
        arm_b_run=arm_b,
        arm_a_path="data/a",
        arm_b_path="data/b",
    )
    project_root = tmp_path
    safe_json = project_root / "data/05_evaluation/expanded_corpus_frozen_regression_v1.json"
    safe_md = project_root / "tests/09_expanded_corpus_frozen_regression_results.md"
    write_expanded_corpus_regression_outputs(
        regression,
        output_json=safe_json,
        output_markdown=safe_md,
        project_root=project_root,
    )
    assert safe_json.is_file()
    for protected in PROTECTED_ARTIFACT_PATHS:
        with pytest.raises(EvaluationOutputError, match="protected"):
            write_expanded_corpus_regression_outputs(
                regression,
                output_json=protected,
                output_markdown=safe_md,
                project_root=project_root,
            )


def test_compute_evaluation_result_id_is_stable() -> None:
    cases = [_case("T001")]
    arm_a = _pool_run(fingerprint="fp-a", chunk_count=215, document_count=10, case_results=cases)
    arm_b = _pool_run(fingerprint="fp-b", chunk_count=333, document_count=15, case_results=cases)
    regression = build_expanded_corpus_regression(
        arm_a_run=arm_a,
        arm_b_run=arm_b,
        arm_a_path="data/a",
        arm_b_path="data/b",
    )
    first = compute_evaluation_result_id(regression)
    second = compute_evaluation_result_id(regression)
    assert first == second
