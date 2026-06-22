"""Unit tests for hybrid reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from customer_claims_rag.evaluation.ab_models import ArmMetrics
from customer_claims_rag.evaluation.hybrid_models import (
    AcceptanceCriteriaResult,
    ChannelReachabilityComparison,
    ExperimentMetadata,
    FusionPoolSnapshot,
    HybridCaseResult,
    HybridEvaluationRun,
    HybridRankingArmResult,
    LexicalIndexMetadata,
    SharedContext,
    VectorPoolSnapshot,
)
from customer_claims_rag.evaluation.hybrid_reporting import render_hybrid_markdown
from customer_claims_rag.evaluation.models import AggregateMetrics, EvaluationRun, RunMetadata
from customer_claims_rag.evaluation.pool_expansion_models import (
    CaseComparisonDetail,
    PoolReachability,
    RankingAggregateComparison,
    RankingComparison,
)


def _minimal_run() -> HybridEvaluationRun:
    metadata = RunMetadata(
        timestamp=datetime.now(timezone.utc),
        evaluation_result_id="id",
        index_fingerprint="fp",
        index_format_version="1.0.0",
        metadata_schema_version="1.0.0",
        embedding_model="text-embedding-3-small",
        collection="customer_claims",
        evaluation_case_count=1,
        threshold=0.0,
        top_k=12,
        fetch_k=24,
    )
    eval_run = EvaluationRun(
        run_metadata=metadata,
        aggregate_metrics=AggregateMetrics(
            total_cases=1,
            successfully_evaluated_cases=1,
            technical_error_count=0,
            source_recall_case_count=1,
            fallback_case_count=0,
            hit_rate_at_1=1.0,
            hit_rate_at_4=1.0,
            hit_rate_at_12=1.0,
            document_recall_at_1=1.0,
            document_recall_at_4=1.0,
            document_recall_at_12=1.0,
            mrr=1.0,
            primary_source_hit_rate_at_1=1.0,
            primary_source_hit_rate_at_4=1.0,
            cases_with_supporting_documents=0,
            supporting_source_hits_at_4=0,
            supporting_source_hit_rate_at_4=None,
            no_result_rate=0.0,
        ),
        risk_metrics=[],
        category_metrics=[],
        threshold_analysis=[],
        case_results=[],
    )
    reach = PoolReachability(primary_reachable=True, any_expected_source_reachable=True)
    case = HybridCaseResult(
        case_id="T004",
        risk="low",
        category="general",
        query="CVV",
        expected_primary_documents=["07_complaint_handling_procedure"],
        baseline_pool=VectorPoolSnapshot(pool_k=24, reachability=reach),
        candidate_fusion_pool=FusionPoolSnapshot(pool_k=24, reachability=reach),
        baseline_ranking=HybridRankingArmResult(metrics=ArmMetrics()),
        candidate_ranking=HybridRankingArmResult(metrics=ArmMetrics()),
        comparison=CaseComparisonDetail(),
    )
    return HybridEvaluationRun(
        timestamp=datetime.now(timezone.utc),
        experiment=ExperimentMetadata(
            experiment_id="hybrid-lexical-vector-v1",
            version="1.0.0",
            experiment_mode="production-like",
            config={},
            config_hash="hash",
            reranker_id="source-authority-v1",
            reranker_config_hash="c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957",
            score_adapter_id="rrf-base-score-adapter-v1",
        ),
        shared_context=SharedContext(
            index_fingerprint="fp",
            embedding_model="text-embedding-3-small",
            evaluation_dataset_fingerprint="ds",
            case_count=1,
            vector_k=24,
            lexical_k=24,
            fusion_k=24,
            final_top_k=12,
            threshold=0.0,
            collection="customer_claims",
        ),
        lexical_index=LexicalIndexMetadata(
            lexical_algorithm="BM25",
            tokenizer_version="tokenizer-v1",
            bm25_k1=1.5,
            bm25_b=0.75,
            chunk_count=215,
            corpus_fingerprint="fp",
            lexical_index_fingerprint="lexfp",
        ),
        baseline=eval_run,
        candidate=eval_run,
        candidate_generation_comparison=ChannelReachabilityComparison(
            baseline_primary_reachable=56,
            baseline_primary_total=57,
            candidate_primary_reachable=57,
            candidate_primary_total=57,
            baseline_supporting_reachable=35,
            baseline_supporting_total=38,
            candidate_supporting_reachable=36,
            candidate_supporting_total=38,
            high_baseline_primary_reachable=15,
            high_candidate_primary_reachable=15,
            high_primary_total=15,
            critical_baseline_primary_reachable=8,
            critical_candidate_primary_reachable=8,
            critical_primary_total=8,
        ),
        ranking_comparison=RankingComparison(
            aggregate=RankingAggregateComparison(primary_source_hit_rate_at_4_delta=0.0)
        ),
        case_results=[case],
        acceptance=AcceptanceCriteriaResult(
            hard_invariants_pass=True,
            single_retrieval_pass=True,
            technical_errors_zero=True,
            reranker_config_hash_match=True,
            lexical_index_valid=True,
            reachability_verdict="accepted",
            reachability_pass=True,
            high_primary_reachability_non_regressed=True,
            critical_primary_reachability_non_regressed=True,
            fully_unreachable_non_increased=True,
            high_critical_reachability_regressions=0,
            ranking_verdict="neutral",
            ranking_pass=True,
            overall_primary_hit_at_4_delta=0.0,
            overall_mrr_delta=0.0,
            critical_primary_regressions=0,
            low_medium_slice_guardrails_pass=True,
        ),
    )


def test_markdown_from_json_round_trip() -> None:
    run = _minimal_run()
    payload = run.model_dump(mode="json")
    restored = HybridEvaluationRun.model_validate(payload)
    markdown = render_hybrid_markdown(restored)
    assert "normalized_rrf_score" in markdown
    assert "is **not** vector similarity" in markdown
    assert "### T004" in markdown
    assert "Retrieval configuration selection" in markdown
    assert "rejected for MVP selection" in markdown
    assert "C:\\\\" not in markdown
    assert "/Users/" not in markdown


def test_json_round_trip() -> None:
    run = _minimal_run()
    restored = HybridEvaluationRun.model_validate(json.loads(run.model_dump_json()))
    assert restored.experiment.experiment_id == "hybrid-lexical-vector-v1"
