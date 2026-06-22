"""Vector pool expansion A/B evaluator with shared deep retrieval."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from customer_claims_rag.evaluation.git_state import read_git_state
from customer_claims_rag.evaluation.metrics import (
    aggregate_case_metrics,
    aggregate_category_metrics,
    aggregate_risk_metrics,
    compute_case_metrics,
    summarize_fallback_cases,
)
from customer_claims_rag.evaluation.models import CaseResult, EvaluationCase, EvaluationRun, RunMetadata
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    build_case_comparison_detail,
    build_ranking_arm_result,
    build_ranking_comparison,
    build_reachability_comparison,
    chunks_from_search_results,
    classify_failure,
    compute_evaluation_dataset_fingerprint,
    compute_pool_expansion_config_hash,
    compute_pool_reachability,
    evaluate_acceptance_criteria,
    load_pool_expansion_config,
    prepare_pool,
    rerank_to_final_top_k,
)
from customer_claims_rag.evaluation.pool_expansion_models import (
    ExperimentMetadata,
    PoolArmSnapshot,
    PoolExpansionCaseResult,
    PoolExpansionEvaluationRun,
    PoolExpansionExperimentConfig,
    SharedContext,
)
from customer_claims_rag.evaluation.threshold_analysis import analyze_thresholds
from customer_claims_rag.exceptions import SearchError
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.reranker import (
    SourceAuthorityV1Reranker,
    compute_config_hash,
    load_reranker_config,
)
from customer_claims_rag.retrieval.retriever import BaselineRetriever


class PoolExpansionEvaluator:
    """Run pool@12 vs pool@24 expansion experiment with one retrieval call per case."""

    def __init__(
        self,
        *,
        retriever: BaselineRetriever,
        reranker: SourceAuthorityV1Reranker,
        config: PoolExpansionExperimentConfig,
        index_dir: Path,
        cases: list[EvaluationCase],
        questions_path: Path,
        expected_path: Path,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.config = config
        self.index_dir = index_dir
        self.cases = cases
        self.questions_path = questions_path
        self.expected_path = expected_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.project_root = project_root

    @classmethod
    def from_paths(
        cls,
        *,
        retriever: BaselineRetriever,
        experiment_config_path: Path,
        reranker_config_path: Path,
        index_dir: Path,
        questions_path: Path,
        expected_path: Path,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
    ) -> "PoolExpansionEvaluator":
        config = load_pool_expansion_config(experiment_config_path)
        reranker = SourceAuthorityV1Reranker(load_reranker_config(reranker_config_path))
        cases = load_evaluation_corpus(
            questions_path=questions_path,
            expected_path=expected_path,
        )
        return cls(
            retriever=retriever,
            reranker=reranker,
            config=config,
            index_dir=index_dir,
            cases=cases,
            questions_path=questions_path,
            expected_path=expected_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            project_root=project_root,
        )

    def validate_index(self) -> None:
        self.retriever.validate_index()

    def evaluate(self) -> PoolExpansionEvaluationRun:
        self.validate_index()
        manifest = load_manifest(self.index_dir)
        git_state = read_git_state(self.project_root)
        dataset_fingerprint = compute_evaluation_dataset_fingerprint(
            questions_path=self.questions_path,
            expected_path=self.expected_path,
        )

        case_results: list[PoolExpansionCaseResult] = []
        baseline_pool_cases: list[CaseResult] = []
        candidate_pool_cases: list[CaseResult] = []
        baseline_ranking_cases: list[CaseResult] = []
        candidate_ranking_cases: list[CaseResult] = []
        shared_prefix_pass = True
        retrieval_calls = 0

        for case in self.cases:
            outcome = self._evaluate_case(case)
            retrieval_calls += 1
            case_results.append(outcome["case_result"])
            baseline_pool_cases.append(outcome["baseline_pool_case"])
            candidate_pool_cases.append(outcome["candidate_pool_case"])
            baseline_ranking_cases.append(outcome["baseline_ranking_case"])
            candidate_ranking_cases.append(outcome["candidate_ranking_case"])
            if not outcome["case_result"].comparison.baseline_pool_is_prefix_of_candidate_pool:
                shared_prefix_pass = False

        if retrieval_calls != len(self.cases):
            raise RuntimeError("retrieval call count mismatch")

        run_metadata_base = self._build_run_metadata(
            manifest=manifest,
            git_state=git_state,
        )
        baseline_pool_run = self._build_evaluation_run(baseline_pool_cases, run_metadata_base)
        candidate_pool_run = self._build_evaluation_run(candidate_pool_cases, run_metadata_base)
        baseline_ranking_run = self._build_evaluation_run(baseline_ranking_cases, run_metadata_base)
        candidate_ranking_run = self._build_evaluation_run(candidate_ranking_cases, run_metadata_base)

        reachability = build_reachability_comparison(case_results)
        ranking = build_ranking_comparison(
            baseline_cases=baseline_ranking_cases,
            candidate_cases=candidate_ranking_cases,
            case_results=case_results,
        )
        technical_errors_zero = candidate_ranking_run.aggregate_metrics.technical_error_count == 0
        acceptance = evaluate_acceptance_criteria(
            config=self.config,
            reranker_config_hash=compute_config_hash(self.reranker.config),
            reachability=reachability,
            ranking=ranking,
            case_results=case_results,
            shared_prefix_pass=shared_prefix_pass,
            single_retrieval_pass=True,
            technical_errors_zero=technical_errors_zero,
        )

        experiment_config_payload = json.loads(
            Path(self.config.config_path or "configs/retrieval/vector_pool_expansion_v1.json").read_text(
                encoding="utf-8",
            )
        )
        experiment = ExperimentMetadata(
            experiment_id=self.config.experiment_id,
            version=self.config.version,
            experiment_mode=self.config.experiment_mode,
            config=experiment_config_payload,
            config_hash=compute_pool_expansion_config_hash(self.config),
            reranker_id=self.config.reranker_id,
            reranker_config_hash=self.config.reranker_config_hash,
            git_commit=git_state.commit,
            git_dirty=git_state.dirty,
            git_status_summary=git_state.status_summary,
        )
        shared_context = SharedContext(
            index_fingerprint=manifest.corpus_fingerprint,
            embedding_model=self.embedding_model,
            evaluation_dataset_fingerprint=dataset_fingerprint,
            case_count=len(self.cases),
            shared_fetch_k=self.config.candidate_pool_k,
            baseline_pool_k=self.config.baseline_pool_k,
            candidate_pool_k=self.config.candidate_pool_k,
            final_top_k=self.config.final_top_k,
            threshold=self.config.threshold,
            collection=self.collection_name,
            chunk_count=manifest.chunk_count,
            document_count=manifest.document_count,
        )

        return PoolExpansionEvaluationRun(
            timestamp=datetime.now(timezone.utc),
            experiment=experiment,
            shared_context=shared_context,
            baseline_pool=baseline_pool_run,
            candidate_pool=candidate_pool_run,
            baseline_ranking=baseline_ranking_run,
            candidate_ranking=candidate_ranking_run,
            reachability_comparison=reachability,
            ranking_comparison=ranking,
            case_results=case_results,
            acceptance=acceptance,
        )

    def _evaluate_case(self, case: EvaluationCase) -> dict[str, object]:
        try:
            response = self.retriever.search(case.query)
        except SearchError as exc:
            error_case = CaseResult(
                test_id=case.test_id,
                query=case.query,
                expected_risk=case.expected_risk,
                expected_primary_documents=case.expected_primary_documents,
                expected_supporting_documents=case.expected_supporting_documents,
                fallback_expected=case.fallback_expected,
                category=case.category,
                status="technical_error",
                error_message=str(exc),
            )
            empty_reachability = compute_pool_reachability(
                [],
                expected_primary_documents=case.expected_primary_documents,
                expected_supporting_documents=case.expected_supporting_documents,
                fallback_expected=case.fallback_expected,
            )
            empty_snapshot = PoolArmSnapshot(pool_k=0, candidate_ids=[], reachability=empty_reachability)
            empty_ranking = build_ranking_arm_result(
                pool=[],
                final_results=[],
                reranked=[],
                case=error_case,
                pool_reachability=empty_reachability,
                expected_primary_documents=case.expected_primary_documents,
                expected_supporting_documents=case.expected_supporting_documents,
            )
            comparison = build_case_comparison_detail(
                baseline_metrics=error_case,
                candidate_metrics=error_case,
                baseline_pool_ids=[],
                candidate_pool_ids=[],
                baseline_ranking_ids=[],
                candidate_ranking_ids=[],
            )
            case_result = PoolExpansionCaseResult(
                case_id=case.test_id,
                risk=case.expected_risk,
                category=case.category,
                query=case.query,
                expected_primary_documents=case.expected_primary_documents,
                expected_supporting_documents=case.expected_supporting_documents,
                fallback_expected=case.fallback_expected,
                shared_deep_candidate_ids=[],
                baseline_pool=empty_snapshot,
                candidate_pool=empty_snapshot,
                baseline_ranking=empty_ranking,
                candidate_ranking=empty_ranking,
                comparison=comparison,
                failure_classification="none",
            )
            return {
                "case_result": case_result,
                "baseline_pool_case": error_case,
                "candidate_pool_case": error_case,
                "baseline_ranking_case": error_case,
                "candidate_ranking_case": error_case,
            }

        deep_results = list(response.results)
        baseline_pool = prepare_pool(deep_results, self.config.baseline_pool_k)
        candidate_pool = prepare_pool(deep_results, self.config.candidate_pool_k)

        baseline_pool_reachability = compute_pool_reachability(
            baseline_pool,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
        )
        candidate_pool_reachability = compute_pool_reachability(
            candidate_pool,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
        )

        baseline_final, baseline_reranked = rerank_to_final_top_k(
            self.reranker,
            case.query,
            baseline_pool,
            final_top_k=self.config.final_top_k,
        )
        candidate_final, candidate_reranked = rerank_to_final_top_k(
            self.reranker,
            case.query,
            candidate_pool,
            final_top_k=self.config.final_top_k,
        )

        baseline_pool_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=chunks_from_search_results(baseline_pool),
        )
        candidate_pool_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=chunks_from_search_results(candidate_pool),
        )
        baseline_ranking_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=chunks_from_search_results(baseline_final),
        )
        candidate_ranking_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=chunks_from_search_results(candidate_final),
        )

        baseline_ranking = build_ranking_arm_result(
            pool=baseline_pool,
            final_results=baseline_final,
            reranked=baseline_reranked,
            case=baseline_ranking_case,
            pool_reachability=baseline_pool_reachability,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
        )
        candidate_ranking = build_ranking_arm_result(
            pool=candidate_pool,
            final_results=candidate_final,
            reranked=candidate_reranked,
            case=candidate_ranking_case,
            pool_reachability=candidate_pool_reachability,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
        )

        comparison = build_case_comparison_detail(
            baseline_metrics=baseline_ranking_case,
            candidate_metrics=candidate_ranking_case,
            baseline_pool_ids=[item.chunk_id for item in baseline_pool],
            candidate_pool_ids=[item.chunk_id for item in candidate_pool],
            baseline_ranking_ids=[item.chunk_id for item in baseline_final],
            candidate_ranking_ids=[item.chunk_id for item in candidate_final],
        )
        failure_classification = classify_failure(
            fallback_expected=case.fallback_expected,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            baseline_pool=baseline_pool_reachability,
            candidate_pool=candidate_pool_reachability,
            baseline_flags=baseline_ranking.flags,
            candidate_flags=candidate_ranking.flags,
            baseline_metrics=baseline_ranking_case,
            candidate_metrics=candidate_ranking_case,
        )

        case_result = PoolExpansionCaseResult(
            case_id=case.test_id,
            risk=case.expected_risk,
            category=case.category,
            query=case.query,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            shared_deep_candidate_ids=[item.chunk_id for item in deep_results],
            baseline_pool=PoolArmSnapshot(
                pool_k=self.config.baseline_pool_k,
                candidate_ids=[item.chunk_id for item in baseline_pool],
                reachability=baseline_pool_reachability,
            ),
            candidate_pool=PoolArmSnapshot(
                pool_k=self.config.candidate_pool_k,
                candidate_ids=[item.chunk_id for item in candidate_pool],
                reachability=candidate_pool_reachability,
            ),
            baseline_ranking=baseline_ranking,
            candidate_ranking=candidate_ranking,
            comparison=comparison,
            failure_classification=failure_classification,
        )
        return {
            "case_result": case_result,
            "baseline_pool_case": baseline_pool_case,
            "candidate_pool_case": candidate_pool_case,
            "baseline_ranking_case": baseline_ranking_case,
            "candidate_ranking_case": candidate_ranking_case,
        }

    def _build_run_metadata(self, *, manifest, git_state) -> RunMetadata:
        return RunMetadata(
            timestamp=datetime.now(timezone.utc),
            evaluation_result_id="pending",
            git_commit=git_state.commit,
            git_dirty=git_state.dirty,
            git_status_summary=git_state.status_summary,
            index_fingerprint=manifest.corpus_fingerprint,
            index_format_version=manifest.index_format_version,
            metadata_schema_version=manifest.metadata_schema_version,
            embedding_model=self.embedding_model,
            vector_dimension=manifest.vector_dimension,
            collection=self.collection_name,
            chunk_count=manifest.chunk_count,
            document_count=manifest.document_count,
            evaluation_case_count=len(self.cases),
            threshold=self.config.threshold,
            top_k=self.config.final_top_k,
            fetch_k=self.config.candidate_pool_k,
        )

    def _build_evaluation_run(
        self,
        case_results: list[CaseResult],
        run_metadata_base: RunMetadata,
    ) -> EvaluationRun:
        technical_errors = [
            f"{case.test_id}: {case.error_message}"
            for case in case_results
            if case.status == "technical_error" and case.error_message
        ]
        return EvaluationRun(
            run_metadata=run_metadata_base,
            aggregate_metrics=aggregate_case_metrics(case_results),
            risk_metrics=aggregate_risk_metrics(case_results),
            category_metrics=aggregate_category_metrics(case_results),
            threshold_analysis=analyze_thresholds(case_results),
            case_results=case_results,
            technical_errors=technical_errors,
            fallback_analysis=summarize_fallback_cases(case_results),
        )
