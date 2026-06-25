"""Vector pool cap A/B evaluator (baseline fetch@24 vs candidate fetch@48 cap@4 pool@36)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from customer_claims_rag.evaluation.diversity_metrics import (
    build_faq_comparison,
    build_new_document_footprint,
    build_required_case_diagnostics,
    build_saturation_metrics,
    evaluate_diversity_acceptance,
    faq_flags_for_final,
    load_vector_pool_cap_config,
    compute_vector_pool_cap_config_hash,
)
from customer_claims_rag.evaluation.diversity_models import (
    DiversityCaseResult,
    DiversityEvaluationRun,
    ExperimentMetadata,
    PoolArmSnapshot,
    RetrievalArmConfig,
    SharedContext,
    VectorPoolCapExperimentConfig,
)
from customer_claims_rag.evaluation.diversity_pool import apply_per_document_cap
from customer_claims_rag.evaluation.git_state import read_git_state
from customer_claims_rag.evaluation.metrics import (
    aggregate_case_metrics,
    aggregate_category_metrics,
    aggregate_risk_metrics,
    compute_case_metrics,
    summarize_fallback_cases,
)
from customer_claims_rag.evaluation.models import CaseResult, EvaluationRun, RunMetadata
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    build_case_comparison_detail,
    build_ranking_arm_result,
    build_ranking_comparison,
    build_reachability_comparison,
    chunks_from_search_results,
    compute_evaluation_dataset_fingerprint,
    compute_pool_reachability,
    rerank_to_final_top_k,
)
from customer_claims_rag.evaluation.pool_expansion_models import PoolExpansionCaseResult
from customer_claims_rag.evaluation.threshold_analysis import analyze_thresholds
from customer_claims_rag.exceptions import SearchError
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.reranker import (
    SourceAuthorityV1Reranker,
    compute_config_hash,
    load_reranker_config,
)
from customer_claims_rag.retrieval.retriever import BaselineRetriever


def _vector_ranks_by_document(results) -> dict[str, list[int]]:
    ranks: dict[str, list[int]] = {}
    for index, result in enumerate(results, start=1):
        ranks.setdefault(result.document_id, []).append(index)
    return ranks


class VectorPoolCapAbEvaluator:
    """Run baseline production retrieval vs capped deep-pool candidate generation."""

    def __init__(
        self,
        *,
        baseline_retriever: BaselineRetriever,
        candidate_retriever: BaselineRetriever,
        reranker: SourceAuthorityV1Reranker,
        config: VectorPoolCapExperimentConfig,
        index_dir: Path,
        cases: list,
        questions_path: Path,
        expected_path: Path,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
    ) -> None:
        self.baseline_retriever = baseline_retriever
        self.candidate_retriever = candidate_retriever
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
        baseline_retriever: BaselineRetriever,
        candidate_retriever: BaselineRetriever,
        experiment_config_path: Path,
        reranker_config_path: Path,
        index_dir: Path,
        questions_path: Path,
        expected_path: Path,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
    ) -> "VectorPoolCapAbEvaluator":
        config = load_vector_pool_cap_config(experiment_config_path)
        reranker = SourceAuthorityV1Reranker(load_reranker_config(reranker_config_path))
        cases = load_evaluation_corpus(
            questions_path=questions_path,
            expected_path=expected_path,
        )
        return cls(
            baseline_retriever=baseline_retriever,
            candidate_retriever=candidate_retriever,
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
        self.baseline_retriever.validate_index()

    def evaluate(self) -> DiversityEvaluationRun:
        started = time.perf_counter()
        self.validate_index()
        manifest = load_manifest(self.index_dir)
        git_state = read_git_state(self.project_root)
        dataset_fingerprint = compute_evaluation_dataset_fingerprint(
            questions_path=self.questions_path,
            expected_path=self.expected_path,
        )

        case_results: list[DiversityCaseResult] = []
        baseline_ranking_cases: list[CaseResult] = []
        candidate_ranking_cases: list[CaseResult] = []
        pool_expansion_cases: list[PoolExpansionCaseResult] = []

        for case in self.cases:
            outcome = self._evaluate_case(case)
            case_results.append(outcome["case_result"])
            baseline_ranking_cases.append(outcome["baseline_ranking_case"])
            candidate_ranking_cases.append(outcome["candidate_ranking_case"])
            pool_expansion_cases.append(outcome["pool_expansion_adapter"])

        run_metadata_base = self._build_run_metadata(manifest=manifest, git_state=git_state)
        baseline_ranking_run = self._build_evaluation_run(baseline_ranking_cases, run_metadata_base)
        candidate_ranking_run = self._build_evaluation_run(candidate_ranking_cases, run_metadata_base)

        reachability = build_reachability_comparison(pool_expansion_cases)
        ranking = build_ranking_comparison(
            baseline_cases=baseline_ranking_cases,
            candidate_cases=candidate_ranking_cases,
            case_results=pool_expansion_cases,
        )

        technical_errors_zero = candidate_ranking_run.aggregate_metrics.technical_error_count == 0
        acceptance_checks, hard_regressions, verdict, boundary = evaluate_diversity_acceptance(
            baseline_metrics=baseline_ranking_run.aggregate_metrics,
            candidate_metrics=candidate_ranking_run.aggregate_metrics,
            reachability=reachability,
            faq_comparison=build_faq_comparison(case_results),
            case_results=case_results,
            reranker_config_hash=compute_config_hash(self.reranker.config),
            technical_errors_zero=technical_errors_zero,
        )

        experiment_config_payload = json.loads(
            Path(
                self.config.config_path or "configs/retrieval/vector_pool_36_cap4_v1.json",
            ).read_text(encoding="utf-8"),
        )
        experiment = ExperimentMetadata(
            experiment_id=self.config.experiment_id,
            version=self.config.version,
            experiment_mode=self.config.experiment_mode,
            config=experiment_config_payload,
            config_hash=compute_vector_pool_cap_config_hash(self.config),
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
            final_top_k=self.config.final_top_k,
            threshold=self.config.threshold,
            collection=self.collection_name,
            chunk_count=manifest.chunk_count,
            document_count=manifest.document_count,
            baseline_arm=RetrievalArmConfig(
                fetch_k=self.config.baseline_fetch_k,
                candidate_pool_k=self.config.baseline_candidate_pool_k,
                per_document_cap=self.config.baseline_per_document_cap,
            ),
            candidate_arm=RetrievalArmConfig(
                fetch_k=self.config.candidate_fetch_k,
                candidate_pool_k=self.config.candidate_candidate_pool_k,
                per_document_cap=self.config.candidate_per_document_cap,
            ),
        )

        return DiversityEvaluationRun(
            timestamp=datetime.now(timezone.utc),
            experiment=experiment,
            shared_context=shared_context,
            baseline_ranking=baseline_ranking_run,
            candidate_ranking=candidate_ranking_run,
            reachability_comparison=reachability,
            baseline_saturation=build_saturation_metrics(
                case_results,
                arm="baseline",
                cap_threshold=None,
                target_pool_k=self.config.baseline_candidate_pool_k,
            ),
            candidate_saturation=build_saturation_metrics(
                case_results,
                arm="candidate",
                cap_threshold=self.config.candidate_per_document_cap,
                target_pool_k=self.config.candidate_candidate_pool_k,
            ),
            faq_comparison=build_faq_comparison(case_results),
            new_document_footprint=build_new_document_footprint(case_results, arm="candidate"),
            required_case_diagnostics=build_required_case_diagnostics(case_results),
            acceptance_checks=acceptance_checks,
            hard_regressions=hard_regressions,
            interpretation_boundary=boundary,
            case_results=case_results,
            verdict=verdict,
            latency_seconds_total=time.perf_counter() - started,
        )

    def _evaluate_case(self, case) -> dict[str, object]:
        try:
            baseline_response = self.baseline_retriever.search(case.query)
            candidate_response = self.candidate_retriever.search(case.query)
        except SearchError as exc:
            return self._error_outcome(case, str(exc))

        baseline_vector = list(baseline_response.results)
        candidate_vector = list(candidate_response.results)

        baseline_pool, baseline_cap = apply_per_document_cap(
            baseline_vector,
            fetch_k=self.config.baseline_fetch_k,
            pool_k=self.config.baseline_candidate_pool_k,
            per_document_cap=self.config.baseline_per_document_cap,
        )
        candidate_pool, candidate_cap = apply_per_document_cap(
            candidate_vector,
            fetch_k=self.config.candidate_fetch_k,
            pool_k=self.config.candidate_candidate_pool_k,
            per_document_cap=self.config.candidate_per_document_cap,
        )

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

        faq_baseline, faq_baseline_no_primary = faq_flags_for_final(
            baseline_final,
            expected_primary_documents=case.expected_primary_documents,
            fallback_expected=case.fallback_expected,
        )
        faq_candidate, faq_candidate_no_primary = faq_flags_for_final(
            candidate_final,
            expected_primary_documents=case.expected_primary_documents,
            fallback_expected=case.fallback_expected,
        )

        baseline_snapshot = PoolArmSnapshot(
            fetch_k=self.config.baseline_fetch_k,
            pool_k=self.config.baseline_candidate_pool_k,
            per_document_cap=self.config.baseline_per_document_cap,
            candidate_ids=[item.chunk_id for item in baseline_pool],
            reachability=baseline_pool_reachability,
            cap_diagnostics=baseline_cap,
        )
        candidate_snapshot = PoolArmSnapshot(
            fetch_k=self.config.candidate_fetch_k,
            pool_k=self.config.candidate_candidate_pool_k,
            per_document_cap=self.config.candidate_per_document_cap,
            candidate_ids=[item.chunk_id for item in candidate_pool],
            reachability=candidate_pool_reachability,
            cap_diagnostics=candidate_cap,
        )

        case_result = DiversityCaseResult(
            case_id=case.test_id,
            risk=case.expected_risk,
            category=case.category,
            query=case.query,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            baseline_vector_chunk_ids=[item.chunk_id for item in baseline_vector],
            candidate_vector_chunk_ids=[item.chunk_id for item in candidate_vector],
            baseline_vector_ranks_by_document=_vector_ranks_by_document(baseline_vector),
            candidate_vector_ranks_by_document=_vector_ranks_by_document(candidate_vector),
            baseline_pool=baseline_snapshot,
            candidate_pool=candidate_snapshot,
            baseline_ranking=baseline_ranking,
            candidate_ranking=candidate_ranking,
            comparison=comparison,
            faq_in_baseline_top4=faq_baseline,
            faq_in_candidate_top4=faq_candidate,
            faq_in_top4_without_primary_baseline=faq_baseline_no_primary,
            faq_in_top4_without_primary_candidate=faq_candidate_no_primary,
        )

        from customer_claims_rag.evaluation.pool_expansion_models import (
            PoolArmSnapshot as ExpansionPoolArmSnapshot,
        )

        pool_expansion_adapter = PoolExpansionCaseResult(
            case_id=case.test_id,
            risk=case.expected_risk,
            category=case.category,
            query=case.query,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            shared_deep_candidate_ids=[item.chunk_id for item in candidate_vector],
            baseline_pool=ExpansionPoolArmSnapshot(
                pool_k=self.config.baseline_candidate_pool_k,
                candidate_ids=[item.chunk_id for item in baseline_pool],
                reachability=baseline_pool_reachability,
            ),
            candidate_pool=ExpansionPoolArmSnapshot(
                pool_k=self.config.candidate_candidate_pool_k,
                candidate_ids=[item.chunk_id for item in candidate_pool],
                reachability=candidate_pool_reachability,
            ),
            baseline_ranking=baseline_ranking,
            candidate_ranking=candidate_ranking,
            comparison=comparison,
            failure_classification="none",
        )

        return {
            "case_result": case_result,
            "baseline_ranking_case": baseline_ranking_case,
            "candidate_ranking_case": candidate_ranking_case,
            "pool_expansion_adapter": pool_expansion_adapter,
        }

    def _error_outcome(self, case, error_message: str) -> dict[str, object]:
        from customer_claims_rag.evaluation.pool_expansion_models import (
            PoolArmSnapshot as ExpansionPoolArmSnapshot,
        )

        error_case = CaseResult(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            status="technical_error",
            error_message=error_message,
        )
        empty_reachability = compute_pool_reachability(
            [],
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
        )
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
        empty_cap = apply_per_document_cap(
            [],
            fetch_k=self.config.baseline_fetch_k,
            pool_k=self.config.baseline_candidate_pool_k,
            per_document_cap=None,
        )[1]
        case_result = DiversityCaseResult(
            case_id=case.test_id,
            risk=case.expected_risk,
            category=case.category,
            query=case.query,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            baseline_pool=PoolArmSnapshot(
                fetch_k=self.config.baseline_fetch_k,
                pool_k=self.config.baseline_candidate_pool_k,
                candidate_ids=[],
                reachability=empty_reachability,
                cap_diagnostics=empty_cap,
            ),
            candidate_pool=PoolArmSnapshot(
                fetch_k=self.config.candidate_fetch_k,
                pool_k=self.config.candidate_candidate_pool_k,
                per_document_cap=self.config.candidate_per_document_cap,
                candidate_ids=[],
                reachability=empty_reachability,
                cap_diagnostics=empty_cap,
            ),
            baseline_ranking=empty_ranking,
            candidate_ranking=empty_ranking,
            comparison=comparison,
        )
        pool_expansion_adapter = PoolExpansionCaseResult(
            case_id=case.test_id,
            risk=case.expected_risk,
            category=case.category,
            query=case.query,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            shared_deep_candidate_ids=[],
            baseline_pool=ExpansionPoolArmSnapshot(
                pool_k=self.config.baseline_candidate_pool_k,
                candidate_ids=[],
                reachability=empty_reachability,
            ),
            candidate_pool=ExpansionPoolArmSnapshot(
                pool_k=self.config.candidate_candidate_pool_k,
                candidate_ids=[],
                reachability=empty_reachability,
            ),
            baseline_ranking=empty_ranking,
            candidate_ranking=empty_ranking,
            comparison=comparison,
            failure_classification="none",
        )
        return {
            "case_result": case_result,
            "baseline_ranking_case": error_case,
            "candidate_ranking_case": error_case,
            "pool_expansion_adapter": pool_expansion_adapter,
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
            fetch_k=self.config.candidate_fetch_k,
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
