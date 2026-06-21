"""Reranking A/B evaluator with shared candidate pool."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from customer_claims_rag.evaluation.ab_metrics import (
    baseline_identity_pass,
    build_ab_comparison,
    build_baseline_identity_report,
    build_case_arm_result,
    build_case_comparison,
    case_result_to_arm_metrics,
    chunks_from_search_results,
    compute_evaluation_dataset_fingerprint,
)
from customer_claims_rag.evaluation.ab_models import (
    AbCaseResult,
    AbEvaluationRun,
    CaseComparison,
    ExperimentMetadata,
    SharedContext,
)
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
from customer_claims_rag.evaluation.result_id import compute_evaluation_result_id
from customer_claims_rag.evaluation.threshold_analysis import analyze_thresholds
from customer_claims_rag.exceptions import SearchError
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.reranker import (
    RerankerConfig,
    SourceAuthorityV1Reranker,
    compute_config_hash,
    load_reranker_config,
)
from customer_claims_rag.retrieval.retriever import BaselineRetriever


FROZEN_BASELINE_RESULT_ID = "2132c861d4d03b3b99fb5413ae69fd3f0b45f3326d7b71671b4ea3a2a8106e0a"
FROZEN_INDEX_FINGERPRINT = "bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3"


class RerankingAbEvaluator:
    """Run production-like reranking A/B evaluation with one retrieval call per case."""

    def __init__(
        self,
        *,
        retriever: BaselineRetriever,
        reranker: SourceAuthorityV1Reranker,
        config: RerankerConfig,
        index_dir: Path,
        cases: list[EvaluationCase],
        questions_path: Path,
        expected_path: Path,
        top_k: int = 12,
        fetch_k: int = 12,
        threshold: float = 0.0,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
        frozen_baseline_path: Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.config = config
        self.index_dir = index_dir
        self.cases = cases
        self.questions_path = questions_path
        self.expected_path = expected_path
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.threshold = threshold
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.project_root = project_root
        self.frozen_baseline_path = frozen_baseline_path

    @classmethod
    def from_paths(
        cls,
        *,
        retriever: BaselineRetriever,
        config_path: Path,
        index_dir: Path,
        questions_path: Path,
        expected_path: Path,
        top_k: int = 12,
        fetch_k: int = 12,
        threshold: float = 0.0,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
        frozen_baseline_path: Path | None = None,
    ) -> "RerankingAbEvaluator":
        config = load_reranker_config(config_path)
        reranker = SourceAuthorityV1Reranker(config)
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
            top_k=top_k,
            fetch_k=fetch_k,
            threshold=threshold,
            collection_name=collection_name,
            embedding_model=embedding_model,
            project_root=project_root,
            frozen_baseline_path=frozen_baseline_path,
        )

    def validate_index(self) -> None:
        self.retriever.validate_index()

    def evaluate(self) -> AbEvaluationRun:
        self.validate_index()
        manifest = load_manifest(self.index_dir)
        git_state = read_git_state(self.project_root)
        dataset_fingerprint = compute_evaluation_dataset_fingerprint(
            questions_path=self.questions_path,
            expected_path=self.expected_path,
        )

        baseline_case_results: list[CaseResult] = []
        candidate_case_results: list[CaseResult] = []
        ab_case_results: list[AbCaseResult] = []
        shared_pool_pass = True
        retrieval_calls = 0

        for case in self.cases:
            arm_outputs = self._evaluate_case(case)
            retrieval_calls += 1
            baseline_case_results.append(arm_outputs["baseline_case"])
            candidate_case_results.append(arm_outputs["candidate_case"])
            ab_case_results.append(arm_outputs["ab_case"])
            if not arm_outputs["ab_case"].comparison.shared_pool_match:
                shared_pool_pass = False

        if retrieval_calls != len(self.cases):
            raise RuntimeError("retrieval call count mismatch")

        run_metadata_base = self._build_run_metadata(
            manifest=manifest,
            git_state=git_state,
            dataset_fingerprint=dataset_fingerprint,
        )
        baseline_template = self._build_template_run(
            case_results=baseline_case_results,
            run_metadata_base=run_metadata_base,
        )
        candidate_template = self._build_template_run(
            case_results=candidate_case_results,
            run_metadata_base=run_metadata_base,
        )
        baseline_run = self._finalize_run(baseline_template)
        candidate_run = self._finalize_run(candidate_template)

        frozen_run = self._load_frozen_baseline()
        identity_report = build_baseline_identity_report(
            computed_run=baseline_run,
            frozen_run=frozen_run,
            case_results=ab_case_results,
        )
        baseline_identity_pass_flag = baseline_identity_pass(identity_report)

        comparison = build_ab_comparison(
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            case_results=ab_case_results,
            baseline_identity_pass=baseline_identity_pass_flag,
            shared_pool_pass=shared_pool_pass,
        )

        config_payload = json.loads(
            (self.config.config_path or Path("configs/reranking/source_authority_v1.json")).read_text(
                encoding="utf-8"
            )
        )
        experiment = ExperimentMetadata(
            reranker_id=self.config.reranker_id,
            version=self.config.version,
            experiment_mode=self.config.experiment_mode,
            config=config_payload,
            config_hash=compute_config_hash(self.config),
            git_commit=git_state.commit,
            git_dirty=git_state.dirty,
            git_status_summary=git_state.status_summary,
        )
        shared_context = SharedContext(
            index_fingerprint=manifest.corpus_fingerprint,
            embedding_model=self.embedding_model,
            evaluation_dataset_fingerprint=dataset_fingerprint,
            case_count=len(self.cases),
            fetch_k=self.fetch_k,
            top_k=self.top_k,
            threshold=self.threshold,
            collection=self.collection_name,
            chunk_count=manifest.chunk_count,
            document_count=manifest.document_count,
        )

        return AbEvaluationRun(
            timestamp=datetime.now(timezone.utc),
            experiment=experiment,
            shared_context=shared_context,
            baseline=baseline_run,
            candidate=candidate_run,
            comparison=comparison,
            case_results=ab_case_results,
            baseline_identity=identity_report,
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
            empty_arm = build_case_arm_result(results=[], reranked=None, case=error_case)
            comparison = CaseComparison(
                shared_pool_match=True,
                reranker_not_applicable=True,
            )
            ab_case = AbCaseResult(
                case_id=case.test_id,
                risk=case.expected_risk,
                category=case.category,
                query=case.query,
                expected_primary_documents=case.expected_primary_documents,
                expected_supporting_documents=case.expected_supporting_documents,
                fallback_expected=case.fallback_expected,
                shared_candidate_ids=[],
                baseline=empty_arm,
                candidate=empty_arm,
                comparison=comparison,
            )
            return {
                "baseline_case": error_case,
                "candidate_case": error_case,
                "ab_case": ab_case,
            }

        baseline_results = list(response.results)
        baseline_chunks = chunks_from_search_results(baseline_results)
        baseline_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=baseline_chunks,
        )

        candidate_input = deepcopy(baseline_results)
        reranked = self.reranker.rerank(case.query, candidate_input)
        candidate_results = [item.result for item in reranked]
        candidate_chunks = chunks_from_search_results(candidate_results)
        candidate_case = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=candidate_chunks,
        )

        baseline_arm = build_case_arm_result(
            results=baseline_results,
            reranked=None,
            case=baseline_case,
        )
        candidate_arm = build_case_arm_result(
            results=baseline_results,
            reranked=reranked,
            case=candidate_case,
        )
        comparison = build_case_comparison(
            baseline_metrics=case_result_to_arm_metrics(baseline_case),
            candidate_metrics=case_result_to_arm_metrics(candidate_case),
            baseline_audits=baseline_arm.ordered_candidates,
            candidate_audits=candidate_arm.ordered_candidates,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
        )

        shared_candidate_ids = [result.chunk_id for result in baseline_results]
        ab_case = AbCaseResult(
            case_id=case.test_id,
            risk=case.expected_risk,
            category=case.category,
            query=case.query,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            shared_candidate_ids=shared_candidate_ids,
            baseline=baseline_arm,
            candidate=candidate_arm,
            comparison=comparison,
        )
        return {
            "baseline_case": baseline_case,
            "candidate_case": candidate_case,
            "ab_case": ab_case,
        }

    def _build_run_metadata(self, *, manifest, git_state, dataset_fingerprint: str) -> RunMetadata:
        _ = dataset_fingerprint
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
            threshold=self.threshold,
            top_k=self.top_k,
            fetch_k=self.fetch_k,
        )

    def _build_template_run(
        self,
        *,
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

    def _finalize_run(self, run: EvaluationRun) -> EvaluationRun:
        result_id = compute_evaluation_result_id(run)
        metadata = run.run_metadata.model_copy(
            update={"evaluation_result_id": result_id},
        )
        return run.model_copy(update={"run_metadata": metadata})

    def _load_frozen_baseline(self) -> EvaluationRun:
        if self.frozen_baseline_path is None:
            root = self.project_root or Path.cwd()
            path = root / "data" / "05_evaluation" / "retrieval_results.json"
        else:
            path = self.frozen_baseline_path
        payload = json.loads(path.read_text(encoding="utf-8"))
        return EvaluationRun.model_validate(payload)
