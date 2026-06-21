"""Baseline retrieval evaluator."""

from __future__ import annotations

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
from customer_claims_rag.evaluation.models import (
    CaseResult,
    EvaluationCase,
    EvaluationRun,
    RetrievedChunkResult,
    RunMetadata,
)
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.result_id import compute_evaluation_result_id
from customer_claims_rag.evaluation.threshold_analysis import analyze_thresholds
from customer_claims_rag.exceptions import SearchError
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.models import SearchResponse
from customer_claims_rag.retrieval.retriever import BaselineRetriever


class RetrievalEvaluator:
    """Evaluate baseline retrieval against the 60-case corpus."""

    def __init__(
        self,
        *,
        retriever: BaselineRetriever,
        index_dir: Path,
        cases: list[EvaluationCase],
        top_k: int = 12,
        fetch_k: int = 12,
        threshold: float = 0.0,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.index_dir = index_dir
        self.cases = cases
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.threshold = threshold
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.project_root = project_root

    @classmethod
    def from_paths(
        cls,
        *,
        retriever: BaselineRetriever,
        index_dir: Path,
        questions_path: Path,
        expected_path: Path,
        top_k: int = 12,
        fetch_k: int = 12,
        threshold: float = 0.0,
        collection_name: str,
        embedding_model: str,
        project_root: Path | None = None,
        cases: list[EvaluationCase] | None = None,
    ) -> "RetrievalEvaluator":
        loaded_cases = cases or load_evaluation_corpus(
            questions_path=questions_path,
            expected_path=expected_path,
        )
        return cls(
            retriever=retriever,
            index_dir=index_dir,
            cases=loaded_cases,
            top_k=top_k,
            fetch_k=fetch_k,
            threshold=threshold,
            collection_name=collection_name,
            embedding_model=embedding_model,
            project_root=project_root,
        )

    def validate_index(self) -> None:
        """Run full static manifest preflight via retriever public API."""
        self.retriever.validate_index()

    def evaluate(self) -> EvaluationRun:
        self.validate_index()
        manifest = load_manifest(self.index_dir)
        git_state = read_git_state(self.project_root)
        case_results: list[CaseResult] = []
        technical_errors: list[str] = []

        for case in self.cases:
            case_results.append(self._evaluate_case(case))

        for case in case_results:
            if case.status == "technical_error" and case.error_message:
                technical_errors.append(f"{case.test_id}: {case.error_message}")

        aggregate = aggregate_case_metrics(case_results)
        risk_metrics = aggregate_risk_metrics(case_results)
        category_metrics = aggregate_category_metrics(case_results)
        threshold_analysis = analyze_thresholds(case_results)
        fallback_analysis = summarize_fallback_cases(case_results)

        provisional = EvaluationRun(
            run_metadata=RunMetadata(
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
            ),
            aggregate_metrics=aggregate,
            risk_metrics=risk_metrics,
            category_metrics=category_metrics,
            threshold_analysis=threshold_analysis,
            case_results=case_results,
            technical_errors=technical_errors,
            fallback_analysis=fallback_analysis,
        )
        result_id = compute_evaluation_result_id(provisional)
        metadata = provisional.run_metadata.model_copy(
            update={"evaluation_result_id": result_id},
        )
        return provisional.model_copy(update={"run_metadata": metadata})

    def _evaluate_case(self, case: EvaluationCase) -> CaseResult:
        try:
            response = self.retriever.search(case.query)
        except SearchError as exc:
            return CaseResult(
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
        chunks = _response_to_chunks(response)
        return compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=chunks,
        )


def _response_to_chunks(response: SearchResponse) -> list[RetrievedChunkResult]:
    chunks: list[RetrievedChunkResult] = []
    for result in response.results:
        chunks.append(
            RetrievedChunkResult(
                rank=result.rank,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                source_path=_to_posix_source_path(result.source_path),
                similarity=result.similarity,
                distance=result.distance,
                heading=result.heading,
            )
        )
    return chunks


def _to_posix_source_path(source_path: str) -> str:
    return Path(source_path).as_posix()
