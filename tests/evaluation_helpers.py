"""Shared helpers for evaluation tests."""

from __future__ import annotations

from datetime import datetime, timezone

from customer_claims_rag.evaluation.models import (
    AggregateMetrics,
    EvaluationRun,
    RetrievedChunkResult,
    RunMetadata,
)
from customer_claims_rag.retrieval.models import SearchResponse, SearchResult


def make_search_result(
    *,
    rank: int,
    document_id: str,
    chunk_id: str | None = None,
    similarity: float = 0.5,
    distance: float = 0.5,
) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id or f"{document_id}::chunk-{rank:03d}",
        document_id=document_id,
        content=f"content for {document_id}",
        source_path=f"data/02_clean_markdown/{document_id}.md",
        chunk_type="policy",
        topic=None,
        risk_level="low",
        heading="Section",
        heading_path=["Section"],
        section="Section",
        subsection=None,
        similarity=similarity,
        distance=distance,
    )


def make_search_response(
    query: str,
    results: list[SearchResult],
    *,
    top_k: int = 12,
    fetch_k: int = 12,
    threshold: float = 0.0,
) -> SearchResponse:
    return SearchResponse(
        query=query,
        top_k=top_k,
        fetch_k=fetch_k,
        similarity_threshold=threshold,
        results=results,
        candidates_fetched=len(results),
        candidates_above_threshold=len(results),
        embedding_model="fake-embedding-model",
        collection_name="customer_claims",
    )


def make_retrieved_chunk(
    *,
    rank: int,
    document_id: str,
    similarity: float = 0.5,
) -> RetrievedChunkResult:
    return RetrievedChunkResult(
        rank=rank,
        chunk_id=f"{document_id}::chunk-{rank:03d}",
        document_id=document_id,
        source_path=f"data/02_clean_markdown/{document_id}.md",
        similarity=similarity,
        distance=1.0 - similarity,
        heading="Section",
    )


def make_fake_evaluation_run() -> EvaluationRun:
    return EvaluationRun(
        run_metadata=RunMetadata(
            timestamp=datetime.now(timezone.utc),
            evaluation_result_id="test-result-id",
            git_commit="abc",
            git_dirty=True,
            git_status_summary="3 changed path(s)",
            index_fingerprint="fp",
            index_format_version="1.0.0",
            metadata_schema_version="1.0.0",
            embedding_model="fake-embedding-model",
            vector_dimension=8,
            collection="customer_claims",
            chunk_count=215,
            document_count=10,
            evaluation_case_count=60,
            threshold=0.0,
            top_k=12,
            fetch_k=12,
        ),
        aggregate_metrics=AggregateMetrics(
            total_cases=60,
            successfully_evaluated_cases=60,
            technical_error_count=0,
            source_recall_case_count=58,
            fallback_case_count=2,
            hit_rate_at_1=0.5,
            hit_rate_at_4=0.6,
            hit_rate_at_12=0.7,
            document_recall_at_1=0.5,
            document_recall_at_4=0.6,
            document_recall_at_12=0.7,
            mrr=0.4,
            primary_source_hit_rate_at_1=0.4,
            primary_source_hit_rate_at_4=0.5,
            cases_with_supporting_documents=38,
            supporting_source_hits_at_4=26,
            supporting_source_hit_rate_at_4=26 / 38,
            no_result_rate=0.0,
        ),
        risk_metrics=[],
        category_metrics=[],
        threshold_analysis=[],
        case_results=[],
    )
