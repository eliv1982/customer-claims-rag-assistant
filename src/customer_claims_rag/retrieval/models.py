"""Retrieval layer data models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IndexManifest(BaseModel):
    """Persisted metadata about a built vector index."""

    model_config = ConfigDict(extra="forbid")

    index_format_version: str
    collection_name: str
    embedding_model: str
    corpus_fingerprint: str
    chunk_count: int
    document_count: int
    metadata_schema_version: str
    vector_dimension: int | None = None
    created_at: datetime | None = None
    chunk_payload_digest: str | None = None
    embedding_digest: str | None = None
    collection_content_digest: str | None = None
    build_run_id: str | None = None


class VectorSearchHit(BaseModel):
    """Single vector store search hit before threshold filtering."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    content: str
    source_path: str
    chunk_type: str
    topic: str | None = None
    risk_level: str | None = None
    heading: str
    heading_path: list[str]
    section: str | None = None
    subsection: str | None = None
    distance: float
    similarity: float
    backend_rank: int = 0


class SearchResult(BaseModel):
    """Ranked retrieval result returned to callers."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    chunk_id: str
    document_id: str
    content: str
    source_path: str
    chunk_type: str
    topic: str | None = None
    risk_level: str | None = None
    heading: str
    heading_path: list[str]
    section: str | None = None
    subsection: str | None = None
    similarity: float
    distance: float


class SearchResponse(BaseModel):
    """Structured search output with diagnostics."""

    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int
    fetch_k: int
    similarity_threshold: float
    results: list[SearchResult]
    candidates_fetched: int
    candidates_above_threshold: int
    embedding_model: str
    collection_name: str


class IndexBuildReport(BaseModel):
    """Summary of a successful index build."""

    model_config = ConfigDict(extra="forbid")

    documents: int
    chunks: int
    embedding_model: str
    collection_name: str
    index_dir: str
    vector_dimension: int
    fingerprint: str
    elapsed_seconds: float
    status: str = "success"
