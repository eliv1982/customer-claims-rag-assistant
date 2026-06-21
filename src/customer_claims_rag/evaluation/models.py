"""Evaluation data models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "medium", "high", "critical"]
EvaluationStatus = Literal["success", "technical_error", "no_grounding"]


class EvaluationCase(BaseModel):
    """Single evaluation case parsed from Markdown corpus."""

    model_config = ConfigDict(extra="forbid")

    test_id: str
    title: str
    query: str
    category: str | None = None
    check_types: list[str] = Field(default_factory=list)
    complexity: str | None = None
    has_dialog_history: bool = False
    expected_risk: RiskLevel
    expected_primary_documents: list[str] = Field(default_factory=list)
    expected_supporting_documents: list[str] = Field(default_factory=list)
    fallback_expected: bool = False
    notes: str | None = None


class RetrievedChunkResult(BaseModel):
    """Chunk-level retrieval output stored for metrics and threshold analysis."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    chunk_id: str
    document_id: str
    source_path: str
    similarity: float
    distance: float
    heading: str


class CaseResult(BaseModel):
    """Evaluation outcome for a single test case."""

    model_config = ConfigDict(extra="forbid")

    test_id: str
    query: str
    expected_risk: RiskLevel
    expected_primary_documents: list[str]
    expected_supporting_documents: list[str]
    fallback_expected: bool
    category: str | None = None
    retrieved_chunks: list[RetrievedChunkResult] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids_at_4: list[str] = Field(default_factory=list)
    top1_similarity: float | None = None
    hit_at_1: bool = False
    hit_at_4: bool = False
    hit_at_12: bool = False
    document_recall_at_1: float = 0.0
    document_recall_at_4: float = 0.0
    document_recall_at_12: float = 0.0
    reciprocal_rank: float = 0.0
    primary_hit_at_1: bool = False
    primary_hit_at_4: bool = False
    supporting_hit_at_4: bool = False
    expected_in_top12_outside_top4: bool = False
    missing_expected_sources: list[str] = Field(default_factory=list)
    unexpected_top_sources: list[str] = Field(default_factory=list)
    status: EvaluationStatus = "success"
    error_message: str | None = None


class AggregateMetrics(BaseModel):
    """Aggregate retrieval metrics."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    successfully_evaluated_cases: int
    technical_error_count: int
    source_recall_case_count: int
    fallback_case_count: int
    hit_rate_at_1: float
    hit_rate_at_4: float
    hit_rate_at_12: float
    document_recall_at_1: float
    document_recall_at_4: float
    document_recall_at_12: float
    mrr: float
    primary_source_hit_rate_at_1: float
    primary_source_hit_rate_at_4: float
    cases_with_supporting_documents: int
    supporting_source_hits_at_4: int
    supporting_source_hit_rate_at_4: float | None = None
    no_result_rate: float
    mean_top1_similarity: float | None = None
    median_top1_similarity: float | None = None
    min_top1_similarity: float | None = None
    max_top1_similarity: float | None = None


class RiskSliceMetrics(BaseModel):
    """Retrieval metrics grouped by expected risk level."""

    model_config = ConfigDict(extra="forbid")

    risk_level: RiskLevel
    case_count: int
    hit_rate_at_1: float
    hit_rate_at_4: float
    hit_rate_at_12: float
    mrr: float
    mean_top1_similarity: float | None = None


class CategorySliceMetrics(BaseModel):
    """Retrieval metrics grouped by evaluation category tag."""

    model_config = ConfigDict(extra="forbid")

    category: str
    case_count: int
    hit_rate_at_1: float
    hit_rate_at_4: float
    hit_rate_at_12: float
    mrr: float
    mean_top1_similarity: float | None = None


class ThresholdSliceMetrics(BaseModel):
    """Metrics at a single similarity threshold from post-hoc filtering."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    cases_with_at_least_one_result: int
    no_result_rate: float
    hit_rate_at_1: float
    hit_rate_at_4: float
    primary_source_hit_rate_at_4: float
    mrr: float
    critical_case_hit_rate_at_4: float
    high_case_hit_rate_at_4: float


class RunMetadata(BaseModel):
    """Metadata describing an evaluation run."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    evaluation_result_id: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    git_status_summary: str | None = None
    index_fingerprint: str | None = None
    index_format_version: str | None = None
    metadata_schema_version: str | None = None
    embedding_model: str
    vector_dimension: int | None = None
    collection: str
    chunk_count: int | None = None
    document_count: int | None = None
    evaluation_case_count: int
    threshold: float
    top_k: int
    fetch_k: int


class EvaluationRun(BaseModel):
    """Complete evaluation run output."""

    model_config = ConfigDict(extra="forbid")

    run_metadata: RunMetadata
    aggregate_metrics: AggregateMetrics
    risk_metrics: list[RiskSliceMetrics]
    category_metrics: list[CategorySliceMetrics]
    threshold_analysis: list[ThresholdSliceMetrics]
    case_results: list[CaseResult]
    technical_errors: list[str] = Field(default_factory=list)
    fallback_analysis: dict[str, float | int | None] = Field(default_factory=dict)
