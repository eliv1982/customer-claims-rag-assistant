"""Data models for vector pool diversity (cap) A/B evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from customer_claims_rag.evaluation.ab_models import ArmMetrics, RankedCandidateAudit
from customer_claims_rag.evaluation.models import AggregateMetrics, EvaluationRun
from customer_claims_rag.evaluation.pool_expansion_models import (
    CaseComparisonDetail,
    FinalRankingFlags,
    PoolReachability,
    RankingArmResult,
    ReachabilityComparison,
)
from customer_claims_rag.retrieval.models import SearchResult

ExperimentMode = Literal["production-like"]
DiversityVerdict = Literal[
    "ACCEPTED AS PARTIAL CANDIDATE-GENERATION REPAIR",
    "REJECTED",
]


class VectorPoolCapExperimentConfig(BaseModel):
    """Versioned vector pool cap experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    version: str
    experiment_mode: ExperimentMode
    baseline_fetch_k: int
    baseline_candidate_pool_k: int
    baseline_per_document_cap: int | None
    candidate_fetch_k: int
    candidate_candidate_pool_k: int
    candidate_per_document_cap: int
    final_top_k: int
    threshold: float
    reranker_id: str
    reranker_config_hash: str
    tie_breaking: list[str]
    config_path: str | None = None


class RetrievalArmConfig(BaseModel):
    """Resolved retrieval settings for one experiment arm."""

    model_config = ConfigDict(extra="forbid")

    fetch_k: int
    candidate_pool_k: int
    per_document_cap: int | None = None


class ExperimentMetadata(BaseModel):
    """Experiment metadata stored in diversity artifacts."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    version: str
    experiment_mode: ExperimentMode
    config: dict[str, Any]
    config_hash: str
    reranker_id: str
    reranker_config_hash: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    git_status_summary: str | None = None


class SharedContext(BaseModel):
    """Shared evaluation context for both arms."""

    model_config = ConfigDict(extra="forbid")

    index_fingerprint: str
    embedding_model: str
    evaluation_dataset_fingerprint: str
    case_count: int
    final_top_k: int
    threshold: float
    collection: str
    chunk_count: int | None = None
    document_count: int | None = None
    baseline_arm: RetrievalArmConfig
    candidate_arm: RetrievalArmConfig


class CappedPoolEntry(BaseModel):
    """One capped-pool item with preserved vector rank."""

    model_config = ConfigDict(extra="forbid")

    result: SearchResult
    vector_rank: int


class CapPoolDiagnostics(BaseModel):
    """Per-case cap and pool shaping diagnostics."""

    model_config = ConfigDict(extra="forbid")

    pool_size: int
    removed_by_cap_count: int
    cap_applied: bool
    unique_document_count: int
    max_chunks_from_one_document: int
    document_counts: dict[str, int] = Field(default_factory=dict)
    pool_shorter_than_target: bool = False
    entries: list[CappedPoolEntry] = Field(default_factory=list)


class PoolArmSnapshot(BaseModel):
    """Pre-rerank pool snapshot for one arm."""

    model_config = ConfigDict(extra="forbid")

    fetch_k: int
    pool_k: int
    per_document_cap: int | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    reachability: PoolReachability
    cap_diagnostics: CapPoolDiagnostics | None = None


class DiversityCaseResult(BaseModel):
    """Combined per-case diversity experiment record."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    risk: str
    category: str | None = None
    query: str
    expected_primary_documents: list[str] = Field(default_factory=list)
    expected_supporting_documents: list[str] = Field(default_factory=list)
    fallback_expected: bool = False
    baseline_vector_chunk_ids: list[str] = Field(default_factory=list)
    candidate_vector_chunk_ids: list[str] = Field(default_factory=list)
    baseline_vector_ranks_by_document: dict[str, list[int]] = Field(default_factory=dict)
    candidate_vector_ranks_by_document: dict[str, list[int]] = Field(default_factory=dict)
    baseline_pool: PoolArmSnapshot
    candidate_pool: PoolArmSnapshot
    baseline_ranking: RankingArmResult
    candidate_ranking: RankingArmResult
    comparison: CaseComparisonDetail
    faq_in_baseline_top4: bool = False
    faq_in_candidate_top4: bool = False
    faq_in_top4_without_primary_baseline: bool = False
    faq_in_top4_without_primary_candidate: bool = False


class SaturationMetrics(BaseModel):
    """Aggregate saturation and diversity metrics for one arm."""

    model_config = ConfigDict(extra="forbid")

    average_unique_documents_in_pool: float
    average_max_chunks_from_one_document: float
    questions_with_document_count_ge_cap: int
    questions_where_cap_removed_chunk: int
    total_removed_by_cap_chunks: int
    average_pool_size_after_cap: float
    questions_with_pool_shorter_than_target: int


class FaqComparison(BaseModel):
    """FAQ presence comparison across arms."""

    model_config = ConfigDict(extra="forbid")

    baseline_questions_with_faq_in_top4: int
    candidate_questions_with_faq_in_top4: int
    baseline_faq_top4_without_primary: int
    candidate_faq_top4_without_primary: int
    faq_top4_case_ids_changed_vs_baseline: list[str] = Field(default_factory=list)


class NewDocumentFootprint(BaseModel):
    """Presence of new corpus documents in pools and finals."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    pools_containing_document: int
    final_top12_containing_document: int


class RequiredCaseDiagnostic(BaseModel):
    """Mandatory per-case diagnostic for selected benchmark cases."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    expected_primary_documents: list[str] = Field(default_factory=list)
    baseline_vector_ranks_by_document: dict[str, list[int]] = Field(default_factory=dict)
    candidate_vector_ranks_by_document: dict[str, list[int]] = Field(default_factory=dict)
    baseline_pool_ranks_by_document: dict[str, list[int]] = Field(default_factory=dict)
    candidate_pool_ranks_by_document: dict[str, list[int]] = Field(default_factory=dict)
    candidate_pool_document_counts: dict[str, int] = Field(default_factory=dict)
    baseline_final_ranks_by_document: dict[str, list[int]] = Field(default_factory=dict)
    candidate_final_ranks_by_document: dict[str, list[int]] = Field(default_factory=dict)
    baseline_primary_hit_at_4: bool = False
    candidate_primary_hit_at_4: bool = False
    baseline_primary_hit_at_12: bool = False
    candidate_primary_hit_at_12: bool = False
    baseline_primary_reachable: bool = False
    candidate_primary_reachable: bool = False
    change_explanation: str = ""


class AcceptanceCheck(BaseModel):
    """Single acceptance criterion evaluation."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str
    passed: bool
    baseline_value: str | None = None
    candidate_value: str | None = None
    hard_regression: bool = False


class HardRegressionFlags(BaseModel):
    """Explicit hard regression indicators."""

    model_config = ConfigDict(extra="forbid")

    primary_hit_at_4_below_baseline: bool = False
    mrr_below_floor: bool = False
    critical_pool_reach_below_floor: bool = False
    faq_top4_above_ceiling: bool = False
    new_unreachable_primary_outside_allowed: bool = False


class InterpretationBoundary(BaseModel):
    """Post-verdict interpretation constraints."""

    model_config = ConfigDict(extra="forbid")

    candidate_is_partial_repair_only: bool = True
    candidate_not_release_ready: bool = True
    no_automatic_production_promotion: bool = True
    production_config_unchanged: bool = True
    unresolved_cases: list[str] = Field(default_factory=list)
    primary_hit_at_4_not_restored_to_arm_a: bool = True
    next_stage_required: str = (
        "Stage 4C.3C ranking/metadata refinement or targeted corpus repair"
    )


class DiversityEvaluationRun(BaseModel):
    """Complete vector pool cap A/B artifact."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    experiment: ExperimentMetadata
    shared_context: SharedContext
    baseline_ranking: EvaluationRun
    candidate_ranking: EvaluationRun
    reachability_comparison: ReachabilityComparison
    baseline_saturation: SaturationMetrics
    candidate_saturation: SaturationMetrics
    faq_comparison: FaqComparison
    new_document_footprint: list[NewDocumentFootprint] = Field(default_factory=list)
    required_case_diagnostics: list[RequiredCaseDiagnostic] = Field(default_factory=list)
    acceptance_checks: list[AcceptanceCheck] = Field(default_factory=list)
    hard_regressions: HardRegressionFlags
    interpretation_boundary: InterpretationBoundary
    case_results: list[DiversityCaseResult] = Field(default_factory=list)
    verdict: DiversityVerdict
    latency_seconds_total: float | None = None
