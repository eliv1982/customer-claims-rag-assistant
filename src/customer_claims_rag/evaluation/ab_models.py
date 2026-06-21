"""Data models for reranking A/B evaluation artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from customer_claims_rag.evaluation.models import (
    AggregateMetrics,
    CaseResult,
    CategorySliceMetrics,
    EvaluationRun,
    RiskSliceMetrics,
    RunMetadata,
)

ExperimentMode = Literal["production-like"]
CaseIdentityClass = Literal[
    "exact",
    "same_order_float_drift",
    "same_set_different_order",
    "substantive_mismatch",
]
BaselineIdentityStatus = Literal[
    "exact",
    "equivalent_with_embedding_drift",
    "substantive_mismatch",
]


class ExperimentMetadata(BaseModel):
    """Experiment-level metadata stored in A/B artifacts."""

    model_config = ConfigDict(extra="forbid")

    reranker_id: str
    version: str
    experiment_mode: ExperimentMode
    config: dict[str, Any]
    config_hash: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    git_status_summary: str | None = None


class SharedContext(BaseModel):
    """Shared context for both A/B arms."""

    model_config = ConfigDict(extra="forbid")

    index_fingerprint: str
    embedding_model: str
    evaluation_dataset_fingerprint: str
    case_count: int
    fetch_k: int
    top_k: int
    threshold: float
    collection: str
    chunk_count: int | None = None
    document_count: int | None = None


class RankedCandidateAudit(BaseModel):
    """Audit record for a single ranked candidate."""

    model_config = ConfigDict(extra="forbid")

    baseline_rank: int | None = None
    candidate_rank: int | None = None
    chunk_id: str
    document_id: str
    chunk_type: str
    similarity: float
    distance: float
    source_authority_bonus: float | None = None
    rerank_score: float | None = None


class ArmMetrics(BaseModel):
    """Metrics for one evaluation arm."""

    model_config = ConfigDict(extra="forbid")

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
    status: str = "success"
    error_message: str | None = None


class CaseArmResult(BaseModel):
    """One arm's ranked output for a case."""

    model_config = ConfigDict(extra="forbid")

    ordered_candidates: list[RankedCandidateAudit] = Field(default_factory=list)
    metrics: ArmMetrics


class CaseComparison(BaseModel):
    """Per-case comparison between baseline and candidate arms."""

    model_config = ConfigDict(extra="forbid")

    promotions: list[str] = Field(default_factory=list)
    demotions: list[str] = Field(default_factory=list)
    rank_changes: dict[str, int] = Field(default_factory=dict)
    primary_hit_at_4_delta: int = 0
    supporting_hit_at_4_delta: int = 0
    mrr_delta: float = 0.0
    shared_pool_match: bool = True
    primary_candidate_generation_failure: bool = False
    supporting_candidate_generation_failure: bool = False
    candidate_generation_gap: bool = False
    reranker_not_applicable: bool = False


class AbCaseResult(BaseModel):
    """Combined per-case A/B record."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    risk: str
    category: str | None = None
    query: str
    expected_primary_documents: list[str] = Field(default_factory=list)
    expected_supporting_documents: list[str] = Field(default_factory=list)
    fallback_expected: bool = False
    shared_candidate_ids: list[str] = Field(default_factory=list)
    baseline: CaseArmResult
    candidate: CaseArmResult
    comparison: CaseComparison


class AggregateComparison(BaseModel):
    """Aggregate metric deltas between arms."""

    model_config = ConfigDict(extra="forbid")

    hit_rate_at_1_delta: float = 0.0
    hit_rate_at_4_delta: float = 0.0
    hit_rate_at_12_delta: float = 0.0
    document_recall_at_1_delta: float = 0.0
    document_recall_at_4_delta: float = 0.0
    document_recall_at_12_delta: float = 0.0
    mrr_delta: float = 0.0
    primary_source_hit_rate_at_1_delta: float = 0.0
    primary_source_hit_rate_at_4_delta: float = 0.0
    supporting_source_hit_rate_at_4_delta: float | None = None


class RiskComparison(BaseModel):
    """Risk-slice comparison."""

    model_config = ConfigDict(extra="forbid")

    risk_level: str
    case_count: int
    baseline_hit_rate_at_4: float
    candidate_hit_rate_at_4: float
    hit_rate_at_4_delta: float
    baseline_mrr: float
    candidate_mrr: float
    mrr_delta: float
    baseline_primary_hit_rate_at_4: float
    candidate_primary_hit_rate_at_4: float
    primary_hit_rate_at_4_delta: float


class CategoryComparison(BaseModel):
    """Category-slice comparison."""

    model_config = ConfigDict(extra="forbid")

    category: str
    case_count: int
    baseline_hit_rate_at_4: float
    candidate_hit_rate_at_4: float
    hit_rate_at_4_delta: float
    baseline_mrr: float
    candidate_mrr: float
    mrr_delta: float
    baseline_primary_hit_rate_at_4: float
    candidate_primary_hit_rate_at_4: float
    primary_hit_rate_at_4_delta: float


class DocumentMovementSummary(BaseModel):
    """Document-level promotion/demotion counts."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    promotions: int = 0
    demotions: int = 0
    net_rank_change: int = 0


class FaqPolicyMovement(BaseModel):
    """FAQ vs policy chunk movement summary."""

    model_config = ConfigDict(extra="forbid")

    faq_chunks_promoted: int = 0
    faq_chunks_demoted: int = 0
    policy_chunks_promoted: int = 0
    policy_chunks_demoted: int = 0
    escalation_chunks_promoted: int = 0
    escalation_chunks_demoted: int = 0


class AcceptanceCriteriaResult(BaseModel):
    """Acceptance criteria evaluation outcome."""

    model_config = ConfigDict(extra="forbid")

    hard_invariants_pass: bool
    baseline_identity_pass: bool
    shared_pool_pass: bool
    technical_errors_zero: bool
    high_critical_net_primary_promotions: int
    high_critical_net_primary_promotions_pass: bool
    critical_primary_regressions: int
    critical_primary_regressions_pass: bool
    overall_primary_hit_at_4_delta: float
    overall_primary_hit_at_4_pass: bool
    overall_mrr_delta: float
    overall_mrr_pass: bool
    low_hit_at_4_regressions: int
    low_hit_at_4_regressions_pass: bool
    low_mrr_delta: float
    low_mrr_pass: bool
    medium_hit_at_4_regressions: int
    medium_hit_at_4_regressions_pass: bool
    medium_mrr_delta: float
    medium_mrr_pass: bool
    fallback_status_preserved: bool
    candidate_accepted: bool
    failure_reasons: list[str] = Field(default_factory=list)


class AbComparison(BaseModel):
    """Top-level comparison section."""

    model_config = ConfigDict(extra="forbid")

    aggregate: AggregateComparison
    risk_slices: list[RiskComparison] = Field(default_factory=list)
    category_slices: list[CategoryComparison] = Field(default_factory=list)
    document_movements: list[DocumentMovementSummary] = Field(default_factory=list)
    faq_policy_movement: FaqPolicyMovement = Field(default_factory=FaqPolicyMovement)
    acceptance: AcceptanceCriteriaResult


class CaseIdentityClassification(BaseModel):
    """Per-case baseline identity classification."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    classification: CaseIdentityClass


class BaselineIdentityReport(BaseModel):
    """Baseline arm comparison against frozen 2B artifact."""

    model_config = ConfigDict(extra="forbid")

    frozen_evaluation_result_id: str
    computed_evaluation_result_id: str
    frozen_index_fingerprint: str
    evaluation_result_id_match: bool
    case_count: int
    exact_count: int
    same_order_float_drift_count: int
    same_set_different_order_count: int
    substantive_mismatch_count: int
    case_classifications: list[CaseIdentityClassification] = Field(default_factory=list)
    max_similarity_drift: float
    max_distance_drift: float
    similarity_tolerance: float
    shared_pool_exact_match: bool
    frozen_metrics_match: bool
    frozen_exact_order_match: bool
    identity_status: BaselineIdentityStatus
    per_case_match: bool
    mismatched_cases: list[str] = Field(default_factory=list)


class AbEvaluationRun(BaseModel):
    """Complete reranking A/B evaluation artifact."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    experiment: ExperimentMetadata
    shared_context: SharedContext
    baseline: EvaluationRun
    candidate: EvaluationRun
    comparison: AbComparison
    case_results: list[AbCaseResult] = Field(default_factory=list)
    baseline_identity: BaselineIdentityReport
