"""Data models for vector pool expansion A/B evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from customer_claims_rag.evaluation.ab_models import ArmMetrics, RankedCandidateAudit
from customer_claims_rag.evaluation.models import AggregateMetrics, EvaluationRun

ExperimentMode = Literal["production-like"]
ReachabilityVerdict = Literal["accepted", "rejected"]
RankingVerdict = Literal["improved", "neutral", "rejected"]
FailureClassification = Literal[
    "none",
    "fallback_case",
    "fully_unreachable_in_baseline_pool",
    "fully_unreachable_in_candidate_pool",
    "reachable_only_beyond_selected_pool",
    "candidate_generation_fixed_ranking_limited",
    "already_reachable_at_baseline",
]


class PoolExpansionExperimentConfig(BaseModel):
    """Versioned pool expansion experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    version: str
    experiment_mode: ExperimentMode
    baseline_pool_k: int
    candidate_pool_k: int
    final_top_k: int
    threshold: float
    reranker_id: str
    reranker_config_hash: str
    tie_breaking: list[str]
    config_path: str | None = None


class ExperimentMetadata(BaseModel):
    """Experiment metadata stored in pool expansion artifacts."""

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
    """Shared retrieval context for both arms."""

    model_config = ConfigDict(extra="forbid")

    index_fingerprint: str
    embedding_model: str
    evaluation_dataset_fingerprint: str
    case_count: int
    shared_fetch_k: int
    baseline_pool_k: int
    candidate_pool_k: int
    final_top_k: int
    threshold: float
    collection: str
    chunk_count: int | None = None
    document_count: int | None = None


class PoolReachability(BaseModel):
    """Reachability diagnostics for one candidate pool prefix."""

    model_config = ConfigDict(extra="forbid")

    primary_reachable: bool = False
    supporting_reachable: bool = False
    any_expected_source_reachable: bool = False
    primary_best_pool_rank: int | None = None
    supporting_best_pool_rank: int | None = None
    fully_unreachable: bool = False


class FinalRankingFlags(BaseModel):
    """Post-rerank outcome flags for one arm."""

    model_config = ConfigDict(extra="forbid")

    primary_in_final_top12: bool = False
    supporting_in_final_top12: bool = False
    pool_reachable_but_not_in_final_top12: bool = False
    final_top12_but_not_top4: bool = False


class RankingArmResult(BaseModel):
    """Final ranked output for one arm after reranking."""

    model_config = ConfigDict(extra="forbid")

    ordered_candidates: list[RankedCandidateAudit] = Field(default_factory=list)
    metrics: ArmMetrics
    flags: FinalRankingFlags = Field(default_factory=FinalRankingFlags)


class PoolArmSnapshot(BaseModel):
    """Pre-rerank pool snapshot for one arm."""

    model_config = ConfigDict(extra="forbid")

    pool_k: int
    candidate_ids: list[str] = Field(default_factory=list)
    reachability: PoolReachability


class CaseComparisonDetail(BaseModel):
    """Per-case pool and ranking comparison."""

    model_config = ConfigDict(extra="forbid")

    pool_promotions: list[str] = Field(default_factory=list)
    pool_regressions: list[str] = Field(default_factory=list)
    ranking_promotions: list[str] = Field(default_factory=list)
    ranking_regressions: list[str] = Field(default_factory=list)
    primary_hit_at_4_delta: int = 0
    supporting_hit_at_4_delta: int = 0
    mrr_delta: float = 0.0
    baseline_pool_is_prefix_of_candidate_pool: bool = True


class PoolExpansionCaseResult(BaseModel):
    """Combined per-case pool expansion record."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    risk: str
    category: str | None = None
    query: str
    expected_primary_documents: list[str] = Field(default_factory=list)
    expected_supporting_documents: list[str] = Field(default_factory=list)
    fallback_expected: bool = False
    shared_deep_candidate_ids: list[str] = Field(default_factory=list)
    baseline_pool: PoolArmSnapshot
    candidate_pool: PoolArmSnapshot
    baseline_ranking: RankingArmResult
    candidate_ranking: RankingArmResult
    comparison: CaseComparisonDetail
    failure_classification: FailureClassification = "none"


class ReachabilitySliceComparison(BaseModel):
    """Reachability counts for one risk slice."""

    model_config = ConfigDict(extra="forbid")

    risk_level: str
    case_count: int
    baseline_primary_reachable: int
    candidate_primary_reachable: int
    baseline_primary_total: int
    candidate_primary_total: int


class ReachabilityComparison(BaseModel):
    """Aggregate reachability comparison."""

    model_config = ConfigDict(extra="forbid")

    baseline_primary_reachable: int
    baseline_primary_total: int
    candidate_primary_reachable: int
    candidate_primary_total: int
    baseline_supporting_reachable: int
    baseline_supporting_total: int
    candidate_supporting_reachable: int
    candidate_supporting_total: int
    baseline_fully_unreachable_cases: list[str] = Field(default_factory=list)
    candidate_fully_unreachable_cases: list[str] = Field(default_factory=list)
    high_baseline_primary_reachable: int
    high_candidate_primary_reachable: int
    high_primary_total: int
    critical_baseline_primary_reachable: int
    critical_candidate_primary_reachable: int
    critical_primary_total: int
    risk_slices: list[ReachabilitySliceComparison] = Field(default_factory=list)


class RankingAggregateComparison(BaseModel):
    """Aggregate ranking metric deltas."""

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


class RiskRankingComparison(BaseModel):
    """Risk-slice ranking comparison."""

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


class RankingComparison(BaseModel):
    """Top-level ranking comparison."""

    model_config = ConfigDict(extra="forbid")

    aggregate: RankingAggregateComparison
    risk_slices: list[RiskRankingComparison] = Field(default_factory=list)
    primary_hit_at_4_promotions: list[str] = Field(default_factory=list)
    primary_hit_at_4_regressions: list[str] = Field(default_factory=list)


class AcceptanceCriteriaResult(BaseModel):
    """Dual verdict acceptance outcome."""

    model_config = ConfigDict(extra="forbid")

    hard_invariants_pass: bool
    shared_prefix_pass: bool
    single_retrieval_pass: bool
    technical_errors_zero: bool
    reranker_config_hash_match: bool
    reachability_verdict: ReachabilityVerdict
    reachability_pass: bool
    critical_primary_reachability_improved: bool
    high_primary_reachability_non_regressed: bool
    fully_unreachable_decreased: bool
    high_critical_reachability_regressions: int
    ranking_verdict: RankingVerdict
    ranking_pass: bool
    overall_primary_hit_at_4_delta: float
    overall_mrr_delta: float
    critical_primary_regressions: int
    failure_reasons: list[str] = Field(default_factory=list)


class PoolExpansionEvaluationRun(BaseModel):
    """Complete pool expansion A/B artifact."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    experiment: ExperimentMetadata
    shared_context: SharedContext
    baseline_pool: EvaluationRun
    candidate_pool: EvaluationRun
    baseline_ranking: EvaluationRun
    candidate_ranking: EvaluationRun
    reachability_comparison: ReachabilityComparison
    ranking_comparison: RankingComparison
    case_results: list[PoolExpansionCaseResult] = Field(default_factory=list)
    acceptance: AcceptanceCriteriaResult
