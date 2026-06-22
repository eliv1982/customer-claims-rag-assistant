"""Data models for hybrid lexical + vector A/B evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from customer_claims_rag.evaluation.ab_models import ArmMetrics
from customer_claims_rag.evaluation.models import AggregateMetrics, EvaluationRun
from customer_claims_rag.evaluation.pool_expansion_models import (
    CaseComparisonDetail,
    FinalRankingFlags,
    PoolReachability,
    RankingAggregateComparison,
    RankingComparison,
    RiskRankingComparison,
)

ExperimentMode = Literal["production-like"]
ReachabilityVerdict = Literal["accepted", "rejected"]
RankingVerdict = Literal["improved", "neutral", "rejected"]
RetrievalChannel = Literal["both", "vector_only", "lexical_only"]
PrimaryRetrievalChannel = Literal["both", "vector_only", "lexical_only", "neither"]


class HybridExperimentConfig(BaseModel):
    """Versioned hybrid retrieval experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    version: str
    experiment_mode: ExperimentMode
    vector_k: int
    lexical_k: int
    fusion_k: int
    rrf_k: int
    vector_weight: float
    lexical_weight: float
    final_top_k: int
    threshold: float
    lexical_algorithm: str
    bm25_k1: float
    bm25_b: float
    reranker_id: str
    reranker_config_hash: str
    score_adapter_id: str
    tie_breaking: list[str]
    config_path: str | None = None


class ExperimentMetadata(BaseModel):
    """Experiment metadata stored in hybrid artifacts."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    version: str
    experiment_mode: ExperimentMode
    config: dict[str, Any]
    config_hash: str
    reranker_id: str
    reranker_config_hash: str
    score_adapter_id: str
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
    vector_k: int
    lexical_k: int
    fusion_k: int
    final_top_k: int
    threshold: float
    collection: str
    chunk_count: int | None = None
    document_count: int | None = None
    vector_retrieval_calls_per_case: int = 1


class LexicalIndexMetadata(BaseModel):
    """Metadata about the lexical index used in the experiment."""

    model_config = ConfigDict(extra="forbid")

    lexical_algorithm: str
    tokenizer_version: str
    bm25_k1: float
    bm25_b: float
    chunk_count: int
    corpus_fingerprint: str
    lexical_index_fingerprint: str
    document_text_template: str = "heading + newline + content"


class HybridCandidateAudit(BaseModel):
    """Audit record for one hybrid reranked candidate."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    chunk_type: str
    vector_rank: int | None = None
    vector_similarity: float | None = None
    vector_distance: float | None = None
    lexical_rank: int | None = None
    bm25_score: float | None = None
    matched_tokens: list[str] = Field(default_factory=list)
    retrieval_channel: RetrievalChannel
    raw_rrf_score: float
    normalized_rrf_score: float
    fusion_rank: int
    source_authority_bonus: float
    final_rerank_score: float
    final_rank: int


class BaselineCandidateAudit(BaseModel):
    """Audit record for baseline arm using vector similarity."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    chunk_type: str
    vector_rank: int
    vector_similarity: float
    vector_distance: float
    source_authority_bonus: float
    final_rerank_score: float
    final_rank: int


class HybridRankingArmResult(BaseModel):
    """Final ranked output for one arm."""

    model_config = ConfigDict(extra="forbid")

    ordered_candidates: list[BaselineCandidateAudit | HybridCandidateAudit] = Field(default_factory=list)
    metrics: ArmMetrics
    flags: FinalRankingFlags = Field(default_factory=FinalRankingFlags)


class FusionPoolSnapshot(BaseModel):
    """Pre-rerank fusion pool snapshot for candidate arm."""

    model_config = ConfigDict(extra="forbid")

    pool_k: int
    candidate_ids: list[str] = Field(default_factory=list)
    reachability: PoolReachability
    vector_reachable: bool = False
    lexical_reachable: bool = False
    retrieval_channel: PrimaryRetrievalChannel | None = None
    vector_only_reachable: bool = False
    lexical_only_reachable: bool = False
    both_reachable: bool = False


class VectorPoolSnapshot(BaseModel):
    """Pre-rerank vector pool snapshot for baseline arm."""

    model_config = ConfigDict(extra="forbid")

    pool_k: int
    candidate_ids: list[str] = Field(default_factory=list)
    reachability: PoolReachability


class LexicalPoolCandidate(BaseModel):
    """Single ordered lexical pool candidate."""

    model_config = ConfigDict(extra="forbid")

    lexical_rank: int
    chunk_id: str
    document_id: str
    bm25_score: float
    matched_tokens: list[str] = Field(default_factory=list)


class LexicalPoolSnapshot(BaseModel):
    """Pre-rerank lexical pool snapshot for candidate arm."""

    model_config = ConfigDict(extra="forbid")

    pool_k: int
    candidate_ids: list[str] = Field(default_factory=list)
    candidates: list[LexicalPoolCandidate] = Field(default_factory=list)
    exact: bool = False


class ChannelReachabilityComparison(BaseModel):
    """Candidate-generation channel reachability metrics."""

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
    vector_only_reachable_cases: list[str] = Field(default_factory=list)
    lexical_only_reachable_cases: list[str] = Field(default_factory=list)
    both_reachable_cases: list[str] = Field(default_factory=list)
    neither_reachable_cases: list[str] = Field(default_factory=list)
    high_baseline_primary_reachable: int
    high_candidate_primary_reachable: int
    high_primary_total: int
    critical_baseline_primary_reachable: int
    critical_candidate_primary_reachable: int
    critical_primary_total: int


class AcceptanceCriteriaResult(BaseModel):
    """Dual verdict acceptance outcome for hybrid experiment."""

    model_config = ConfigDict(extra="forbid")

    hard_invariants_pass: bool
    single_retrieval_pass: bool
    technical_errors_zero: bool
    reranker_config_hash_match: bool
    lexical_index_valid: bool
    reachability_verdict: ReachabilityVerdict
    reachability_pass: bool
    high_primary_reachability_non_regressed: bool
    critical_primary_reachability_non_regressed: bool
    fully_unreachable_non_increased: bool
    high_critical_reachability_regressions: int
    ranking_verdict: RankingVerdict
    ranking_pass: bool
    overall_primary_hit_at_4_delta: float
    overall_mrr_delta: float
    critical_primary_regressions: int
    low_medium_slice_guardrails_pass: bool
    failure_reasons: list[str] = Field(default_factory=list)


class HybridCaseResult(BaseModel):
    """Combined per-case hybrid experiment record."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    risk: str
    category: str | None = None
    query: str
    expected_primary_documents: list[str] = Field(default_factory=list)
    expected_supporting_documents: list[str] = Field(default_factory=list)
    fallback_expected: bool = False
    shared_vector_candidate_ids: list[str] = Field(default_factory=list)
    baseline_pool: VectorPoolSnapshot
    lexical_pool: LexicalPoolSnapshot | None = None
    candidate_fusion_pool: FusionPoolSnapshot
    baseline_ranking: HybridRankingArmResult
    candidate_ranking: HybridRankingArmResult
    comparison: CaseComparisonDetail


class HybridEvaluationRun(BaseModel):
    """Complete hybrid lexical + vector A/B artifact."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    experiment: ExperimentMetadata
    shared_context: SharedContext
    lexical_index: LexicalIndexMetadata
    baseline: EvaluationRun
    candidate: EvaluationRun
    candidate_generation_comparison: ChannelReachabilityComparison
    ranking_comparison: RankingComparison
    case_results: list[HybridCaseResult] = Field(default_factory=list)
    acceptance: AcceptanceCriteriaResult
