"""Data models for doc08 atomic corpus A/B experiment."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from customer_claims_rag.evaluation.models import AggregateMetrics, CaseResult
from customer_claims_rag.evaluation.pool_expansion_models import (
    PoolReachability,
    RankingArmResult,
    ReachabilityComparison,
)
from customer_claims_rag.ingestion.corpus_overlay import Doc08ChunkDiff


class Doc08ChunkDiffModel(BaseModel):
    """JSON-serializable chunk diff."""

    model_config = ConfigDict(extra="forbid")

    baseline_chunk_ids: list[str]
    candidate_chunk_ids: list[str]
    added_chunk_ids: list[str]
    removed_chunk_ids: list[str]
    changed_chunk_ids: list[str]
    unchanged_non_doc08_chunk_ids: list[str]
    non_doc08_byte_identical: bool

    @classmethod
    def from_dataclass(cls, diff: Doc08ChunkDiff) -> "Doc08ChunkDiffModel":
        return cls(
            baseline_chunk_ids=diff.baseline_chunk_ids,
            candidate_chunk_ids=diff.candidate_chunk_ids,
            added_chunk_ids=diff.added_chunk_ids,
            removed_chunk_ids=diff.removed_chunk_ids,
            changed_chunk_ids=diff.changed_chunk_ids,
            unchanged_non_doc08_chunk_ids=diff.unchanged_non_doc08_chunk_ids,
            non_doc08_byte_identical=diff.non_doc08_byte_identical,
        )

Doc08Verdict = Literal["ACCEPTED AS TARGETED CORPUS REPAIR", "REJECTED"]


class RetrievalArmMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arm: Literal["baseline", "candidate"]
    index_dir: str
    index_fingerprint: str
    corpus_fingerprint: str | None = None
    doc08_fingerprint: str | None = None
    chunk_count: int
    document_count: int
    fetch_k: int
    candidate_pool_k: int
    per_document_cap: int | None
    final_top_k: int
    threshold: float
    reranker_id: str
    reranker_config_hash: str
    collection: str
    embedding_model: str
    benchmark_fingerprint: str | None = None
    execution_timestamp: datetime | None = None
    chunk_payload_digest: str | None = None
    embedding_digest: str | None = None
    collection_content_digest: str | None = None
    build_run_id: str | None = None


class FrozenReachabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_reachable: int
    primary_denominator: int
    primary_unreachable_case_ids: list[str] = Field(default_factory=list)
    high_risk_reachable: int
    high_risk_denominator: int
    critical_reachable: int
    critical_denominator: int
    fully_unreachable_case_ids: list[str] = Field(default_factory=list)


class BaselineReproductionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_experiment_id: str
    reference_arm: str
    reference_artifact_path: str
    passed: bool
    checks: list["AcceptanceCheck"] = Field(default_factory=list)


class ExtensionCaseDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    query_preview: str
    expected_primary: list[str] = Field(default_factory=list)
    expected_supporting: list[str] = Field(default_factory=list)
    baseline_vector_ranks_doc08: list[int] = Field(default_factory=list)
    candidate_vector_ranks_doc08: list[int] = Field(default_factory=list)
    baseline_vector_ranks_doc12: list[int] = Field(default_factory=list)
    candidate_vector_ranks_doc12: list[int] = Field(default_factory=list)
    baseline_pool_rank_doc08: int | None = None
    candidate_pool_rank_doc08: int | None = None
    baseline_pool_rank_doc12: int | None = None
    candidate_pool_rank_doc12: int | None = None
    baseline_final_rank_doc08: int | None = None
    candidate_final_rank_doc08: int | None = None
    baseline_final_rank_doc12: int | None = None
    candidate_final_rank_doc12: int | None = None
    baseline_primary_hit_at_4: bool = False
    candidate_primary_hit_at_4: bool = False
    baseline_primary_hit_at_12: bool = False
    candidate_primary_hit_at_12: bool = False
    delta_classification: str


class Doc08CaseDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    query_preview: str
    expected_primary: list[str]
    baseline_vector_ranks_doc08: list[int] = Field(default_factory=list)
    candidate_vector_ranks_doc08: list[int] = Field(default_factory=list)
    baseline_vector_ranks_doc12: list[int] = Field(default_factory=list)
    candidate_vector_ranks_doc12: list[int] = Field(default_factory=list)
    baseline_pool_rank_doc08: int | None = None
    candidate_pool_rank_doc08: int | None = None
    baseline_final_rank_doc08: int | None = None
    candidate_final_rank_doc08: int | None = None
    baseline_final_rank_doc12: int | None = None
    candidate_final_rank_doc12: int | None = None
    candidate_best_doc08_heading: str | None = None
    candidate_best_doc08_similarity: float | None = None
    top_competing_docs_candidate: list[str] = Field(default_factory=list)
    doc08_grounding_relevant: bool | None = None
    baseline_primary_reachable: bool = False
    candidate_primary_reachable: bool = False
    baseline_primary_hit_at_4: bool = False
    candidate_primary_hit_at_4: bool = False
    baseline_primary_hit_at_12: bool = False
    candidate_primary_hit_at_12: bool = False


class Doc08Footprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_appearances: int = 0
    top12_appearances: int = 0
    top4_appearances: int = 0
    top1_appearances: int = 0
    cases_in_pool: list[str] = Field(default_factory=list)
    cases_in_top4: list[str] = Field(default_factory=list)


class AcceptanceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str
    passed: bool
    baseline_value: str | None = None
    candidate_value: str | None = None
    hard_rejection: bool = False


class ExtensionArmMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_count: int
    privacy_hit_at_4: int
    privacy_hit_at_12: int
    threat_doc12_hit_at_4: int
    threat_doc12_in_top4: int
    threat_doc12_rank_vs_doc08_ok: int
    negative_domain_hit_at_4: int
    negative_doc08_top1: int
    negative_doc08_top4: int


class ExtensionEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    fingerprint: str
    baseline: ExtensionArmMetrics
    candidate: ExtensionArmMetrics
    case_diagnostics: list[ExtensionCaseDiagnostic] = Field(default_factory=list)
    verdict: Doc08Verdict
    acceptance_checks: list[AcceptanceCheck] = Field(default_factory=list)


class Doc08AtomicCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    baseline_case: CaseResult
    candidate_case: CaseResult
    baseline_ranking: RankingArmResult
    candidate_ranking: RankingArmResult
    baseline_pool: PoolReachability
    candidate_pool: PoolReachability
    baseline_pool_doc_ids: list[str] = Field(default_factory=list)
    candidate_pool_doc_ids: list[str] = Field(default_factory=list)
    baseline_final_doc_ids: list[str] = Field(default_factory=list)
    candidate_final_doc_ids: list[str] = Field(default_factory=list)


class Doc08AtomicEvaluationRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    experiment_id: str
    source_commit: str | None = None
    source_dirty: bool = False
    artifact_commit: str | None = None
    reference_experiment_id: str
    reference_arm: str
    reference_artifact_path: str
    baseline_reproduction: BaselineReproductionResult
    baseline_arm: RetrievalArmMetadata
    candidate_arm: RetrievalArmMetadata
    frozen_benchmark_fingerprint: str
    extension_benchmark_fingerprint: str
    production_retrieval_config_hash: str
    chunk_diff: Doc08ChunkDiffModel
    frozen_baseline_metrics: AggregateMetrics
    frozen_candidate_metrics: AggregateMetrics
    frozen_baseline_primary_hit_at_12: float
    frozen_candidate_primary_hit_at_12: float
    frozen_reachability: ReachabilityComparison
    frozen_reachability_baseline: FrozenReachabilitySnapshot
    frozen_reachability_candidate: FrozenReachabilitySnapshot
    faq_top4_baseline: int
    faq_top4_candidate: int
    case_results: list[Doc08AtomicCaseResult] = Field(default_factory=list)
    required_diagnostics: list[Doc08CaseDiagnostic] = Field(default_factory=list)
    doc08_footprint_baseline: Doc08Footprint
    doc08_footprint_candidate: Doc08Footprint
    promoted_cases: list[str] = Field(default_factory=list)
    regressed_cases: list[str] = Field(default_factory=list)
    primary_unreachable_baseline: list[str] = Field(default_factory=list)
    primary_unreachable_candidate: list[str] = Field(default_factory=list)
    fully_unreachable_baseline: list[str] = Field(default_factory=list)
    fully_unreachable_candidate: list[str] = Field(default_factory=list)
    extension: ExtensionEvaluationResult
    acceptance_checks: list[AcceptanceCheck] = Field(default_factory=list)
    interpretation_boundary: list[str] = Field(default_factory=list)
    verdict: Doc08Verdict
    latency_seconds_total: float = 0.0
    config: dict[str, Any] = Field(default_factory=dict)
