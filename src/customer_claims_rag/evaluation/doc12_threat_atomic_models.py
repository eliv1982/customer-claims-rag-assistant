"""Data models for doc12 threat atomic corpus A/B experiment."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from customer_claims_rag.evaluation.doc08_atomic_models import (
    AcceptanceCheck,
    BaselineReproductionResult,
    ExtensionArmMetrics,
    ExtensionCaseDiagnostic,
    ExtensionEvaluationResult,
    FrozenReachabilitySnapshot,
    RetrievalArmMetadata,
)
from customer_claims_rag.evaluation.models import AggregateMetrics, CaseResult
from customer_claims_rag.evaluation.pool_expansion_models import (
    PoolReachability,
    RankingArmResult,
    ReachabilityComparison,
)
from customer_claims_rag.ingestion.corpus_overlay import DocumentChunkDiff


class Doc12ChunkDiffModel(BaseModel):
    """JSON-serializable doc12 chunk diff."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    baseline_chunk_ids: list[str]
    candidate_chunk_ids: list[str]
    added_chunk_ids: list[str]
    removed_chunk_ids: list[str]
    changed_chunk_ids: list[str]
    unchanged_non_target_chunk_ids: list[str]
    non_target_byte_identical: bool

    @classmethod
    def from_dataclass(cls, diff: DocumentChunkDiff) -> "Doc12ChunkDiffModel":
        return cls(
            document_id=diff.document_id,
            baseline_chunk_ids=diff.baseline_chunk_ids,
            candidate_chunk_ids=diff.candidate_chunk_ids,
            added_chunk_ids=diff.added_chunk_ids,
            removed_chunk_ids=diff.removed_chunk_ids,
            changed_chunk_ids=diff.changed_chunk_ids,
            unchanged_non_target_chunk_ids=diff.unchanged_non_target_chunk_ids,
            non_target_byte_identical=diff.non_target_byte_identical,
        )


Doc12Verdict = Literal["ACCEPTED AS COMBINED TARGETED CORPUS REPAIR", "REJECTED"]


class HoldoutArmMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_count: int
    positive_doc12_hit_at_4: int
    positive_doc12_hit_at_12: int
    positive_doc12_rank_vs_doc08_ok: int
    negative_domain_hit_at_4: int
    negative_doc12_top1: int
    negative_doc12_top4: int


class HoldoutEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    fingerprint: str
    baseline: HoldoutArmMetrics
    candidate: HoldoutArmMetrics
    case_diagnostics: list[ExtensionCaseDiagnostic] = Field(default_factory=list)
    verdict: Doc12Verdict
    acceptance_checks: list[AcceptanceCheck] = Field(default_factory=list)


class Doc12ThreatAtomicCaseResult(BaseModel):
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


class Doc12ThreatAtomicEvaluationRun(BaseModel):
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
    holdout_benchmark_fingerprint: str
    production_retrieval_config_hash: str
    chunk_diff: Doc12ChunkDiffModel
    doc08_fingerprint_unchanged: bool
    expected_doc08_fingerprint: str
    frozen_baseline_metrics: AggregateMetrics
    frozen_candidate_metrics: AggregateMetrics
    frozen_baseline_primary_hit_at_12: float
    frozen_candidate_primary_hit_at_12: float
    frozen_reachability: ReachabilityComparison
    frozen_reachability_baseline: FrozenReachabilitySnapshot
    frozen_reachability_candidate: FrozenReachabilitySnapshot
    faq_top4_baseline: int
    faq_top4_candidate: int
    case_results: list[Doc12ThreatAtomicCaseResult] = Field(default_factory=list)
    primary_unreachable_baseline: list[str] = Field(default_factory=list)
    primary_unreachable_candidate: list[str] = Field(default_factory=list)
    fully_unreachable_baseline: list[str] = Field(default_factory=list)
    fully_unreachable_candidate: list[str] = Field(default_factory=list)
    promoted_cases_hit_at_4: list[str] = Field(default_factory=list)
    regressed_cases_hit_at_4: list[str] = Field(default_factory=list)
    promoted_cases_hit_at_12: list[str] = Field(default_factory=list)
    regressed_cases_hit_at_12: list[str] = Field(default_factory=list)
    extension: ExtensionEvaluationResult
    holdout: HoldoutEvaluationResult
    acceptance_checks: list[AcceptanceCheck] = Field(default_factory=list)
    interpretation_boundary: list[str] = Field(default_factory=list)
    verdict: Doc12Verdict
    latency_seconds_total: float = 0.0
    config: dict[str, Any] = Field(default_factory=dict)
