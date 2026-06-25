"""Data models for expanded-corpus frozen benchmark regression."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from customer_claims_rag.evaluation.models import AggregateMetrics, RiskSliceMetrics
from customer_claims_rag.evaluation.pool_expansion_models import PoolExpansionEvaluationRun

PairedChangeClass = Literal[
    "stable",
    "promotion",
    "regression",
    "mixed",
    "strict_frozen_regression_with_semantically_stronger_new_source",
]

ComparativeVerdict = Literal[
    "ACCEPT",
    "ACCEPT_WITH_TARGETED_FOLLOW-UP",
    "REJECT / REPAIR REQUIRED",
]

NEW_DOCUMENT_IDS: tuple[str, ...] = (
    "11_payment_security_and_dispute_handling",
    "12_staff_safety_and_threat_handling",
    "13_physical_hazard_and_foreign_body_protocol",
    "14_evidence_standards_and_incomplete_information",
    "15_conflicting_rules_and_remedy_priority",
)

SEMANTIC_OVERLAY_BY_CASE: dict[str, str] = {
    "T004": "11_payment_security_and_dispute_handling",
    "T040": "13_physical_hazard_and_foreign_body_protocol",
    "T047": "12_staff_safety_and_threat_handling",
    "T055": "14_evidence_standards_and_incomplete_information",
    "T023": "15_conflicting_rules_and_remedy_priority",
    "T027": "15_conflicting_rules_and_remedy_priority",
    "T053": "15_conflicting_rules_and_remedy_priority",
}


class IndexArmSummary(BaseModel):
    """Frozen retrieval metrics for one index arm."""

    model_config = ConfigDict(extra="forbid")

    arm_id: str
    arm_label: str
    index_path: str
    index_fingerprint: str
    chunk_count: int
    document_count: int
    retrieval_chain: str
    frozen_retrieval_config_id: str
    frozen_retrieval_config_hash: str
    reranker_id: str
    candidate_pool_k: int
    final_top_k: int
    threshold: float
    aggregate_metrics: AggregateMetrics
    risk_metrics: list[RiskSliceMetrics]
    pool24_primary_reachable: int
    pool24_primary_total: int
    pool24_supporting_reachable: int
    pool24_supporting_total: int
    pool24_fully_unreachable_cases: list[str] = Field(default_factory=list)
    high_primary_reachable_pool24: int
    critical_primary_reachable_pool24: int
    high_primary_total: int
    critical_primary_total: int


class MetricDelta(BaseModel):
    """Aggregate metric delta (arm B minus arm A)."""

    model_config = ConfigDict(extra="forbid")

    hit_rate_at_1: float
    hit_rate_at_4: float
    hit_rate_at_12: float
    document_recall_at_1: float
    document_recall_at_4: float
    document_recall_at_12: float
    mrr: float
    primary_source_hit_rate_at_1: float
    primary_source_hit_rate_at_4: float
    supporting_source_hit_rate_at_4: float | None = None
    pool24_primary_reachable_delta: int
    pool24_fully_unreachable_delta: int
    high_primary_reachable_pool24_delta: int
    critical_primary_reachable_pool24_delta: int


class PairedCaseResult(BaseModel):
    """Per-question paired comparison between index arms."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    risk: str
    category: str | None = None
    expected_primary_documents: list[str] = Field(default_factory=list)
    expected_supporting_documents: list[str] = Field(default_factory=list)
    fallback_expected: bool = False
    arm_a_primary_rank_final: int | None = None
    arm_b_primary_rank_final: int | None = None
    arm_a_top4_document_ids: list[str] = Field(default_factory=list)
    arm_b_top4_document_ids: list[str] = Field(default_factory=list)
    arm_a_top12_document_ids: list[str] = Field(default_factory=list)
    arm_b_top12_document_ids: list[str] = Field(default_factory=list)
    arm_a_primary_in_pool24: bool = False
    arm_b_primary_in_pool24: bool = False
    arm_a_primary_in_final_top12: bool = False
    arm_b_primary_in_final_top12: bool = False
    arm_a_primary_hit_at_4: bool = False
    arm_b_primary_hit_at_4: bool = False
    faq_in_arm_a_top4: bool = False
    faq_in_arm_b_top4: bool = False
    new_documents_in_arm_b_top4: list[str] = Field(default_factory=list)
    new_documents_in_arm_b_top12: list[str] = Field(default_factory=list)
    new_documents_in_arm_b_pool24: list[str] = Field(default_factory=list)
    frozen_primary_displaced_by_new_doc: bool = False
    harmful_regression: bool = False
    classification: PairedChangeClass = "stable"
    explanation: str = ""


class SemanticOverlayReview(BaseModel):
    """Diagnostic overlay for semantically stronger new sources."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    frozen_expected_primary: list[str] = Field(default_factory=list)
    overlay_document_id: str
    overlay_in_arm_b_top4: bool = False
    overlay_in_arm_b_top12: bool = False
    overlay_in_arm_b_pool24: bool = False
    frozen_primary_in_arm_b_top12: bool = False
    frozen_primary_in_arm_b_pool24: bool = False
    more_specific_than_frozen: bool = False
    frozen_should_remain_supporting: bool = False
    recommend_extension_set: bool = False
    recommend_benchmark_modification: bool = False
    notes: str = ""


class NewDocumentFootprint(BaseModel):
    """Retrieval footprint for one new policy document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    top1_questions: list[str] = Field(default_factory=list)
    top4_questions: list[str] = Field(default_factory=list)
    top12_questions: list[str] = Field(default_factory=list)
    pool24_questions: list[str] = Field(default_factory=list)
    likely_helpful_questions: list[str] = Field(default_factory=list)
    irrelevant_noise_questions: list[str] = Field(default_factory=list)
    displaces_frozen_primary_questions: list[str] = Field(default_factory=list)


class FaqDominanceComparison(BaseModel):
    """FAQ presence comparison across arms."""

    model_config = ConfigDict(extra="forbid")

    arm_a_faq_top1_count: int
    arm_b_faq_top1_count: int
    arm_a_faq_top4_chunk_slots: int
    arm_b_faq_top4_chunk_slots: int
    arm_a_questions_with_faq_in_top4: int
    arm_b_questions_with_faq_in_top4: int
    arm_a_faq_outranks_all_primaries: int
    arm_b_faq_outranks_all_primaries: int
    by_risk: dict[str, dict[str, int]] = Field(default_factory=dict)


class ExpandedCorpusRegressionRun(BaseModel):
    """Immutable paired regression artifact for expanded corpus evaluation."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    timestamp: datetime
    evaluation_dataset_fingerprint: str
    frozen_question_set_id: str
    retrieval_chain: str
    arm_a: IndexArmSummary
    arm_b: IndexArmSummary
    metric_deltas: MetricDelta
    paired_cases: list[PairedCaseResult]
    semantic_overlay: list[SemanticOverlayReview] = Field(default_factory=list)
    new_document_footprints: list[NewDocumentFootprint] = Field(default_factory=list)
    faq_dominance: FaqDominanceComparison
    harmful_regressions: list[str] = Field(default_factory=list)
    promotions: list[str] = Field(default_factory=list)
    verdict: ComparativeVerdict
    verdict_rationale: str
    arm_a_source_run: PoolExpansionEvaluationRun
    arm_b_source_run: PoolExpansionEvaluationRun
