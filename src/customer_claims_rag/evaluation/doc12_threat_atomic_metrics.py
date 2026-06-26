"""Footprint and holdout metrics for doc12 threat atomic experiment."""

from __future__ import annotations

from customer_claims_rag.evaluation.doc08_atomic_contract import (
    NEGATIVE_EXTENSION_IDS,
    PRIVACY_EXTENSION_IDS,
    THREAT_EXTENSION_IDS,
    build_extension_case_diagnostic,
    classify_threat_case_delta,
    compute_extension_metrics,
    evaluate_extension_acceptance,
)
from customer_claims_rag.evaluation.doc12_threat_atomic_contract import (
    HOLDOUT_NEGATIVE_IDS,
    HOLDOUT_POSITIVE_IDS,
    assert_baseline_reproduction,
    build_doc12_reachability_comparison,
    build_frozen_reachability_snapshot,
    classify_holdout_positive_delta,
    compute_holdout_metrics,
    compute_primary_hit_at_12,
    count_faq_in_top4,
    evaluate_frozen_acceptance,
    evaluate_holdout_acceptance,
    fully_unreachable_case_ids,
    load_doc08_experimental_oracle,
    load_holdout_corpus,
    primary_unreachable_case_ids,
    repo_relative_path,
    validate_baseline_reproduction,
)
from customer_claims_rag.ingestion.corpus_overlay import DOC12_DOCUMENT_ID

__all__ = [
    "DOC12_DOCUMENT_ID",
    "HOLDOUT_NEGATIVE_IDS",
    "HOLDOUT_POSITIVE_IDS",
    "NEGATIVE_EXTENSION_IDS",
    "PRIVACY_EXTENSION_IDS",
    "THREAT_EXTENSION_IDS",
    "assert_baseline_reproduction",
    "build_doc12_reachability_comparison",
    "build_extension_case_diagnostic",
    "build_frozen_reachability_snapshot",
    "classify_holdout_positive_delta",
    "classify_threat_case_delta",
    "compute_extension_metrics",
    "compute_holdout_metrics",
    "compute_primary_hit_at_12",
    "count_faq_in_top4",
    "evaluate_extension_acceptance",
    "evaluate_frozen_acceptance",
    "evaluate_holdout_acceptance",
    "fully_unreachable_case_ids",
    "load_doc08_experimental_oracle",
    "load_holdout_corpus",
    "primary_unreachable_case_ids",
    "repo_relative_path",
    "validate_baseline_reproduction",
]
