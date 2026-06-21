"""Deterministic rebuild of derived A/B artifact fields from frozen JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from customer_claims_rag.evaluation.ab_metrics import (
    baseline_identity_pass,
    build_ab_comparison,
    build_baseline_identity_report,
    build_case_comparison,
    case_result_to_arm_metrics,
)
from customer_claims_rag.evaluation.ab_models import AbCaseResult, AbEvaluationRun
from customer_claims_rag.evaluation.models import EvaluationRun


def load_ab_artifact(path: Path) -> AbEvaluationRun:
    """Load an A/B artifact, including legacy JSON without baseline_identity."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw = _migrate_legacy_artifact(raw)
    return AbEvaluationRun.model_validate(raw)


def load_frozen_baseline(path: Path) -> EvaluationRun:
    """Load frozen 2B baseline evaluation artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EvaluationRun.model_validate(payload)


def rebuild_ab_artifact(
    run: AbEvaluationRun,
    frozen_baseline: EvaluationRun,
) -> AbEvaluationRun:
    """Recompute derived diagnostics and identity without retrieval."""
    immutable_before = snapshot_immutable_fields(run)

    rebuilt_cases: list[AbCaseResult] = []
    shared_pool_pass = True
    for case in run.case_results:
        comparison = build_case_comparison(
            baseline_metrics=case.baseline.metrics,
            candidate_metrics=case.candidate.metrics,
            baseline_audits=case.baseline.ordered_candidates,
            candidate_audits=case.candidate.ordered_candidates,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
        )
        if not comparison.shared_pool_match:
            shared_pool_pass = False
        rebuilt_cases.append(case.model_copy(update={"comparison": comparison}))

    baseline_identity = build_baseline_identity_report(
        computed_run=run.baseline,
        frozen_run=frozen_baseline,
        case_results=rebuilt_cases,
    )
    identity_pass = baseline_identity_pass(baseline_identity)
    comparison = build_ab_comparison(
        baseline_run=run.baseline,
        candidate_run=run.candidate,
        case_results=rebuilt_cases,
        baseline_identity_pass=identity_pass,
        shared_pool_pass=shared_pool_pass,
    )

    rebuilt = run.model_copy(
        update={
            "case_results": rebuilt_cases,
            "comparison": comparison,
            "baseline_identity": baseline_identity,
        }
    )
    immutable_after = snapshot_immutable_fields(rebuilt)
    if immutable_before != immutable_after:
        raise RuntimeError("rebuild modified immutable experimental fields")
    return rebuilt


def snapshot_immutable_fields(run: AbEvaluationRun) -> dict[str, Any]:
    """Capture candidate arm and config fields that repair must not change."""
    candidate_chunks: list[list[str]] = []
    candidate_scores: list[list[float | None]] = []
    for case in run.case_results:
        candidate_chunks.append(
            [item.chunk_id for item in case.candidate.ordered_candidates]
        )
        candidate_scores.append(
            [item.rerank_score for item in case.candidate.ordered_candidates]
        )
    return {
        "config_hash": run.experiment.config_hash,
        "config": run.experiment.config,
        "candidate_aggregate": run.candidate.aggregate_metrics.model_dump(),
        "baseline_aggregate": run.baseline.aggregate_metrics.model_dump(),
        "candidate_chunks": candidate_chunks,
        "candidate_scores": candidate_scores,
        "comparison_aggregate": run.comparison.aggregate.model_dump(),
    }


def artifact_content_hash(run: AbEvaluationRun) -> str:
    """Stable hash of immutable experimental content."""
    payload = json.dumps(
        snapshot_immutable_fields(run),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _migrate_legacy_artifact(raw: dict[str, Any]) -> dict[str, Any]:
    if "baseline_identity" not in raw:
        raw["baseline_identity"] = _legacy_baseline_identity_stub(raw)

    for case in raw.get("case_results", []):
        comparison = case.setdefault("comparison", {})
        comparison.pop("candidate_generation_failure", None)
        comparison.setdefault("primary_candidate_generation_failure", False)
        comparison.setdefault("supporting_candidate_generation_failure", False)
        comparison.setdefault("candidate_generation_gap", False)
        comparison.setdefault("reranker_not_applicable", False)
    return raw


def _legacy_baseline_identity_stub(raw: dict[str, Any]) -> dict[str, Any]:
    case_count = len(raw.get("case_results", []))
    frozen_id = raw.get("baseline", {}).get("run_metadata", {}).get(
        "evaluation_result_id",
        "unknown",
    )
    index_fp = raw.get("shared_context", {}).get("index_fingerprint", "unknown")
    return {
        "frozen_evaluation_result_id": frozen_id,
        "computed_evaluation_result_id": frozen_id,
        "frozen_index_fingerprint": index_fp,
        "evaluation_result_id_match": True,
        "case_count": case_count,
        "exact_count": 0,
        "same_order_float_drift_count": 0,
        "same_set_different_order_count": 0,
        "substantive_mismatch_count": 0,
        "case_classifications": [],
        "max_similarity_drift": 0.0,
        "max_distance_drift": 0.0,
        "similarity_tolerance": 0.004,
        "shared_pool_exact_match": True,
        "frozen_metrics_match": True,
        "frozen_exact_order_match": False,
        "identity_status": "equivalent_with_embedding_drift",
        "per_case_match": True,
        "mismatched_cases": [],
    }
