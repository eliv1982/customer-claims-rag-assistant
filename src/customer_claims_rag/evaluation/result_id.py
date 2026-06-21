"""Stable evaluation result identifier."""

from __future__ import annotations

import hashlib
import json

from customer_claims_rag.evaluation.models import EvaluationRun


def compute_evaluation_result_id(run: EvaluationRun) -> str:
    """Hash canonical run payload excluding timestamp and absolute paths."""
    metadata = run.run_metadata.model_dump(mode="json")
    metadata.pop("timestamp", None)
    metadata.pop("evaluation_result_id", None)
    payload = {
        "run_metadata": metadata,
        "aggregate_metrics": run.aggregate_metrics.model_dump(mode="json"),
        "risk_metrics": [item.model_dump(mode="json") for item in run.risk_metrics],
        "category_metrics": [item.model_dump(mode="json") for item in run.category_metrics],
        "threshold_analysis": [item.model_dump(mode="json") for item in run.threshold_analysis],
        "case_results": [item.model_dump(mode="json") for item in run.case_results],
        "technical_errors": run.technical_errors,
        "fallback_analysis": run.fallback_analysis,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
