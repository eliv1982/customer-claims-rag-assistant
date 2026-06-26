"""ANN vs exact robustness diagnostics for doc12 threat experiment."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from customer_claims_rag.evaluation.doc12_threat_atomic_contract import DOC12_DOCUMENT_ID
from customer_claims_rag.evaluation.doc12_threat_atomic_evaluator import Doc12ThreatAtomicAbEvaluator
from customer_claims_rag.evaluation.doc12_threat_atomic_models import AnnRobustnessResult


def _find_case_diagnostic(run, case_id: str):
    for collection in (run.extension.case_diagnostics, run.holdout.case_diagnostics):
        for item in collection:
            if item.case_id == case_id:
                return item
    return None


@dataclass(frozen=True)
class CaseRankSnapshot:
    case_id: str
    exact_vector_rank_doc12: int | None
    ann_vector_rank_doc12: int | None
    exact_final_rank_doc12: int | None
    ann_final_rank_doc12: int | None
    top4_membership_changed: bool
    top12_membership_changed: bool


def _doc12_rank(results, *, document_id: str = DOC12_DOCUMENT_ID) -> int | None:
    for index, item in enumerate(results, start=1):
        if item.document_id == document_id:
            return index
    return None


def _top_doc_ids(results, limit: int) -> set[str]:
    return {item.document_id for item in results[:limit]}


def compare_exact_vs_ann_case(
    *,
    case_id: str,
    exact_vector,
    ann_vector,
    exact_final,
    ann_final,
) -> CaseRankSnapshot:
    exact_top4 = _top_doc_ids(exact_final, 4)
    ann_top4 = _top_doc_ids(ann_final, 4)
    exact_top12 = _top_doc_ids(exact_final, 12)
    ann_top12 = _top_doc_ids(ann_final, 12)
    return CaseRankSnapshot(
        case_id=case_id,
        exact_vector_rank_doc12=_doc12_rank(exact_vector),
        ann_vector_rank_doc12=_doc12_rank(ann_vector),
        exact_final_rank_doc12=_doc12_rank(exact_final),
        ann_final_rank_doc12=_doc12_rank(ann_final),
        top4_membership_changed=exact_top4 != ann_top4,
        top12_membership_changed=exact_top12 != ann_top12,
    )


def _final_doc_ids_for_case(run, case_id: str) -> list[str] | None:
    for item in run.case_results:
        if item.case_id == case_id:
            return list(item.candidate_final_doc_ids)
    return None


def summarize_ann_build(
    *,
    build_label: str,
    run,
    exact_run,
) -> dict[str, Any]:
    frozen = run.frozen_candidate_metrics
    extension = run.extension.candidate
    holdout = run.holdout.candidate
    e008_exact = next(
        item for item in exact_run.extension.case_diagnostics if item.case_id == "E008"
    )
    e008_ann = next(item for item in run.extension.case_diagnostics if item.case_id == "E008")
    h005_exact = next(
        item for item in exact_run.holdout.case_diagnostics if item.case_id == "H005"
    )
    h005_ann = next(item for item in run.holdout.case_diagnostics if item.case_id == "H005")

    case_ids = [item.case_id for item in exact_run.case_results]
    case_ids += [item.case_id for item in exact_run.extension.case_diagnostics]
    case_ids += [item.case_id for item in exact_run.holdout.case_diagnostics]

    case_comparisons: list[dict[str, Any]] = []
    top4_changes = 0
    top12_changes = 0
    for case_id in case_ids:
        exact_final_ids = _final_doc_ids_for_case(exact_run, case_id)
        ann_final_ids = _final_doc_ids_for_case(run, case_id)
        exact_diag = _find_case_diagnostic(exact_run, case_id)
        ann_diag = _find_case_diagnostic(run, case_id)
        if exact_final_ids is not None and ann_final_ids is not None:
            top4_changed = set(exact_final_ids[:4]) != set(ann_final_ids[:4])
            top12_changed = set(exact_final_ids[:12]) != set(ann_final_ids[:12])
        elif exact_diag is not None and ann_diag is not None:
            top4_changed = (
                exact_diag.candidate_primary_hit_at_4 != ann_diag.candidate_primary_hit_at_4
                or exact_diag.candidate_final_rank_doc12 != ann_diag.candidate_final_rank_doc12
            )
            top12_changed = (
                exact_diag.candidate_primary_hit_at_12 != ann_diag.candidate_primary_hit_at_12
                or exact_diag.candidate_final_rank_doc12 != ann_diag.candidate_final_rank_doc12
            )
        else:
            top4_changed = False
            top12_changed = False
        if top4_changed:
            top4_changes += 1
        if top12_changed:
            top12_changes += 1
        case_comparisons.append(
            {
                "case_id": case_id,
                "exact_final_rank_doc12": (
                    exact_diag.candidate_final_rank_doc12 if exact_diag else None
                ),
                "ann_final_rank_doc12": ann_diag.candidate_final_rank_doc12 if ann_diag else None,
                "top4_membership_changed": top4_changed,
                "top12_membership_changed": top12_changed,
            }
        )

    return {
        "build_label": build_label,
        "frozen_primary_hit_at_4": frozen.primary_source_hit_rate_at_4,
        "frozen_primary_hit_at_12": run.frozen_candidate_primary_hit_at_12,
        "extension_threat_doc12_hit_at_4": extension.threat_doc12_hit_at_4,
        "holdout_negative_doc12_top4": holdout.negative_doc12_top4,
        "e008_final_rank_doc12": e008_ann.candidate_final_rank_doc12,
        "h005_final_rank_doc12": h005_ann.candidate_final_rank_doc12,
        "exact_e008_final_rank_doc12": e008_exact.candidate_final_rank_doc12,
        "exact_h005_final_rank_doc12": h005_exact.candidate_final_rank_doc12,
        "experiment_verdict": run.verdict,
        "top4_membership_differences": top4_changes,
        "top12_membership_differences": top12_changes,
        "case_comparisons": case_comparisons,
    }


def run_ann_robustness_matrix(
    *,
    config: dict,
    canonical_dir: Path,
    parent_dir: Path,
    project_root: Path,
    ann_evaluator: Doc12ThreatAtomicAbEvaluator,
    exact_run,
    embedding_snapshot: Path,
    embedding_snapshot_manifest: Path,
    builds: int = 3,
) -> AnnRobustnessResult:
    from customer_claims_rag.evaluation.doc12_replay_integrity import build_candidate_index_to_dir

    if parent_dir.exists():
        shutil.rmtree(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    original_candidate_index = ann_evaluator.candidate_index_dir
    build_summaries: list[dict[str, Any]] = []
    cases_with_top4_variation: dict[str, list[str]] = {}
    cases_with_top12_variation: dict[str, list[str]] = {}

    for label in ("A", "B", "C")[:builds]:
        index_dir = parent_dir / f"BUILD_{label}"
        build_candidate_index_to_dir(
            config=config,
            canonical_dir=canonical_dir,
            index_dir=index_dir,
            project_root=project_root,
            run_id=f"ann-robust-{label.lower()}-{uuid.uuid4().hex[:8]}",
            embedding_snapshot=embedding_snapshot,
            embedding_snapshot_manifest=embedding_snapshot_manifest,
            live_provider=False,
        )
        ann_evaluator.replace_candidate_index(index_dir)
        ann_run = ann_evaluator.evaluate()
        summary = summarize_ann_build(
            build_label=label,
            run=ann_run,
            exact_run=exact_run,
        )
        build_summaries.append(summary)
        for item in summary["case_comparisons"]:
            case_id = item["case_id"]
            if item["top4_membership_changed"]:
                cases_with_top4_variation.setdefault(case_id, []).append(label)
            if item["top12_membership_changed"]:
                cases_with_top12_variation.setdefault(case_id, []).append(label)

    ann_evaluator.replace_candidate_index(original_candidate_index)

    e008_ranks = sorted(
        {
            item["e008_final_rank_doc12"]
            for item in build_summaries
            if item["e008_final_rank_doc12"] is not None
        }
    )
    h005_ranks = sorted(
        {
            item["h005_final_rank_doc12"]
            for item in build_summaries
            if item["h005_final_rank_doc12"] is not None
        }
    )
    verdicts = {item["experiment_verdict"] for item in build_summaries}
    exact_verdict = exact_run.verdict

    return AnnRobustnessResult(
        authoritative=False,
        builds=builds,
        build_summaries=build_summaries,
        cases_with_top4_variation=[
            {"case_id": case_id, "builds": builds_list}
            for case_id, builds_list in sorted(cases_with_top4_variation.items())
        ],
        cases_with_top12_variation=[
            {"case_id": case_id, "builds": builds_list}
            for case_id, builds_list in sorted(cases_with_top12_variation.items())
        ],
        e008_ann_rank_range=e008_ranks,
        h005_ann_rank_range=h005_ranks,
        exact_e008_final_rank_doc12=next(
            item.candidate_final_rank_doc12
            for item in exact_run.extension.case_diagnostics
            if item.case_id == "E008"
        ),
        exact_h005_final_rank_doc12=next(
            item.candidate_final_rank_doc12
            for item in exact_run.holdout.case_diagnostics
            if item.case_id == "H005"
        ),
        ann_verdict_variation=sorted(verdicts),
        exact_authoritative_verdict=exact_verdict,
        verdict_changes_experiment_acceptance=exact_verdict not in verdicts
        or len(verdicts) > 1,
    )


def digest_exact_run(run) -> str:
    import hashlib

    payload = {
        "verdict": run.verdict,
        "frozen_primary_hit_at_4": run.frozen_candidate_metrics.primary_source_hit_rate_at_4,
        "frozen_primary_hit_at_12": run.frozen_candidate_primary_hit_at_12,
        "extension_threat_hit_at_4": run.extension.candidate.threat_doc12_hit_at_4,
        "holdout_negative_doc12_top4": run.holdout.candidate.negative_doc12_top4,
        "e008": next(
            item.model_dump(mode="json")
            for item in run.extension.case_diagnostics
            if item.case_id == "E008"
        ),
        "h005": next(
            item.model_dump(mode="json")
            for item in run.holdout.case_diagnostics
            if item.case_id == "H005"
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "CaseRankSnapshot",
    "compare_exact_vs_ann_case",
    "digest_exact_run",
    "run_ann_robustness_matrix",
    "summarize_ann_build",
]
