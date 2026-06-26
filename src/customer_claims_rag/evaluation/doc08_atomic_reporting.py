"""Reporting for doc08 atomic corpus experiment."""

from __future__ import annotations

import json
from pathlib import Path

from customer_claims_rag.evaluation.doc08_atomic_models import Doc08AtomicEvaluationRun

DEFAULT_JSON = Path("data/05_evaluation/doc08_atomic_risk_units_v1.json")
DEFAULT_MARKDOWN = Path("tests/11_doc08_atomic_risk_units_results.md")


def render_doc08_markdown(run: Doc08AtomicEvaluationRun) -> str:
    lines = [
        f"# Doc08 atomic risk units: {run.experiment_id}",
        "",
        f"**Timestamp:** {run.timestamp.isoformat()}",
        f"**Source commit:** `{run.source_commit or 'unknown'}`",
        f"**Artifact commit:** `{run.artifact_commit or 'pending'}`",
        f"**Verdict:** `{run.verdict}`",
        "",
        "## Retrieval arms",
        "",
        "| Arm | Index fingerprint | fetch_k | pool_k | cap | chunks |",
        "|-----|-------------------|--------:|-------:|----:|-------:|",
        f"| baseline | `{run.baseline_arm.index_fingerprint[:16]}...` | "
        f"{run.baseline_arm.fetch_k} | {run.baseline_arm.candidate_pool_k} | "
        f"{run.baseline_arm.per_document_cap} | {run.baseline_arm.chunk_count} |",
        f"| candidate | `{run.candidate_arm.index_fingerprint[:16]}...` | "
        f"{run.candidate_arm.fetch_k} | {run.candidate_arm.candidate_pool_k} | "
        f"{run.candidate_arm.per_document_cap} | {run.candidate_arm.chunk_count} |",
        "",
        "## Frozen metrics",
        "",
        "| Metric | Baseline | Candidate |",
        "|--------|----------|-----------|",
        f"| Primary hit@4 | {run.frozen_baseline_metrics.primary_source_hit_rate_at_4:.3f} | "
        f"{run.frozen_candidate_metrics.primary_source_hit_rate_at_4:.3f} |",
        f"| Primary hit@12 | {run.frozen_baseline_metrics.hit_rate_at_12:.3f} | "
        f"{run.frozen_candidate_metrics.hit_rate_at_12:.3f} |",
        f"| MRR | {run.frozen_baseline_metrics.mrr:.3f} | "
        f"{run.frozen_candidate_metrics.mrr:.3f} |",
        "",
        f"- Primary unreachable baseline: {run.primary_unreachable_baseline}",
        f"- Primary unreachable candidate: {run.primary_unreachable_candidate}",
        f"- Fully unreachable candidate: {run.fully_unreachable_candidate}",
        "",
        "## Doc08 chunk diff",
        "",
        f"- Baseline doc08 chunks: {len(run.chunk_diff.baseline_chunk_ids)}",
        f"- Candidate doc08 chunks: {len(run.chunk_diff.candidate_chunk_ids)}",
        f"- Added: {run.chunk_diff.added_chunk_ids}",
        f"- Removed: {run.chunk_diff.removed_chunk_ids}",
        f"- Changed: {len(run.chunk_diff.changed_chunk_ids)}",
        f"- Non-doc08 byte-identical: {run.chunk_diff.non_doc08_byte_identical}",
        "",
        "## Doc08 footprint (candidate)",
        "",
        f"- Pool: {run.doc08_footprint_candidate.pool_appearances}",
        f"- Top-12: {run.doc08_footprint_candidate.top12_appearances}",
        f"- Top-4: {run.doc08_footprint_candidate.top4_appearances}",
        "",
        "## T044 / T047 diagnostics",
        "",
    ]
    for diag in run.required_diagnostics:
        if diag.case_id not in {"T044", "T047"}:
            continue
        lines.extend(
            [
                f"### {diag.case_id}",
                "",
                f"- Expected primary: {diag.expected_primary}",
                f"- Baseline vector rank doc08: {diag.baseline_vector_ranks_doc08}",
                f"- Candidate vector rank doc08: {diag.candidate_vector_ranks_doc08}",
                f"- Candidate pool rank doc08: {diag.candidate_pool_rank_doc08}",
                f"- Candidate final rank doc08: {diag.candidate_final_rank_doc08}",
                f"- Baseline/Candidate final rank doc12: {diag.baseline_final_rank_doc12} / {diag.candidate_final_rank_doc12}",
                f"- Best doc08 heading: {diag.candidate_best_doc08_heading}",
                f"- Best doc08 similarity: {diag.candidate_best_doc08_similarity}",
                f"- Top competing docs: {diag.top_competing_docs_candidate}",
                f"- Grounding relevant: {diag.doc08_grounding_relevant}",
                "",
            ]
        )

    lines.extend(
        [
            "## Extension metrics (candidate)",
            "",
            f"- Privacy hit@4: {run.extension.candidate.privacy_hit_at_4}/4",
            f"- Threat doc12 in top-4: {run.extension.candidate.threat_doc12_in_top4}/4",
            f"- Negative doc08 top-1: {run.extension.candidate.negative_doc08_top1}",
            "",
            "## Acceptance checklist",
            "",
        ]
    )
    for check in run.acceptance_checks:
        mark = "PASS" if check.passed else "FAIL"
        lines.append(
            f"- [{mark}] `{check.criterion_id}`: {check.description} "
            f"(baseline={check.baseline_value}, candidate={check.candidate_value})"
        )
    lines.extend(["", "## Interpretation boundary", ""])
    lines.extend(f"- {item}" for item in run.interpretation_boundary)
    return "\n".join(lines) + "\n"


def write_doc08_outputs(
    run: Doc08AtomicEvaluationRun,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_doc08_markdown(run), encoding="utf-8")
