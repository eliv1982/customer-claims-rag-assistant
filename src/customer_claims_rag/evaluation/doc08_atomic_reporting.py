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
        f"**Source dirty:** `{run.source_dirty}`",
        f"**Artifact commit:** `{run.artifact_commit or 'pending'}`",
        f"**Reference:** `{run.reference_experiment_id}` / `{run.reference_arm}`",
        f"**Verdict:** `{run.verdict}`",
        "",
        "## Baseline reproduction",
        "",
        f"- Passed: **{run.baseline_reproduction.passed}**",
        f"- Reference artifact: `{run.baseline_reproduction.reference_artifact_path}`",
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
        f"| Primary hit@4 | {run.frozen_baseline_metrics.primary_source_hit_rate_at_4:.6f} | "
        f"{run.frozen_candidate_metrics.primary_source_hit_rate_at_4:.6f} |",
        f"| Primary hit@12 | {run.frozen_baseline_primary_hit_at_12:.6f} | "
        f"{run.frozen_candidate_primary_hit_at_12:.6f} |",
        f"| MRR | {run.frozen_baseline_metrics.mrr:.6f} | "
        f"{run.frozen_candidate_metrics.mrr:.6f} |",
        f"| Primary reach | {run.frozen_reachability_baseline.primary_reachable}/"
        f"{run.frozen_reachability_baseline.primary_denominator} | "
        f"{run.frozen_reachability_candidate.primary_reachable}/"
        f"{run.frozen_reachability_candidate.primary_denominator} |",
        f"| High-risk reach | {run.frozen_reachability_baseline.high_risk_reachable}/"
        f"{run.frozen_reachability_baseline.high_risk_denominator} | "
        f"{run.frozen_reachability_candidate.high_risk_reachable}/"
        f"{run.frozen_reachability_candidate.high_risk_denominator} |",
        f"| Critical reach | {run.frozen_reachability_baseline.critical_reachable}/"
        f"{run.frozen_reachability_baseline.critical_denominator} | "
        f"{run.frozen_reachability_candidate.critical_reachable}/"
        f"{run.frozen_reachability_candidate.critical_denominator} |",
        f"| FAQ top-4 | {run.faq_top4_baseline} | {run.faq_top4_candidate} |",
        "",
        f"- Primary unreachable baseline: {run.primary_unreachable_baseline}",
        f"- Primary unreachable candidate: {run.primary_unreachable_candidate}",
        f"- Fully unreachable baseline: {run.fully_unreachable_baseline}",
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
        "## Extension metrics",
        "",
        "| Metric | Baseline | Candidate |",
        "|--------|----------|-----------|",
        f"| Privacy hit@4 | {run.extension.baseline.privacy_hit_at_4}/4 | "
        f"{run.extension.candidate.privacy_hit_at_4}/4 |",
        f"| Privacy hit@12 | {run.extension.baseline.privacy_hit_at_12}/4 | "
        f"{run.extension.candidate.privacy_hit_at_12}/4 |",
        f"| Threat doc12 hit@4 | {run.extension.baseline.threat_doc12_hit_at_4}/4 | "
        f"{run.extension.candidate.threat_doc12_hit_at_4}/4 |",
        f"| Threat doc12 in top-4 | {run.extension.baseline.threat_doc12_in_top4}/4 | "
        f"{run.extension.candidate.threat_doc12_in_top4}/4 |",
        f"| Negative domain hit@4 | {run.extension.baseline.negative_domain_hit_at_4}/4 | "
        f"{run.extension.candidate.negative_domain_hit_at_4}/4 |",
        "",
        "## Extension case diagnostics",
        "",
        "| Case | Baseline hit@4 | Candidate hit@4 | Delta |",
        "|------|----------------|-----------------|-------|",
    ]
    for diag in run.extension.case_diagnostics:
        lines.append(
            f"| {diag.case_id} | {diag.baseline_primary_hit_at_4} | "
            f"{diag.candidate_primary_hit_at_4} | {diag.delta_classification} |"
        )

    lines.extend(["", "## T044 / T047 diagnostics", ""])
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

    lines.extend(["", "## Acceptance checklist", ""])
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
