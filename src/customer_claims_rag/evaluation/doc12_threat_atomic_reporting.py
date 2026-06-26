"""Reporting for doc12 threat atomic corpus experiment."""

from __future__ import annotations

import json
from pathlib import Path

from customer_claims_rag.evaluation.doc12_threat_atomic_models import Doc12ThreatAtomicEvaluationRun

DEFAULT_JSON = Path("data/05_evaluation/doc12_threat_atomic_units_v1.json")
DEFAULT_MARKDOWN = Path("tests/12_doc12_threat_atomic_units_results.md")


def render_doc12_markdown(run: Doc12ThreatAtomicEvaluationRun) -> str:
    lines = [
        f"# Doc12 threat atomic units: {run.experiment_id}",
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
        f"- Doc08 fingerprint unchanged: **{run.doc08_fingerprint_unchanged}**",
        f"- Expected doc08 fingerprint: `{run.expected_doc08_fingerprint}`",
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
        "| Metric | Baseline (doc08 exp.) | Candidate (doc12 overlay) |",
        "|--------|----------------------:|--------------------------:|",
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
        f"| FAQ top-4 | {run.faq_top4_baseline} | {run.faq_top4_candidate} |",
        "",
    ]
    if run.replay_stability is not None:
        lines.extend(
            [
                "## Replay stability",
                "",
                f"- Integrity verdict: `{run.replay_stability.integrity_verdict}`",
                f"- All identical: **{run.replay_stability.all_identical}**",
                f"- Repeated query runs identical: {run.replay_stability.repeated_query_runs.get('all_identical')}",
                f"- Repeated full runs identical: {run.replay_stability.repeated_full_runs.get('all_identical')}",
                f"- Independent rebuilds identical: {run.replay_stability.independent_rebuilds.get('all_identical')}",
                f"- Candidate collection digest: `{run.candidate_arm.collection_content_digest}`",
                f"- Candidate embedding digest: `{run.candidate_arm.embedding_digest}`",
                "",
            ]
        )
    lines.extend(
        [
        f"- Promoted hit@4: {run.promoted_cases_hit_at_4}",
        f"- Regressed hit@4: {run.regressed_cases_hit_at_4}",
        f"- Promoted hit@12: {run.promoted_cases_hit_at_12}",
        f"- Regressed hit@12: {run.regressed_cases_hit_at_12}",
        "",
        "## Doc12 chunk diff",
        "",
        f"- Document: `{run.chunk_diff.document_id}`",
        f"- Baseline doc12 chunks: {len(run.chunk_diff.baseline_chunk_ids)}",
        f"- Candidate doc12 chunks: {len(run.chunk_diff.candidate_chunk_ids)}",
        f"- Added: {run.chunk_diff.added_chunk_ids}",
        f"- Removed: {run.chunk_diff.removed_chunk_ids}",
        f"- Changed: {len(run.chunk_diff.changed_chunk_ids)}",
        f"- Non-doc12 byte-identical: {run.chunk_diff.non_target_byte_identical}",
        "",
        "## Extension metrics",
        "",
        "| Metric | Baseline | Candidate |",
        "|--------|----------|-----------|",
        f"| Privacy hit@4 | {run.extension.baseline.privacy_hit_at_4}/4 | "
        f"{run.extension.candidate.privacy_hit_at_4}/4 |",
        f"| Threat doc12 hit@4 | {run.extension.baseline.threat_doc12_hit_at_4}/4 | "
        f"{run.extension.candidate.threat_doc12_hit_at_4}/4 |",
        f"| Threat doc12 in top-4 | {run.extension.baseline.threat_doc12_in_top4}/4 | "
        f"{run.extension.candidate.threat_doc12_in_top4}/4 |",
        f"| Negative domain hit@4 | {run.extension.baseline.negative_domain_hit_at_4}/4 | "
        f"{run.extension.candidate.negative_domain_hit_at_4}/4 |",
        "",
        "## Holdout metrics",
        "",
        "| Metric | Baseline | Candidate |",
        "|--------|----------|-----------|",
        f"| Positive doc12 hit@4 | {run.holdout.baseline.positive_doc12_hit_at_4}/6 | "
        f"{run.holdout.candidate.positive_doc12_hit_at_4}/6 |",
        f"| Positive doc12 hit@12 | {run.holdout.baseline.positive_doc12_hit_at_12}/6 | "
        f"{run.holdout.candidate.positive_doc12_hit_at_12}/6 |",
        f"| Positive rank vs doc08 ok | {run.holdout.baseline.positive_doc12_rank_vs_doc08_ok}/6 | "
        f"{run.holdout.candidate.positive_doc12_rank_vs_doc08_ok}/6 |",
        f"| Negative domain hit@4 | {run.holdout.baseline.negative_domain_hit_at_4}/6 | "
        f"{run.holdout.candidate.negative_domain_hit_at_4}/6 |",
        f"| Negative doc12 top-1 | {run.holdout.baseline.negative_doc12_top1} | "
        f"{run.holdout.candidate.negative_doc12_top1} |",
        f"| Negative doc12 top-4 | {run.holdout.baseline.negative_doc12_top4}/6 | "
        f"{run.holdout.candidate.negative_doc12_top4}/6 |",
        "",
        "## Holdout case diagnostics",
        "",
        "| Case | Baseline hit@4 | Candidate hit@4 | Delta |",
        "|------|----------------|-----------------|-------|",
        ]
    )
    for diag in run.holdout.case_diagnostics:
        lines.append(
            f"| {diag.case_id} | {diag.baseline_primary_hit_at_4} | "
            f"{diag.candidate_primary_hit_at_4} | {diag.delta_classification} |"
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


def write_doc12_outputs(
    run: Doc12ThreatAtomicEvaluationRun,
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
    markdown_path.write_text(render_doc12_markdown(run), encoding="utf-8")
