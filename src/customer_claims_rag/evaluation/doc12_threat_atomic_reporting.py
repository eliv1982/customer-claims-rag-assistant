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
        f"**Experiment verdict source:** `{run.experiment_verdict_source}`",
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
    if run.environment is not None:
        lines.extend(
            [
                "## Environment",
                "",
                f"- Kind: `{run.environment.kind}`",
                f"- Python: `{run.environment.python_version}`",
                f"- Pip: `{run.environment.pip_version}`",
                f"- Pip check exit code: `{run.environment.pip_check_exit_code}`",
                f"- Dependency check: `{run.environment.dependency_check_summary}`",
                "",
            ]
        )
    if run.replay_integrity is not None:
        replay = run.replay_integrity
        frozen = replay.frozen_snapshot_replay
        lines.extend(
            [
                "## Replay integrity",
                "",
                f"- Integrity verdict: `{replay.integrity_verdict}`",
            ]
        )
        if frozen is not None:
            lines.extend(
                [
                    f"- ANN frozen rebuild identical: **{frozen.all_identical}** (authoritative={frozen.authoritative})",
                    f"- ANN frozen runs: {frozen.runs}",
                    f"- ANN repeated query runs identical: {frozen.repeated_query_runs.get('all_identical')}",
                    f"- ANN repeated full runs identical: {frozen.repeated_full_runs.get('all_identical')}",
                ]
            )
        if replay.exact_replay is not None:
            exact = replay.exact_replay
            lines.extend(
                [
                    f"- Exact repeated full runs identical: **{exact.repeated_full_identical}**",
                    f"- Exact independent loader roots identical: **{exact.independent_loader_identical}**",
                    f"- Exact source-commit reconstruction identical: **{exact.source_commit_reconstruction_identical}**",
                    f"- Baseline snapshot digest: `{exact.baseline_snapshot_digest}`",
                    f"- Candidate snapshot digest: `{exact.candidate_snapshot_digest}`",
                ]
            )
        if replay.ann_robustness is not None:
            ann = replay.ann_robustness
            lines.extend(
                [
                    f"- ANN robustness builds: {ann.builds} (authoritative={ann.authoritative})",
                    f"- E008 ANN rank range: {ann.e008_ann_rank_range}",
                    f"- H005 ANN rank range: {ann.h005_ann_rank_range}",
                    f"- Exact E008 final rank doc12: {ann.exact_e008_final_rank_doc12}",
                    f"- Exact H005 final rank doc12: {ann.exact_h005_final_rank_doc12}",
                    f"- ANN verdict variation: {ann.ann_verdict_variation}",
                ]
            )
        if replay.ann_rebuild_stability is not None:
            lines.append(
                f"- ANN rebuild stability: `{replay.ann_rebuild_stability.status}` "
                f"(authoritative={replay.ann_rebuild_stability.authoritative})"
            )
        lines.extend(
            [
                f"- Candidate embedding snapshot: `{replay.embedding_snapshot_path}`",
                f"- Baseline embedding snapshot: `{replay.baseline_snapshot_path}`",
                f"- Snapshot digest: `{replay.embedding_snapshot_digest}`",
                f"- Candidate collection digest: `{replay.collection_content_digest}`",
                f"- Candidate embedding digest: `{replay.embedding_digest}`",
            ]
        )
        live = replay.live_provider_robustness
        if live is not None:
            lines.extend(
                [
                    f"- Live provider runs: {live.runs} (authoritative={live.authoritative})",
                    f"- Unique embedding digests: {live.unique_embedding_digests}",
                    f"- E008 hit@4 pass/fail: {live.e008_hit4_pass_count}/{live.e008_hit4_fail_count}",
                    f"- Experiment REJECTED count: {live.experiment_verdict_rejected_count}",
                    f"- {live.observed_metric_variability.get('wording', '')}",
                    "",
                ]
            )
        else:
            lines.append("")
    elif run.replay_stability is not None:
        lines.extend(
            [
                "## Replay stability (legacy)",
                "",
                f"- Integrity verdict: `{run.replay_stability.integrity_verdict}`",
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
