"""Markdown and JSON reporting for vector pool cap A/B evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from customer_claims_rag.evaluation.diversity_metrics import (
    STAGE_4C2_BASELINE_TARGETS,
    compute_primary_hit_rate_at_12_from_cases,
)
from customer_claims_rag.evaluation.diversity_models import DiversityEvaluationRun
from customer_claims_rag.exceptions import EvaluationOutputError

DEFAULT_DIVERSITY_JSON = Path("data/05_evaluation/vector_pool_36_cap4_v1.json")
DEFAULT_DIVERSITY_MARKDOWN = Path("tests/10_vector_pool_36_cap4_results.md")

PROTECTED_ARTIFACT_PATHS = (
    Path("data/05_evaluation/hybrid_lexical_vector_v1.json"),
    Path("data/05_evaluation/vector_pool_expansion_v1.json"),
    Path("data/05_evaluation/reranking_ab_source_authority_v1.json"),
    Path("data/05_evaluation/retrieval_results.json"),
    Path("data/05_evaluation/expanded_corpus_frozen_regression_v1.json"),
    Path("tests/03_test_results.md"),
    Path("tests/07_vector_pool_expansion_results.md"),
    Path("tests/09_expanded_corpus_frozen_regression_results.md"),
)


def write_diversity_outputs(
    run: DiversityEvaluationRun,
    *,
    output_json: Path,
    output_markdown: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write diversity experiment JSON and Markdown artifacts atomically."""
    json_path = _validate_output_path(output_json, project_root=project_root)
    markdown_path = _validate_output_path(output_markdown, project_root=project_root)
    _ensure_not_protected(json_path, project_root=project_root)
    _ensure_not_protected(markdown_path, project_root=project_root)

    json_content = json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    markdown_content = render_diversity_markdown(run)

    temp_paths: list[Path] = []
    try:
        _atomic_write_text(json_path, json_content, temp_paths=temp_paths)
        _atomic_write_text(markdown_path, markdown_content, temp_paths=temp_paths)
    except Exception as exc:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        raise EvaluationOutputError(f"Diversity output write failed: {exc}") from None
    return json_path, markdown_path


def render_diversity_markdown(run: DiversityEvaluationRun) -> str:
    """Render human-readable diversity experiment report."""
    experiment = run.experiment
    shared = run.shared_context
    baseline = run.baseline_ranking.aggregate_metrics
    candidate = run.candidate_ranking.aggregate_metrics
    reach = run.reachability_comparison
    faq = run.faq_comparison
    boundary = run.interpretation_boundary

    baseline_primary_hit12 = baseline.hit_rate_at_12
    candidate_primary_hit12 = candidate.hit_rate_at_12

    lines = [
        f"# Vector pool cap A/B: {experiment.experiment_id}",
        "",
        f"**Timestamp:** {run.timestamp.isoformat()}",
        f"**Experiment ID:** `{experiment.experiment_id}`",
        f"**Version:** `{experiment.version}`",
        f"**Git commit:** `{experiment.git_commit}`",
        f"**Config hash:** `{experiment.config_hash}`",
        f"**Verdict:** `{run.verdict}`",
        "",
        "## 1. Executive summary",
        "",
        (
            "Controlled candidate-generation experiment comparing production retrieval "
            f"(`fetch_k=24`, pool@24, no cap) against deep retrieval with per-document cap "
            f"(`fetch_k=48`, cap@4, pool@36) before frozen `source-authority-v1` reranking."
        ),
        "",
        f"- Baseline primary hit@4: **{baseline.primary_source_hit_rate_at_4:.3f}**",
        f"- Candidate primary hit@4: **{candidate.primary_source_hit_rate_at_4:.3f}**",
        f"- Baseline MRR: **{baseline.mrr:.3f}** | Candidate MRR: **{candidate.mrr:.3f}**",
        f"- Primary pool reach: **{reach.baseline_primary_reachable}/{reach.baseline_primary_total}** "
        f"-> **{reach.candidate_primary_reachable}/{reach.candidate_primary_total}**",
        "",
        "## 2. Repository and artifact identity",
        "",
        f"- Index fingerprint: `{shared.index_fingerprint}`",
        f"- Corpus: **{shared.document_count}** documents / **{shared.chunk_count}** chunks",
        f"- Benchmark fingerprint: `{shared.evaluation_dataset_fingerprint}`",
        f"- Reranker: `{experiment.reranker_id}` (hash `{experiment.reranker_config_hash}`)",
        "",
        "## 3. Exact baseline/candidate configurations",
        "",
        "### Baseline arm",
        "",
        f"- `fetch_k = {shared.baseline_arm.fetch_k}`",
        f"- `candidate_pool_k = {shared.baseline_arm.candidate_pool_k}`",
        f"- `per_document_cap = none`",
        f"- `reranker = source-authority-v1`",
        f"- `final_top_k = {shared.final_top_k}`",
        f"- `threshold = {shared.threshold}`",
        "",
        "### Candidate arm",
        "",
        f"- `fetch_k = {shared.candidate_arm.fetch_k}`",
        f"- `per_document_cap = {shared.candidate_arm.per_document_cap}`",
        f"- `candidate_pool_k = {shared.candidate_arm.candidate_pool_k}`",
        f"- `reranker = source-authority-v1`",
        f"- `final_top_k = {shared.final_top_k}`",
        f"- `threshold = {shared.threshold}`",
        "",
        "## 4. Exact cap algorithm",
        "",
        "1. Retrieve vector results at `fetch_k=48` in similarity order.",
        "2. Iterate results in original vector order.",
        "3. Keep at most four chunks per `document_id`.",
        "4. Stop at 36 pooled results or after all 48 retrieved results.",
        "5. No backfill beyond `fetch@48`.",
        "6. Reassign pool ranks from 1 without mutating input `SearchResult` objects.",
        "7. Preserve original vector rank in diagnostics.",
        "8. Pass shaped pool to `SourceAuthorityV1Reranker`; final results are top-12.",
        "",
        "## 5. Aggregate A/B table (final top-12)",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|--------|--------:|----------:|------:|",
        f"| Hit@1 | {baseline.hit_rate_at_1:.3f} | {candidate.hit_rate_at_1:.3f} | "
        f"{candidate.hit_rate_at_1 - baseline.hit_rate_at_1:+.3f} |",
        f"| Hit@4 | {baseline.hit_rate_at_4:.3f} | {candidate.hit_rate_at_4:.3f} | "
        f"{candidate.hit_rate_at_4 - baseline.hit_rate_at_4:+.3f} |",
        f"| Hit@12 | {baseline.hit_rate_at_12:.3f} | {candidate.hit_rate_at_12:.3f} | "
        f"{candidate.hit_rate_at_12 - baseline.hit_rate_at_12:+.3f} |",
        f"| Primary hit@4 | {baseline.primary_source_hit_rate_at_4:.3f} | "
        f"{candidate.primary_source_hit_rate_at_4:.3f} | "
        f"{candidate.primary_source_hit_rate_at_4 - baseline.primary_source_hit_rate_at_4:+.3f} |",
        f"| Primary hit@12 | {baseline_primary_hit12:.3f} | "
        f"{candidate_primary_hit12:.3f} | "
        f"{candidate_primary_hit12 - baseline_primary_hit12:+.3f} |",
        f"| MRR | {baseline.mrr:.3f} | {candidate.mrr:.3f} | "
        f"{candidate.mrr - baseline.mrr:+.3f} |",
        f"| Document recall@4 | {baseline.document_recall_at_4:.3f} | "
        f"{candidate.document_recall_at_4:.3f} | "
        f"{candidate.document_recall_at_4 - baseline.document_recall_at_4:+.3f} |",
        "",
        "### Stage 4C.2 baseline reproduction (baseline arm)",
        "",
        f"- Target primary hit@4: {STAGE_4C2_BASELINE_TARGETS['primary_hit_at_4']:.3f} "
        f"(actual {baseline.primary_source_hit_rate_at_4:.3f})",
        f"- Target primary hit@12: {STAGE_4C2_BASELINE_TARGETS['primary_hit_at_12']:.3f} "
        f"(actual {baseline_primary_hit12:.3f})",
        f"- Target MRR: {STAGE_4C2_BASELINE_TARGETS['mrr']:.3f} (actual {baseline.mrr:.3f})",
        f"- Target primary reach: {STAGE_4C2_BASELINE_TARGETS['primary_reach'][0]}/"
        f"{STAGE_4C2_BASELINE_TARGETS['primary_reach'][1]} "
        f"(actual {reach.baseline_primary_reachable}/{reach.baseline_primary_total})",
        f"- Target FAQ top-4 count: {STAGE_4C2_BASELINE_TARGETS['faq_top4_count']} "
        f"(actual {faq.baseline_questions_with_faq_in_top4})",
        "",
        "## 6. Reachability table",
        "",
        f"- Primary reachable: {reach.baseline_primary_reachable}/{reach.baseline_primary_total} "
        f"-> {reach.candidate_primary_reachable}/{reach.candidate_primary_total}",
        f"- High-risk primary reachable: {reach.high_baseline_primary_reachable}/"
        f"{reach.high_primary_total} -> {reach.high_candidate_primary_reachable}/"
        f"{reach.high_primary_total}",
        f"- Critical primary reachable: {reach.critical_baseline_primary_reachable}/"
        f"{reach.critical_primary_total} -> {reach.critical_candidate_primary_reachable}/"
        f"{reach.critical_primary_total}",
        f"- Baseline fully unreachable: {reach.baseline_fully_unreachable_cases}",
        f"- Candidate fully unreachable: {reach.candidate_fully_unreachable_cases}",
        "",
        "## 7. Saturation/diversity table",
        "",
        "| Metric | Baseline | Candidate |",
        "|--------|--------:|----------:|",
        f"| Avg unique documents in pool | {run.baseline_saturation.average_unique_documents_in_pool:.2f} | "
        f"{run.candidate_saturation.average_unique_documents_in_pool:.2f} |",
        f"| Avg max chunks from one document | "
        f"{run.baseline_saturation.average_max_chunks_from_one_document:.2f} | "
        f"{run.candidate_saturation.average_max_chunks_from_one_document:.2f} |",
        f"| Questions with doc count >= cap | "
        f"{run.baseline_saturation.questions_with_document_count_ge_cap} | "
        f"{run.candidate_saturation.questions_with_document_count_ge_cap} |",
        f"| Questions where cap removed chunk | "
        f"{run.baseline_saturation.questions_where_cap_removed_chunk} | "
        f"{run.candidate_saturation.questions_where_cap_removed_chunk} |",
        f"| Total removed by cap | {run.baseline_saturation.total_removed_by_cap_chunks} | "
        f"{run.candidate_saturation.total_removed_by_cap_chunks} |",
        f"| Avg pool size | {run.baseline_saturation.average_pool_size_after_cap:.2f} | "
        f"{run.candidate_saturation.average_pool_size_after_cap:.2f} |",
        f"| Pools shorter than target | "
        f"{run.baseline_saturation.questions_with_pool_shorter_than_target} | "
        f"{run.candidate_saturation.questions_with_pool_shorter_than_target} |",
        "",
        "## 8. FAQ comparison",
        "",
        f"- Questions with FAQ in final top-4: baseline **{faq.baseline_questions_with_faq_in_top4}** "
        f"-> candidate **{faq.candidate_questions_with_faq_in_top4}**",
        f"- FAQ in top-4 without expected primary: baseline **{faq.baseline_faq_top4_without_primary}** "
        f"-> candidate **{faq.candidate_faq_top4_without_primary}**",
        f"- Case IDs with changed FAQ top-4 presence: {faq.faq_top4_case_ids_changed_vs_baseline}",
        "",
        "## 9. Documents 11–15 footprint (candidate arm)",
        "",
        "| Document | Pools | Final top-12 |",
        "|----------|------:|-------------:|",
    ]
    for item in run.new_document_footprint:
        lines.append(
            f"| `{item.document_id}` | {item.pools_containing_document} | "
            f"{item.final_top12_containing_document} |"
        )

    lines.extend(["", "## 10. Required per-case analysis", ""])
    for diag in run.required_case_diagnostics:
        lines.extend(
            [
                f"### {diag.case_id}",
                "",
                f"- Expected primary: {diag.expected_primary_documents}",
                f"- Baseline vector ranks: {diag.baseline_vector_ranks_by_document}",
                f"- Candidate vector ranks: {diag.candidate_vector_ranks_by_document}",
                f"- Baseline pool ranks: {diag.baseline_pool_ranks_by_document}",
                f"- Candidate pool ranks: {diag.candidate_pool_ranks_by_document}",
                f"- Candidate pool document counts: {diag.candidate_pool_document_counts}",
                f"- Baseline final ranks: {diag.baseline_final_ranks_by_document}",
                f"- Candidate final ranks: {diag.candidate_final_ranks_by_document}",
                f"- Primary hit@4: {diag.baseline_primary_hit_at_4} -> {diag.candidate_primary_hit_at_4}",
                f"- Primary hit@12: {diag.baseline_primary_hit_at_12} -> {diag.candidate_primary_hit_at_12}",
                f"- Primary reachable: {diag.baseline_primary_reachable} -> {diag.candidate_primary_reachable}",
                f"- Change: {diag.change_explanation}",
                "",
            ]
        )

    lines.extend(["## 11. Acceptance checklist", ""])
    for check in run.acceptance_checks:
        status = "PASS" if check.passed else "FAIL"
        hard = " (hard regression)" if check.hard_regression else ""
        lines.append(
            f"- [{status}] `{check.criterion_id}`: {check.description}{hard}"
            + (
                f" (baseline={check.baseline_value}, candidate={check.candidate_value})"
                if check.baseline_value or check.candidate_value
                else ""
            )
        )

    lines.extend(
        [
            "",
            "## 12. Verdict",
            "",
            f"**{run.verdict}**",
            "",
            "## 13. Production unchanged",
            "",
            "Current production retrieval config (`vector-pool-expansion-v1`, fetch@24, pool@24, "
            "no cap) and `FrozenRetrievalService` semantics were not modified by this experiment.",
            "",
            "## 14. Remaining T044/T047 backlog",
            "",
            f"Unresolved cases explicitly allowed to remain unreachable: {boundary.unresolved_cases}",
            "",
            "## 15. Recommended next stage",
            "",
            f"{boundary.next_stage_required} (not implemented in this stage).",
            "",
            "### Interpretation boundary",
            "",
            "- Candidate is only a partial candidate-generation repair.",
            "- Candidate is not release-ready and receives no automatic production promotion.",
            "- Primary hit@4 restoration to historical Arm A levels is not achieved.",
        ]
    )
    if run.latency_seconds_total is not None:
        lines.extend(
            [
                "",
                f"_Informational total evaluation latency: {run.latency_seconds_total:.1f}s_",
            ]
        )
    return "\n".join(lines) + "\n"


def _validate_output_path(path: Path, *, project_root: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _ensure_not_protected(path: Path, *, project_root: Path) -> None:
    for protected in PROTECTED_ARTIFACT_PATHS:
        if path.resolve() == (project_root / protected).resolve():
            raise EvaluationOutputError(f"refusing to overwrite protected artifact: {protected}")


def _atomic_write_text(path: Path, content: str, *, temp_paths: list[Path]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_paths.append(temp_path)
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
