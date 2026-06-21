"""Evaluation reporting and artifact persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from customer_claims_rag.evaluation.metrics import primary_found_in_top12_outside_top4
from customer_claims_rag.evaluation.models import EvaluationRun, ThresholdSliceMetrics
from customer_claims_rag.evaluation.path_helpers import (
    COMMITTED_IMPROVEMENT_LOG,
    COMMITTED_RESULTS_REPORT,
    validate_committed_report_path,
    validate_evaluation_output_path,
)
from customer_claims_rag.exceptions import EvaluationOutputError


def write_evaluation_outputs(
    run: EvaluationRun,
    *,
    output_json: Path,
    output_results: Path,
    output_improvement_log: Path,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    """Write JSON and Markdown artifacts with best-effort atomic replace."""
    json_path = validate_evaluation_output_path(output_json, project_root=project_root)
    results_path = validate_committed_report_path(
        output_results,
        project_root=project_root,
        expected=COMMITTED_RESULTS_REPORT,
    )
    improvement_path = validate_committed_report_path(
        output_improvement_log,
        project_root=project_root,
        expected=COMMITTED_IMPROVEMENT_LOG,
    )

    json_content = json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    results_content = render_results_markdown(run)
    improvement_content = render_improvement_log_markdown(run)

    written: list[str] = []
    temp_paths: list[Path] = []
    try:
        _atomic_write_text(json_path, json_content, temp_paths=temp_paths)
        written.append("json")
        _atomic_write_text(results_path, results_content, temp_paths=temp_paths)
        written.append("results_markdown")
        _atomic_write_text(improvement_path, improvement_content, temp_paths=temp_paths)
        written.append("improvement_log")
    except Exception as exc:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        raise EvaluationOutputError(
            "evaluation output write failed; output set may be partially updated. "
            f"Completed writes: {', '.join(written) or 'none'}. "
            "Compare evaluation_result_id across artifacts if present."
        ) from None
    return json_path, results_path, improvement_path


def render_results_markdown(run: EvaluationRun) -> str:
    metadata = run.run_metadata
    aggregate = run.aggregate_metrics
    failed_cases = [
        case
        for case in run.case_results
        if case.status == "success"
        and not case.fallback_expected
        and case.expected_primary_documents
        and not case.primary_hit_at_4
    ]
    failed_cases.sort(key=lambda case: case.test_id)
    dirty_lines: list[str] = []
    if metadata.git_dirty is True:
        dirty_lines = [
            "> **Примечание по Git:** commit hash alone does not fully identify evaluation "
            "implementation because the working tree was dirty.",
            "",
        ]
    lines = [
        "# Результаты baseline retrieval evaluation",
        "",
        f"**Evaluation result ID:** `{metadata.evaluation_result_id}`",
        f"**Дата прогона:** {metadata.timestamp.isoformat()}",
        f"**Git commit:** `{metadata.git_commit or 'unknown'}`",
        f"**Working tree dirty:** {metadata.git_dirty if metadata.git_dirty is not None else 'unknown'}",
        f"**Git status summary:** {metadata.git_status_summary or 'unknown'}",
        f"**Index fingerprint:** `{metadata.index_fingerprint}`",
        f"**Embedding model:** `{metadata.embedding_model}`",
        f"**Collection:** `{metadata.collection}`",
        f"**Vector dimension:** {metadata.vector_dimension}",
        f"**Chunk count:** {metadata.chunk_count}",
        f"**Document count:** {metadata.document_count}",
        f"**Threshold:** {metadata.threshold:.2f} (filtering disabled for baseline recall)",
        f"**top_k / fetch_k:** {metadata.top_k} / {metadata.fetch_k}",
        "",
        *dirty_lines,
        "> **Ограничение:** это retrieval-only evaluation. Метрики не оценивают качество LLM-ответов, "
        "risk/handoff classification, answer factuality или Markdown output contract.",
        "",
        "> **Output consistency:** каждый файл записывается атомарно, но набор из трёх файлов "
        "не является общей транзакцией. `evaluation_result_id` должен совпадать во всех артефактах.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Total cases | {aggregate.total_cases} |",
        f"| Successfully evaluated | {aggregate.successfully_evaluated_cases} |",
        f"| Technical errors | {aggregate.technical_error_count} |",
        f"| Source-recall cases | {aggregate.source_recall_case_count} |",
        f"| Fallback cases | {aggregate.fallback_case_count} |",
        f"| Hit@1 | {aggregate.hit_rate_at_1:.3f} |",
        f"| Hit@4 | {aggregate.hit_rate_at_4:.3f} |",
        f"| Hit@12 | {aggregate.hit_rate_at_12:.3f} |",
        f"| Document recall@1 | {aggregate.document_recall_at_1:.3f} |",
        f"| Document recall@4 | {aggregate.document_recall_at_4:.3f} |",
        f"| Document recall@12 | {aggregate.document_recall_at_12:.3f} |",
        f"| MRR | {aggregate.mrr:.3f} |",
        f"| Primary source hit@1 | {aggregate.primary_source_hit_rate_at_1:.3f} |",
        f"| Primary source hit@4 | {aggregate.primary_source_hit_rate_at_4:.3f} |",
        f"| Supporting source hit@4 | {_format_supporting_hit(aggregate)} |",
        f"| No-result rate | {aggregate.no_result_rate:.3f} |",
        f"| Mean top-1 similarity | {_fmt(aggregate.mean_top1_similarity)} |",
        f"| Median top-1 similarity | {_fmt(aggregate.median_top1_similarity)} |",
        f"| Min top-1 similarity | {_fmt(aggregate.min_top1_similarity)} |",
        f"| Max top-1 similarity | {_fmt(aggregate.max_top1_similarity)} |",
        "",
        "## Metrics by risk",
        "",
        "| Risk | Cases | Hit@1 | Hit@4 | Hit@12 | MRR | Mean top-1 sim |",
        "|------|------:|------:|------:|-------:|----:|---------------:|",
    ]
    for slice_metrics in run.risk_metrics:
        lines.append(
            f"| {slice_metrics.risk_level} | {slice_metrics.case_count} | "
            f"{slice_metrics.hit_rate_at_1:.3f} | {slice_metrics.hit_rate_at_4:.3f} | "
            f"{slice_metrics.hit_rate_at_12:.3f} | {slice_metrics.mrr:.3f} | "
            f"{_fmt(slice_metrics.mean_top1_similarity)} |"
        )
    lines.extend(["", "## Metrics by category", ""])
    if run.category_metrics:
        lines.extend(
            [
                "| Category | Cases | Hit@1 | Hit@4 | Hit@12 | MRR | Mean top-1 sim |",
                "|----------|------:|------:|------:|-------:|----:|---------------:|",
            ]
        )
        for slice_metrics in run.category_metrics:
            lines.append(
                f"| {slice_metrics.category} | {slice_metrics.case_count} | "
                f"{slice_metrics.hit_rate_at_1:.3f} | {slice_metrics.hit_rate_at_4:.3f} | "
                f"{slice_metrics.hit_rate_at_12:.3f} | {slice_metrics.mrr:.3f} | "
                f"{_fmt(slice_metrics.mean_top1_similarity)} |"
            )
    else:
        lines.append("_Category slices unavailable._")
    lines.extend(["", "## Threshold sweep", ""])
    lines.extend(_threshold_table(run.threshold_analysis))
    lines.extend(["", "## Fallback / no-grounding separation", ""])
    lines.extend(_fallback_section(run))
    lines.extend(["", "## Failed cases (primary source miss@4)", ""])
    if failed_cases:
        for case in failed_cases[:20]:
            extra = ""
            if primary_found_in_top12_outside_top4(
                case.retrieved_chunks,
                case.expected_primary_documents,
            ):
                extra = "; primary source found in top-12 but outside top-4"
            lines.append(
                f"- **{case.test_id}** ({case.expected_risk}): expected primary="
                f"{case.expected_primary_documents}; top retrieved @4="
                f"{case.retrieved_document_ids_at_4 or ['<none>']}; "
                f"top-1 sim={_fmt(case.top1_similarity)}{extra}"
            )
        if len(failed_cases) > 20:
            lines.append(f"- ... and {len(failed_cases) - 20} more")
    else:
        lines.append("_No primary source miss@4 among source-recall cases._")
    lines.extend(["", "## Short interpretation", ""])
    lines.extend(_interpretation(run))
    lines.extend(["", "---", "", "*Generated by baseline retrieval evaluation (stage 2B).*"])
    return "\n".join(lines) + "\n"


def render_improvement_log_markdown(run: EvaluationRun) -> str:
    metadata = run.run_metadata
    aggregate = run.aggregate_metrics
    critical_misses = [
        case
        for case in run.case_results
        if case.expected_risk == "critical"
        and case.status == "success"
        and not case.fallback_expected
        and case.expected_primary_documents
        and not case.primary_hit_at_4
    ]
    high_misses = [
        case
        for case in run.case_results
        if case.expected_risk == "high"
        and case.status == "success"
        and not case.fallback_expected
        and case.expected_primary_documents
        and not case.primary_hit_at_4
    ]
    lines = [
        "# Improvement log: baseline retrieval",
        "",
        f"**Evaluation result ID:** `{metadata.evaluation_result_id}`",
        f"**Дата baseline run:** {metadata.timestamp.date().isoformat()}",
        f"**Git commit:** `{metadata.git_commit or 'unknown'}`",
        f"**Working tree dirty:** {metadata.git_dirty if metadata.git_dirty is not None else 'unknown'}",
        "",
        "## Baseline configuration",
        "",
        "- Baseline dense retrieval without reranking, source priority or risk-aware boosting",
        f"- Embedding model: `{metadata.embedding_model}`",
        f"- Collection: `{metadata.collection}`",
        f"- Index fingerprint: `{metadata.index_fingerprint}`",
        f"- Evaluation threshold: `{metadata.threshold:.2f}` (no filtering for baseline recall)",
        f"- Candidate pool: fetch_k={metadata.fetch_k}, metrics reported at k=1/4/12",
        "- All @k metrics use the first k raw chunks, then deduplicate documents inside that window",
        "- Retrieval-only metrics; LLM answer pipeline not evaluated",
        "",
        "## Main failure patterns",
        "",
        f"- Overall primary source hit@4: **{aggregate.primary_source_hit_rate_at_4:.3f}** "
        f"on {aggregate.source_recall_case_count} source-recall cases",
        f"- Supporting source hit@4: **{_format_supporting_hit(aggregate)}**",
        f"- Critical primary miss@4: **{len(critical_misses)}** cases",
        f"- High primary miss@4: **{len(high_misses)}** cases",
        f"- FAQ or generic docs may appear instead of profile documents in top-4 chunks",
        f"- Technical errors: **{aggregate.technical_error_count}**",
        "",
        "## Possible causes",
        "",
        "- Semantic similarity favors FAQ chunks with overlapping wording",
        "- Multiple valid documents share vocabulary without source-priority reranking",
        "- Chunk boundaries split policy sections away from query-specific terms",
        "- Boundary/adversarial cases still retrieve plausible but non-primary documents",
        "",
        "## Candidate improvements (not implemented in 2B)",
        "",
        "- Source-priority reranking after baseline recall measurement",
        "- Risk-aware reranking for critical/high cases",
        "- Hybrid BM25 + dense retrieval",
        "- Query rewriting for long mixed messages",
        "- Production threshold selection after separate review",
        "",
        "## Explicitly not implemented yet",
        "",
        "- LLM answer generation and factuality scoring",
        "- Risk/handoff classification model evaluation",
        "- Critical safety pass rate based on generated answers",
        "- Markdown output contract validation",
        "",
        "## Improvement policy",
        "",
        "Do **not** claim retrieval improvement until an A/B rerun on this same 60-case corpus "
        "shows measurable delta with unchanged ingestion and corpus fingerprint.",
        "",
        "## Critical/high primary misses",
        "",
    ]
    for case in critical_misses + high_misses:
        lines.append(
            f"- **{case.test_id}** ({case.expected_risk}): expected primary "
            f"{case.expected_primary_documents}; top-4 docs @4 "
            f"{case.retrieved_document_ids_at_4 or ['<none>']}"
        )
    if not critical_misses and not high_misses:
        lines.append("_No critical/high primary miss@4 in this run._")
    lines.extend(["", "---", "", "*Baseline captured at stage 2B; changes require A/B rerun.*"])
    return "\n".join(lines) + "\n"


def _threshold_table(slices: list[ThresholdSliceMetrics]) -> list[str]:
    lines = [
        "| Threshold | Cases w/ result | No-result rate | Hit@1 | Hit@4 | Primary hit@4 | MRR | Critical hit@4 | High hit@4 |",
        "|----------:|----------------:|---------------:|------:|------:|--------------:|----:|---------------:|-----------:|",
    ]
    for item in slices:
        lines.append(
            f"| {item.threshold:.2f} | {item.cases_with_at_least_one_result} | "
            f"{item.no_result_rate:.3f} | {item.hit_rate_at_1:.3f} | {item.hit_rate_at_4:.3f} | "
            f"{item.primary_source_hit_rate_at_4:.3f} | {item.mrr:.3f} | "
            f"{item.critical_case_hit_rate_at_4:.3f} | {item.high_case_hit_rate_at_4:.3f} |"
        )
    return lines


def _fallback_section(run: EvaluationRun) -> list[str]:
    fallback = run.fallback_analysis
    fallback_cases = [case for case in run.case_results if case.fallback_expected]
    lines = [
        f"- Fallback cases (`fallback_expected=true`): **{fallback.get('case_count', 0)}**",
        f"- Mean top-1 similarity: {_fmt(fallback.get('mean_top1_similarity'))}",
        f"- Min / max top-1 similarity: {_fmt(fallback.get('min_top1_similarity'))} / "
        f"{_fmt(fallback.get('max_top1_similarity'))}",
        "- These cases are excluded from source-recall aggregates; similarity distribution "
        "is tracked separately and is **not** fallback accuracy.",
    ]
    for case in fallback_cases:
        lines.append(
            f"- **{case.test_id}**: top-1 sim={_fmt(case.top1_similarity)}, "
            f"top doc={case.retrieved_document_ids[:1] or ['<none>']}"
        )
    return lines


def _interpretation(run: EvaluationRun) -> list[str]:
    aggregate = run.aggregate_metrics
    critical = next(
        (item for item in run.risk_metrics if item.risk_level == "critical"),
        None,
    )
    high = next((item for item in run.risk_metrics if item.risk_level == "high"), None)
    lines = [
        f"- Baseline hit@4 across source-recall cases: **{aggregate.hit_rate_at_4:.1%}**; "
        f"primary source hit@4: **{aggregate.primary_source_hit_rate_at_4:.1%}**.",
        f"- Supporting source hit@4: **{_format_supporting_hit(aggregate)}**.",
        "- All @k metrics are computed from the first k raw chunks only.",
        "- Similarity scores are cosine-derived distances, not probabilities.",
        "- Threshold filtering was disabled (`threshold=0.0`) to measure raw recall.",
    ]
    if critical:
        lines.append(
            f"- Critical cases: hit@4 **{critical.hit_rate_at_4:.1%}**, MRR **{critical.mrr:.3f}**."
        )
    if high:
        lines.append(f"- High cases: hit@4 **{high.hit_rate_at_4:.1%}**, MRR **{high.mrr:.3f}**.")
    lines.append(
        "- This run does **not** establish production readiness; review critical/high primary misses "
        "before any threshold or reranking decision."
    )
    return lines


def _format_supporting_hit(aggregate) -> str:
    if aggregate.cases_with_supporting_documents == 0:
        return "n/a (no cases with supporting sources)"
    rate = aggregate.supporting_source_hit_rate_at_4
    assert rate is not None
    return (
        f"{rate:.3f} ({aggregate.supporting_source_hits_at_4}/"
        f"{aggregate.cases_with_supporting_documents})"
    )


def _fmt(value: object | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _atomic_write_text(path: Path, content: str, *, temp_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        suffix=path.suffix or ".tmp",
        dir=str(path.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    temp_paths.append(temp_path)
    try:
        import os

        os.close(fd)
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
        temp_paths.remove(temp_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if temp_path in temp_paths:
            temp_paths.remove(temp_path)
        raise
