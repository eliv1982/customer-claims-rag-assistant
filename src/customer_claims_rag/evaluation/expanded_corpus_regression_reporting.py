"""Markdown and JSON reporting for expanded corpus frozen regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from customer_claims_rag.evaluation.expanded_corpus_regression_models import (
    ExpandedCorpusRegressionRun,
    SEMANTIC_OVERLAY_BY_CASE,
)
from customer_claims_rag.exceptions import EvaluationOutputError

DEFAULT_REGRESSION_JSON = Path("data/05_evaluation/expanded_corpus_frozen_regression_v1.json")
DEFAULT_REGRESSION_MARKDOWN = Path("tests/09_expanded_corpus_frozen_regression_results.md")

PROTECTED_ARTIFACT_PATHS = (
    Path("data/05_evaluation/hybrid_lexical_vector_v1.json"),
    Path("data/05_evaluation/vector_pool_expansion_v1.json"),
    Path("data/05_evaluation/reranking_ab_source_authority_v1.json"),
    Path("data/05_evaluation/retrieval_results.json"),
    Path("tests/03_test_results.md"),
    Path("tests/07_vector_pool_expansion_results.md"),
)


def write_expanded_corpus_regression_outputs(
    run: ExpandedCorpusRegressionRun,
    *,
    output_json: Path,
    output_markdown: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    json_path = _validate_output_path(output_json, project_root=project_root)
    markdown_path = _validate_output_path(output_markdown, project_root=project_root)
    _ensure_not_protected(json_path, project_root=project_root)
    _ensure_not_protected(markdown_path, project_root=project_root)

    json_content = json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    markdown_content = render_expanded_corpus_regression_markdown(run)
    temp_paths: list[Path] = []
    try:
        _atomic_write_text(json_path, json_content, temp_paths=temp_paths)
        _atomic_write_text(markdown_path, markdown_content, temp_paths=temp_paths)
    except Exception as exc:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        raise EvaluationOutputError(
            f"expanded corpus regression output write failed: {exc}",
        ) from None
    return json_path, markdown_path


def render_expanded_corpus_regression_markdown(run: ExpandedCorpusRegressionRun) -> str:
    lines: list[str] = [
        "# Expanded corpus frozen regression: expanded_corpus_frozen_regression_v1",
        "",
        f"**Timestamp:** {run.timestamp.isoformat()}",
        f"**Evaluation ID:** `{run.evaluation_id}`",
        f"**Frozen question set:** `{run.frozen_question_set_id}`",
        f"**Retrieval chain:** `{run.retrieval_chain}`",
        f"**Verdict:** `{run.verdict}`",
        "",
        run.verdict_rationale,
        "",
        "## 1. Index arms",
        "",
        "| Arm | Path | Fingerprint | Chunks | Documents |",
        "|-----|------|-------------|-------:|----------:|",
        f"| A historical | `{run.arm_a.index_path}` | `{run.arm_a.index_fingerprint[:16]}...` | "
        f"{run.arm_a.chunk_count} | {run.arm_a.document_count} |",
        f"| B production | `{run.arm_b.index_path}` | `{run.arm_b.index_fingerprint[:16]}...` | "
        f"{run.arm_b.chunk_count} | {run.arm_b.document_count} |",
        "",
        "Both arms used identical frozen retrieval config "
        f"`{run.arm_a.frozen_retrieval_config_id}` "
        f"(hash `{run.arm_a.frozen_retrieval_config_hash[:16]}...`), "
        f"reranker `{run.arm_a.reranker_id}`, pool@{run.arm_a.candidate_pool_k}, "
        f"final top-{run.arm_a.final_top_k}, threshold {run.arm_a.threshold}.",
        "",
        "## 2. Aggregate metrics (final top-12 after rerank)",
        "",
        "| Metric | Arm A | Arm B | Delta (B−A) |",
        "|--------|------:|------:|------------:|",
    ]
    a = run.arm_a.aggregate_metrics
    b = run.arm_b.aggregate_metrics
    d = run.metric_deltas
    metric_rows = [
        ("Hit@1", a.hit_rate_at_1, b.hit_rate_at_1, d.hit_rate_at_1),
        ("Hit@4", a.hit_rate_at_4, b.hit_rate_at_4, d.hit_rate_at_4),
        ("Hit@12", a.hit_rate_at_12, b.hit_rate_at_12, d.hit_rate_at_12),
        ("Primary hit@4", a.primary_source_hit_rate_at_4, b.primary_source_hit_rate_at_4, d.primary_source_hit_rate_at_4),
        ("MRR", a.mrr, b.mrr, d.mrr),
        ("Document recall@4", a.document_recall_at_4, b.document_recall_at_4, d.document_recall_at_4),
        ("Document recall@12", a.document_recall_at_12, b.document_recall_at_12, d.document_recall_at_12),
    ]
    for label, av, bv, delta in metric_rows:
        lines.append(f"| {label} | {av:.3f} | {bv:.3f} | {delta:+.3f} |")

    lines.extend(
        [
            "",
            "## 3. Pool@24 reachability",
            "",
            f"- Primary reachable: Arm A {run.arm_a.pool24_primary_reachable}/"
            f"{run.arm_a.pool24_primary_total} -> Arm B {run.arm_b.pool24_primary_reachable}/"
            f"{run.arm_b.pool24_primary_total} (delta {d.pool24_primary_reachable_delta:+d})",
            f"- Fully unreachable primaries (A): {run.arm_a.pool24_fully_unreachable_cases}",
            f"- Fully unreachable primaries (B): {run.arm_b.pool24_fully_unreachable_cases}",
            f"- High-risk primary reachable@24: {run.arm_a.high_primary_reachable_pool24}/"
            f"{run.arm_a.high_primary_total} -> {run.arm_b.high_primary_reachable_pool24}/"
            f"{run.arm_b.high_primary_total}",
            f"- Critical primary reachable@24: {run.arm_a.critical_primary_reachable_pool24}/"
            f"{run.arm_a.critical_primary_total} -> {run.arm_b.critical_primary_reachable_pool24}/"
            f"{run.arm_b.critical_primary_total}",
            "",
            "## 4. Risk slices (final top-12)",
            "",
            "| Risk | Arm A Hit@4 | Arm B Hit@4 | Arm A MRR | Arm B MRR |",
            "|------|------------:|------------:|--------:|--------:|",
        ]
    )
    risk_a = {item.risk_level: item for item in run.arm_a.risk_metrics}
    risk_b = {item.risk_level: item for item in run.arm_b.risk_metrics}
    for risk in ("low", "medium", "high", "critical"):
        if risk in risk_a and risk in risk_b:
            lines.append(
                f"| {risk} | {risk_a[risk].hit_rate_at_4:.3f} | {risk_b[risk].hit_rate_at_4:.3f} | "
                f"{risk_a[risk].mrr:.3f} | {risk_b[risk].mrr:.3f} |"
            )

    lines.extend(["", "## 5. FAQ dominance", ""])
    faq = run.faq_dominance
    lines.extend(
        [
            f"- FAQ top-1 slots: Arm A {faq.arm_a_faq_top1_count} -> Arm B {faq.arm_b_faq_top1_count}",
            f"- Questions with FAQ in top-4: Arm A {faq.arm_a_questions_with_faq_in_top4} "
            f"-> Arm B {faq.arm_b_questions_with_faq_in_top4}",
            f"- FAQ outranks all primaries in top-4: Arm A {faq.arm_a_faq_outranks_all_primaries} "
            f"-> Arm B {faq.arm_b_faq_outranks_all_primaries}",
            "",
            "## 6. New document footprint (Arm B)",
            "",
        ]
    )
    for footprint in run.new_document_footprints:
        lines.extend(
            [
                f"### `{footprint.document_id}`",
                "",
                f"- top-1: {len(footprint.top1_questions)} ({', '.join(footprint.top1_questions) or '-'})",
                f"- top-4: {len(footprint.top4_questions)}",
                f"- top-12: {len(footprint.top12_questions)}",
                f"- pool@24: {len(footprint.pool24_questions)}",
                f"- likely helpful: {', '.join(footprint.likely_helpful_questions) or '-'}",
                f"- irrelevant noise top-4: {', '.join(footprint.irrelevant_noise_questions) or '-'}",
                f"- displaces frozen primary: {', '.join(footprint.displaces_frozen_primary_questions) or '-'}",
                "",
            ]
        )

    lines.extend(["## 7. Special cases", ""])
    paired_by_id = {case.case_id: case for case in run.paired_cases}
    for case_id in ("T004", "T040", "T047", "T055", "T023", "T027", "T053"):
        case = paired_by_id.get(case_id)
        if case is None:
            continue
        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- Risk: {case.risk}; expected primary: {case.expected_primary_documents}",
                f"- Arm A primary rank (final): {case.arm_a_primary_rank_final}; "
                f"Arm B: {case.arm_b_primary_rank_final}",
                f"- Arm A top-4: {case.arm_a_top4_document_ids}",
                f"- Arm B top-4: {case.arm_b_top4_document_ids}",
                f"- Primary in pool@24: A={case.arm_a_primary_in_pool24}, B={case.arm_b_primary_in_pool24}",
                f"- Classification: `{case.classification}`",
                f"- {case.explanation}",
                "",
            ]
        )

    lines.extend(["## 8. Semantic displacement overlay", ""])
    for review in run.semantic_overlay:
        lines.extend(
            [
                f"### {review.case_id} -> `{review.overlay_document_id}`",
                "",
                f"- more specific than frozen: {review.more_specific_than_frozen}",
                f"- frozen primary still in Arm B top-12: {review.frozen_primary_in_arm_b_top12}",
                f"- recommend extension set: {review.recommend_extension_set}",
                f"- recommend benchmark modification: {review.recommend_benchmark_modification}",
                f"- {review.notes}",
                "",
            ]
        )

    lines.extend(["## 9. Paired per-question summary", ""])
    lines.append("| Case | Class | Harmful | Arm A rank | Arm B rank | Arm B top-4 |")
    lines.append("|------|-------|---------|----------:|----------:|-------------|")
    for case in run.paired_cases:
        top4 = ", ".join(case.arm_b_top4_document_ids[:4]) or "-"
        lines.append(
            f"| {case.case_id} | {case.classification} | {case.harmful_regression} | "
            f"{case.arm_a_primary_rank_final} | {case.arm_b_primary_rank_final} | {top4} |"
        )

    lines.extend(
        [
            "",
            "## 10. Notes",
            "",
            "- Strict frozen metrics use the unchanged 60-question benchmark expectations.",
            "- Semantic overlay candidates: "
            + ", ".join(f"{k}->{v}" for k, v in SEMANTIC_OVERLAY_BY_CASE.items())
            + ".",
            "- Release-validation extension set remains pending for Stage 4C.3.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_output_path(path: Path, *, project_root: Path) -> Path:
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (project_root / path).resolve()
    if not str(resolved).startswith(str(project_root.resolve())):
        raise EvaluationOutputError(f"output path must stay inside project root: {path}")
    return resolved


def _ensure_not_protected(path: Path, *, project_root: Path) -> None:
    for protected in PROTECTED_ARTIFACT_PATHS:
        if path.resolve() == (project_root / protected).resolve():
            raise EvaluationOutputError(f"refusing to overwrite protected artifact: {protected}")


def _atomic_write_text(path: Path, content: str, *, temp_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        suffix=path.suffix,
        dir=str(path.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    temp_paths.append(temp_path)
    import os

    os.close(fd)
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
