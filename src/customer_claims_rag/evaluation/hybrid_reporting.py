"""Markdown and JSON reporting for hybrid lexical + vector evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from customer_claims_rag.evaluation.hybrid_metrics import rebuild_hybrid_diagnostics
from customer_claims_rag.evaluation.hybrid_models import HybridCaseResult, HybridEvaluationRun
from customer_claims_rag.exceptions import EvaluationOutputError


DEFAULT_HYBRID_JSON = Path("data/05_evaluation/hybrid_lexical_vector_v1.json")
DEFAULT_HYBRID_MARKDOWN = Path("tests/08_hybrid_lexical_vector_results.md")

PROTECTED_BASELINE_PATHS = (
    Path("tests/03_test_results.md"),
    Path("tests/04_improvement_log.md"),
    Path("tests/05_answer_level_test_results.md"),
    Path("data/05_evaluation/retrieval_results.json"),
    Path("data/05_evaluation/reranking_ab_source_authority_v1.json"),
    Path("data/05_evaluation/vector_pool_expansion_v1.json"),
)


def write_hybrid_outputs(
    run: HybridEvaluationRun,
    *,
    output_json: Path,
    output_markdown: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write hybrid JSON and Markdown artifacts atomically."""
    json_path = _validate_output_path(output_json, project_root=project_root)
    markdown_path = _validate_output_path(output_markdown, project_root=project_root)
    _ensure_not_protected(json_path, project_root=project_root)
    _ensure_not_protected(markdown_path, project_root=project_root)

    json_content = json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    markdown_content = render_hybrid_markdown(run)

    temp_paths: list[Path] = []
    try:
        _atomic_write_text(json_path, json_content, temp_paths=temp_paths)
        _atomic_write_text(markdown_path, markdown_content, temp_paths=temp_paths)
    except Exception as exc:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        raise EvaluationOutputError(f"Hybrid output write failed: {exc}") from None
    return json_path, markdown_path


def rebuild_hybrid_outputs(
    *,
    artifact_path: Path,
    output_json: Path,
    output_markdown: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    """Rebuild channel diagnostics from self-contained JSON and write outputs."""
    resolved_artifact = _validate_output_path(artifact_path, project_root=project_root)
    run = HybridEvaluationRun.model_validate(
        json.loads(resolved_artifact.read_text(encoding="utf-8"))
    )
    rebuilt = rebuild_hybrid_diagnostics(run)
    return write_hybrid_outputs(
        rebuilt,
        output_json=output_json,
        output_markdown=output_markdown,
        project_root=project_root,
    )


def rebuild_report_from_artifact(
    *,
    artifact_path: Path,
    output_markdown: Path,
    project_root: Path,
) -> Path:
    """Render Markdown from a self-contained JSON artifact without BM25 or retrieval."""
    resolved_artifact = _validate_output_path(artifact_path, project_root=project_root)
    markdown_path = _validate_output_path(output_markdown, project_root=project_root)
    _ensure_not_protected(markdown_path, project_root=project_root)
    run = HybridEvaluationRun.model_validate(
        json.loads(resolved_artifact.read_text(encoding="utf-8"))
    )
    _atomic_write_text(markdown_path, render_hybrid_markdown(run), temp_paths=[])
    return markdown_path


def rebuild_markdown_from_artifact(
    *,
    artifact_path: Path,
    output_markdown: Path,
    project_root: Path,
) -> Path:
    """Backward-compatible alias for report-only rebuild from JSON."""
    return rebuild_report_from_artifact(
        artifact_path=artifact_path,
        output_markdown=output_markdown,
        project_root=project_root,
    )


def render_hybrid_markdown(run: HybridEvaluationRun) -> str:
    """Render human-readable hybrid experiment report from artifact."""
    experiment = run.experiment
    shared = run.shared_context
    reach = run.candidate_generation_comparison
    ranking = run.ranking_comparison
    acceptance = run.acceptance
    baseline = run.baseline.aggregate_metrics
    candidate = run.candidate.aggregate_metrics
    aggregate = ranking.aggregate
    lexical = run.lexical_index

    lines = [
        "# Hybrid lexical + vector results: hybrid-lexical-vector-v1",
        "",
        f"**Timestamp:** {run.timestamp.isoformat()}",
        f"**Experiment ID:** `{experiment.experiment_id}`",
        f"**Version:** `{experiment.version}`",
        f"**Experiment mode:** `{experiment.experiment_mode}`",
        f"**Config hash:** `{experiment.config_hash}`",
        f"**Reranker:** `{experiment.reranker_id}` (hash `{experiment.reranker_config_hash}`)",
        f"**Score adapter:** `{experiment.score_adapter_id}`",
        "",
        "## 1. Experiment contract",
        "",
        "- One shared vector retrieval per case (`vector_k=24`, `threshold=0.0`)",
        "- Lexical BM25 top-24 per case (no embeddings, no LLM)",
        "- Equal-weight RRF (`rrf_k=60`, weights 1.0/1.0) with normalized RRF base score",
        f"- Baseline arm: vector top-{shared.vector_k} → source-authority-v1 (vector similarity) → final top-{shared.final_top_k}",
        f"- Candidate arm: vector + lexical → fusion top-{shared.fusion_k} → {experiment.score_adapter_id} → source-authority-v1 → final top-{shared.final_top_k}",
        f"- Index fingerprint: `{shared.index_fingerprint}`",
        f"- Lexical index fingerprint: `{lexical.lexical_index_fingerprint}`",
        f"- Embedding model: `{shared.embedding_model}`",
        "- Expected sources used only for post-retrieval evaluation diagnostics",
        "",
        "Exact lexical top-24 pools were reconstructed once using the frozen tokenizer, "
        "BM25 parameters, queries, and verified corpus fingerprint. "
        "The reconstruction did not rerun vector retrieval, fusion, source-authority ranking, "
        "or final ranking.",
        "",
        "**Score semantics:** `normalized_rrf_score` is the reranker base score in the candidate arm. "
        "It is **not** vector similarity. Vector similarity is preserved separately in audit fields.",
        "",
        "### Baseline arm terminology",
        "",
        "Baseline arm этапа 2C.3 — закрытый candidate arm этапа 2C.2: "
        "vector pool@24, source-authority-v1 reranking, final top-12.",
        "",
        "## 2. Candidate-generation reachability",
        "",
        "| Pool | Primary reachable | Supporting reachable | Fully unreachable |",
        "|------|------------------:|-------------------:|-------------------|",
        f"| vector@24 (baseline) | {reach.baseline_primary_reachable}/{reach.baseline_primary_total} | "
        f"{reach.baseline_supporting_reachable}/{reach.baseline_supporting_total} | "
        f"{reach.baseline_fully_unreachable_cases} |",
        f"| fusion@24 (candidate) | {reach.candidate_primary_reachable}/{reach.candidate_primary_total} | "
        f"{reach.candidate_supporting_reachable}/{reach.candidate_supporting_total} | "
        f"{reach.candidate_fully_unreachable_cases} |",
        "",
        f"- High primary reachability: {reach.high_baseline_primary_reachable}/{reach.high_primary_total} → "
        f"{reach.high_candidate_primary_reachable}/{reach.high_primary_total}",
        f"- Critical primary reachability: {reach.critical_baseline_primary_reachable}/{reach.critical_primary_total} → "
        f"{reach.critical_candidate_primary_reachable}/{reach.critical_primary_total}",
        f"- Vector-only primary reachable cases: {reach.vector_only_reachable_cases or 'none'}",
        f"- Lexical-only primary reachable cases: {reach.lexical_only_reachable_cases or 'none'}",
        f"- Both-channel primary reachable cases: {reach.both_reachable_cases or 'none'}",
        f"- Neither-channel primary reachable cases: {reach.neither_reachable_cases or 'none'}",
        "",
        "## 3. Final ranking metrics",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|--------|---------:|----------:|------:|",
        f"| Hit@1 | {baseline.hit_rate_at_1:.3f} | {candidate.hit_rate_at_1:.3f} | {aggregate.hit_rate_at_1_delta:+.3f} |",
        f"| Hit@4 | {baseline.hit_rate_at_4:.3f} | {candidate.hit_rate_at_4:.3f} | {aggregate.hit_rate_at_4_delta:+.3f} |",
        f"| Hit@12 | {baseline.hit_rate_at_12:.3f} | {candidate.hit_rate_at_12:.3f} | {aggregate.hit_rate_at_12_delta:+.3f} |",
        f"| Document recall@4 | {baseline.document_recall_at_4:.3f} | {candidate.document_recall_at_4:.3f} | {aggregate.document_recall_at_4_delta:+.3f} |",
        f"| Primary hit@4 | {baseline.primary_source_hit_rate_at_4:.3f} | {candidate.primary_source_hit_rate_at_4:.3f} | {aggregate.primary_source_hit_rate_at_4_delta:+.3f} |",
        f"| Supporting hit@4 | {_supporting(baseline)} | {_supporting(candidate)} | {_supporting_delta(aggregate.supporting_source_hit_rate_at_4_delta)} |",
        f"| MRR | {baseline.mrr:.3f} | {candidate.mrr:.3f} | {aggregate.mrr_delta:+.3f} |",
        "",
        "## 4. Promotions and regressions",
        "",
        f"- Primary hit@4 promotions: {ranking.primary_hit_at_4_promotions or 'none'}",
        f"- Primary hit@4 regressions: {ranking.primary_hit_at_4_regressions or 'none'}",
        "",
        "## 5. Focus case diagnostics (reporting only)",
        "",
    ]
    for case_id in ("T004", "T040", "T047", "T053", "T057"):
        lines.extend(_render_focus_case(run, case_id))

    lines.extend(
        [
            "",
            "## 6. Acceptance verdicts",
            "",
            f"**Candidate-generation verdict:** **{acceptance.reachability_verdict}**",
            f"**Final-ranking verdict:** **{acceptance.ranking_verdict}**",
            "",
            f"- Hard invariants pass: **{acceptance.hard_invariants_pass}**",
            f"- Single vector retrieval pass: **{acceptance.single_retrieval_pass}**",
            f"- Reranker config hash match: **{acceptance.reranker_config_hash_match}**",
            f"- Lexical index valid: **{acceptance.lexical_index_valid}**",
            f"- High primary reachability non-regressed: **{acceptance.high_primary_reachability_non_regressed}**",
            f"- Critical primary reachability non-regressed: **{acceptance.critical_primary_reachability_non_regressed}**",
            f"- Fully unreachable non-increased: **{acceptance.fully_unreachable_non_increased}**",
            f"- Low/medium guardrails pass: **{acceptance.low_medium_slice_guardrails_pass}**",
            f"- Overall primary hit@4 delta: **{acceptance.overall_primary_hit_at_4_delta:+.3f}**",
            f"- Overall MRR delta: **{acceptance.overall_mrr_delta:+.3f}**",
            f"- Critical primary regressions: **{acceptance.critical_primary_regressions}**",
        ]
    )
    if acceptance.failure_reasons:
        lines.append("")
        lines.append("Failure reasons:")
        for reason in acceptance.failure_reasons:
            lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## 7. Retrieval configuration selection",
            "",
            "**hybrid-lexical-vector-v1:** rejected for MVP selection",
            "",
            "**Reason:**",
            "- critical primary regression T057",
            "- primary regressions T028/T057",
            "- high/critical reachability regression",
            "- supporting hit@4 decrease",
            "",
            "**Selected retrieval configuration:** stage 2C.2 candidate",
            "",
            "```text",
            "vector top-24",
            "→ source-authority-v1",
            "→ final top-12",
            "```",
            "",
            "No weighted RRF or post-hoc fusion tuning was performed.",
            "Retrieval experimentation is frozen after stage 2C.3.",
            "",
            "## 8. Limitations",
            "",
            "- Hybrid fusion improves candidate reachability but does not guarantee top-4 gains.",
            "- Normalized RRF base scores are not comparable to vector similarity magnitudes.",
            "- Frozen source-authority-v1 max bonus (+0.03) cannot overcome large base-score gaps.",
            "",
            "---",
            "",
            "*Generated by hybrid lexical + vector evaluation (stage 2C.3).*",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_focus_case(run: HybridEvaluationRun, case_id: str) -> list[str]:
    case = _find_case(run.case_results, case_id)
    if case is None:
        return [f"### {case_id}", "", "- Case not found in artifact.", ""]

    candidate_audit = case.candidate_ranking.ordered_candidates
    primary_lex_rank = _best_lexical_rank(candidate_audit, case.expected_primary_documents)
    primary_fusion_rank = case.candidate_fusion_pool.reachability.primary_best_pool_rank
    primary_final_rank = _best_final_rank(candidate_audit, case.expected_primary_documents)

    return [
        f"### {case_id}",
        "",
        f"- **Risk / category:** {case.risk} / {case.category}",
        f"- **Expected primary:** {case.expected_primary_documents}",
        f"- **Baseline vector pool@24 primary reachable:** {case.baseline_pool.reachability.primary_reachable} "
        f"(best rank {case.baseline_pool.reachability.primary_best_pool_rank})",
        f"- **Candidate fusion pool@24 primary reachable:** {case.candidate_fusion_pool.reachability.primary_reachable} "
        f"(best rank {primary_fusion_rank})",
        f"- **Lexical primary best rank (pool):** {_best_lexical_pool_rank(case)}",
        f"- **Lexical primary best rank (candidate audit):** {primary_lex_rank}",
        f"- **Primary in final top-12 (candidate):** {case.candidate_ranking.flags.primary_in_final_top12}",
        f"- **Candidate primary hit@4:** {case.candidate_ranking.metrics.primary_hit_at_4}",
        f"- **Vector reachable (primary):** {case.candidate_fusion_pool.vector_reachable}",
        f"- **Lexical reachable (primary):** {case.candidate_fusion_pool.lexical_reachable}",
        f"- **Retrieval channel (primary):** {case.candidate_fusion_pool.retrieval_channel}",
        "",
    ]


def _find_case(case_results: list[HybridCaseResult], case_id: str) -> HybridCaseResult | None:
    return next((case for case in case_results if case.case_id == case_id), None)


def _best_lexical_pool_rank(case: HybridCaseResult) -> int | None:
    if case.lexical_pool is None or not case.expected_primary_documents:
        return None
    primary = set(case.expected_primary_documents)
    best: int | None = None
    for candidate in case.lexical_pool.candidates:
        if candidate.document_id in primary:
            best = candidate.lexical_rank if best is None else min(best, candidate.lexical_rank)
    return best


def _best_lexical_rank(candidates, primary_docs: list[str]) -> int | None:
    if not primary_docs:
        return None
    primary = set(primary_docs)
    best: int | None = None
    for item in candidates:
        if item.document_id in primary and item.lexical_rank is not None:
            best = item.lexical_rank if best is None else min(best, item.lexical_rank)
    return best


def _best_final_rank(candidates, primary_docs: list[str]) -> int | None:
    if not primary_docs:
        return None
    primary = set(primary_docs)
    best: int | None = None
    for item in candidates:
        if item.document_id in primary:
            best = item.final_rank if best is None else min(best, item.final_rank)
    return best


def _supporting(metrics) -> str:
    if metrics.supporting_source_hit_rate_at_4 is None:
        return "n/a"
    return f"{metrics.supporting_source_hit_rate_at_4:.3f}"


def _supporting_delta(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.3f}"


def _validate_output_path(path: Path, *, project_root: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not str(resolved).startswith(str(project_root.resolve())):
        raise EvaluationOutputError(f"output path escapes project root: {path}")
    return resolved


def _ensure_not_protected(path: Path, *, project_root: Path) -> None:
    for protected in PROTECTED_BASELINE_PATHS:
        if path.resolve() == (project_root / protected).resolve():
            raise EvaluationOutputError(f"refusing to overwrite protected artifact: {protected}")


def _atomic_write_text(path: Path, content: str, *, temp_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_paths.append(temp_path)
    temp_path.replace(path)
