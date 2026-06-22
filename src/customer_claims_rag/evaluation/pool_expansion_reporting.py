"""Markdown and JSON reporting for pool expansion evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from customer_claims_rag.evaluation.pool_expansion_models import PoolExpansionEvaluationRun
from customer_claims_rag.exceptions import EvaluationOutputError


DEFAULT_POOL_EXPANSION_JSON = Path("data/05_evaluation/vector_pool_expansion_v1.json")
DEFAULT_POOL_EXPANSION_MARKDOWN = Path("tests/07_vector_pool_expansion_results.md")

PROTECTED_BASELINE_PATHS = (
    Path("tests/03_test_results.md"),
    Path("tests/04_improvement_log.md"),
    Path("tests/05_answer_level_test_results.md"),
    Path("data/05_evaluation/retrieval_results.json"),
    Path("data/05_evaluation/reranking_ab_source_authority_v1.json"),
)


def write_pool_expansion_outputs(
    run: PoolExpansionEvaluationRun,
    *,
    output_json: Path,
    output_markdown: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write pool expansion JSON and Markdown artifacts atomically."""
    json_path = _validate_output_path(output_json, project_root=project_root)
    markdown_path = _validate_output_path(output_markdown, project_root=project_root)
    _ensure_not_protected(json_path, project_root=project_root)
    _ensure_not_protected(markdown_path, project_root=project_root)

    json_content = json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    markdown_content = render_pool_expansion_markdown(run)

    temp_paths: list[Path] = []
    try:
        _atomic_write_text(json_path, json_content, temp_paths=temp_paths)
        _atomic_write_text(markdown_path, markdown_content, temp_paths=temp_paths)
    except Exception as exc:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        raise EvaluationOutputError(f"Pool expansion output write failed: {exc}") from None
    return json_path, markdown_path


def rebuild_markdown_from_artifact(
    *,
    artifact_path: Path,
    output_markdown: Path,
    project_root: Path,
) -> Path:
    """Rebuild Markdown report from an existing self-contained JSON artifact."""
    resolved_artifact = _validate_output_path(artifact_path, project_root=project_root)
    markdown_path = _validate_output_path(output_markdown, project_root=project_root)
    _ensure_not_protected(markdown_path, project_root=project_root)
    run = PoolExpansionEvaluationRun.model_validate(
        json.loads(resolved_artifact.read_text(encoding="utf-8"))
    )
    _atomic_write_text(markdown_path, render_pool_expansion_markdown(run), temp_paths=[])
    return markdown_path


def render_pool_expansion_markdown(run: PoolExpansionEvaluationRun) -> str:
    """Render human-readable pool expansion report from a self-contained artifact."""
    experiment = run.experiment
    shared = run.shared_context
    reach = run.reachability_comparison
    ranking = run.ranking_comparison
    acceptance = run.acceptance
    baseline_rank = run.baseline_ranking.aggregate_metrics
    candidate_rank = run.candidate_ranking.aggregate_metrics
    aggregate = ranking.aggregate

    lines = [
        "# Vector pool expansion results: vector-pool-expansion-v1",
        "",
        f"**Timestamp:** {run.timestamp.isoformat()}",
        f"**Experiment ID:** `{experiment.experiment_id}`",
        f"**Version:** `{experiment.version}`",
        f"**Experiment mode:** `{experiment.experiment_mode}`",
        f"**Config hash:** `{experiment.config_hash}`",
        f"**Reranker:** `{experiment.reranker_id}` (hash `{experiment.reranker_config_hash}`)",
        "",
        "## 1. Experiment contract",
        "",
        "- One shared vector retrieval per case (`fetch_k=24`, `threshold=0.0`)",
        f"- Baseline arm: pool@**{shared.baseline_pool_k}** → frozen reranker → final top-{shared.final_top_k}",
        f"- Candidate arm: pool@**{shared.candidate_pool_k}** → frozen reranker → final top-{shared.final_top_k}",
        f"- Index fingerprint: `{shared.index_fingerprint}`",
        f"- Embedding model: `{shared.embedding_model}`",
        "- Expected sources used only for post-retrieval evaluation diagnostics",
        "",
        "### Терминология baseline arm",
        "",
        "Baseline arm этапа 2C.2 — это frozen конфигурация candidate arm этапа 2C.1: "
        "vector pool@12, затем source-authority-v1 reranking и final top-12.",
        "",
        "Это **не** no-reranker baseline arm из исходного A/B-артефакта 2C.1.",
        "",
        "```text",
        "2C.1 no-rerank baseline  ≠  2C.2 baseline arm",
        "2C.2 baseline arm        =  2C.1 candidate arm  =  pool@12 + source-authority-v1",
        "```",
        "",
        "## 2. Pool reachability",
        "",
        "| Pool | Primary reachable | Supporting reachable | Fully unreachable |",
        "|------|------------------:|-------------------:|-------------------|",
        f"| pool@{shared.baseline_pool_k} | {reach.baseline_primary_reachable}/{reach.baseline_primary_total} | "
        f"{reach.baseline_supporting_reachable}/{reach.baseline_supporting_total} | "
        f"{reach.baseline_fully_unreachable_cases} |",
        f"| pool@{shared.candidate_pool_k} | {reach.candidate_primary_reachable}/{reach.candidate_primary_total} | "
        f"{reach.candidate_supporting_reachable}/{reach.candidate_supporting_total} | "
        f"{reach.candidate_fully_unreachable_cases} |",
        "",
        f"- High primary reachability: {reach.high_baseline_primary_reachable}/{reach.high_primary_total} → "
        f"{reach.high_candidate_primary_reachable}/{reach.high_primary_total}",
        f"- Critical primary reachability: {reach.critical_baseline_primary_reachable}/{reach.critical_primary_total} → "
        f"{reach.critical_candidate_primary_reachable}/{reach.critical_primary_total}",
        "",
        "## 3. Final ranking metrics",
        "",
        "| Metric | Baseline pool@12 | Candidate pool@24 | Delta |",
        "|--------|-----------------:|------------------:|------:|",
        "",
        "_Baseline pool@12 в таблице ниже — baseline arm 2C.2 (см. терминологию выше)._",
        "",
        f"| Hit@1 | {baseline_rank.hit_rate_at_1:.3f} | {candidate_rank.hit_rate_at_1:.3f} | {aggregate.hit_rate_at_1_delta:+.3f} |",
        f"| Hit@4 | {baseline_rank.hit_rate_at_4:.3f} | {candidate_rank.hit_rate_at_4:.3f} | {aggregate.hit_rate_at_4_delta:+.3f} |",
        f"| Primary hit@4 | {baseline_rank.primary_source_hit_rate_at_4:.3f} | {candidate_rank.primary_source_hit_rate_at_4:.3f} | {aggregate.primary_source_hit_rate_at_4_delta:+.3f} |",
        f"| Supporting hit@4 | {_supporting(baseline_rank)} | {_supporting(candidate_rank)} | {_supporting_delta(aggregate.supporting_source_hit_rate_at_4_delta)} |",
        f"| MRR | {baseline_rank.mrr:.3f} | {candidate_rank.mrr:.3f} | {aggregate.mrr_delta:+.3f} |",
        "",
        "**Note:** Primary hit@4 did not increase in this run; reachability gains are separate from ranking metrics.",
        "",
        "## 4. Promotions and regressions",
        "",
        f"- Primary hit@4 promotions: {ranking.primary_hit_at_4_promotions or 'none'}",
        f"- Primary hit@4 regressions: {ranking.primary_hit_at_4_regressions or 'none'}",
        "",
        "## 5. Critical case diagnostics",
        "",
    ]
    for case_id in ("T004", "T040", "T047"):
        lines.extend(_render_critical_case(run, case_id))

    lines.extend(
        [
            "",
            "## 6. Acceptance verdicts",
            "",
            f"**Candidate-generation verdict:** **{acceptance.reachability_verdict}**",
            f"**Final-ranking verdict:** **{acceptance.ranking_verdict}**",
            "",
            f"- Hard invariants pass: **{acceptance.hard_invariants_pass}**",
            f"- Shared prefix pass: **{acceptance.shared_prefix_pass}**",
            f"- Single retrieval pass: **{acceptance.single_retrieval_pass}**",
            f"- Reranker config hash match: **{acceptance.reranker_config_hash_match}**",
            f"- Critical primary reachability improved: **{acceptance.critical_primary_reachability_improved}**",
            f"- High primary reachability non-regressed: **{acceptance.high_primary_reachability_non_regressed}**",
            f"- Fully unreachable decreased: **{acceptance.fully_unreachable_decreased}**",
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
            "## 7. Limitations",
            "",
            "- Pool expansion improves candidate reachability but does not guarantee top-4 ranking improvements.",
            "- Frozen source-authority-v1 max bonus (+0.03) cannot overtake large similarity gaps.",
            "- Cases beyond pool@24 require a separate experiment or hybrid retrieval.",
            "",
            "## 8. Production threshold",
            "",
            "Production similarity threshold в рамках этапа 2C.2 не выбирался и не изменялся. "
            "Эксперимент выполнялся с `threshold = 0.0` только для контролируемого сравнения "
            "глубины candidate pool. Значение `0.0` не является production-настройкой.",
            "",
            "---",
            "",
            "*Generated by vector pool expansion evaluation (stage 2C.2).*",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_critical_case(run: PoolExpansionEvaluationRun, case_id: str) -> list[str]:
    case = next((item for item in run.case_results if item.case_id == case_id), None)
    if case is None:
        return [f"### {case_id}", "", f"_Case {case_id} not found._", ""]
    lines = [
        f"### {case_id}",
        "",
        f"- **Risk / category:** {case.risk} / {case.category}",
        f"- **Expected primary:** {case.expected_primary_documents}",
        f"- **pool@{run.shared_context.baseline_pool_k} primary reachable:** {case.baseline_pool.reachability.primary_reachable} "
        f"(best rank {case.baseline_pool.reachability.primary_best_pool_rank})",
        f"- **pool@{run.shared_context.candidate_pool_k} primary reachable:** {case.candidate_pool.reachability.primary_reachable} "
        f"(best rank {case.candidate_pool.reachability.primary_best_pool_rank})",
        f"- **Primary in final top-12 (candidate arm):** {case.candidate_ranking.flags.primary_in_final_top12}",
        f"- **Candidate primary hit@4:** {case.candidate_ranking.metrics.primary_hit_at_4}",
        f"- **Classification:** `{case.failure_classification}`",
    ]
    if case_id == "T004":
        lines.append(
            "- Expected primary is reachable only beyond the selected candidate pool@24; "
            "this is not classified as absent from the full vector index."
        )
    if case_id in {"T040", "T047"} and case.failure_classification == "candidate_generation_fixed_ranking_limited":
        lines.append("- Candidate-generation fixed; remaining limitation is final ranking within top-4.")
    lines.append("")
    return lines


def _supporting(aggregate) -> str:
    if aggregate.cases_with_supporting_documents == 0:
        return "n/a"
    rate = aggregate.supporting_source_hit_rate_at_4
    assert rate is not None
    return f"{rate:.3f}"


def _supporting_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _validate_output_path(path: Path, *, project_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _ensure_not_protected(path: Path, *, project_root: Path) -> None:
    root = project_root.resolve()
    for protected in PROTECTED_BASELINE_PATHS:
        if path == (root / protected).resolve():
            raise EvaluationOutputError(
                f"Output must not overwrite protected artifact: {protected.as_posix()}"
            )


def _atomic_write_text(path: Path, content: str, *, temp_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(suffix=path.suffix or ".tmp", dir=str(path.parent), text=True)
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
