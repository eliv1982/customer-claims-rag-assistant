"""Markdown and JSON reporting for reranking A/B evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from customer_claims_rag.evaluation.ab_models import AbEvaluationRun
from customer_claims_rag.exceptions import EvaluationOutputError


DEFAULT_AB_JSON = Path("data/05_evaluation/reranking_ab_source_authority_v1.json")
DEFAULT_AB_MARKDOWN = Path("tests/06_reranking_ab_results.md")

PROTECTED_BASELINE_PATHS = (
    Path("tests/03_test_results.md"),
    Path("tests/04_improvement_log.md"),
    Path("tests/05_answer_level_test_results.md"),
    Path("data/05_evaluation/retrieval_results.json"),
)


def write_ab_outputs(
    run: AbEvaluationRun,
    *,
    output_json: Path,
    output_markdown: Path,
    project_root: Path,
) -> tuple[Path, Path]:
    """Write A/B JSON and Markdown artifacts atomically."""
    json_path = _validate_ab_output_path(output_json, project_root=project_root)
    markdown_path = _validate_ab_output_path(output_markdown, project_root=project_root)
    _ensure_not_protected(json_path, project_root=project_root)
    _ensure_not_protected(markdown_path, project_root=project_root)

    json_content = json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    markdown_content = render_ab_markdown(run)

    temp_paths: list[Path] = []
    try:
        _atomic_write_text(json_path, json_content, temp_paths=temp_paths)
        _atomic_write_text(markdown_path, markdown_content, temp_paths=temp_paths)
    except Exception as exc:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        raise EvaluationOutputError(f"A/B output write failed: {exc}") from None
    return json_path, markdown_path


def render_ab_markdown(run: AbEvaluationRun) -> str:
    """Render human-readable A/B report from a self-contained artifact."""
    experiment = run.experiment
    shared = run.shared_context
    baseline = run.baseline.aggregate_metrics
    candidate = run.candidate.aggregate_metrics
    aggregate = run.comparison.aggregate
    acceptance = run.comparison.acceptance
    faq_policy = run.comparison.faq_policy_movement
    identity = run.baseline_identity

    order_diff_cases = identity.same_set_different_order_count
    lines = [
        "# Reranking A/B results: source-authority-v1",
        "",
        f"**Timestamp:** {run.timestamp.isoformat()}",
        f"**Reranker ID:** `{experiment.reranker_id}`",
        f"**Version:** `{experiment.version}`",
        f"**Experiment mode:** `{experiment.experiment_mode}`",
        f"**Config hash:** `{experiment.config_hash}`",
        f"**Git commit:** `{experiment.git_commit or 'unknown'}`",
        f"**Working tree dirty:** {experiment.git_dirty if experiment.git_dirty is not None else 'unknown'}",
        "",
        "## 1. Experiment contract",
        "",
        "- Shared candidate pool from one baseline retrieval call per case",
        f"- `fetch_k={shared.fetch_k}`, `top_k={shared.top_k}`, `threshold={shared.threshold:.2f}`",
        f"- Embedding model: `{shared.embedding_model}`",
        f"- Index fingerprint: `{shared.index_fingerprint}`",
        f"- Evaluation dataset fingerprint: `{shared.evaluation_dataset_fingerprint}`",
        "- Reranker inputs: query, baseline candidates, `chunk_type`, similarity metadata only",
        "- No oracle risk/category labels and no expected-source features in reranker",
        "",
        "## 2. Baseline identity",
        "",
        f"- Frozen baseline evaluation result ID: `{identity.frozen_evaluation_result_id}`",
        f"- Computed baseline evaluation result ID: `{identity.computed_evaluation_result_id}`",
        f"- Evaluation result ID match: **{identity.evaluation_result_id_match}**",
        f"- Frozen index fingerprint: `{identity.frozen_index_fingerprint}`",
        f"- Run index fingerprint: `{shared.index_fingerprint}`",
        f"- Case count: **{identity.case_count}**",
        f"- Exact ordered match: **{identity.exact_count}**",
        f"- Same-order float drift: **{identity.same_order_float_drift_count}**",
        f"- Same-set different-order: **{identity.same_set_different_order_count}**",
        f"- Substantive mismatches: **{identity.substantive_mismatch_count}**",
        f"- Max similarity drift: **{identity.max_similarity_drift:.6f}** (tolerance {identity.similarity_tolerance})",
        f"- Max distance drift: **{identity.max_distance_drift:.6f}**",
        f"- Frozen metrics match: **{identity.frozen_metrics_match}**",
        f"- Shared pool exact match: **{identity.shared_pool_exact_match}**",
        f"- **Status:** **{identity.identity_status}**",
        "",
        "Frozen baseline metrics and candidate sets are equivalent.",
        f"Exact ordered identity was not reproduced in {order_diff_cases}/{identity.case_count} cases due live embedding drift.",
        "No substantive candidate-set mismatch was found.",
        "",
        "## 3. Overall baseline vs candidate metrics",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|--------|---------:|----------:|------:|",
        f"| Hit@1 | {baseline.hit_rate_at_1:.3f} | {candidate.hit_rate_at_1:.3f} | {aggregate.hit_rate_at_1_delta:+.3f} |",
        f"| Hit@4 | {baseline.hit_rate_at_4:.3f} | {candidate.hit_rate_at_4:.3f} | {aggregate.hit_rate_at_4_delta:+.3f} |",
        f"| Hit@12 | {baseline.hit_rate_at_12:.3f} | {candidate.hit_rate_at_12:.3f} | {aggregate.hit_rate_at_12_delta:+.3f} |",
        f"| Document recall@1 | {baseline.document_recall_at_1:.3f} | {candidate.document_recall_at_1:.3f} | {aggregate.document_recall_at_1_delta:+.3f} |",
        f"| Document recall@4 | {baseline.document_recall_at_4:.3f} | {candidate.document_recall_at_4:.3f} | {aggregate.document_recall_at_4_delta:+.3f} |",
        f"| Document recall@12 | {baseline.document_recall_at_12:.3f} | {candidate.document_recall_at_12:.3f} | {aggregate.document_recall_at_12_delta:+.3f} |",
        f"| Primary hit@1 | {baseline.primary_source_hit_rate_at_1:.3f} | {candidate.primary_source_hit_rate_at_1:.3f} | {aggregate.primary_source_hit_rate_at_1_delta:+.3f} |",
        f"| Primary hit@4 | {baseline.primary_source_hit_rate_at_4:.3f} | {candidate.primary_source_hit_rate_at_4:.3f} | {aggregate.primary_source_hit_rate_at_4_delta:+.3f} |",
        f"| Supporting hit@4 | {_supporting(baseline)} | {_supporting(candidate)} | {_supporting_delta(aggregate.supporting_source_hit_rate_at_4_delta)} |",
        f"| MRR | {baseline.mrr:.3f} | {candidate.mrr:.3f} | {aggregate.mrr_delta:+.3f} |",
        f"| Technical errors | {baseline.technical_error_count} | {candidate.technical_error_count} | — |",
        "",
        "## 4. Risk-level deltas",
        "",
        "| Risk | Cases | Baseline Hit@4 | Candidate Hit@4 | Δ Hit@4 | Baseline Primary@4 | Candidate Primary@4 | Δ Primary@4 | Δ MRR |",
        "|------|------:|---------------:|----------------:|--------:|-------------------:|--------------------:|------------:|------:|",
    ]
    for item in run.comparison.risk_slices:
        lines.append(
            f"| {item.risk_level} | {item.case_count} | {item.baseline_hit_rate_at_4:.3f} | "
            f"{item.candidate_hit_rate_at_4:.3f} | {item.hit_rate_at_4_delta:+.3f} | "
            f"{item.baseline_primary_hit_rate_at_4:.3f} | {item.candidate_primary_hit_rate_at_4:.3f} | "
            f"{item.primary_hit_rate_at_4_delta:+.3f} | {item.mrr_delta:+.3f} |"
        )
    lines.extend(["", "## 5. Category-level deltas", ""])
    if run.comparison.category_slices:
        lines.extend(
            [
                "| Category | Cases | Baseline Hit@4 | Candidate Hit@4 | Δ Hit@4 | Baseline Primary@4 | Candidate Primary@4 | Δ Primary@4 | Δ MRR |",
                "|----------|------:|---------------:|----------------:|--------:|-------------------:|--------------------:|------------:|------:|",
            ]
        )
        for item in run.comparison.category_slices:
            lines.append(
                f"| {item.category} | {item.case_count} | {item.baseline_hit_rate_at_4:.3f} | "
                f"{item.candidate_hit_rate_at_4:.3f} | {item.hit_rate_at_4_delta:+.3f} | "
                f"{item.baseline_primary_hit_rate_at_4:.3f} | {item.candidate_primary_hit_rate_at_4:.3f} | "
                f"{item.primary_hit_rate_at_4_delta:+.3f} | {item.mrr_delta:+.3f} |"
            )
    else:
        lines.append("_No category slices available._")

    promotions = [
        case
        for case in run.case_results
        if case.comparison.primary_hit_at_4_delta > 0 and not case.fallback_expected
    ]
    regressions = [
        case
        for case in run.case_results
        if case.comparison.primary_hit_at_4_delta < 0 and not case.fallback_expected
    ]
    lines.extend(["", "## 6. Promotions (primary hit@4 gained)", ""])
    if promotions:
        for case in sorted(promotions, key=lambda item: item.case_id):
            lines.append(
                f"- **{case.case_id}** ({case.risk}): baseline primary@4="
                f"{case.baseline.metrics.primary_hit_at_4} → candidate="
                f"{case.candidate.metrics.primary_hit_at_4}"
            )
    else:
        lines.append("_No primary hit@4 promotions._")

    lines.extend(["", "## 7. Regressions (primary hit@4 lost)", ""])
    if regressions:
        for case in sorted(regressions, key=lambda item: item.case_id):
            lines.append(
                f"- **{case.case_id}** ({case.risk}): baseline primary@4="
                f"{case.baseline.metrics.primary_hit_at_4} → candidate="
                f"{case.candidate.metrics.primary_hit_at_4}"
            )
    else:
        lines.append("_No primary hit@4 regressions._")

    lines.extend(["", "## 8. Document promotion/demotion counts", ""])
    for item in run.comparison.document_movements:
        if item.promotions or item.demotions:
            lines.append(
                f"- `{item.document_id}`: promotions={item.promotions}, "
                f"demotions={item.demotions}, net_rank_change={item.net_rank_change}"
            )
    if not any(item.promotions or item.demotions for item in run.comparison.document_movements):
        lines.append("_No document-level rank movement._")

    primary_failures = sorted(
        case.case_id
        for case in run.case_results
        if case.comparison.primary_candidate_generation_failure
    )
    supporting_failures = sorted(
        case.case_id
        for case in run.case_results
        if case.comparison.supporting_candidate_generation_failure
    )
    reranker_not_applicable = sorted(
        case.case_id
        for case in run.case_results
        if case.comparison.reranker_not_applicable
    )

    lines.extend(
        [
            "",
            "## 9. Candidate-generation diagnostics",
            "",
            "### Primary candidate-generation failures",
            "",
        ]
    )
    if primary_failures:
        for case_id in primary_failures:
            lines.append(f"- **{case_id}**")
    else:
        lines.append("_None._")

    lines.extend(["", "### Supporting candidate-generation failures", ""])
    if supporting_failures:
        for case_id in supporting_failures:
            lines.append(f"- **{case_id}**")
    else:
        lines.append("_None._")

    lines.extend(["", "### Reranker fully not applicable", ""])
    if reranker_not_applicable:
        for case_id in reranker_not_applicable:
            lines.append(f"- **{case_id}**")
    else:
        lines.append("_None._")

    lines.extend(
        [
            "",
            "## 10. FAQ vs policy movement",
            "",
            f"- FAQ chunks promoted: **{faq_policy.faq_chunks_promoted}**",
            f"- FAQ chunks demoted: **{faq_policy.faq_chunks_demoted}**",
            f"- Policy chunks promoted: **{faq_policy.policy_chunks_promoted}**",
            f"- Policy chunks demoted: **{faq_policy.policy_chunks_demoted}**",
            f"- Escalation chunks promoted: **{faq_policy.escalation_chunks_promoted}**",
            f"- Escalation chunks demoted: **{faq_policy.escalation_chunks_demoted}**",
            "",
            "## 11. T040 analysis",
            "",
        ]
    )
    lines.extend(_render_case_analysis(run, "T040"))
    lines.extend(["", "## 12. T047 analysis", ""])
    lines.extend(_render_case_analysis(run, "T047"))

    verdict = "candidate accepted" if acceptance.candidate_accepted else "candidate rejected"
    lines.extend(
        [
            "",
            "## 13. Acceptance criteria result",
            "",
            f"**Verdict:** **{verdict}**",
            "",
            f"- Hard invariants pass: **{acceptance.hard_invariants_pass}**",
            f"- Baseline identity pass: **{acceptance.baseline_identity_pass}**",
            f"- Shared pool pass: **{acceptance.shared_pool_pass}**",
            f"- Technical errors zero: **{acceptance.technical_errors_zero}**",
            f"- High+critical net primary promotions: **{acceptance.high_critical_net_primary_promotions}** (pass={acceptance.high_critical_net_primary_promotions_pass})",
            f"- Critical primary regressions: **{acceptance.critical_primary_regressions}** (pass={acceptance.critical_primary_regressions_pass})",
            f"- Overall primary hit@4 delta: **{acceptance.overall_primary_hit_at_4_delta:+.3f}** (pass={acceptance.overall_primary_hit_at_4_pass})",
            f"- Overall MRR delta: **{acceptance.overall_mrr_delta:+.3f}** (pass={acceptance.overall_mrr_pass})",
            f"- Low Hit@4 regressions: **{acceptance.low_hit_at_4_regressions}** (pass={acceptance.low_hit_at_4_regressions_pass})",
            f"- Low MRR delta: **{acceptance.low_mrr_delta:+.3f}** (pass={acceptance.low_mrr_pass})",
            f"- Medium Hit@4 regressions: **{acceptance.medium_hit_at_4_regressions}** (pass={acceptance.medium_hit_at_4_regressions_pass})",
            f"- Medium MRR delta: **{acceptance.medium_mrr_delta:+.3f}** (pass={acceptance.medium_mrr_pass})",
            f"- Fallback status preserved: **{acceptance.fallback_status_preserved}**",
        ]
    )
    if acceptance.failure_reasons:
        lines.append("")
        lines.append("Failure reasons:")
        for reason in acceptance.failure_reasons:
            lines.append(f"- {reason}")

    max_bonus = _max_source_bonus(run.experiment.config)
    lines.extend(
        [
            "",
            "## 14. Limitations",
            "",
            "- Reranking only reorders the existing top-12 candidate pool; it cannot recover documents absent from baseline retrieval.",
            "- `source-authority-v1` uses chunk_type authority only; no lexical, risk, or category signals.",
            "- Production-like mode does not use predicted request risk or category.",
            (
                "- Source bonus can change ordering only within the configured "
                f"maximum bonus zone; candidates with similarity gaps greater "
                f"than max_source_bonus ({max_bonus:.2f}) cannot be overtaken solely by this bonus."
            ),
            "",
            "## 15. Production threshold",
            "",
            "No production similarity threshold was selected in this experiment.",
            "",
            "---",
            "",
            "*Generated by reranking A/B evaluation (stage 2C.1).*",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_case_analysis(run: AbEvaluationRun, case_id: str) -> list[str]:
    case = next((item for item in run.case_results if item.case_id == case_id), None)
    if case is None:
        return [f"_Case {case_id} not found._"]
    lines = [
        f"- **Risk / category:** {case.risk} / {case.category}",
        f"- **Expected primary:** {case.expected_primary_documents}",
        f"- **Expected supporting:** {case.expected_supporting_documents}",
        f"- **Shared candidate IDs:** {len(case.shared_candidate_ids)} chunks",
        f"- **Baseline primary hit@4:** {case.baseline.metrics.primary_hit_at_4}",
        f"- **Candidate primary hit@4:** {case.candidate.metrics.primary_hit_at_4}",
        f"- **primary_candidate_generation_failure:** {case.comparison.primary_candidate_generation_failure}",
        f"- **supporting_candidate_generation_failure:** {case.comparison.supporting_candidate_generation_failure}",
        f"- **reranker_not_applicable:** {case.comparison.reranker_not_applicable}",
    ]
    if case.comparison.reranker_not_applicable:
        lines.append(
            "- Expected document(s) absent from shared top-12 pool; reranking cannot recover them."
        )
    return lines


def _max_source_bonus(config: dict) -> float:
    mapping = config.get("source_authority_mapping", {})
    bonuses = [float(value) for value in mapping.values()]
    return max(bonuses) if bonuses else 0.0


def _supporting(aggregate) -> str:
    if aggregate.cases_with_supporting_documents == 0:
        return "n/a"
    rate = aggregate.supporting_source_hit_rate_at_4
    assert rate is not None
    return f"{rate:.3f} ({aggregate.supporting_source_hits_at_4}/{aggregate.cases_with_supporting_documents})"


def _supporting_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _validate_ab_output_path(path: Path, *, project_root: Path) -> Path:
    root = project_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    return resolved


def _ensure_not_protected(path: Path, *, project_root: Path) -> None:
    root = project_root.resolve()
    for protected in PROTECTED_BASELINE_PATHS:
        if path == (root / protected).resolve():
            raise EvaluationOutputError(
                f"A/B output must not overwrite protected baseline artifact: {protected.as_posix()}"
            )


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
