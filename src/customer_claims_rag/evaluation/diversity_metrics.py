"""Metrics, acceptance criteria, and aggregation for diversity A/B evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from customer_claims_rag.evaluation.diversity_models import (
    AcceptanceCheck,
    ArmReachabilityConsistency,
    DiversityCaseResult,
    DiversityVerdict,
    FaqComparison,
    HardRegressionFlags,
    InterpretationBoundary,
    NewDocumentFootprint,
    ReachabilityConsistencySummary,
    RequiredCaseDiagnostic,
    SaturationMetrics,
    VectorPoolCapExperimentConfig,
)
from customer_claims_rag.evaluation.expanded_corpus_regression_models import (
    NEW_DOCUMENT_IDS,
    SEMANTIC_OVERLAY_BY_CASE,
)
from customer_claims_rag.evaluation.models import CaseResult
from customer_claims_rag.evaluation.pool_expansion_metrics import FROZEN_RERANKER_CONFIG_HASH
from customer_claims_rag.evaluation.pool_expansion_models import ReachabilityComparison
from customer_claims_rag.retrieval.models import SearchResult

FAQ_DOCUMENT_ID = "10_customer_faq"

REQUIRED_CASE_IDS: tuple[str, ...] = (
    "T004",
    "T016",
    "T039",
    "T040",
    "T044",
    "T046",
    "T047",
    "T051",
)

ALLOWED_UNREACHABLE_CASES: frozenset[str] = frozenset({"T044", "T047"})

MUST_BECOME_REACHABLE_CASES: frozenset[str] = frozenset({"T004", "T016", "T040", "T046"})

STAGE_4C2_BASELINE_TARGETS = {
    "primary_hit_at_4": 0.621,
    "primary_hit_at_12": 0.897,
    "mrr": 0.643,
    "primary_reach": (51, 57),
    "high_risk_reach": (14, 15),
    "critical_reach": (5, 8),
    "faq_top4_count": 36,
}


class ReachabilityConsistencyError(ValueError):
    """Raised when reachability aggregates disagree with case-level records."""


def build_reachability_consistency_summary(
    case_results: list[DiversityCaseResult],
) -> ReachabilityConsistencySummary:
    """Derive canonical primary reach counts and unreachable sets from case records."""
    return ReachabilityConsistencySummary(
        baseline=_arm_reachability_consistency(case_results, arm="baseline"),
        candidate=_arm_reachability_consistency(case_results, arm="candidate"),
    )


def validate_reachability_consistency(
    summary: ReachabilityConsistencySummary,
    case_results: list[DiversityCaseResult],
    *,
    reachability_comparison: ReachabilityComparison | None = None,
) -> None:
    """Validate arithmetic and set invariants for reachability reporting."""
    for arm_name, arm_summary in (
        ("baseline", summary.baseline),
        ("candidate", summary.candidate),
    ):
        if (
            arm_summary.primary_reachable_count
            + len(arm_summary.primary_unreachable_case_ids)
            != arm_summary.primary_denominator
        ):
            raise ReachabilityConsistencyError(
                f"{arm_name} primary reach arithmetic failed: "
                f"{arm_summary.primary_reachable_count} + "
                f"{len(arm_summary.primary_unreachable_case_ids)} != "
                f"{arm_summary.primary_denominator}"
            )

        expected_unreachable = _primary_unreachable_case_ids(case_results, arm=arm_name)
        if set(arm_summary.primary_unreachable_case_ids) != expected_unreachable:
            raise ReachabilityConsistencyError(
                f"{arm_name} primary unreachable set mismatch: "
                f"{arm_summary.primary_unreachable_case_ids} != {sorted(expected_unreachable)}"
            )

    if reachability_comparison is not None:
        if summary.baseline.primary_reachable_count != reachability_comparison.baseline_primary_reachable:
            raise ReachabilityConsistencyError(
                "baseline primary reach numerator disagrees with reachability_comparison"
            )
        if summary.candidate.primary_reachable_count != reachability_comparison.candidate_primary_reachable:
            raise ReachabilityConsistencyError(
                "candidate primary reach numerator disagrees with reachability_comparison"
            )
        if summary.baseline.primary_denominator != reachability_comparison.baseline_primary_total:
            raise ReachabilityConsistencyError(
                "baseline primary reach denominator disagrees with reachability_comparison"
            )
        if summary.candidate.primary_denominator != reachability_comparison.candidate_primary_total:
            raise ReachabilityConsistencyError(
                "candidate primary reach denominator disagrees with reachability_comparison"
            )


def _arm_reachability_consistency(
    case_results: list[DiversityCaseResult],
    *,
    arm: str,
) -> ArmReachabilityConsistency:
    primary_denominator = 0
    primary_reachable_count = 0
    primary_unreachable_case_ids: list[str] = []
    fully_unreachable_case_ids: list[str] = []

    for case in case_results:
        if case.fallback_expected:
            continue
        pool = case.baseline_pool if arm == "baseline" else case.candidate_pool
        if case.expected_primary_documents:
            primary_denominator += 1
            if pool.reachability.primary_reachable:
                primary_reachable_count += 1
            else:
                primary_unreachable_case_ids.append(case.case_id)
        if pool.reachability.fully_unreachable:
            fully_unreachable_case_ids.append(case.case_id)

    return ArmReachabilityConsistency(
        primary_reachable_count=primary_reachable_count,
        primary_denominator=primary_denominator,
        primary_unreachable_case_ids=sorted(primary_unreachable_case_ids),
        fully_unreachable_case_ids=sorted(fully_unreachable_case_ids),
    )


def _primary_unreachable_case_ids(
    case_results: list[DiversityCaseResult],
    *,
    arm: str,
) -> set[str]:
    unreachable: set[str] = set()
    for case in case_results:
        if case.fallback_expected or not case.expected_primary_documents:
            continue
        pool = case.baseline_pool if arm == "baseline" else case.candidate_pool
        if not pool.reachability.primary_reachable:
            unreachable.add(case.case_id)
    return unreachable


def load_vector_pool_cap_config(path: Path) -> VectorPoolCapExperimentConfig:
    """Load vector pool cap experiment config from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return VectorPoolCapExperimentConfig.model_validate(
        {**payload, "config_path": str(path.resolve())},
    )


def compute_vector_pool_cap_config_hash(config: VectorPoolCapExperimentConfig) -> str:
    """Return SHA-256 hash of canonical experiment config payload."""
    payload = {
        "experiment_id": config.experiment_id,
        "version": config.version,
        "experiment_mode": config.experiment_mode,
        "baseline_fetch_k": config.baseline_fetch_k,
        "baseline_candidate_pool_k": config.baseline_candidate_pool_k,
        "baseline_per_document_cap": config.baseline_per_document_cap,
        "candidate_fetch_k": config.candidate_fetch_k,
        "candidate_candidate_pool_k": config.candidate_candidate_pool_k,
        "candidate_per_document_cap": config.candidate_per_document_cap,
        "final_top_k": config.final_top_k,
        "threshold": config.threshold,
        "reranker_id": config.reranker_id,
        "reranker_config_hash": config.reranker_config_hash,
        "tie_breaking": config.tie_breaking,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_primary_hit_rate_at_12_from_cases(case_results: list[DiversityCaseResult], *, arm: str) -> float:
    """Compute final top-12 hit@12 rate (any expected source), matching Stage 4C.2 reporting."""
    filtered = [
        case
        for case in case_results
        if not case.fallback_expected
    ]
    if not filtered:
        return 0.0
    hits = 0
    for case in filtered:
        ranking = case.baseline_ranking if arm == "baseline" else case.candidate_ranking
        if ranking.metrics.hit_at_12:
            hits += 1
    return hits / len(filtered)


def build_saturation_metrics(
    case_results: list[DiversityCaseResult],
    *,
    arm: str,
    cap_threshold: int | None,
    target_pool_k: int,
) -> SaturationMetrics:
    """Aggregate saturation metrics for baseline or candidate arm."""
    if not case_results:
        return SaturationMetrics(
            average_unique_documents_in_pool=0.0,
            average_max_chunks_from_one_document=0.0,
            questions_with_document_count_ge_cap=0,
            questions_where_cap_removed_chunk=0,
            total_removed_by_cap_chunks=0,
            average_pool_size_after_cap=0.0,
            questions_with_pool_shorter_than_target=0,
        )

    unique_counts: list[int] = []
    max_per_doc: list[int] = []
    ge_cap = 0
    cap_removed = 0
    total_removed = 0
    pool_sizes: list[int] = []
    shorter_than_target = 0

    for case in case_results:
        snapshot = case.baseline_pool if arm == "baseline" else case.candidate_pool
        diagnostics = snapshot.cap_diagnostics
        if diagnostics is None:
            continue
        unique_counts.append(diagnostics.unique_document_count)
        max_per_doc.append(diagnostics.max_chunks_from_one_document)
        pool_sizes.append(diagnostics.pool_size)
        if diagnostics.pool_shorter_than_target:
            shorter_than_target += 1
        if cap_threshold is not None and diagnostics.max_chunks_from_one_document >= cap_threshold:
            ge_cap += 1
        if diagnostics.removed_by_cap_count > 0:
            cap_removed += 1
        total_removed += diagnostics.removed_by_cap_count

    count = len(case_results)
    return SaturationMetrics(
        average_unique_documents_in_pool=sum(unique_counts) / count,
        average_max_chunks_from_one_document=sum(max_per_doc) / count,
        questions_with_document_count_ge_cap=ge_cap,
        questions_where_cap_removed_chunk=cap_removed,
        total_removed_by_cap_chunks=total_removed,
        average_pool_size_after_cap=sum(pool_sizes) / count,
        questions_with_pool_shorter_than_target=shorter_than_target,
    )


def build_faq_comparison(case_results: list[DiversityCaseResult]) -> FaqComparison:
    """Compare FAQ presence in final top-4 across arms."""
    baseline_faq = 0
    candidate_faq = 0
    baseline_without_primary = 0
    candidate_without_primary = 0
    changed_cases: list[str] = []

    for case in case_results:
        if case.fallback_expected:
            continue
        if case.faq_in_baseline_top4:
            baseline_faq += 1
        if case.faq_in_candidate_top4:
            candidate_faq += 1
        if case.faq_in_top4_without_primary_baseline:
            baseline_without_primary += 1
        if case.faq_in_top4_without_primary_candidate:
            candidate_without_primary += 1
        if case.faq_in_baseline_top4 != case.faq_in_candidate_top4:
            changed_cases.append(case.case_id)

    return FaqComparison(
        baseline_questions_with_faq_in_top4=baseline_faq,
        candidate_questions_with_faq_in_top4=candidate_faq,
        baseline_faq_top4_without_primary=baseline_without_primary,
        candidate_faq_top4_without_primary=candidate_without_primary,
        faq_top4_case_ids_changed_vs_baseline=sorted(changed_cases),
    )


def build_new_document_footprint(
    case_results: list[DiversityCaseResult],
    *,
    arm: str,
) -> list[NewDocumentFootprint]:
    """Count new-document presence in pools and final top-12."""
    pool_counts = dict.fromkeys(NEW_DOCUMENT_IDS, 0)
    final_counts = dict.fromkeys(NEW_DOCUMENT_IDS, 0)

    for case in case_results:
        snapshot = case.baseline_pool if arm == "baseline" else case.candidate_pool
        ranking = case.baseline_ranking if arm == "baseline" else case.candidate_ranking
        pool_doc_ids: set[str] = set()
        if snapshot.cap_diagnostics is not None:
            pool_doc_ids = set(snapshot.cap_diagnostics.document_counts)
        final_doc_ids = {item.document_id for item in ranking.ordered_candidates[:12]}
        for doc_id in NEW_DOCUMENT_IDS:
            if doc_id in pool_doc_ids:
                pool_counts[doc_id] += 1
            if doc_id in final_doc_ids:
                final_counts[doc_id] += 1

    return [
        NewDocumentFootprint(
            document_id=doc_id,
            pools_containing_document=pool_counts[doc_id],
            final_top12_containing_document=final_counts[doc_id],
        )
        for doc_id in NEW_DOCUMENT_IDS
    ]


def _vector_ranks_by_document(results: list) -> dict[str, list[int]]:
    ranks: dict[str, list[int]] = {}
    for index, result in enumerate(results, start=1):
        ranks.setdefault(result.document_id, []).append(index)
    return ranks


def build_required_case_diagnostics(
    case_results: list[DiversityCaseResult],
) -> list[RequiredCaseDiagnostic]:
    """Build mandatory diagnostics for selected benchmark cases."""
    by_id = {case.case_id: case for case in case_results}
    diagnostics: list[RequiredCaseDiagnostic] = []
    for case_id in REQUIRED_CASE_IDS:
        case = by_id.get(case_id)
        if case is None:
            continue
        diagnostics.append(
            RequiredCaseDiagnostic(
                case_id=case_id,
                expected_primary_documents=case.expected_primary_documents,
                baseline_vector_ranks_by_document=case.baseline_vector_ranks_by_document,
                candidate_vector_ranks_by_document=case.candidate_vector_ranks_by_document,
                baseline_pool_ranks_by_document=_pool_ranks_from_snapshot(case.baseline_pool),
                candidate_pool_ranks_by_document=_pool_ranks_from_snapshot(case.candidate_pool),
                candidate_pool_document_counts=(
                    case.candidate_pool.cap_diagnostics.document_counts
                    if case.candidate_pool.cap_diagnostics
                    else {}
                ),
                baseline_final_ranks_by_document=_final_ranks(case.baseline_ranking),
                candidate_final_ranks_by_document=_final_ranks(case.candidate_ranking),
                baseline_primary_hit_at_4=case.baseline_ranking.metrics.primary_hit_at_4,
                candidate_primary_hit_at_4=case.candidate_ranking.metrics.primary_hit_at_4,
                baseline_primary_hit_at_12=case.baseline_ranking.metrics.hit_at_12,
                candidate_primary_hit_at_12=case.candidate_ranking.metrics.hit_at_12,
                baseline_primary_reachable=case.baseline_pool.reachability.primary_reachable,
                candidate_primary_reachable=case.candidate_pool.reachability.primary_reachable,
                change_explanation=_explain_case_change(case),
            )
        )
    return diagnostics


def evaluate_diversity_acceptance(
    *,
    baseline_metrics,
    candidate_metrics,
    reachability: ReachabilityComparison,
    reachability_consistency: ReachabilityConsistencySummary,
    faq_comparison: FaqComparison,
    case_results: list[DiversityCaseResult],
    reranker_config_hash: str,
    technical_errors_zero: bool,
) -> tuple[list[AcceptanceCheck], HardRegressionFlags, DiversityVerdict, InterpretationBoundary]:
    """Evaluate acceptance criteria and derive verdict."""
    if case_results:
        validate_reachability_consistency(
            reachability_consistency,
            case_results,
            reachability_comparison=reachability,
        )
    candidate_unreachable = reachability_consistency.candidate.primary_unreachable_case_ids
    baseline_unreachable = reachability_consistency.baseline.primary_unreachable_case_ids
    baseline_primary_hit4 = baseline_metrics.primary_source_hit_rate_at_4
    candidate_primary_hit4 = candidate_metrics.primary_source_hit_rate_at_4
    baseline_mrr = baseline_metrics.mrr
    candidate_mrr = candidate_metrics.mrr
    candidate_primary_hit12 = candidate_metrics.hit_rate_at_12

    hard = HardRegressionFlags(
        primary_hit_at_4_below_baseline=candidate_primary_hit4 < baseline_primary_hit4,
        mrr_below_floor=candidate_mrr < 0.643,
        critical_pool_reach_below_floor=reachability.critical_candidate_primary_reachable < 7,
        faq_top4_above_ceiling=faq_comparison.candidate_questions_with_faq_in_top4 > 40,
        new_unreachable_primary_outside_allowed=_new_unreachable_primary_outside_allowed(
            baseline_unreachable,
            candidate_unreachable,
        ),
    )

    checks: list[AcceptanceCheck] = []

    def add_check(
        criterion_id: str,
        description: str,
        passed: bool,
        *,
        baseline_value: str | None = None,
        candidate_value: str | None = None,
        hard_regression: bool = False,
    ) -> None:
        checks.append(
            AcceptanceCheck(
                criterion_id=criterion_id,
                description=description,
                passed=passed,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                hard_regression=hard_regression,
            )
        )

    add_check(
        "primary_hit_at_4",
        "Primary hit@4 >= baseline raw value",
        candidate_primary_hit4 >= baseline_primary_hit4,
        baseline_value=str(baseline_primary_hit4),
        candidate_value=str(candidate_primary_hit4),
        hard_regression=hard.primary_hit_at_4_below_baseline,
    )
    add_check(
        "mrr",
        "MRR >= 0.645",
        candidate_mrr >= 0.645,
        candidate_value=str(candidate_mrr),
        hard_regression=hard.mrr_below_floor,
    )
    add_check(
        "primary_hit_at_12",
        "Primary hit@12 >= 0.910",
        candidate_primary_hit12 >= 0.910,
        candidate_value=str(candidate_primary_hit12),
    )
    add_check(
        "primary_pool_reach",
        "Primary pool reach >= 55/57",
        reachability.candidate_primary_reachable >= 55,
        candidate_value=f"{reachability.candidate_primary_reachable}/{reachability.candidate_primary_total}",
    )
    add_check(
        "high_risk_pool_reach",
        "High-risk pool reach >= 14/15",
        reachability.high_candidate_primary_reachable >= 14,
        candidate_value=f"{reachability.high_candidate_primary_reachable}/{reachability.high_primary_total}",
    )
    add_check(
        "critical_pool_reach",
        "Critical pool reach >= 7/8",
        reachability.critical_candidate_primary_reachable >= 7,
        candidate_value=(
            f"{reachability.critical_candidate_primary_reachable}/"
            f"{reachability.critical_primary_total}"
        ),
        hard_regression=hard.critical_pool_reach_below_floor,
    )
    add_check(
        "faq_top4_count",
        "FAQ top-4 count <= 36",
        faq_comparison.candidate_questions_with_faq_in_top4 <= 36,
        baseline_value=str(faq_comparison.baseline_questions_with_faq_in_top4),
        candidate_value=str(faq_comparison.candidate_questions_with_faq_in_top4),
        hard_regression=hard.faq_top4_above_ceiling,
    )

    unreachable_subset_ok = set(candidate_unreachable).issubset(ALLOWED_UNREACHABLE_CASES)
    add_check(
        "unreachable_subset",
        "Unreachable primaries subset of {T044, T047}",
        unreachable_subset_ok,
        candidate_value=str(candidate_unreachable),
        hard_regression=hard.new_unreachable_primary_outside_allowed,
    )

    for case_id in sorted(MUST_BECOME_REACHABLE_CASES):
        case = next((item for item in case_results if item.case_id == case_id), None)
        reachable = case.candidate_pool.reachability.primary_reachable if case else False
        add_check(
            f"reachability_{case_id}",
            f"{case_id} must become reachable",
            reachable,
            candidate_value=str(reachable),
        )

    no_reachability_regression = _no_baseline_reachable_lost(case_results)
    add_check(
        "no_reachability_regression",
        "No previously reachable primary becomes unreachable",
        no_reachability_regression,
    )

    for case_id in ("T040", "T047"):
        overlay_doc = SEMANTIC_OVERLAY_BY_CASE.get(case_id)
        preserved = _overlay_document_preserved(case_results, case_id, overlay_doc)
        add_check(
            f"new_doc_preserved_{case_id}",
            f"Cap must not fully remove dedicated new document from {case_id}",
            preserved,
            candidate_value=overlay_doc,
        )

    add_check(
        "technical_errors_zero",
        "No technical retrieval errors",
        technical_errors_zero,
    )
    add_check(
        "reranker_hash",
        "Frozen reranker config hash matches",
        reranker_config_hash == FROZEN_RERANKER_CONFIG_HASH,
        candidate_value=reranker_config_hash,
    )

    all_passed = all(check.passed for check in checks)
    verdict: DiversityVerdict = (
        "ACCEPTED AS PARTIAL CANDIDATE-GENERATION REPAIR" if all_passed else "REJECTED"
    )

    boundary = InterpretationBoundary(
        unresolved_cases=sorted(candidate_unreachable),
        primary_unreachable_cases=candidate_unreachable,
        fully_unreachable_cases=reachability_consistency.candidate.fully_unreachable_case_ids,
    )
    return checks, hard, verdict, boundary


def faq_flags_for_final(
    final_results: list[SearchResult],
    *,
    expected_primary_documents: list[str],
    fallback_expected: bool,
) -> tuple[bool, bool]:
    """Return FAQ-in-top4 and FAQ-without-primary flags for one arm."""
    if fallback_expected:
        return False, False
    top4_docs = [item.document_id for item in final_results[:4]]
    faq_in_top4 = FAQ_DOCUMENT_ID in top4_docs
    primary_in_top4 = any(doc in top4_docs for doc in expected_primary_documents)
    faq_without_primary = faq_in_top4 and not primary_in_top4
    return faq_in_top4, faq_without_primary


def _new_unreachable_primary_outside_allowed(
    baseline_unreachable: list[str],
    candidate_unreachable: list[str],
) -> bool:
    newly_unreachable = set(candidate_unreachable) - set(baseline_unreachable)
    return bool(newly_unreachable - ALLOWED_UNREACHABLE_CASES)


def _no_baseline_reachable_lost(case_results: list[DiversityCaseResult]) -> bool:
    for case in case_results:
        if case.fallback_expected or not case.expected_primary_documents:
            continue
        if (
            case.baseline_pool.reachability.primary_reachable
            and not case.candidate_pool.reachability.primary_reachable
        ):
            return False
    return True


def _overlay_document_preserved(
    case_results: list[DiversityCaseResult],
    case_id: str,
    overlay_doc: str | None,
) -> bool:
    if overlay_doc is None:
        return True
    case = next((item for item in case_results if item.case_id == case_id), None)
    if case is None:
        return False
    pool_docs = case.candidate_pool.cap_diagnostics.document_counts if case.candidate_pool.cap_diagnostics else {}
    return overlay_doc in pool_docs



def _pool_ranks_from_snapshot(snapshot) -> dict[str, list[int]]:
    ranks: dict[str, list[int]] = {}
    if snapshot.cap_diagnostics is None:
        return ranks
    for entry in snapshot.cap_diagnostics.entries:
        doc_id = entry.result.document_id
        ranks.setdefault(doc_id, []).append(entry.result.rank)
    return ranks


def _final_ranks(ranking) -> dict[str, list[int]]:
    ranks: dict[str, list[int]] = {}
    for item in ranking.ordered_candidates:
        if item.candidate_rank is None:
            continue
        ranks.setdefault(item.document_id, []).append(item.candidate_rank)
    return ranks


def _explain_case_change(case: DiversityCaseResult) -> str:
    parts: list[str] = []
    if case.baseline_pool.reachability.primary_reachable != case.candidate_pool.reachability.primary_reachable:
        parts.append(
            "primary reachability "
            f"{case.baseline_pool.reachability.primary_reachable} -> "
            f"{case.candidate_pool.reachability.primary_reachable}"
        )
    if case.comparison.primary_hit_at_4_delta != 0:
        parts.append(f"primary hit@4 delta {case.comparison.primary_hit_at_4_delta:+d}")
    if case.candidate_pool.cap_diagnostics and case.candidate_pool.cap_diagnostics.removed_by_cap_count > 0:
        parts.append(
            f"cap removed {case.candidate_pool.cap_diagnostics.removed_by_cap_count} chunks"
        )
    if not parts:
        return "no material change"
    return "; ".join(parts)


__all__ = [
    "ALLOWED_UNREACHABLE_CASES",
    "FAQ_DOCUMENT_ID",
    "MUST_BECOME_REACHABLE_CASES",
    "REQUIRED_CASE_IDS",
    "ReachabilityConsistencyError",
    "build_reachability_consistency_summary",
    "validate_reachability_consistency",
    "STAGE_4C2_BASELINE_TARGETS",
    "build_faq_comparison",
    "build_new_document_footprint",
    "build_required_case_diagnostics",
    "build_saturation_metrics",
    "compute_primary_hit_rate_at_12_from_cases",
    "compute_vector_pool_cap_config_hash",
    "evaluate_diversity_acceptance",
    "faq_flags_for_final",
    "load_vector_pool_cap_config",
]
