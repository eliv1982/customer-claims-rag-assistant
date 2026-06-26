"""Stage 4C.3B metric contract and reference oracle for doc08 experiment."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from customer_claims_rag.evaluation.diversity_metrics import ALLOWED_UNREACHABLE_CASES
from customer_claims_rag.evaluation.doc08_atomic_models import (
    AcceptanceCheck,
    Doc08AtomicCaseResult,
    Doc08Verdict,
    ExtensionArmMetrics,
    ExtensionCaseDiagnostic,
    FrozenReachabilitySnapshot,
)
from customer_claims_rag.evaluation.models import AggregateMetrics, CaseResult
from customer_claims_rag.evaluation.pool_expansion_models import ReachabilityComparison
from customer_claims_rag.ingestion.corpus_overlay import DOC08_DOCUMENT_ID

DOC12_DOCUMENT_ID = "12_staff_safety_and_threat_handling"
FAQ_DOCUMENT_ID = "10_customer_faq"

REFERENCE_EXPERIMENT_ID = "vector-pool-36-cap4-v1"
REFERENCE_ARM = "candidate"
DEFAULT_REFERENCE_ARTIFACT = Path("data/05_evaluation/vector_pool_36_cap4_v1.json")
ALLOWED_UNTRACKED_PATHS = frozenset(
    {
        "data/04_index_backup_10docs_215chunks/",
    }
)

PRIVACY_EXTENSION_IDS = ("E001", "E002", "E003", "E004")
THREAT_EXTENSION_IDS = ("E005", "E006", "E007", "E008")
NEGATIVE_EXTENSION_IDS = ("E009", "E010", "E011", "E012")

MRR_FLOOR = 0.645
PRIMARY_POOL_REACH_FLOOR = 55
PRIMARY_POOL_REACH_DENOMINATOR = 57
HIGH_RISK_REACH_FLOOR = 14
HIGH_RISK_DENOMINATOR = 15
CRITICAL_REACH_FLOOR = 7
CRITICAL_DENOMINATOR = 8
FAQ_TOP4_CEILING = 36
FAQ_TOP4_HARD_CEILING = 40


class BaselineReproductionError(RuntimeError):
    """Raised when doc08 baseline arm fails to reproduce Stage 4C.3B reference."""


class DirtySourceTreeError(RuntimeError):
    """Raised when evaluation is attempted from a dirty git working tree."""


@dataclass(frozen=True)
class BaselineReferenceOracle:
    """Immutable Stage 4C.3B comparison oracle (vector-pool candidate arm)."""

    experiment_id: str
    reference_arm: str
    artifact_path: str
    evaluation_dataset_fingerprint: str
    aggregate_metrics: AggregateMetrics
    reachability: ReachabilityComparison
    primary_unreachable_case_ids: tuple[str, ...]
    fully_unreachable_case_ids: tuple[str, ...]
    faq_top4_count: int

    @property
    def primary_hit_at_4(self) -> float:
        return self.aggregate_metrics.primary_source_hit_rate_at_4

    @property
    def primary_hit_at_12(self) -> float:
        return self.aggregate_metrics.hit_rate_at_12

    @property
    def mrr(self) -> float:
        return self.aggregate_metrics.mrr


def _has_disallowed_worktree_changes(project_root: Path | None) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_root.resolve()) if project_root is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().replace("\\", "/")
        if path in ALLOWED_UNTRACKED_PATHS:
            continue
        return True
    return False


def ensure_clean_source_tree(
    git_state,
    *,
    allow_dirty: bool = False,
    project_root: Path | None = None,
) -> None:
    """Refuse artifact generation from a dirty working tree unless explicitly allowed."""
    if allow_dirty:
        return
    if project_root is not None:
        if _has_disallowed_worktree_changes(project_root):
            raise DirtySourceTreeError(
                "refusing to generate doc08 experiment artifact from dirty source tree"
            )
        return
    if git_state.dirty:
        raise DirtySourceTreeError(
            "refusing to generate doc08 experiment artifact from dirty source tree"
        )


def load_baseline_reference_oracle(
    path: Path,
    *,
    arm: str = REFERENCE_ARM,
) -> BaselineReferenceOracle:
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment_id = payload["experiment"]["experiment_id"]
    ranking_key = f"{arm}_ranking"
    if ranking_key not in payload:
        raise ValueError(f"reference artifact missing {ranking_key!r}")
    aggregate = AggregateMetrics.model_validate(payload[ranking_key]["aggregate_metrics"])
    reachability = ReachabilityComparison.model_validate(payload["reachability_comparison"])
    consistency_arm = payload["reachability_consistency"][arm]
    faq = payload["faq_comparison"]
    faq_key = f"{arm}_questions_with_faq_in_top4"
    return BaselineReferenceOracle(
        experiment_id=experiment_id,
        reference_arm=arm,
        artifact_path=path.as_posix(),
        evaluation_dataset_fingerprint=payload["shared_context"]["evaluation_dataset_fingerprint"],
        aggregate_metrics=aggregate,
        reachability=reachability,
        primary_unreachable_case_ids=tuple(consistency_arm["primary_unreachable_case_ids"]),
        fully_unreachable_case_ids=tuple(consistency_arm["fully_unreachable_case_ids"]),
        faq_top4_count=int(faq[faq_key]),
    )


def build_doc08_reachability_comparison(
    case_results: list[Doc08AtomicCaseResult],
) -> ReachabilityComparison:
    """Aggregate pool reachability using Stage 4C.3B eligibility rules."""
    baseline_primary = candidate_primary = primary_total = 0
    baseline_supporting = candidate_supporting = supporting_total = 0
    high_baseline = high_candidate = high_total = 0
    critical_baseline = critical_candidate = critical_total = 0

    for item in case_results:
        case = item.baseline_case
        if case.fallback_expected:
            continue
        if case.expected_primary_documents:
            primary_total += 1
            if item.baseline_pool.primary_reachable:
                baseline_primary += 1
            if item.candidate_pool.primary_reachable:
                candidate_primary += 1
            if case.expected_risk == "high":
                high_total += 1
                if item.baseline_pool.primary_reachable:
                    high_baseline += 1
                if item.candidate_pool.primary_reachable:
                    high_candidate += 1
            if case.expected_risk == "critical":
                critical_total += 1
                if item.baseline_pool.primary_reachable:
                    critical_baseline += 1
                if item.candidate_pool.primary_reachable:
                    critical_candidate += 1
        if case.expected_supporting_documents:
            supporting_total += 1
            if item.baseline_pool.supporting_reachable:
                baseline_supporting += 1
            if item.candidate_pool.supporting_reachable:
                candidate_supporting += 1

    baseline_fully = sorted(
        item.case_id
        for item in case_results
        if not item.baseline_case.fallback_expected and item.baseline_pool.fully_unreachable
    )
    candidate_fully = sorted(
        item.case_id
        for item in case_results
        if not item.candidate_case.fallback_expected and item.candidate_pool.fully_unreachable
    )

    return ReachabilityComparison(
        baseline_primary_reachable=baseline_primary,
        baseline_primary_total=primary_total,
        candidate_primary_reachable=candidate_primary,
        candidate_primary_total=primary_total,
        baseline_supporting_reachable=baseline_supporting,
        baseline_supporting_total=supporting_total,
        candidate_supporting_reachable=candidate_supporting,
        candidate_supporting_total=supporting_total,
        baseline_fully_unreachable_cases=baseline_fully,
        candidate_fully_unreachable_cases=candidate_fully,
        high_baseline_primary_reachable=high_baseline,
        high_candidate_primary_reachable=high_candidate,
        high_primary_total=high_total,
        critical_baseline_primary_reachable=critical_baseline,
        critical_candidate_primary_reachable=critical_candidate,
        critical_primary_total=critical_total,
        risk_slices=[],
    )


def build_frozen_reachability_snapshot(
    case_results: list[Doc08AtomicCaseResult],
    *,
    arm: str,
) -> FrozenReachabilitySnapshot:
    primary_unreachable = primary_unreachable_case_ids(case_results, arm=arm)
    fully_unreachable = fully_unreachable_case_ids(case_results, arm=arm)
    reachability = build_doc08_reachability_comparison(case_results)
    if arm == "baseline":
        return FrozenReachabilitySnapshot(
            primary_reachable=reachability.baseline_primary_reachable,
            primary_denominator=reachability.baseline_primary_total,
            primary_unreachable_case_ids=primary_unreachable,
            high_risk_reachable=reachability.high_baseline_primary_reachable,
            high_risk_denominator=reachability.high_primary_total,
            critical_reachable=reachability.critical_baseline_primary_reachable,
            critical_denominator=reachability.critical_primary_total,
            fully_unreachable_case_ids=fully_unreachable,
        )
    return FrozenReachabilitySnapshot(
        primary_reachable=reachability.candidate_primary_reachable,
        primary_denominator=reachability.candidate_primary_total,
        primary_unreachable_case_ids=primary_unreachable,
        high_risk_reachable=reachability.high_candidate_primary_reachable,
        high_risk_denominator=reachability.high_primary_total,
        critical_reachable=reachability.critical_candidate_primary_reachable,
        critical_denominator=reachability.critical_primary_total,
        fully_unreachable_case_ids=fully_unreachable,
    )


def primary_unreachable_case_ids(
    case_results: list[Doc08AtomicCaseResult],
    *,
    arm: str,
) -> list[str]:
    unreachable: list[str] = []
    for item in case_results:
        case = item.baseline_case if arm == "baseline" else item.candidate_case
        pool = item.baseline_pool if arm == "baseline" else item.candidate_pool
        if case.fallback_expected or not case.expected_primary_documents:
            continue
        if not pool.primary_reachable:
            unreachable.append(item.case_id)
    return sorted(unreachable)


def fully_unreachable_case_ids(
    case_results: list[Doc08AtomicCaseResult],
    *,
    arm: str,
) -> list[str]:
    return sorted(
        item.case_id
        for item in case_results
        if not (item.baseline_case if arm == "baseline" else item.candidate_case).fallback_expected
        and (item.baseline_pool if arm == "baseline" else item.candidate_pool).fully_unreachable
    )


def count_faq_in_top4(case_results: list[Doc08AtomicCaseResult], *, arm: str) -> int:
    count = 0
    for item in case_results:
        case = item.baseline_case if arm == "baseline" else item.candidate_case
        if case.fallback_expected:
            continue
        chunks = case.retrieved_chunks[:4]
        if any(chunk.document_id == FAQ_DOCUMENT_ID for chunk in chunks):
            count += 1
    return count


def validate_baseline_reproduction(
    *,
    oracle: BaselineReferenceOracle,
    baseline_metrics: AggregateMetrics,
    reachability: ReachabilityComparison,
    primary_unreachable: list[str],
    fully_unreachable: list[str],
    faq_top4: int,
    primary_hit_at_12: float,
) -> list[AcceptanceCheck]:
    """Verify baseline arm reproduces the immutable reference oracle."""
    checks: list[AcceptanceCheck] = []

    def add(
        criterion_id: str,
        description: str,
        passed: bool,
        *,
        actual: str,
        expected: str,
    ) -> None:
        checks.append(
            AcceptanceCheck(
                criterion_id=criterion_id,
                description=description,
                passed=passed,
                baseline_value=expected,
                candidate_value=actual,
                hard_rejection=not passed,
            )
        )

    ref = oracle
    add(
        "reference_identity",
        "Reference artifact experiment identity",
        ref.experiment_id == REFERENCE_EXPERIMENT_ID,
        actual=ref.experiment_id,
        expected=REFERENCE_EXPERIMENT_ID,
    )
    add(
        "baseline_primary_hit_at_4",
        "Baseline primary hit@4 reproduces reference",
        baseline_metrics.primary_source_hit_rate_at_4 == ref.primary_hit_at_4,
        actual=str(baseline_metrics.primary_source_hit_rate_at_4),
        expected=str(ref.primary_hit_at_4),
    )
    add(
        "baseline_primary_hit_at_12",
        "Baseline primary hit@12 reproduces reference",
        primary_hit_at_12 == ref.primary_hit_at_12,
        actual=str(primary_hit_at_12),
        expected=str(ref.primary_hit_at_12),
    )
    add(
        "baseline_mrr",
        "Baseline MRR reproduces reference",
        baseline_metrics.mrr == ref.mrr,
        actual=str(baseline_metrics.mrr),
        expected=str(ref.mrr),
    )
    expected_reach = (
        f"{ref.reachability.candidate_primary_reachable}/"
        f"{ref.reachability.candidate_primary_total}"
    )
    actual_reach = (
        f"{reachability.baseline_primary_reachable}/{reachability.baseline_primary_total}"
    )
    add(
        "baseline_primary_reach",
        "Baseline primary pool reach reproduces reference",
        (
            reachability.baseline_primary_reachable == ref.reachability.candidate_primary_reachable
            and reachability.baseline_primary_total == ref.reachability.candidate_primary_total
        ),
        actual=actual_reach,
        expected=expected_reach,
    )
    add(
        "baseline_high_risk_reach",
        "Baseline high-risk reach reproduces reference",
        (
            reachability.high_baseline_primary_reachable
            == ref.reachability.high_candidate_primary_reachable
            and reachability.high_primary_total == ref.reachability.high_primary_total
        ),
        actual=(
            f"{reachability.high_baseline_primary_reachable}/"
            f"{reachability.high_primary_total}"
        ),
        expected=(
            f"{ref.reachability.high_candidate_primary_reachable}/"
            f"{ref.reachability.high_primary_total}"
        ),
    )
    add(
        "baseline_critical_reach",
        "Baseline critical reach reproduces reference",
        (
            reachability.critical_baseline_primary_reachable
            == ref.reachability.critical_candidate_primary_reachable
            and reachability.critical_primary_total == ref.reachability.critical_primary_total
        ),
        actual=(
            f"{reachability.critical_baseline_primary_reachable}/"
            f"{reachability.critical_primary_total}"
        ),
        expected=(
            f"{ref.reachability.critical_candidate_primary_reachable}/"
            f"{ref.reachability.critical_primary_total}"
        ),
    )
    add(
        "baseline_primary_unreachable",
        "Baseline primary-unreachable set reproduces reference",
        primary_unreachable == list(ref.primary_unreachable_case_ids),
        actual=str(primary_unreachable),
        expected=str(list(ref.primary_unreachable_case_ids)),
    )
    add(
        "baseline_fully_unreachable",
        "Baseline fully-unreachable set reproduces reference",
        fully_unreachable == list(ref.fully_unreachable_case_ids),
        actual=str(fully_unreachable),
        expected=str(list(ref.fully_unreachable_case_ids)),
    )
    add(
        "baseline_faq_top4",
        "Baseline FAQ top-4 reproduces reference",
        faq_top4 == ref.faq_top4_count,
        actual=str(faq_top4),
        expected=str(ref.faq_top4_count),
    )
    return checks


def assert_baseline_reproduction(checks: list[AcceptanceCheck]) -> None:
    failed = [check for check in checks if not check.passed]
    if failed:
        details = "; ".join(f"{check.criterion_id}={check.candidate_value}" for check in failed)
        raise BaselineReproductionError(
            f"baseline arm failed Stage 4C.3B reference reproduction: {details}"
        )


def compute_primary_hit_at_12(case_results: list[Doc08AtomicCaseResult], *, arm: str) -> float:
    """Hit@12 over non-fallback cases using final ranking metrics."""
    filtered = [
        item
        for item in case_results
        if not item.baseline_case.fallback_expected
    ]
    if not filtered:
        return 0.0
    hits = 0
    for item in filtered:
        ranking = item.baseline_ranking if arm == "baseline" else item.candidate_ranking
        if ranking.metrics.hit_at_12:
            hits += 1
    return hits / len(filtered)


def compute_extension_metrics(
    cases: list[CaseResult],
    *,
    privacy_ids: tuple[str, ...],
    threat_ids: tuple[str, ...],
    negative_ids: tuple[str, ...],
) -> ExtensionArmMetrics:
    by_id = {case.test_id: case for case in cases}

    def hit4(ids: tuple[str, ...]) -> int:
        return sum(1 for case_id in ids if by_id[case_id].primary_hit_at_4)

    def hit12(ids: tuple[str, ...]) -> int:
        return sum(1 for case_id in ids if by_id[case_id].hit_at_12)

    threat_doc12_top4 = 0
    threat_rank_ok = 0
    for case_id in threat_ids:
        case = by_id[case_id]
        docs = [chunk.document_id for chunk in case.retrieved_chunks]
        ranks12 = [index + 1 for index, doc_id in enumerate(docs) if doc_id == DOC12_DOCUMENT_ID]
        ranks08 = [index + 1 for index, doc_id in enumerate(docs) if doc_id == DOC08_DOCUMENT_ID]
        if ranks12 and ranks12[0] <= 4:
            threat_doc12_top4 += 1
        if ranks12 and (not ranks08 or ranks12[0] <= ranks08[0]):
            threat_rank_ok += 1

    neg_doc08_top1 = sum(
        1
        for case_id in negative_ids
        if by_id[case_id].retrieved_chunks
        and by_id[case_id].retrieved_chunks[0].document_id == DOC08_DOCUMENT_ID
    )
    neg_doc08_top4 = sum(
        1
        for case_id in negative_ids
        if any(
            chunk.document_id == DOC08_DOCUMENT_ID
            for chunk in by_id[case_id].retrieved_chunks[:4]
        )
    )

    return ExtensionArmMetrics(
        case_count=len(cases),
        privacy_hit_at_4=hit4(privacy_ids),
        privacy_hit_at_12=hit12(privacy_ids),
        threat_doc12_hit_at_4=hit4(threat_ids),
        threat_doc12_in_top4=threat_doc12_top4,
        threat_doc12_rank_vs_doc08_ok=threat_rank_ok,
        negative_domain_hit_at_4=hit4(negative_ids),
        negative_doc08_top1=neg_doc08_top1,
        negative_doc08_top4=neg_doc08_top4,
    )


def classify_threat_case_delta(
    *,
    case_id: str,
    baseline: CaseResult,
    candidate: CaseResult,
) -> str:
    """Classify E005-E008 threat-case delta without blaming doc08 for pre-existing failures."""

    def threat_pass(case: CaseResult) -> bool:
        docs = [chunk.document_id for chunk in case.retrieved_chunks]
        ranks12 = [index + 1 for index, doc_id in enumerate(docs) if doc_id == DOC12_DOCUMENT_ID]
        ranks08 = [index + 1 for index, doc_id in enumerate(docs) if doc_id == DOC08_DOCUMENT_ID]
        return bool(
            case.primary_hit_at_4
            and ranks12
            and ranks12[0] <= 4
            and (not ranks08 or ranks12[0] <= ranks08[0])
        )

    baseline_pass = threat_pass(baseline)
    candidate_pass = threat_pass(candidate)
    if baseline_pass and candidate_pass:
        return "unchanged_pass"
    if not baseline_pass and not candidate_pass:
        return "both_fail_not_doc08_regression"
    if not baseline_pass and candidate_pass:
        return "candidate_improves"
    if baseline_pass and not candidate_pass:
        return "candidate_regression"
    return "rank_order_changed"


def build_extension_case_diagnostic(
    *,
    case_id: str,
    query: str,
    expected_primary: list[str],
    expected_supporting: list[str],
    baseline_vector_ranks: dict[str, list[int]],
    candidate_vector_ranks: dict[str, list[int]],
    baseline_pool_doc_ids: list[str],
    candidate_pool_doc_ids: list[str],
    baseline_final_doc_ids: list[str],
    candidate_final_doc_ids: list[str],
    baseline_case: CaseResult,
    candidate_case: CaseResult,
) -> ExtensionCaseDiagnostic:
    def pool_rank(doc_ids: list[str], document_id: str) -> int | None:
        for index, doc_id in enumerate(doc_ids, start=1):
            if doc_id == document_id:
                return index
        return None

    def final_rank(doc_ids: list[str], document_id: str) -> int | None:
        for index, doc_id in enumerate(doc_ids, start=1):
            if doc_id == document_id:
                return index
        return None

    if case_id in THREAT_EXTENSION_IDS:
        delta = classify_threat_case_delta(
            case_id=case_id,
            baseline=baseline_case,
            candidate=candidate_case,
        )
    else:
        if baseline_case.primary_hit_at_4 and not candidate_case.primary_hit_at_4:
            delta = "candidate_regression"
        elif not baseline_case.primary_hit_at_4 and candidate_case.primary_hit_at_4:
            delta = "candidate_improves"
        elif baseline_case.primary_hit_at_4 == candidate_case.primary_hit_at_4:
            delta = "unchanged_pass" if baseline_case.primary_hit_at_4 else "both_fail_not_doc08_regression"
        else:
            delta = "rank_order_changed"

    return ExtensionCaseDiagnostic(
        case_id=case_id,
        query_preview=query[:120],
        expected_primary=expected_primary,
        expected_supporting=expected_supporting,
        baseline_vector_ranks_doc08=baseline_vector_ranks.get(DOC08_DOCUMENT_ID, []),
        candidate_vector_ranks_doc08=candidate_vector_ranks.get(DOC08_DOCUMENT_ID, []),
        baseline_vector_ranks_doc12=baseline_vector_ranks.get(DOC12_DOCUMENT_ID, []),
        candidate_vector_ranks_doc12=candidate_vector_ranks.get(DOC12_DOCUMENT_ID, []),
        baseline_pool_rank_doc08=pool_rank(baseline_pool_doc_ids, DOC08_DOCUMENT_ID),
        candidate_pool_rank_doc08=pool_rank(candidate_pool_doc_ids, DOC08_DOCUMENT_ID),
        baseline_pool_rank_doc12=pool_rank(baseline_pool_doc_ids, DOC12_DOCUMENT_ID),
        candidate_pool_rank_doc12=pool_rank(candidate_pool_doc_ids, DOC12_DOCUMENT_ID),
        baseline_final_rank_doc08=final_rank(baseline_final_doc_ids, DOC08_DOCUMENT_ID),
        candidate_final_rank_doc08=final_rank(candidate_final_doc_ids, DOC08_DOCUMENT_ID),
        baseline_final_rank_doc12=final_rank(baseline_final_doc_ids, DOC12_DOCUMENT_ID),
        candidate_final_rank_doc12=final_rank(candidate_final_doc_ids, DOC12_DOCUMENT_ID),
        baseline_primary_hit_at_4=baseline_case.primary_hit_at_4,
        candidate_primary_hit_at_4=candidate_case.primary_hit_at_4,
        baseline_primary_hit_at_12=baseline_case.hit_at_12,
        candidate_primary_hit_at_12=candidate_case.hit_at_12,
        delta_classification=delta,
    )


def evaluate_extension_acceptance(
    candidate: ExtensionArmMetrics,
) -> tuple[list[AcceptanceCheck], Doc08Verdict]:
    checks: list[AcceptanceCheck] = []

    def add(
        criterion_id: str,
        description: str,
        passed: bool,
        val: str,
        *,
        hard: bool = False,
    ) -> None:
        checks.append(
            AcceptanceCheck(
                criterion_id=criterion_id,
                description=description,
                passed=passed,
                candidate_value=val,
                hard_rejection=not passed and hard,
            )
        )

    add(
        "ext_privacy_hit4",
        "Privacy doc08 hit@4 = 4/4",
        candidate.privacy_hit_at_4 == 4,
        f"{candidate.privacy_hit_at_4}/4",
        hard=True,
    )
    add(
        "ext_privacy_hit12",
        "Privacy doc08 hit@12 = 4/4",
        candidate.privacy_hit_at_12 == 4,
        f"{candidate.privacy_hit_at_12}/4",
    )
    add(
        "ext_threat_doc12_hit4",
        "Threat doc12 hit@4 = 4/4",
        candidate.threat_doc12_hit_at_4 == 4,
        f"{candidate.threat_doc12_hit_at_4}/4",
        hard=True,
    )
    add(
        "ext_threat_doc12_top4",
        "Threat doc12 in final top-4 = 4/4",
        candidate.threat_doc12_in_top4 == 4,
        f"{candidate.threat_doc12_in_top4}/4",
        hard=True,
    )
    add(
        "ext_threat_rank",
        "Threat doc12 rank not worse than doc08 = 4/4",
        candidate.threat_doc12_rank_vs_doc08_ok == 4,
        f"{candidate.threat_doc12_rank_vs_doc08_ok}/4",
        hard=True,
    )
    add(
        "ext_negative_hit4",
        "Negative domain hit@4 >= 3/4",
        candidate.negative_domain_hit_at_4 >= 3,
        f"{candidate.negative_domain_hit_at_4}/4",
    )
    add(
        "ext_negative_doc08_top1",
        "Doc08 not top-1 in negative cases",
        candidate.negative_doc08_top1 == 0,
        str(candidate.negative_doc08_top1),
        hard=True,
    )
    add(
        "ext_negative_doc08_top4",
        "Doc08 top-4 in <= 1/4 negative cases",
        candidate.negative_doc08_top4 <= 1,
        f"{candidate.negative_doc08_top4}/4",
    )

    verdict: Doc08Verdict = (
        "ACCEPTED AS TARGETED CORPUS REPAIR" if all(check.passed for check in checks) else "REJECTED"
    )
    return checks, verdict


def evaluate_frozen_acceptance(
    *,
    baseline_metrics: AggregateMetrics,
    candidate_metrics: AggregateMetrics,
    baseline_primary_hit_at_12: float,
    candidate_primary_hit_at_12: float,
    reachability: ReachabilityComparison,
    faq_top4_baseline: int,
    faq_top4_candidate: int,
    primary_unreachable_baseline: list[str],
    primary_unreachable_candidate: list[str],
    t044_reachable: bool,
    t044_hit12: bool,
    t047_doc12_final_rank: int | None,
) -> tuple[list[AcceptanceCheck], Doc08Verdict]:
    checks: list[AcceptanceCheck] = []

    def add(
        criterion_id: str,
        description: str,
        passed: bool,
        *,
        baseline_value: str | None = None,
        candidate_value: str | None = None,
        hard: bool = False,
    ) -> None:
        checks.append(
            AcceptanceCheck(
                criterion_id=criterion_id,
                description=description,
                passed=passed,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                hard_rejection=not passed and hard,
            )
        )

    add(
        "t044_reachable",
        "T044 primary reachable",
        t044_reachable,
        candidate_value=str(t044_reachable),
        hard=True,
    )
    add(
        "t044_hit12",
        "T044 primary hit@12",
        t044_hit12,
        candidate_value=str(t044_hit12),
        hard=True,
    )
    add(
        "primary_hit_at_4",
        "Primary hit@4 >= baseline",
        candidate_metrics.primary_source_hit_rate_at_4 >= baseline_metrics.primary_source_hit_rate_at_4,
        baseline_value=str(baseline_metrics.primary_source_hit_rate_at_4),
        candidate_value=str(candidate_metrics.primary_source_hit_rate_at_4),
        hard=True,
    )
    add(
        "primary_hit_at_12",
        "Primary hit@12 >= baseline",
        candidate_primary_hit_at_12 >= baseline_primary_hit_at_12,
        baseline_value=str(baseline_primary_hit_at_12),
        candidate_value=str(candidate_primary_hit_at_12),
    )
    add(
        "mrr",
        "MRR >= 0.645",
        candidate_metrics.mrr >= MRR_FLOOR,
        baseline_value=str(baseline_metrics.mrr),
        candidate_value=str(candidate_metrics.mrr),
        hard=True,
    )
    add(
        "primary_pool_reach",
        "Primary pool reach >= 56/57",
        reachability.candidate_primary_reachable >= 56,
        baseline_value=(
            f"{reachability.baseline_primary_reachable}/"
            f"{reachability.baseline_primary_total}"
        ),
        candidate_value=(
            f"{reachability.candidate_primary_reachable}/"
            f"{reachability.candidate_primary_total}"
        ),
    )
    add(
        "high_risk_reach",
        "High-risk reach = 15/15",
        reachability.high_candidate_primary_reachable == HIGH_RISK_DENOMINATOR,
        baseline_value=(
            f"{reachability.high_baseline_primary_reachable}/"
            f"{reachability.high_primary_total}"
        ),
        candidate_value=(
            f"{reachability.high_candidate_primary_reachable}/"
            f"{reachability.high_primary_total}"
        ),
        hard=True,
    )
    add(
        "critical_reach",
        "Critical reach >= 7/8",
        reachability.critical_candidate_primary_reachable >= CRITICAL_REACH_FLOOR,
        baseline_value=(
            f"{reachability.critical_baseline_primary_reachable}/"
            f"{reachability.critical_primary_total}"
        ),
        candidate_value=(
            f"{reachability.critical_candidate_primary_reachable}/"
            f"{reachability.critical_primary_total}"
        ),
        hard=True,
    )
    add(
        "faq_top4",
        "FAQ top-4 <= 36",
        faq_top4_candidate <= FAQ_TOP4_CEILING,
        baseline_value=str(faq_top4_baseline),
        candidate_value=str(faq_top4_candidate),
        hard=faq_top4_candidate > FAQ_TOP4_HARD_CEILING,
    )

    new_unreachable = sorted(set(primary_unreachable_candidate) - set(primary_unreachable_baseline))
    add(
        "no_new_unreachable",
        "No new unreachable primary",
        len(new_unreachable) == 0,
        candidate_value=str(new_unreachable),
        hard=bool(new_unreachable),
    )

    doc12_top4 = t047_doc12_final_rank is not None and t047_doc12_final_rank <= 4
    add(
        "t047_doc12_top4",
        "T047 doc12 remains in final top-4",
        doc12_top4,
        candidate_value=str(t047_doc12_final_rank),
        hard=True,
    )

    unreachable_subset_ok = set(primary_unreachable_candidate).issubset(ALLOWED_UNREACHABLE_CASES)
    add(
        "unreachable_subset",
        "Unreachable primaries subset of {T044, T047}",
        unreachable_subset_ok,
        baseline_value=str(primary_unreachable_baseline),
        candidate_value=str(primary_unreachable_candidate),
        hard=not unreachable_subset_ok,
    )

    verdict: Doc08Verdict = (
        "ACCEPTED AS TARGETED CORPUS REPAIR" if all(check.passed for check in checks) else "REJECTED"
    )
    return checks, verdict
