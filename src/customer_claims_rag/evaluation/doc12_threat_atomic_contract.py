"""Stage 4C.3D metric contract and doc08 experimental oracle for doc12 experiment."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from customer_claims_rag.evaluation.doc08_atomic_contract import (
    ALLOWED_UNTRACKED_PATHS,
    DOC12_DOCUMENT_ID,
    FAQ_DOCUMENT_ID,
    NEGATIVE_EXTENSION_IDS,
    PRIVACY_EXTENSION_IDS,
    THREAT_EXTENSION_IDS,
    build_extension_case_diagnostic,
    classify_threat_case_delta,
    compute_extension_metrics,
    ensure_clean_source_tree,
    evaluate_extension_acceptance,
)
from customer_claims_rag.evaluation.doc08_atomic_models import (
    AcceptanceCheck,
    ExtensionArmMetrics,
)
from customer_claims_rag.evaluation.doc12_threat_atomic_models import (
    Doc12ThreatAtomicCaseResult,
    Doc12Verdict,
    HoldoutArmMetrics,
)
from customer_claims_rag.evaluation.extension_parser import (
    DOCUMENT_ID_RE,
    EXTENSION_KNOWN_DOCUMENT_IDS,
    NOTES_MARKER,
    _extract_user_message,
    _optional_text,
    _parse_fields,
    _validate_fallback_consistency,
)
from customer_claims_rag.evaluation.models import AggregateMetrics, CaseResult
from customer_claims_rag.evaluation.pool_expansion_models import ReachabilityComparison
from customer_claims_rag.exceptions import EvaluationCorpusError
from customer_claims_rag.ingestion.corpus_overlay import DOC08_DOCUMENT_ID

REFERENCE_EXPERIMENT_ID = "doc08-atomic-risk-units-v1"
REFERENCE_ARM = "candidate"
DEFAULT_REFERENCE_ARTIFACT = Path("data/05_evaluation/doc08_atomic_risk_units_v1.json")

EXPECTED_DOC08_FINGERPRINT = (
    "c208e6527facd195497d26e9efe6d8b4f487dfdea59629e603205ff1f4ea0c39"
)
EXPECTED_BASELINE_INDEX_FP = (
    "d3c27f4a72e5f78582c6cc29b8cb6c9f26234f6582970ead602f44e5b375bad6"
)

EXPECTED_BASELINE_PRIMARY_HIT_AT_4 = 0.6379310344827587
EXPECTED_BASELINE_PRIMARY_HIT_AT_12 = 0.9655172413793104
EXPECTED_BASELINE_MRR = 0.6758620689655173
EXPECTED_BASELINE_PRIMARY_REACH = 57
EXPECTED_BASELINE_PRIMARY_DENOMINATOR = 57
EXPECTED_BASELINE_HIGH_RISK_REACH = 15
EXPECTED_BASELINE_HIGH_RISK_DENOMINATOR = 15
EXPECTED_BASELINE_CRITICAL_REACH = 8
EXPECTED_BASELINE_CRITICAL_DENOMINATOR = 8
EXPECTED_BASELINE_FAQ_TOP4 = 36
EXPECTED_BASELINE_PRIMARY_UNREACHABLE: tuple[str, ...] = ()
EXPECTED_BASELINE_FULLY_UNREACHABLE: tuple[str, ...] = ()

HOLDOUT_POSITIVE_IDS = ("H001", "H002", "H003", "H004", "H005", "H006")
HOLDOUT_NEGATIVE_IDS = ("H007", "H008", "H009", "H010", "H011", "H012")

HOLDOUT_CASE_HEADER_RE = re.compile(r"^## (H\d{3}) — (.+)$", re.MULTILINE)
VALID_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})

HOLDOUT_POSITIVE_HIT4_FLOOR = 5
HOLDOUT_POSITIVE_HIT12_TARGET = 6
HOLDOUT_POSITIVE_RANK_TARGET = 6
HOLDOUT_NEGATIVE_HIT4_FLOOR = 5
HOLDOUT_NEGATIVE_DOC12_TOP4_CEILING = 1


class BaselineReproductionError(RuntimeError):
    """Raised when doc12 baseline arm fails to reproduce doc08 experimental candidate."""


class DirtySourceTreeError(RuntimeError):
    """Raised when evaluation is attempted from a dirty git working tree."""


@dataclass(frozen=True)
class Doc08ExperimentalOracle:
    """Immutable doc08 experimental candidate-arm comparison oracle."""

    experiment_id: str
    reference_arm: str
    artifact_path: str
    frozen_metrics: AggregateMetrics
    primary_hit_at_12: float
    reachability: ReachabilityComparison
    primary_unreachable_case_ids: tuple[str, ...]
    fully_unreachable_case_ids: tuple[str, ...]
    faq_top4_count: int
    extension_candidate: ExtensionArmMetrics
    baseline_index_fingerprint: str
    doc08_fingerprint: str

    @property
    def primary_hit_at_4(self) -> float:
        return self.frozen_metrics.primary_source_hit_rate_at_4

    @property
    def mrr(self) -> float:
        return self.frozen_metrics.mrr


def repo_relative_path(path: Path, project_root: Path | None = None) -> str:
    """Return repo-relative POSIX path without drive letter."""
    resolved = path.resolve()
    if project_root is not None:
        try:
            return resolved.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
    text = resolved.as_posix()
    if len(text) > 2 and text[1] == ":":
        return text[2:].lstrip("/")
    return text


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


def load_doc08_experimental_oracle(
    path: Path,
    *,
    arm: str = REFERENCE_ARM,
    project_root: Path | None = None,
) -> Doc08ExperimentalOracle:
    """Load doc08 atomic experiment artifact frozen candidate metrics and extension."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    experiment_id = payload["experiment_id"]
    if arm == "candidate":
        frozen_metrics = AggregateMetrics.model_validate(payload["frozen_candidate_metrics"])
        primary_hit_at_12 = float(payload["frozen_candidate_primary_hit_at_12"])
        reach_snapshot = payload["frozen_reachability_candidate"]
        primary_unreachable = tuple(payload["primary_unreachable_candidate"])
        fully_unreachable = tuple(payload["fully_unreachable_candidate"])
        faq_top4 = int(payload["faq_top4_candidate"])
        extension_candidate = ExtensionArmMetrics.model_validate(payload["extension"]["candidate"])
        index_fp = payload["candidate_arm"]["index_fingerprint"]
        doc08_fp = payload["candidate_arm"]["doc08_fingerprint"]
    else:
        frozen_metrics = AggregateMetrics.model_validate(payload["frozen_baseline_metrics"])
        primary_hit_at_12 = float(payload["frozen_baseline_primary_hit_at_12"])
        reach_snapshot = payload["frozen_reachability_baseline"]
        primary_unreachable = tuple(payload["primary_unreachable_baseline"])
        fully_unreachable = tuple(payload["fully_unreachable_baseline"])
        faq_top4 = int(payload["faq_top4_baseline"])
        extension_candidate = ExtensionArmMetrics.model_validate(payload["extension"]["baseline"])
        index_fp = payload["baseline_arm"]["index_fingerprint"]
        doc08_fp = payload["baseline_arm"]["doc08_fingerprint"]

    reachability = ReachabilityComparison(
        baseline_primary_reachable=reach_snapshot["primary_reachable"],
        baseline_primary_total=reach_snapshot["primary_denominator"],
        candidate_primary_reachable=reach_snapshot["primary_reachable"],
        candidate_primary_total=reach_snapshot["primary_denominator"],
        baseline_supporting_reachable=0,
        baseline_supporting_total=0,
        candidate_supporting_reachable=0,
        candidate_supporting_total=0,
        baseline_fully_unreachable_cases=list(fully_unreachable),
        candidate_fully_unreachable_cases=list(fully_unreachable),
        high_baseline_primary_reachable=reach_snapshot["high_risk_reachable"],
        high_candidate_primary_reachable=reach_snapshot["high_risk_reachable"],
        high_primary_total=reach_snapshot["high_risk_denominator"],
        critical_baseline_primary_reachable=reach_snapshot["critical_reachable"],
        critical_candidate_primary_reachable=reach_snapshot["critical_reachable"],
        critical_primary_total=reach_snapshot["critical_denominator"],
        risk_slices=[],
    )

    return Doc08ExperimentalOracle(
        experiment_id=experiment_id,
        reference_arm=arm,
        artifact_path=repo_relative_path(path, project_root),
        frozen_metrics=frozen_metrics,
        primary_hit_at_12=primary_hit_at_12,
        reachability=reachability,
        primary_unreachable_case_ids=primary_unreachable,
        fully_unreachable_case_ids=fully_unreachable,
        faq_top4_count=faq_top4,
        extension_candidate=extension_candidate,
        baseline_index_fingerprint=index_fp,
        doc08_fingerprint=doc08_fp or EXPECTED_DOC08_FINGERPRINT,
    )


def build_doc12_reachability_comparison(
    case_results: list[Doc12ThreatAtomicCaseResult],
) -> ReachabilityComparison:
    """Aggregate pool reachability for doc12 experiment frozen cases."""
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
    case_results: list[Doc12ThreatAtomicCaseResult],
    *,
    arm: str,
):
    from customer_claims_rag.evaluation.doc08_atomic_models import FrozenReachabilitySnapshot

    primary_unreachable = primary_unreachable_case_ids(case_results, arm=arm)
    fully_unreachable = fully_unreachable_case_ids(case_results, arm=arm)
    reachability = build_doc12_reachability_comparison(case_results)
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
    case_results: list[Doc12ThreatAtomicCaseResult],
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
    case_results: list[Doc12ThreatAtomicCaseResult],
    *,
    arm: str,
) -> list[str]:
    return sorted(
        item.case_id
        for item in case_results
        if not (item.baseline_case if arm == "baseline" else item.candidate_case).fallback_expected
        and (item.baseline_pool if arm == "baseline" else item.candidate_pool).fully_unreachable
    )


def count_faq_in_top4(case_results: list[Doc12ThreatAtomicCaseResult], *, arm: str) -> int:
    count = 0
    for item in case_results:
        case = item.baseline_case if arm == "baseline" else item.candidate_case
        if case.fallback_expected:
            continue
        chunks = case.retrieved_chunks[:4]
        if any(chunk.document_id == FAQ_DOCUMENT_ID for chunk in chunks):
            count += 1
    return count


def compute_primary_hit_at_12(
    case_results: list[Doc12ThreatAtomicCaseResult],
    *,
    arm: str,
) -> float:
    filtered = [item for item in case_results if not item.baseline_case.fallback_expected]
    if not filtered:
        return 0.0
    hits = 0
    for item in filtered:
        ranking = item.baseline_ranking if arm == "baseline" else item.candidate_ranking
        if ranking.metrics.hit_at_12:
            hits += 1
    return hits / len(filtered)


def validate_baseline_reproduction(
    *,
    oracle: Doc08ExperimentalOracle,
    baseline_metrics: AggregateMetrics,
    reachability: ReachabilityComparison,
    primary_unreachable: list[str],
    fully_unreachable: list[str],
    faq_top4: int,
    primary_hit_at_12: float,
    baseline_index_fingerprint: str,
    doc08_fingerprint: str,
) -> list[AcceptanceCheck]:
    """Verify baseline arm reproduces doc08 experimental candidate oracle."""
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
        "baseline_index_fingerprint",
        "Baseline index fingerprint matches doc08 experimental candidate",
        baseline_index_fingerprint == EXPECTED_BASELINE_INDEX_FP,
        actual=baseline_index_fingerprint,
        expected=EXPECTED_BASELINE_INDEX_FP,
    )
    add(
        "baseline_doc08_fingerprint",
        "Baseline doc08 fingerprint matches expected overlay",
        doc08_fingerprint == EXPECTED_DOC08_FINGERPRINT,
        actual=doc08_fingerprint,
        expected=EXPECTED_DOC08_FINGERPRINT,
    )
    add(
        "baseline_primary_hit_at_4",
        "Baseline primary hit@4 reproduces doc08 experimental candidate",
        baseline_metrics.primary_source_hit_rate_at_4 == EXPECTED_BASELINE_PRIMARY_HIT_AT_4,
        actual=str(baseline_metrics.primary_source_hit_rate_at_4),
        expected=str(EXPECTED_BASELINE_PRIMARY_HIT_AT_4),
    )
    add(
        "baseline_primary_hit_at_12",
        "Baseline primary hit@12 reproduces doc08 experimental candidate",
        primary_hit_at_12 == EXPECTED_BASELINE_PRIMARY_HIT_AT_12,
        actual=str(primary_hit_at_12),
        expected=str(EXPECTED_BASELINE_PRIMARY_HIT_AT_12),
    )
    add(
        "baseline_mrr",
        "Baseline MRR reproduces doc08 experimental candidate",
        baseline_metrics.mrr == EXPECTED_BASELINE_MRR,
        actual=str(baseline_metrics.mrr),
        expected=str(EXPECTED_BASELINE_MRR),
    )
    add(
        "baseline_primary_reach",
        "Baseline primary pool reach reproduces doc08 experimental candidate",
        (
            reachability.baseline_primary_reachable == EXPECTED_BASELINE_PRIMARY_REACH
            and reachability.baseline_primary_total == EXPECTED_BASELINE_PRIMARY_DENOMINATOR
        ),
        actual=(
            f"{reachability.baseline_primary_reachable}/"
            f"{reachability.baseline_primary_total}"
        ),
        expected=(
            f"{EXPECTED_BASELINE_PRIMARY_REACH}/"
            f"{EXPECTED_BASELINE_PRIMARY_DENOMINATOR}"
        ),
    )
    add(
        "baseline_high_risk_reach",
        "Baseline high-risk reach reproduces doc08 experimental candidate",
        (
            reachability.high_baseline_primary_reachable == EXPECTED_BASELINE_HIGH_RISK_REACH
            and reachability.high_primary_total == EXPECTED_BASELINE_HIGH_RISK_DENOMINATOR
        ),
        actual=(
            f"{reachability.high_baseline_primary_reachable}/"
            f"{reachability.high_primary_total}"
        ),
        expected=(
            f"{EXPECTED_BASELINE_HIGH_RISK_REACH}/"
            f"{EXPECTED_BASELINE_HIGH_RISK_DENOMINATOR}"
        ),
    )
    add(
        "baseline_critical_reach",
        "Baseline critical reach reproduces doc08 experimental candidate",
        (
            reachability.critical_baseline_primary_reachable == EXPECTED_BASELINE_CRITICAL_REACH
            and reachability.critical_primary_total == EXPECTED_BASELINE_CRITICAL_DENOMINATOR
        ),
        actual=(
            f"{reachability.critical_baseline_primary_reachable}/"
            f"{reachability.critical_primary_total}"
        ),
        expected=(
            f"{EXPECTED_BASELINE_CRITICAL_REACH}/"
            f"{EXPECTED_BASELINE_CRITICAL_DENOMINATOR}"
        ),
    )
    add(
        "baseline_primary_unreachable",
        "Baseline primary-unreachable set reproduces doc08 experimental candidate",
        primary_unreachable == list(EXPECTED_BASELINE_PRIMARY_UNREACHABLE),
        actual=str(primary_unreachable),
        expected=str(list(EXPECTED_BASELINE_PRIMARY_UNREACHABLE)),
    )
    add(
        "baseline_fully_unreachable",
        "Baseline fully-unreachable set reproduces doc08 experimental candidate",
        fully_unreachable == list(EXPECTED_BASELINE_FULLY_UNREACHABLE),
        actual=str(fully_unreachable),
        expected=str(list(EXPECTED_BASELINE_FULLY_UNREACHABLE)),
    )
    add(
        "baseline_faq_top4",
        "Baseline FAQ top-4 reproduces doc08 experimental candidate",
        faq_top4 == EXPECTED_BASELINE_FAQ_TOP4,
        actual=str(faq_top4),
        expected=str(EXPECTED_BASELINE_FAQ_TOP4),
    )
    _ = ref
    return checks


def assert_baseline_reproduction(checks: list[AcceptanceCheck]) -> None:
    failed = [check for check in checks if not check.passed]
    if failed:
        details = "; ".join(f"{check.criterion_id}={check.candidate_value}" for check in failed)
        raise BaselineReproductionError(
            f"baseline arm failed doc08 experimental candidate reproduction: {details}"
        )


def evaluate_frozen_acceptance(
    *,
    case_results: list[Doc12ThreatAtomicCaseResult],
    baseline_metrics: AggregateMetrics,
    candidate_metrics: AggregateMetrics,
    baseline_primary_hit_at_12: float,
    candidate_primary_hit_at_12: float,
) -> tuple[list[AcceptanceCheck], Doc12Verdict, dict[str, list[str]]]:
    checks: list[AcceptanceCheck] = []
    regressed_hit4 = sorted(
        item.case_id
        for item in case_results
        if not item.baseline_case.fallback_expected
        and item.baseline_case.primary_hit_at_4
        and not item.candidate_case.primary_hit_at_4
    )
    regressed_hit12 = sorted(
        item.case_id
        for item in case_results
        if not item.baseline_case.fallback_expected
        and item.baseline_ranking.metrics.hit_at_12
        and not item.candidate_ranking.metrics.hit_at_12
    )
    promoted_hit4 = sorted(
        item.case_id
        for item in case_results
        if not item.baseline_case.fallback_expected
        and not item.baseline_case.primary_hit_at_4
        and item.candidate_case.primary_hit_at_4
    )
    promoted_hit12 = sorted(
        item.case_id
        for item in case_results
        if not item.baseline_case.fallback_expected
        and not item.baseline_ranking.metrics.hit_at_12
        and item.candidate_ranking.metrics.hit_at_12
    )

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
        "frozen_no_hit4_regression",
        "No per-case primary hit@4 regression vs doc08 experimental baseline",
        len(regressed_hit4) == 0,
        candidate_value=str(regressed_hit4),
        hard=True,
    )
    add(
        "frozen_no_hit12_regression",
        "No per-case primary hit@12 regression vs doc08 experimental baseline",
        len(regressed_hit12) == 0,
        candidate_value=str(regressed_hit12),
        hard=True,
    )
    add(
        "frozen_aggregate_hit4",
        "Aggregate primary hit@4 >= doc08 experimental baseline",
        candidate_metrics.primary_source_hit_rate_at_4 >= baseline_metrics.primary_source_hit_rate_at_4,
        baseline_value=str(baseline_metrics.primary_source_hit_rate_at_4),
        candidate_value=str(candidate_metrics.primary_source_hit_rate_at_4),
    )
    add(
        "frozen_aggregate_hit12",
        "Aggregate primary hit@12 >= doc08 experimental baseline",
        candidate_primary_hit_at_12 >= baseline_primary_hit_at_12,
        baseline_value=str(baseline_primary_hit_at_12),
        candidate_value=str(candidate_primary_hit_at_12),
    )

    case_delta = {
        "promoted_hit4": promoted_hit4,
        "regressed_hit4": regressed_hit4,
        "promoted_hit12": promoted_hit12,
        "regressed_hit12": regressed_hit12,
    }
    verdict: Doc12Verdict = (
        "ACCEPTED AS COMBINED TARGETED CORPUS REPAIR"
        if all(check.passed for check in checks)
        else "REJECTED"
    )
    return checks, verdict, case_delta


def compute_holdout_metrics(
    cases: list[CaseResult],
    *,
    positive_ids: tuple[str, ...] = HOLDOUT_POSITIVE_IDS,
    negative_ids: tuple[str, ...] = HOLDOUT_NEGATIVE_IDS,
) -> HoldoutArmMetrics:
    by_id = {case.test_id: case for case in cases}

    def hit4(ids: tuple[str, ...]) -> int:
        return sum(1 for case_id in ids if by_id[case_id].primary_hit_at_4)

    def hit12(ids: tuple[str, ...]) -> int:
        return sum(1 for case_id in ids if by_id[case_id].hit_at_12)

    positive_rank_ok = 0
    for case_id in positive_ids:
        case = by_id[case_id]
        docs = [chunk.document_id for chunk in case.retrieved_chunks]
        ranks12 = [index + 1 for index, doc_id in enumerate(docs) if doc_id == DOC12_DOCUMENT_ID]
        ranks08 = [index + 1 for index, doc_id in enumerate(docs) if doc_id == DOC08_DOCUMENT_ID]
        if ranks12 and (not ranks08 or ranks12[0] <= ranks08[0]):
            positive_rank_ok += 1

    neg_doc12_top1 = sum(
        1
        for case_id in negative_ids
        if by_id[case_id].retrieved_chunks
        and by_id[case_id].retrieved_chunks[0].document_id == DOC12_DOCUMENT_ID
    )
    neg_doc12_top4 = sum(
        1
        for case_id in negative_ids
        if any(
            chunk.document_id == DOC12_DOCUMENT_ID
            for chunk in by_id[case_id].retrieved_chunks[:4]
        )
    )

    return HoldoutArmMetrics(
        case_count=len(cases),
        positive_doc12_hit_at_4=hit4(positive_ids),
        positive_doc12_hit_at_12=hit12(positive_ids),
        positive_doc12_rank_vs_doc08_ok=positive_rank_ok,
        negative_domain_hit_at_4=hit4(negative_ids),
        negative_doc12_top1=neg_doc12_top1,
        negative_doc12_top4=neg_doc12_top4,
    )


def classify_holdout_positive_delta(
    *,
    baseline: CaseResult,
    candidate: CaseResult,
) -> str:
    def positive_pass(case: CaseResult) -> bool:
        docs = [chunk.document_id for chunk in case.retrieved_chunks]
        ranks12 = [index + 1 for index, doc_id in enumerate(docs) if doc_id == DOC12_DOCUMENT_ID]
        ranks08 = [index + 1 for index, doc_id in enumerate(docs) if doc_id == DOC08_DOCUMENT_ID]
        return bool(
            case.primary_hit_at_4
            and case.hit_at_12
            and ranks12
            and (not ranks08 or ranks12[0] <= ranks08[0])
        )

    baseline_pass = positive_pass(baseline)
    candidate_pass = positive_pass(candidate)
    if baseline_pass and candidate_pass:
        return "unchanged_pass"
    if not baseline_pass and not candidate_pass:
        return "both_fail_not_doc12_regression"
    if not baseline_pass and candidate_pass:
        return "candidate_improves"
    if baseline_pass and not candidate_pass:
        return "candidate_regression"
    return "rank_order_changed"


def evaluate_holdout_acceptance(
    candidate: HoldoutArmMetrics,
) -> tuple[list[AcceptanceCheck], Doc12Verdict]:
    checks: list[AcceptanceCheck] = []
    positive_total = len(HOLDOUT_POSITIVE_IDS)
    negative_total = len(HOLDOUT_NEGATIVE_IDS)

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
        "holdout_positive_hit4",
        f"Holdout positive doc12 hit@4 >= {HOLDOUT_POSITIVE_HIT4_FLOOR}/{positive_total}",
        candidate.positive_doc12_hit_at_4 >= HOLDOUT_POSITIVE_HIT4_FLOOR,
        f"{candidate.positive_doc12_hit_at_4}/{positive_total}",
        hard=True,
    )
    add(
        "holdout_positive_hit12",
        f"Holdout positive doc12 hit@12 = {HOLDOUT_POSITIVE_HIT12_TARGET}/{positive_total}",
        candidate.positive_doc12_hit_at_12 == HOLDOUT_POSITIVE_HIT12_TARGET,
        f"{candidate.positive_doc12_hit_at_12}/{positive_total}",
        hard=True,
    )
    add(
        "holdout_positive_rank",
        f"Holdout positive doc12 rank vs doc08 ok = {HOLDOUT_POSITIVE_RANK_TARGET}/{positive_total}",
        candidate.positive_doc12_rank_vs_doc08_ok == HOLDOUT_POSITIVE_RANK_TARGET,
        f"{candidate.positive_doc12_rank_vs_doc08_ok}/{positive_total}",
        hard=True,
    )
    add(
        "holdout_negative_hit4",
        f"Holdout negative domain hit@4 >= {HOLDOUT_NEGATIVE_HIT4_FLOOR}/{negative_total}",
        candidate.negative_domain_hit_at_4 >= HOLDOUT_NEGATIVE_HIT4_FLOOR,
        f"{candidate.negative_domain_hit_at_4}/{negative_total}",
    )
    add(
        "holdout_negative_doc12_top1",
        "Holdout negative doc12 not top-1",
        candidate.negative_doc12_top1 == 0,
        str(candidate.negative_doc12_top1),
        hard=True,
    )
    add(
        "holdout_negative_doc12_top4",
        f"Holdout negative doc12 top-4 <= {HOLDOUT_NEGATIVE_DOC12_TOP4_CEILING}/{negative_total}",
        candidate.negative_doc12_top4 <= HOLDOUT_NEGATIVE_DOC12_TOP4_CEILING,
        f"{candidate.negative_doc12_top4}/{negative_total}",
        hard=True,
    )

    verdict: Doc12Verdict = (
        "ACCEPTED AS COMBINED TARGETED CORPUS REPAIR"
        if all(check.passed for check in checks)
        else "REJECTED"
    )
    return checks, verdict


def compute_holdout_dataset_fingerprint(
    *,
    questions_path: Path,
    expected_path: Path,
    benchmark_id: str,
) -> str:
    import hashlib

    payload = {
        "benchmark_id": benchmark_id,
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_holdout_corpus(
    *,
    questions_path: Path,
    expected_path: Path,
    expected_ids: list[str],
) -> list:
    from customer_claims_rag.evaluation.models import EvaluationCase

    questions_text = questions_path.read_text(encoding="utf-8")
    expected_text = expected_path.read_text(encoding="utf-8")
    question_map = _parse_holdout_questions(_split_holdout_sections(questions_text))
    expected_map = _parse_holdout_expected(_split_holdout_sections(expected_text))

    cases: list[EvaluationCase] = []
    for case_id in expected_ids:
        if case_id not in question_map:
            raise EvaluationCorpusError(f"missing holdout question: {case_id}")
        if case_id not in expected_map:
            raise EvaluationCorpusError(f"missing holdout expected: {case_id}")
        question = question_map[case_id]
        expected = expected_map[case_id]
        cases.append(
            EvaluationCase(
                test_id=case_id,
                title=question["title"],
                query=question["query"],
                category=question.get("category"),
                expected_risk=expected["expected_risk"],
                expected_primary_documents=expected["expected_primary_documents"],
                expected_supporting_documents=expected["expected_supporting_documents"],
                fallback_expected=expected["fallback_expected"],
                notes=question.get("notes"),
            )
        )
    return cases


def _split_holdout_sections(text: str) -> list[tuple[str, str, str]]:
    matches = list(HOLDOUT_CASE_HEADER_RE.finditer(text))
    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        case_id = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((case_id, title, text[start:end]))
    return sections


def _parse_holdout_questions(
    sections: list[tuple[str, str, str]],
) -> dict[str, dict[str, object]]:
    parsed: dict[str, dict[str, object]] = {}
    for case_id, title, body in sections:
        fields = _parse_fields(body)
        query = _extract_user_message(body)
        if not query:
            raise EvaluationCorpusError(f"{case_id}: missing user message")
        notes = None
        notes_index = body.find(NOTES_MARKER)
        if notes_index >= 0:
            notes = body[notes_index + len(NOTES_MARKER) :].split("\n", 1)[0].strip()
        parsed[case_id] = {
            "title": title,
            "query": query,
            "category": _optional_text(fields.get("Группа")),
            "notes": notes,
        }
    return parsed


def _parse_holdout_expected(
    sections: list[tuple[str, str, str]],
) -> dict[str, dict[str, object]]:
    parsed: dict[str, dict[str, object]] = {}
    for case_id, _title, body in sections:
        fields = _parse_fields(body)
        risk_raw = _optional_text(fields.get("Expected risk level"))
        if risk_raw not in VALID_RISK_LEVELS:
            raise EvaluationCorpusError(f"{case_id}: invalid risk: {risk_raw!r}")
        primary_docs = _parse_holdout_sources(fields.get("Primary sources"), case_id=case_id)
        supporting_docs = _parse_holdout_sources(
            fields.get("Acceptable additional sources"),
            case_id=case_id,
            allow_empty=True,
        )
        fallback_raw = _optional_text(fields.get("Fallback expected"))
        if fallback_raw not in {"да", "нет"}:
            raise EvaluationCorpusError(f"{case_id}: invalid fallback: {fallback_raw!r}")
        fallback_expected = fallback_raw == "да"
        _validate_fallback_consistency(
            case_id,
            primary_docs=primary_docs,
            supporting_docs=supporting_docs,
            fallback_expected=fallback_expected,
        )
        parsed[case_id] = {
            "expected_risk": risk_raw,
            "expected_primary_documents": primary_docs,
            "expected_supporting_documents": supporting_docs,
            "fallback_expected": fallback_expected,
        }
    return parsed


def _parse_holdout_sources(
    raw: str | None,
    *,
    case_id: str,
    allow_empty: bool = False,
) -> list[str]:
    if raw is None:
        if allow_empty:
            return []
        raise EvaluationCorpusError(f"{case_id}: missing Primary sources")
    normalized = raw.strip().lower()
    if normalized.startswith("отсутствуют"):
        return []
    document_ids = DOCUMENT_ID_RE.findall(raw)
    if not document_ids:
        if allow_empty:
            return []
        raise EvaluationCorpusError(f"{case_id}: no document IDs in sources")
    ordered: list[str] = []
    seen: set[str] = set()
    for document_id in document_ids:
        if document_id not in EXTENSION_KNOWN_DOCUMENT_IDS:
            raise EvaluationCorpusError(f"{case_id}: unknown document {document_id!r}")
        if document_id not in seen:
            seen.add(document_id)
            ordered.append(document_id)
    return ordered


__all__ = [
    "DEFAULT_REFERENCE_ARTIFACT",
    "DOC08_DOCUMENT_ID",
    "DOC12_DOCUMENT_ID",
    "EXPECTED_BASELINE_INDEX_FP",
    "EXPECTED_DOC08_FINGERPRINT",
    "HOLDOUT_NEGATIVE_IDS",
    "HOLDOUT_POSITIVE_IDS",
    "NEGATIVE_EXTENSION_IDS",
    "PRIVACY_EXTENSION_IDS",
    "REFERENCE_ARM",
    "REFERENCE_EXPERIMENT_ID",
    "THREAT_EXTENSION_IDS",
    "BaselineReproductionError",
    "DirtySourceTreeError",
    "Doc08ExperimentalOracle",
    "assert_baseline_reproduction",
    "build_doc12_reachability_comparison",
    "build_extension_case_diagnostic",
    "build_frozen_reachability_snapshot",
    "classify_holdout_positive_delta",
    "classify_threat_case_delta",
    "compute_extension_metrics",
    "compute_holdout_metrics",
    "compute_primary_hit_at_12",
    "count_faq_in_top4",
    "ensure_clean_source_tree",
    "evaluate_extension_acceptance",
    "evaluate_frozen_acceptance",
    "evaluate_holdout_acceptance",
    "fully_unreachable_case_ids",
    "load_doc08_experimental_oracle",
    "load_holdout_corpus",
    "primary_unreachable_case_ids",
    "repo_relative_path",
    "validate_baseline_reproduction",
]
