"""Acceptance metrics for doc08 atomic corpus experiment."""

from __future__ import annotations

from customer_claims_rag.evaluation.doc08_atomic_models import (
    AcceptanceCheck,
    Doc08AtomicCaseResult,
    Doc08CaseDiagnostic,
    Doc08Footprint,
    Doc08Verdict,
    ExtensionArmMetrics,
)
from customer_claims_rag.evaluation.models import AggregateMetrics, CaseResult
from customer_claims_rag.ingestion.corpus_overlay import DOC08_DOCUMENT_ID

DOC12_DOCUMENT_ID = "12_staff_safety_and_threat_handling"

REQUIRED_CASE_IDS = (
    "T011",
    "T035",
    "T039",
    "T040",
    "T044",
    "T046",
    "T047",
    "T051",
)

PRIVACY_EXTENSION_IDS = ("E001", "E002", "E003", "E004")
THREAT_EXTENSION_IDS = ("E005", "E006", "E007", "E008")
NEGATIVE_EXTENSION_IDS = ("E009", "E010", "E011", "E012")

BASELINE_PRIMARY_HIT4 = 0.6206896551724138
BASELINE_PRIMARY_HIT12 = 0.9137931034482759


def build_doc08_footprint(
    case_results: list[Doc08AtomicCaseResult],
    *,
    arm: str,
) -> Doc08Footprint:
    pool_cases: list[str] = []
    top4_cases: list[str] = []
    pool_count = top12_count = top4_count = top1_count = 0

    for item in case_results:
        pool_docs = (
            item.baseline_pool_doc_ids if arm == "baseline" else item.candidate_pool_doc_ids
        )
        final_docs = (
            item.baseline_final_doc_ids if arm == "baseline" else item.candidate_final_doc_ids
        )
        if DOC08_DOCUMENT_ID in pool_docs:
            pool_count += 1
            pool_cases.append(item.case_id)
        if DOC08_DOCUMENT_ID in final_docs[:12]:
            top12_count += 1
        if DOC08_DOCUMENT_ID in final_docs[:4]:
            top4_count += 1
            top4_cases.append(item.case_id)
        if final_docs and final_docs[0] == DOC08_DOCUMENT_ID:
            top1_count += 1

    return Doc08Footprint(
        pool_appearances=pool_count,
        top12_appearances=top12_count,
        top4_appearances=top4_count,
        top1_appearances=top1_count,
        cases_in_pool=sorted(pool_cases),
        cases_in_top4=sorted(top4_cases),
    )


def _ranks_for_doc(results, document_id: str) -> list[int]:
    return [r.rank for r in results if r.document_id == document_id]


def build_case_diagnostic(
    item: Doc08AtomicCaseResult,
    *,
    baseline_vector,
    candidate_vector,
    query: str,
    expected_primary: list[str],
) -> Doc08CaseDiagnostic:
    def pool_rank(doc_ids: list[str]) -> int | None:
        for index, doc_id in enumerate(doc_ids, start=1):
            if doc_id == DOC08_DOCUMENT_ID:
                return index
        return None

    def final_rank(doc_ids: list[str], doc_id: str) -> int | None:
        for index, did in enumerate(doc_ids, start=1):
            if did == doc_id:
                return index
        return None

    best08 = next((r for r in candidate_vector if r.document_id == DOC08_DOCUMENT_ID), None)
    top_competing = [r.document_id for r in candidate_vector[:5] if r.document_id != DOC08_DOCUMENT_ID]

    heading = best08.heading if best08 else None
    sim = float(best08.similarity) if best08 else None
    content_lower = (best08.content if best08 else "").lower()
    grounding = None
    if item.case_id == "T044" and heading:
        grounding = any(
            token in content_lower
            for token in ("персональн", "утечк", "чуж", "наклейк", "адрес", "possible leak")
        )
    elif item.case_id == "T047" and heading:
        grounding = "угроз" in content_lower or "threat" in heading.lower()

    return Doc08CaseDiagnostic(
        case_id=item.case_id,
        query_preview=query[:120],
        expected_primary=expected_primary,
        baseline_vector_ranks_doc08=_ranks_for_doc(baseline_vector, DOC08_DOCUMENT_ID),
        candidate_vector_ranks_doc08=_ranks_for_doc(candidate_vector, DOC08_DOCUMENT_ID),
        baseline_vector_ranks_doc12=_ranks_for_doc(baseline_vector, DOC12_DOCUMENT_ID),
        candidate_vector_ranks_doc12=_ranks_for_doc(candidate_vector, DOC12_DOCUMENT_ID),
        baseline_pool_rank_doc08=pool_rank(item.baseline_pool_doc_ids),
        candidate_pool_rank_doc08=pool_rank(item.candidate_pool_doc_ids),
        baseline_final_rank_doc08=final_rank(item.baseline_final_doc_ids, DOC08_DOCUMENT_ID),
        candidate_final_rank_doc08=final_rank(item.candidate_final_doc_ids, DOC08_DOCUMENT_ID),
        baseline_final_rank_doc12=final_rank(item.baseline_final_doc_ids, DOC12_DOCUMENT_ID),
        candidate_final_rank_doc12=final_rank(item.candidate_final_doc_ids, DOC12_DOCUMENT_ID),
        candidate_best_doc08_heading=heading,
        candidate_best_doc08_similarity=sim,
        top_competing_docs_candidate=top_competing,
        doc08_grounding_relevant=grounding,
        baseline_primary_reachable=item.baseline_pool.primary_reachable,
        candidate_primary_reachable=item.candidate_pool.primary_reachable,
        baseline_primary_hit_at_4=item.baseline_case.primary_hit_at_4,
        candidate_primary_hit_at_4=item.candidate_case.primary_hit_at_4,
        baseline_primary_hit_at_12=item.baseline_case.hit_at_12,
        candidate_primary_hit_at_12=item.candidate_case.hit_at_12,
    )


def compute_extension_metrics(
    cases: list[CaseResult],
    *,
    privacy_ids: tuple[str, ...],
    threat_ids: tuple[str, ...],
    negative_ids: tuple[str, ...],
) -> ExtensionArmMetrics:
    by_id = {c.test_id: c for c in cases}

    def hit4(ids: tuple[str, ...]) -> int:
        return sum(1 for cid in ids if by_id[cid].primary_hit_at_4)

    def hit12(ids: tuple[str, ...]) -> int:
        return sum(1 for cid in ids if by_id[cid].hit_at_12)

    threat_doc12_top4 = 0
    threat_rank_ok = 0
    for cid in threat_ids:
        c = by_id[cid]
        docs = [ch.document_id for ch in c.retrieved_chunks]
        ranks12 = [i + 1 for i, d in enumerate(docs) if d == DOC12_DOCUMENT_ID]
        ranks08 = [i + 1 for i, d in enumerate(docs) if d == DOC08_DOCUMENT_ID]
        if ranks12 and ranks12[0] <= 4:
            threat_doc12_top4 += 1
        if ranks12 and (not ranks08 or ranks12[0] <= ranks08[0]):
            threat_rank_ok += 1

    neg_doc08_top1 = sum(
        1
        for cid in negative_ids
        if by_id[cid].retrieved_chunks
        and by_id[cid].retrieved_chunks[0].document_id == DOC08_DOCUMENT_ID
    )
    neg_doc08_top4 = sum(
        1
        for cid in negative_ids
        if any(ch.document_id == DOC08_DOCUMENT_ID for ch in by_id[cid].retrieved_chunks[:4])
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


def evaluate_extension_acceptance(
    candidate: ExtensionArmMetrics,
) -> tuple[list[AcceptanceCheck], Doc08Verdict]:
    checks: list[AcceptanceCheck] = []

    def add(cid: str, desc: str, passed: bool, val: str, hard: bool = False) -> None:
        checks.append(
            AcceptanceCheck(
                criterion_id=cid,
                description=desc,
                passed=passed,
                candidate_value=val,
                hard_rejection=not passed and hard,
            )
        )

    add("ext_privacy_hit4", "Privacy doc08 hit@4 = 4/4", candidate.privacy_hit_at_4 == 4, f"{candidate.privacy_hit_at_4}/4", hard=True)
    add("ext_privacy_hit12", "Privacy doc08 hit@12 = 4/4", candidate.privacy_hit_at_12 == 4, f"{candidate.privacy_hit_at_12}/4")
    add("ext_threat_doc12_hit4", "Threat doc12 hit@4 = 4/4", candidate.threat_doc12_hit_at_4 == 4, f"{candidate.threat_doc12_hit_at_4}/4", hard=True)
    add("ext_threat_doc12_top4", "Threat doc12 in final top-4 = 4/4", candidate.threat_doc12_in_top4 == 4, f"{candidate.threat_doc12_in_top4}/4", hard=True)
    add("ext_threat_rank", "Threat doc12 rank not worse than doc08 = 4/4", candidate.threat_doc12_rank_vs_doc08_ok == 4, f"{candidate.threat_doc12_rank_vs_doc08_ok}/4", hard=True)
    add("ext_negative_hit4", "Negative domain hit@4 >= 3/4", candidate.negative_domain_hit_at_4 >= 3, f"{candidate.negative_domain_hit_at_4}/4")
    add("ext_negative_doc08_top1", "Doc08 not top-1 in negative cases", candidate.negative_doc08_top1 == 0, str(candidate.negative_doc08_top1), hard=True)
    add("ext_negative_doc08_top4", "Doc08 top-4 in <= 1/4 negative cases", candidate.negative_doc08_top4 <= 1, f"{candidate.negative_doc08_top4}/4")

    passed_all = all(c.passed for c in checks)
    verdict: Doc08Verdict = "ACCEPTED AS TARGETED CORPUS REPAIR" if passed_all else "REJECTED"
    return checks, verdict


def evaluate_frozen_acceptance(
    *,
    baseline_metrics: AggregateMetrics,
    candidate_metrics: AggregateMetrics,
    faq_top4_candidate: int,
    primary_unreachable_baseline: list[str],
    primary_unreachable_candidate: list[str],
    t044_diag: Doc08CaseDiagnostic | None,
    t047_diag: Doc08CaseDiagnostic | None,
) -> tuple[list[AcceptanceCheck], Doc08Verdict]:
    checks: list[AcceptanceCheck] = []

    def add(
        cid: str,
        desc: str,
        passed: bool,
        *,
        baseline_value: str | None = None,
        candidate_value: str | None = None,
        hard: bool = False,
    ) -> None:
        checks.append(
            AcceptanceCheck(
                criterion_id=cid,
                description=desc,
                passed=passed,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                hard_rejection=not passed and hard,
            )
        )

    t044_reach = t044_diag.candidate_primary_reachable if t044_diag else False
    t044_hit12 = t044_diag.candidate_primary_hit_at_12 if t044_diag else False

    add("t044_reachable", "T044 primary reachable", t044_reach, candidate_value=str(t044_reach), hard=True)
    add("t044_hit12", "T044 primary hit@12", t044_hit12, candidate_value=str(t044_hit12), hard=True)
    add(
        "primary_hit_at_4",
        "Primary hit@4 >= baseline",
        candidate_metrics.primary_source_hit_rate_at_4 >= BASELINE_PRIMARY_HIT4,
        baseline_value=str(baseline_metrics.primary_source_hit_rate_at_4),
        candidate_value=str(candidate_metrics.primary_source_hit_rate_at_4),
        hard=True,
    )
    add(
        "primary_hit_at_12",
        "Primary hit@12 >= baseline",
        candidate_metrics.hit_rate_at_12 >= BASELINE_PRIMARY_HIT12,
        candidate_value=str(candidate_metrics.hit_rate_at_12),
    )
    add(
        "mrr",
        "MRR >= 0.645",
        candidate_metrics.mrr >= 0.645,
        candidate_value=str(candidate_metrics.mrr),
        hard=True,
    )
    add("faq_top4", "FAQ top-4 <= 36", faq_top4_candidate <= 36, candidate_value=str(faq_top4_candidate), hard=faq_top4_candidate > 40)

    new_unreachable = set(primary_unreachable_candidate) - set(primary_unreachable_baseline)
    add(
        "no_new_unreachable",
        "No new unreachable primary",
        len(new_unreachable) == 0,
        candidate_value=str(sorted(new_unreachable)),
        hard=bool(new_unreachable),
    )

    if t047_diag:
        doc12_top4 = t047_diag.candidate_final_rank_doc12 is not None and t047_diag.candidate_final_rank_doc12 <= 4
        add(
            "t047_doc12_top4",
            "T047 doc12 remains in final top-4",
            doc12_top4,
            candidate_value=str(t047_diag.candidate_final_rank_doc12),
            hard=True,
        )

    passed_all = all(c.passed for c in checks)
    verdict: Doc08Verdict = "ACCEPTED AS TARGETED CORPUS REPAIR" if passed_all else "REJECTED"
    return checks, verdict


def list_unreachable(
    case_results: list[Doc08AtomicCaseResult],
    *,
    arm: str,
) -> tuple[list[str], list[str]]:
    primary: list[str] = []
    fully: list[str] = []
    for item in case_results:
        pool = item.baseline_pool if arm == "baseline" else item.candidate_pool
        if not pool.primary_reachable:
            primary.append(item.case_id)
        if pool.fully_unreachable:
            fully.append(item.case_id)
    return sorted(primary), sorted(fully)
