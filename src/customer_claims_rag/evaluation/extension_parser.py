"""Extension evaluation corpus parser (non-frozen benchmarks)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from customer_claims_rag.evaluation.models import EvaluationCase, RiskLevel
from customer_claims_rag.evaluation.parser import (
    DOCUMENT_ID_RE,
    FIELD_RE,
    NOTES_MARKER,
    USER_MESSAGE_MARKER,
    VALID_RISK_LEVELS,
    _extract_user_message,
    _optional_text,
    _parse_fields,
    _validate_fallback_consistency,
)
from customer_claims_rag.exceptions import EvaluationCorpusError

EXTENSION_CASE_HEADER_RE = re.compile(r"^## (E\d{3}) — (.+)$", re.MULTILINE)

EXTENSION_KNOWN_DOCUMENT_IDS: set[str] = {
    "01_service_overview",
    "02_delivery_rules",
    "03_order_changes_and_cancellations",
    "04_refund_policy",
    "05_compensation_policy",
    "06_food_quality_and_packaging",
    "07_complaint_handling_procedure",
    "08_escalation_and_risk_rules",
    "09_response_style_and_templates",
    "10_customer_faq",
    "11_payment_security_and_dispute_handling",
    "12_staff_safety_and_threat_handling",
    "13_physical_hazard_and_foreign_body_protocol",
    "14_evidence_standards_and_incomplete_information",
    "15_conflicting_rules_and_remedy_priority",
}


def compute_extension_dataset_fingerprint(
    *,
    questions_path: Path,
    expected_path: Path,
    benchmark_id: str,
) -> str:
    payload = {
        "benchmark_id": benchmark_id,
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_extension_corpus(
    *,
    questions_path: Path,
    expected_path: Path,
    expected_ids: list[str],
) -> list[EvaluationCase]:
    questions_text = questions_path.read_text(encoding="utf-8")
    expected_text = expected_path.read_text(encoding="utf-8")
    question_map = _parse_extension_questions(_split_extension_sections(questions_text))
    expected_map = _parse_extension_expected(_split_extension_sections(expected_text))

    cases: list[EvaluationCase] = []
    for case_id in expected_ids:
        if case_id not in question_map:
            raise EvaluationCorpusError(f"missing extension question: {case_id}")
        if case_id not in expected_map:
            raise EvaluationCorpusError(f"missing extension expected: {case_id}")
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


def _split_extension_sections(text: str) -> list[tuple[str, str, str]]:
    matches = list(EXTENSION_CASE_HEADER_RE.finditer(text))
    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        case_id = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((case_id, title, text[start:end]))
    return sections


def _parse_extension_questions(
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


def _parse_extension_expected(
    sections: list[tuple[str, str, str]],
) -> dict[str, dict[str, object]]:
    parsed: dict[str, dict[str, object]] = {}
    for case_id, _title, body in sections:
        fields = _parse_fields(body)
        risk_raw = _optional_text(fields.get("Expected risk level"))
        if risk_raw not in VALID_RISK_LEVELS:
            raise EvaluationCorpusError(f"{case_id}: invalid risk: {risk_raw!r}")
        primary_docs = _parse_extension_sources(
            fields.get("Primary sources"),
            case_id=case_id,
        )
        supporting_docs = _parse_extension_sources(
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


def _parse_extension_sources(
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
