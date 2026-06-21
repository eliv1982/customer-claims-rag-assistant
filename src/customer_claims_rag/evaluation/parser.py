"""Strict parser for evaluation Markdown corpus."""

from __future__ import annotations

import re
from pathlib import Path

from customer_claims_rag.evaluation.models import EvaluationCase, RiskLevel
from customer_claims_rag.exceptions import EvaluationCorpusError

CASE_HEADER_RE = re.compile(r"^## (T\d{3}) — (.+)$", re.MULTILINE)
FIELD_RE = re.compile(r"^- \*\*(.+?):\*\* (.+)$", re.MULTILINE)
DOCUMENT_ID_RE = re.compile(r"`(\d{2}_[a-z0-9_]+)`")
VALID_RISK_LEVELS: set[str] = {"low", "medium", "high", "critical"}
KNOWN_DOCUMENT_IDS: set[str] = {
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
}
EXPECTED_TEST_IDS = [f"T{index:03d}" for index in range(1, 61)]
USER_MESSAGE_MARKER = "**Сообщение пользователя:**"
NOTES_MARKER = "**Что проверяется:**"


def load_evaluation_corpus(
    *,
    questions_path: Path,
    expected_path: Path,
) -> list[EvaluationCase]:
    """Load and validate all 60 evaluation cases from Markdown sources."""
    questions_text = questions_path.read_text(encoding="utf-8")
    expected_text = expected_path.read_text(encoding="utf-8")

    question_sections = _split_sections(questions_text)
    expected_sections = _split_sections(expected_text)

    if len(question_sections) != 60:
        raise EvaluationCorpusError(
            f"expected 60 question sections, found {len(question_sections)}"
        )
    if len(expected_sections) != 60:
        raise EvaluationCorpusError(
            f"expected 60 expected-answer sections, found {len(expected_sections)}"
        )

    question_map = _parse_question_sections(question_sections)
    expected_map = _parse_expected_sections(expected_sections)

    missing_questions = [test_id for test_id in EXPECTED_TEST_IDS if test_id not in question_map]
    if missing_questions:
        raise EvaluationCorpusError(
            f"missing question records: {', '.join(missing_questions)}"
        )

    missing_expected = [test_id for test_id in EXPECTED_TEST_IDS if test_id not in expected_map]
    if missing_expected:
        raise EvaluationCorpusError(
            f"missing expected-answer records: {', '.join(missing_expected)}"
        )

    extra_questions = sorted(set(question_map) - set(EXPECTED_TEST_IDS))
    if extra_questions:
        raise EvaluationCorpusError(
            f"unexpected question test IDs: {', '.join(extra_questions)}"
        )

    extra_expected = sorted(set(expected_map) - set(EXPECTED_TEST_IDS))
    if extra_expected:
        raise EvaluationCorpusError(
            f"unexpected expected-answer test IDs: {', '.join(extra_expected)}"
        )

    cases: list[EvaluationCase] = []
    for test_id in EXPECTED_TEST_IDS:
        question = question_map[test_id]
        expected = expected_map[test_id]
        cases.append(
            EvaluationCase(
                test_id=test_id,
                title=question["title"],
                query=question["query"],
                category=question.get("category"),
                check_types=question.get("check_types", []),
                complexity=question.get("complexity"),
                has_dialog_history=question.get("has_dialog_history", False),
                expected_risk=expected["expected_risk"],
                expected_primary_documents=expected["expected_primary_documents"],
                expected_supporting_documents=expected["expected_supporting_documents"],
                fallback_expected=expected["fallback_expected"],
                notes=question.get("notes"),
            )
        )
    return cases


def _split_sections(text: str) -> list[tuple[str, str, str]]:
    matches = list(CASE_HEADER_RE.finditer(text))
    sections: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        test_id = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        sections.append((test_id, title, body))
    return sections


def _parse_question_sections(
    sections: list[tuple[str, str, str]],
) -> dict[str, dict[str, object]]:
    parsed: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for test_id, title, body in sections:
        if test_id in seen:
            raise EvaluationCorpusError(f"duplicate question test ID: {test_id}")
        seen.add(test_id)
        fields = _parse_fields(body)
        category = _optional_text(fields.get("Группа"))
        check_types_raw = fields.get("Тип проверки")
        check_types = (
            [part.strip() for part in str(check_types_raw).split(",") if part.strip()]
            if check_types_raw
            else []
        )
        complexity = _optional_text(fields.get("Сложность"))
        history = _optional_text(fields.get("История диалога"))
        query = _extract_user_message(body)
        if not query:
            raise EvaluationCorpusError(f"{test_id}: missing user message")
        notes = None
        notes_index = body.find(NOTES_MARKER)
        if notes_index >= 0:
            notes = body[notes_index + len(NOTES_MARKER) :].split("\n", 1)[0].strip()
        parsed[test_id] = {
            "title": title,
            "query": query,
            "category": category,
            "check_types": check_types,
            "complexity": complexity,
            "has_dialog_history": history == "присутствует",
            "notes": notes,
        }
    return parsed


def _parse_expected_sections(
    sections: list[tuple[str, str, str]],
) -> dict[str, dict[str, object]]:
    parsed: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for test_id, _title, body in sections:
        if test_id in seen:
            raise EvaluationCorpusError(f"duplicate expected test ID: {test_id}")
        seen.add(test_id)
        fields = _parse_fields(body)
        risk_raw = _optional_text(fields.get("Expected risk level"))
        if risk_raw not in VALID_RISK_LEVELS:
            raise EvaluationCorpusError(
                f"{test_id}: malformed expected risk level: {risk_raw!r}"
            )
        primary_raw = fields.get("Primary sources")
        supporting_raw = fields.get("Acceptable additional sources")
        fallback_raw = _optional_text(fields.get("Fallback expected"))
        if fallback_raw not in {"да", "нет"}:
            raise EvaluationCorpusError(
                f"{test_id}: malformed fallback expected marker: {fallback_raw!r}"
            )
        primary_docs = _parse_source_field(
            primary_raw,
            field_name="Primary sources",
            test_id=test_id,
        )
        supporting_docs = _parse_supporting_sources(
            supporting_raw,
            test_id=test_id,
        )
        fallback_expected = fallback_raw == "да"
        _validate_fallback_consistency(
            test_id,
            primary_docs=primary_docs,
            supporting_docs=supporting_docs,
            fallback_expected=fallback_expected,
        )
        parsed[test_id] = {
            "expected_risk": risk_raw,
            "expected_primary_documents": primary_docs,
            "expected_supporting_documents": supporting_docs,
            "fallback_expected": fallback_expected,
        }
    return parsed


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in FIELD_RE.finditer(body):
        fields[match.group(1)] = match.group(2).strip()
    return fields


def _extract_user_message(body: str) -> str:
    marker_index = body.find(USER_MESSAGE_MARKER)
    if marker_index >= 0:
        search_region = body[marker_index + len(USER_MESSAGE_MARKER) :]
    else:
        search_region = body
    lines: list[str] = []
    started = False
    for line in search_region.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            started = True
            lines.append(stripped[1:].lstrip())
            continue
        if started and stripped:
            break
        if started and not stripped:
            break
    return "\n".join(lines).strip()


def _parse_source_field(
    raw: str | None,
    *,
    field_name: str,
    test_id: str,
) -> list[str]:
    if raw is None:
        raise EvaluationCorpusError(f"{test_id}: missing {field_name}")
    normalized = raw.strip().lower()
    if normalized.startswith("отсутствуют"):
        return []
    return _extract_document_ids(raw, test_id=test_id, field_name=field_name)


def _validate_fallback_consistency(
    test_id: str,
    *,
    primary_docs: list[str],
    supporting_docs: list[str],
    fallback_expected: bool,
) -> None:
    if fallback_expected and primary_docs:
        raise EvaluationCorpusError(
            f"{test_id}: fallback expected but primary sources are defined"
        )
    if not fallback_expected and not primary_docs and not supporting_docs:
        raise EvaluationCorpusError(
            f"{test_id}: non-fallback case must define primary or supporting sources"
        )


def _parse_supporting_sources(raw: str | None, *, test_id: str) -> list[str]:
    if raw is None:
        return []
    normalized = raw.strip().lower()
    if normalized.startswith("отсутствуют"):
        return []
    return _extract_document_ids(raw, test_id=test_id, field_name="Acceptable additional sources")


def _extract_document_ids(raw: str, *, test_id: str, field_name: str) -> list[str]:
    document_ids = DOCUMENT_ID_RE.findall(raw)
    if not document_ids:
        raise EvaluationCorpusError(
            f"{test_id}: malformed {field_name}; no document IDs found in {raw!r}"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for document_id in document_ids:
        if document_id not in KNOWN_DOCUMENT_IDS:
            raise EvaluationCorpusError(
                f"{test_id}: unknown document ID {document_id!r} in {field_name}"
            )
        if document_id not in seen:
            seen.add(document_id)
            ordered.append(document_id)
    return ordered


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
