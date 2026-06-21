"""Evaluation corpus parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.evaluation.parser import (
    EXPECTED_TEST_IDS,
    _parse_question_sections,
    _split_sections,
    load_evaluation_corpus,
)
from customer_claims_rag.exceptions import EvaluationCorpusError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT_ROOT / "tests" / "01_test_questions.md"
EXPECTED = PROJECT_ROOT / "tests" / "02_expected_answers.md"


def test_parser_loads_exactly_60_real_cases() -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    assert len(cases) == 60
    assert [case.test_id for case in cases] == EXPECTED_TEST_IDS


def test_parser_extracts_structured_fields() -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    first = cases[0]
    assert first.test_id == "T001"
    assert first.query
    assert first.category == "general"
    assert first.expected_risk == "low"
    assert first.expected_primary_documents == ["01_service_overview"]
    assert first.fallback_expected is False


def test_parser_fallback_cases() -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    t006 = next(case for case in cases if case.test_id == "T006")
    t060 = next(case for case in cases if case.test_id == "T060")
    assert t006.fallback_expected is True
    assert t006.expected_primary_documents == []
    assert t060.fallback_expected is True
    assert t060.expected_primary_documents == []


def test_parser_multiline_and_unicode_queries() -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    t058 = next(case for case in cases if case.test_id == "T058")
    assert "FF-88888" in t058.query
    assert "<retrieved_context>" in t058.query
    assert any("Итал" in case.query or "FF-" in case.query for case in cases)


def test_parser_multiple_primary_sources() -> None:
    cases = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    t003 = next(case for case in cases if case.test_id == "T003")
    assert set(t003.expected_primary_documents) == {
        "01_service_overview",
        "07_complaint_handling_procedure",
    }


def test_parser_duplicate_question_id_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "broken.md"
    broken.write_text(
        "## T001 — A\n\n> q1\n\n---\n\n## T001 — B\n\n> q2\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationCorpusError, match="duplicate question"):
        _parse_question_sections(_split_sections(broken.read_text(encoding="utf-8")))


def test_parser_missing_expected_record_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing_expected.md"
    missing.write_text("## T001 — A\n\n> q\n", encoding="utf-8")
    with pytest.raises(EvaluationCorpusError, match="expected 60 question sections"):
        load_evaluation_corpus(questions_path=missing, expected_path=EXPECTED)


def test_parser_missing_question_record_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(EvaluationCorpusError, match="expected 60 expected-answer sections"):
        load_evaluation_corpus(questions_path=QUESTIONS, expected_path=empty)


def test_parser_malformed_risk_rejected(tmp_path: Path) -> None:
    text = EXPECTED.read_text(encoding="utf-8").replace(
        "- **Expected risk level:** low",
        "- **Expected risk level:** unknown",
        1,
    )
    broken = tmp_path / "broken_expected.md"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(EvaluationCorpusError, match="malformed expected risk level"):
        load_evaluation_corpus(questions_path=QUESTIONS, expected_path=broken)


def test_parser_malformed_source_id_rejected(tmp_path: Path) -> None:
    text = EXPECTED.read_text(encoding="utf-8").replace(
        "`01_service_overview`",
        "`99_unknown_doc`",
        1,
    )
    broken = tmp_path / "broken_expected.md"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(EvaluationCorpusError, match="unknown document ID"):
        load_evaluation_corpus(questions_path=QUESTIONS, expected_path=broken)
