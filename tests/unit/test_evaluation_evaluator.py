"""Evaluation evaluator tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.evaluation.evaluator import RetrievalEvaluator
from customer_claims_rag.evaluation.models import EvaluationCase
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.retrieval.models import IndexManifest
from tests.evaluation_helpers import make_search_response, make_search_result

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT_ROOT / "tests" / "01_test_questions.md"
EXPECTED = PROJECT_ROOT / "tests" / "02_expected_answers.md"


class FakeRetriever:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[str] = []
        self.vector_store = MagicMock()
        self.vector_store.count.return_value = 215

    def validate_index(self) -> None:
        return None

    def search(self, query: str):
        self.calls.append(query)
        response = self.responses.get(query)
        if isinstance(response, Exception):
            raise response
        return response


def _cases() -> list[EvaluationCase]:
    return load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)


def test_evaluator_calls_search_once_per_case(monkeypatch) -> None:
    cases = _cases()[:3]
    responses = {
        case.query: make_search_response(
            case.query,
            [make_search_result(rank=1, document_id=case.expected_primary_documents[0] or "01_service_overview")],
        )
        for case in cases
    }
    retriever = FakeRetriever(responses)
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.evaluator.load_manifest",
        lambda _index_dir: IndexManifest(
            index_format_version="1.0.0",
            collection_name="customer_claims",
            embedding_model="fake-embedding-model",
            corpus_fingerprint="abc",
            chunk_count=215,
            document_count=10,
            metadata_schema_version="1.0.0",
            vector_dimension=8,
        ),
    )
    evaluator = RetrievalEvaluator(
        retriever=retriever,
        index_dir=Path("data/04_index"),
        cases=cases,
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        project_root=Path("."),
    )
    run = evaluator.evaluate()
    assert len(retriever.calls) == 3
    assert [case.test_id for case in run.case_results] == ["T001", "T002", "T003"]


def test_evaluator_preserves_case_order_for_full_corpus(monkeypatch) -> None:
    cases = _cases()
    responses = {}
    for case in cases:
        primary = case.expected_primary_documents[0] if case.expected_primary_documents else "01_service_overview"
        responses[case.query] = make_search_response(
            case.query,
            [make_search_result(rank=1, document_id=primary, similarity=0.8)],
        )
    retriever = FakeRetriever(responses)
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.evaluator.load_manifest",
        lambda _index_dir: IndexManifest(
            index_format_version="1.0.0",
            collection_name="customer_claims",
            embedding_model="fake-embedding-model",
            corpus_fingerprint="abc",
            chunk_count=215,
            document_count=10,
            metadata_schema_version="1.0.0",
            vector_dimension=8,
        ),
    )
    evaluator = RetrievalEvaluator(
        retriever=retriever,
        index_dir=Path("data/04_index"),
        cases=cases,
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        project_root=Path("."),
    )
    run = evaluator.evaluate()
    assert len(retriever.calls) == 60
    assert [case.test_id for case in run.case_results] == [f"T{index:03d}" for index in range(1, 61)]


def test_evaluator_records_case_specific_error(monkeypatch) -> None:
    from customer_claims_rag.exceptions import SearchError

    cases = _cases()[:2]
    retriever = FakeRetriever({cases[0].query: SearchError("search failed")})
    retriever.responses[cases[1].query] = make_search_response(
        cases[1].query,
        [make_search_result(rank=1, document_id="01_service_overview")],
    )
    monkeypatch.setattr(
        "customer_claims_rag.evaluation.evaluator.load_manifest",
        lambda _index_dir: IndexManifest(
            index_format_version="1.0.0",
            collection_name="customer_claims",
            embedding_model="fake-embedding-model",
            corpus_fingerprint="abc",
            chunk_count=215,
            document_count=10,
            metadata_schema_version="1.0.0",
            vector_dimension=8,
        ),
    )
    evaluator = RetrievalEvaluator(
        retriever=retriever,
        index_dir=Path("data/04_index"),
        cases=cases,
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        project_root=Path("."),
    )
    run = evaluator.evaluate()
    assert run.case_results[0].status == "technical_error"
    assert run.case_results[1].status == "success"
    assert run.aggregate_metrics.technical_error_count == 1
