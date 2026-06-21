"""Evaluate retrieval CLI tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.cli import evaluate_retrieval as eval_cli
from customer_claims_rag.exceptions import EvaluationCorpusError, RetrievalError
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from tests.evaluation_helpers import make_fake_evaluation_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fake_provider_factory(*, model_name: str, api_key: str | None = None):
    return FakeEmbeddingProvider(model_name=model_name, vector_dimension=8)


def _fake_store_factory(*, index_dir: Path, collection_name: str):
    return ChromaVectorStore(index_dir=index_dir, collection_name=collection_name)


def test_cli_success(monkeypatch, temp_project: Path, capsys) -> None:
    monkeypatch.chdir(temp_project)
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = make_fake_evaluation_run()
    monkeypatch.setattr(eval_cli, "RetrievalEvaluator", lambda **kwargs: mock_evaluator)
    monkeypatch.setattr(
        eval_cli,
        "write_evaluation_outputs",
        lambda run, **kwargs: (
            temp_project / "data" / "05_evaluation" / "retrieval_results.json",
            temp_project / "tests" / "03_test_results.md",
            temp_project / "tests" / "04_improvement_log.md",
        ),
    )
    (temp_project / "tests").mkdir(exist_ok=True)
    code, _ = eval_cli.run_evaluation(
        questions_path=PROJECT_ROOT / "tests" / "01_test_questions.md",
        expected_path=PROJECT_ROOT / "tests" / "02_expected_answers.md",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        project_root_path=temp_project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Baseline retrieval evaluation complete" in captured.out


def test_cli_malformed_corpus_nonzero(temp_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(temp_project)
    tests_dir = temp_project / "tests"
    tests_dir.mkdir(exist_ok=True)
    questions_path = tests_dir / "01_test_questions.md"
    expected_path = tests_dir / "02_expected_answers.md"
    questions_path.write_text("not a valid corpus", encoding="utf-8")
    expected_path.write_text("not a valid corpus", encoding="utf-8")
    code = eval_cli.run_evaluation(
        questions_path=questions_path,
        expected_path=expected_path,
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        project_root_path=temp_project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )[0]
    captured = capsys.readouterr()
    assert code == 2
    assert "Error:" in captured.err


def test_cli_rejects_unsafe_output_path(temp_project: Path, capsys) -> None:
    code = eval_cli.run_evaluation(
        questions_path=PROJECT_ROOT / "tests" / "01_test_questions.md",
        expected_path=PROJECT_ROOT / "tests" / "02_expected_answers.md",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        output_json=temp_project / "data" / "02_clean_markdown" / "bad.json",
        project_root_path=temp_project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )[0]
    captured = capsys.readouterr()
    assert code == 1
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_cli_verbose_sanitized_traceback(temp_project: Path, monkeypatch, capsys) -> None:
    SECRET = "sk-test-secret-value"

    def explode(*args, **kwargs):
        raise RuntimeError(f"boom {SECRET}")

    monkeypatch.setattr(eval_cli, "run_evaluation", explode)
    code = eval_cli.main(["--verbose"])
    captured = capsys.readouterr()
    assert code == 1
    assert SECRET not in captured.err
    assert "Traceback" in captured.err
