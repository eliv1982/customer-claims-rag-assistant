"""Fatal preflight and CLI failure tests for evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.cli import evaluate_retrieval as eval_cli
from customer_claims_rag.exceptions import IndexManifestError
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.manifest import build_manifest, write_manifest_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT_ROOT / "tests" / "01_test_questions.md"
EXPECTED = PROJECT_ROOT / "tests" / "02_expected_answers.md"


def _fake_provider_factory(*, model_name: str, api_key: str | None = None):
    return FakeEmbeddingProvider(model_name=model_name, vector_dimension=8)


def _fake_store_factory(*, index_dir: Path, collection_name: str):
    return ChromaVectorStore(index_dir=index_dir, collection_name=collection_name)


def _write_manifest(
    index_dir: Path,
    *,
    embedding_model: str = "fake-embedding-model",
    collection_name: str = "customer_claims",
    chunk_count: int = 0,
    metadata_schema_version: str = "1.0.0",
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_atomic(
        index_dir,
        build_manifest(
            collection_name=collection_name,
            embedding_model=embedding_model,
            corpus_fingerprint="test-fp",
            chunk_count=chunk_count,
            document_count=1,
            metadata_schema_version=metadata_schema_version,
            vector_dimension=8,
        ),
    )


def _run(temp_project: Path, **overrides):
    kwargs = {
        "questions_path": QUESTIONS,
        "expected_path": EXPECTED,
        "index_dir": temp_project / "data" / "04_index",
        "collection_name": "customer_claims",
        "embedding_model": "fake-embedding-model",
        "project_root_path": temp_project.resolve(),
        "embedding_provider_factory": _fake_provider_factory,
        "vector_store_factory": _fake_store_factory,
    }
    kwargs.update(overrides)
    return eval_cli.run_evaluation(**kwargs)


def test_missing_manifest_nonzero_exit(temp_project: Path, capsys) -> None:
    index_dir = temp_project / "data" / "04_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    code, run = _run(temp_project)
    captured = capsys.readouterr()
    assert code == 1
    assert run is None
    assert "manifest not found" in captured.err
    assert "Baseline retrieval evaluation complete" not in captured.out


def test_embedding_model_mismatch_fatal(temp_project: Path, capsys) -> None:
    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    try:
        _write_manifest(index_dir, embedding_model="other-model", chunk_count=store.count())
    finally:
        store.close()
    code, _ = _run(temp_project)
    captured = capsys.readouterr()
    assert code == 1
    assert "embedding model mismatch" in captured.err


def test_collection_mismatch_fatal(temp_project: Path, capsys) -> None:
    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    try:
        _write_manifest(index_dir, collection_name="wrong_collection", chunk_count=store.count())
    finally:
        store.close()
    code, _ = _run(temp_project, collection_name="customer_claims")
    captured = capsys.readouterr()
    assert code == 1
    assert "collection name mismatch" in captured.err


def test_metadata_schema_mismatch_fatal(temp_project: Path, capsys) -> None:
    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    try:
        _write_manifest(
            index_dir,
            metadata_schema_version="9.9.9",
            chunk_count=store.count(),
        )
    finally:
        store.close()
    code, _ = _run(temp_project)
    captured = capsys.readouterr()
    assert code == 1
    assert "metadata schema version mismatch" in captured.err


def test_chunk_count_mismatch_fatal(temp_project: Path, capsys) -> None:
    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    try:
        _write_manifest(index_dir, chunk_count=999)
    finally:
        store.close()
    code, _ = _run(temp_project)
    captured = capsys.readouterr()
    assert code == 1
    assert "chunk count mismatch" in captured.err


def test_missing_api_key_fatal(temp_project: Path, monkeypatch, capsys) -> None:
    from customer_claims_rag import env_bootstrap

    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    try:
        _write_manifest(index_dir, chunk_count=store.count())
    finally:
        store.close()

    def real_factory(*, model_name: str, api_key: str | None = None):
        from customer_claims_rag.retrieval.factory import create_embedding_provider

        return create_embedding_provider(model_name=model_name, api_key=api_key)

    env_bootstrap.reset_project_env()
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: temp_project)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code, _ = _run(temp_project, embedding_provider_factory=real_factory)
    captured = capsys.readouterr()
    assert code == 1
    assert "OPENAI_API_KEY" in captured.err
    assert "Baseline retrieval evaluation complete" not in captured.out


def test_preflight_failure_skips_search(temp_project: Path, monkeypatch) -> None:
    index_dir = temp_project / "data" / "04_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    search_calls = 0

    class SpyRetriever:
        def validate_index(self):
            raise IndexManifestError("manifest not found")

        def search(self, query: str):
            nonlocal search_calls
            search_calls += 1
            raise AssertionError("search should not run")

    monkeypatch.setattr(
        eval_cli,
        "BaselineRetriever",
        lambda **kwargs: SpyRetriever(),
    )
    code, _ = _run(temp_project)
    assert code == 1
    assert search_calls == 0


def test_parser_runs_before_embedding_on_malformed_corpus(temp_project: Path, capsys) -> None:
    broken = temp_project / "broken.md"
    broken.write_text("not a valid corpus", encoding="utf-8")
    embed_calls = 0

    def counting_factory(*, model_name: str, api_key: str | None = None):
        nonlocal embed_calls
        embed_calls += 1
        return _fake_provider_factory(model_name=model_name, api_key=api_key)

    code, _ = eval_cli.run_evaluation(
        questions_path=broken,
        expected_path=EXPECTED,
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        project_root_path=temp_project.resolve(),
        embedding_provider_factory=counting_factory,
        vector_store_factory=_fake_store_factory,
    )
    assert code == 2
    assert embed_calls == 0


def test_no_success_reports_on_fatal_preflight(temp_project: Path, capsys) -> None:
    results_path = temp_project / "tests" / "03_test_results.md"
    results_path.parent.mkdir(parents=True)
    results_path.write_text("PRE-RUN\n", encoding="utf-8")
    index_dir = temp_project / "data" / "04_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    code, _ = _run(
        temp_project,
        output_markdown=results_path,
        output_json=temp_project / "data" / "05_evaluation" / "retrieval_results.json",
    )
    assert code == 1
    assert results_path.read_text(encoding="utf-8") == "PRE-RUN\n"


def test_partial_output_write_nonzero(temp_project: Path, monkeypatch, capsys) -> None:
    from customer_claims_rag.evaluation import reporting
    from tests.evaluation_helpers import make_fake_evaluation_run

    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    try:
        _write_manifest(index_dir, chunk_count=store.count())
    finally:
        store.close()

    mock_evaluator = type("E", (), {"evaluate": lambda self: make_fake_evaluation_run()})()
    monkeypatch.setattr(
        eval_cli,
        "RetrievalEvaluator",
        lambda **kwargs: mock_evaluator,
    )

    original_atomic = reporting._atomic_write_text
    calls = {"count": 0}

    def flaky_atomic(path, content, *, temp_paths):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("write failed")
        return original_atomic(path, content, temp_paths=temp_paths)

    monkeypatch.setattr(reporting, "_atomic_write_text", flaky_atomic)

    tests_dir = temp_project / "tests"
    tests_dir.mkdir(exist_ok=True)
    results_path = tests_dir / "03_test_results.md"
    improvement_path = tests_dir / "04_improvement_log.md"
    results_path.write_text("PRE-RUN\n", encoding="utf-8")
    improvement_path.write_text("PRE-RUN\n", encoding="utf-8")
    json_path = temp_project / "data" / "05_evaluation" / "retrieval_results.json"

    code, _ = _run(
        temp_project,
        output_json=json_path,
        output_markdown=results_path,
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "partially updated" in captured.err
    assert "Baseline retrieval evaluation complete" not in captured.out
    assert json_path.exists()
    assert results_path.read_text(encoding="utf-8") == "PRE-RUN\n"
    assert improvement_path.read_text(encoding="utf-8") == "PRE-RUN\n"
    assert not list(tests_dir.glob("*.tmp"))
    assert not list((temp_project / "data" / "05_evaluation").glob("*.tmp"))


def test_third_output_replace_failure_nonzero(temp_project: Path, monkeypatch, capsys) -> None:
    from customer_claims_rag.evaluation import reporting
    from tests.evaluation_helpers import make_fake_evaluation_run

    index_dir = temp_project / "data" / "04_index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    try:
        _write_manifest(index_dir, chunk_count=store.count())
    finally:
        store.close()

    mock_evaluator = type("E", (), {"evaluate": lambda self: make_fake_evaluation_run()})()
    monkeypatch.setattr(
        eval_cli,
        "RetrievalEvaluator",
        lambda **kwargs: mock_evaluator,
    )

    original_atomic = reporting._atomic_write_text
    calls = {"count": 0}

    def flaky_atomic(path, content, *, temp_paths):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("write failed")
        return original_atomic(path, content, temp_paths=temp_paths)

    monkeypatch.setattr(reporting, "_atomic_write_text", flaky_atomic)

    tests_dir = temp_project / "tests"
    tests_dir.mkdir(exist_ok=True)
    results_path = tests_dir / "03_test_results.md"
    improvement_path = tests_dir / "04_improvement_log.md"
    improvement_path.write_text("PRE-RUN\n", encoding="utf-8")
    json_path = temp_project / "data" / "05_evaluation" / "retrieval_results.json"

    code, _ = _run(
        temp_project,
        output_json=json_path,
        output_markdown=results_path,
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "partially updated" in captured.err
    assert "Baseline retrieval evaluation complete" not in captured.out
    assert json_path.exists()
    json.loads(json_path.read_text(encoding="utf-8"))
    assert "Evaluation result ID" in results_path.read_text(encoding="utf-8")
    assert improvement_path.read_text(encoding="utf-8") == "PRE-RUN\n"
    assert not list(tests_dir.glob("*.tmp"))
    assert not list((temp_project / "data" / "05_evaluation").glob("*.tmp"))
