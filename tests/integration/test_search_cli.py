"""Search index CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.cli import build_index as build_cli
from customer_claims_rag.cli import search_index as search_cli
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider


def _fake_provider_factory(*, model_name: str, api_key: str | None = None):
    return FakeEmbeddingProvider(model_name=model_name, vector_dimension=8)


def _fake_store_factory(*, index_dir: Path, collection_name: str):
    return ChromaVectorStore(index_dir=index_dir, collection_name=collection_name)


@pytest.fixture
def indexed_project(temp_project: Path, monkeypatch) -> Path:
    monkeypatch.chdir(temp_project)
    build_cli.run_build(
        input_dir=temp_project / "data" / "02_clean_markdown",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        batch_size=32,
        rebuild=True,
        permitted_root=temp_project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    return temp_project


def test_cli_json_output_reflects_threshold(indexed_project: Path, capsys) -> None:
    code, _ = search_cli.run_search(
        query="возврат",
        index_dir=indexed_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        top_k=2,
        fetch_k=4,
        threshold=0.35,
        json_output=True,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["similarity_threshold"] == 0.35


def test_cli_default_threshold_without_override(
    indexed_project: Path,
    monkeypatch,
    capsys,
) -> None:
    from customer_claims_rag import env_bootstrap

    env_bootstrap.reset_project_env()
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: indexed_project)
    monkeypatch.delenv("RAG_SIMILARITY_THRESHOLD", raising=False)
    code, _ = search_cli.run_search(
        query="возврат",
        index_dir=indexed_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        json_output=True,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["similarity_threshold"] == 0.0
    assert payload["results"]


def test_cli_json_error_nonzero_without_stdout_json(temp_project: Path, capsys) -> None:
    code, _ = search_cli.run_search(
        query="test",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        json_output=True,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
        project_root=temp_project.resolve(),
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Error:" in captured.err
    assert captured.out.strip() == ""


def test_cli_successful_search(indexed_project: Path, capsys) -> None:
    code, _ = search_cli.run_search(
        query="доставка заказа",
        index_dir=indexed_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        top_k=4,
        fetch_k=12,
        threshold=0.0,
        json_output=False,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "#1 similarity=" in captured.out or "No results above threshold" in captured.out


def test_cli_json_output(indexed_project: Path, capsys) -> None:
    code, response = search_cli.run_search(
        query="возврат",
        index_dir=indexed_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        top_k=2,
        fetch_k=4,
        threshold=0.0,
        json_output=True,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["query"] == "возврат"
    assert "results" in payload


def test_cli_no_results(indexed_project: Path, capsys) -> None:
    code, _ = search_cli.run_search(
        query="возврат",
        index_dir=indexed_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        top_k=4,
        fetch_k=12,
        threshold=0.99,
        json_output=False,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "No results above threshold" in captured.out


def test_cli_no_results_json(indexed_project: Path, capsys) -> None:
    code, _ = search_cli.run_search(
        query="возврат",
        index_dir=indexed_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        threshold=0.99,
        json_output=True,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["results"] == []
    assert payload["candidates_above_threshold"] == 0


def test_cli_missing_api_key_with_real_factory(temp_project: Path, monkeypatch, capsys) -> None:
    from customer_claims_rag import env_bootstrap

    env_bootstrap.reset_project_env()
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: temp_project)
    monkeypatch.chdir(temp_project)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = search_cli.main(["query", "--index-dir", "data/04_index"])
    captured = capsys.readouterr()
    assert code != 0
    assert "OPENAI_API_KEY" in captured.err
    assert "Traceback" not in captured.err
    assert "sk-" not in captured.err


def test_cli_missing_index(temp_project: Path, capsys) -> None:
    code, _ = search_cli.run_search(
        query="test",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        json_output=False,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
        project_root=temp_project.resolve(),
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "manifest not found" in captured.err


def test_cli_manifest_mismatch(indexed_project: Path, capsys) -> None:
    code, _ = search_cli.run_search(
        query="test",
        index_dir=indexed_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="other-model",
        threshold=0.0,
        json_output=False,
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "embedding model mismatch" in captured.err


def test_cli_invalid_numeric_arguments(indexed_project: Path, capsys) -> None:
    code = search_cli.main(
        [
            "query",
            "--index-dir",
            "data/04_index",
            "--top-k",
            "0",
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Error:" in captured.err


def test_cli_no_traceback_without_verbose(indexed_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(indexed_project)
    code = search_cli.main(["query", "--index-dir", "missing-index"])
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" not in captured.err


def test_cli_traceback_with_verbose(indexed_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(indexed_project)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(search_cli, "run_search", explode)
    code = search_cli.main(["query", "--verbose"])
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" in captured.err
