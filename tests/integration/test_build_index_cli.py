"""Build index CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.cli import build_index as cli_module
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.manifest import load_manifest


def _fake_provider_factory(*, model_name: str, api_key: str | None = None):
    return FakeEmbeddingProvider(model_name=model_name, vector_dimension=8)


def _fake_store_factory(*, index_dir: Path, collection_name: str):
    return ChromaVectorStore(index_dir=index_dir, collection_name=collection_name)


BASELINE_CHUNK_COUNT = 215
RELEASE_EXPANSION_DOC11_CHUNK_COUNT = 20
EXPECTED_CORPUS_CHUNK_COUNT = BASELINE_CHUNK_COUNT + RELEASE_EXPANSION_DOC11_CHUNK_COUNT


def test_cli_successful_build(temp_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(temp_project)
    code = cli_module.run_build(
        input_dir=temp_project / "data" / "02_clean_markdown",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        batch_size=16,
        rebuild=True,
        permitted_root=temp_project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "Status: success" in captured.out
    assert f"Chunks: {EXPECTED_CORPUS_CHUNK_COUNT}" in captured.out
    manifest = load_manifest(temp_project / "data" / "04_index")
    assert manifest.chunk_count == EXPECTED_CORPUS_CHUNK_COUNT


def test_cli_missing_rebuild_flag(temp_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(temp_project)
    code = cli_module.run_build(
        input_dir=temp_project / "data" / "02_clean_markdown",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        batch_size=16,
        rebuild=False,
        permitted_root=temp_project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Error:" in captured.err


def test_cli_missing_api_key_with_real_factory(temp_project: Path, monkeypatch, capsys) -> None:
    from customer_claims_rag import env_bootstrap

    env_bootstrap.reset_project_env()
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: temp_project)
    monkeypatch.chdir(temp_project)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--index-dir",
            "data/04_index",
            "--rebuild",
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "OPENAI_API_KEY" in captured.err
    assert "Traceback" not in captured.err
    assert "sk-" not in captured.err


def test_cli_invalid_batch_size(temp_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(temp_project)
    code = cli_module.run_build(
        input_dir=temp_project / "data" / "02_clean_markdown",
        index_dir=temp_project / "data" / "04_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        batch_size=0,
        rebuild=True,
        permitted_root=temp_project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Error:" in captured.err


def test_cli_no_traceback_without_verbose(temp_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(temp_project)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--index-dir",
            "data/04_index",
            "--batch-size",
            "0",
            "--rebuild",
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" not in captured.err


def test_cli_traceback_with_verbose(temp_project: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(temp_project)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "run_build", explode)
    code = cli_module.main(
        [
            "--input-dir",
            "data/02_clean_markdown",
            "--index-dir",
            "data/04_index",
            "--rebuild",
            "--verbose",
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" in captured.err
