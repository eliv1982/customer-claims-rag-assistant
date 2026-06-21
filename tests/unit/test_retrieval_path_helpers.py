"""Retrieval index path safety tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from customer_claims_rag.cli import build_index as build_cli
from customer_claims_rag.config import DEFAULT_INDEX_DIR, DEFAULT_INPUT_DIR
from customer_claims_rag.exceptions import RetrievalError
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.path_helpers import validate_index_dir


def test_default_index_dir_allowed(temp_project: Path) -> None:
    resolved = validate_index_dir(
        DEFAULT_INDEX_DIR,
        project_root=temp_project.resolve(),
        input_dir=DEFAULT_INPUT_DIR,
    )
    assert resolved == (temp_project / "data" / "04_index").resolve()


def test_custom_index_dir_inside_project_allowed(temp_project: Path) -> None:
    custom = temp_project / "data" / "custom_index"
    resolved = validate_index_dir(
        custom,
        project_root=temp_project.resolve(),
    )
    assert resolved == custom.resolve()


def test_index_dir_same_as_input_rejected(temp_project: Path) -> None:
    input_dir = temp_project / "data" / "02_clean_markdown"
    with pytest.raises(RetrievalError, match="must not be the same as input_dir"):
        validate_index_dir(
            input_dir,
            project_root=temp_project.resolve(),
            input_dir=input_dir,
        )


def test_index_dir_inside_input_rejected(temp_project: Path) -> None:
    input_dir = temp_project / "data" / "02_clean_markdown"
    nested = input_dir / "nested_index"
    with pytest.raises(RetrievalError, match="inside input directory"):
        validate_index_dir(
            nested,
            project_root=temp_project.resolve(),
            input_dir=input_dir,
        )


def test_index_dir_inside_clean_markdown_rejected(temp_project: Path) -> None:
    index_dir = temp_project / "data" / "02_clean_markdown" / "index"
    with pytest.raises(RetrievalError, match="clean Markdown directory"):
        validate_index_dir(index_dir, project_root=temp_project.resolve())


def test_index_dir_inside_raw_rejected(temp_project: Path) -> None:
    raw_dir = temp_project / "data" / "01_raw"
    raw_dir.mkdir(parents=True)
    index_dir = raw_dir / "index"
    with pytest.raises(RetrievalError, match="raw source directory"):
        validate_index_dir(index_dir, project_root=temp_project.resolve())


def test_path_outside_project_root_rejected(temp_project: Path) -> None:
    outside = temp_project.parent / "outside_index_for_test"
    outside.mkdir(exist_ok=True)
    with pytest.raises(RetrievalError, match="path traversal rejected"):
        validate_index_dir(outside, project_root=temp_project.resolve())


def test_parent_traversal_rejected(temp_project: Path) -> None:
    with pytest.raises(RetrievalError, match="path traversal rejected"):
        validate_index_dir(
            Path("..") / "outside_index",
            project_root=temp_project.resolve(),
        )


@pytest.mark.skipif(os.name != "posix", reason="symlink escape test requires POSIX")
def test_symlink_escape_rejected(temp_project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside_index"
    outside.mkdir()
    link = temp_project / "escaped_index"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RetrievalError, match="path traversal rejected"):
        validate_index_dir(link, project_root=temp_project.resolve())


def _fake_provider_factory(*, model_name: str, api_key: str | None = None):
    return FakeEmbeddingProvider(model_name=model_name, vector_dimension=8)


def _fake_store_factory(*, index_dir: Path, collection_name: str):
    return ChromaVectorStore(index_dir=index_dir, collection_name=collection_name)


def test_cli_rejects_unsafe_index_before_store_creation(
    temp_project: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(temp_project)
    source_files = list((temp_project / "data" / "02_clean_markdown").glob("*.md"))
    assert source_files, "fixture corpus must contain Markdown files"
    source_file = source_files[0]
    source_mtime_before = source_file.stat().st_mtime

    code = build_cli.run_build(
        input_dir=temp_project / "data" / "02_clean_markdown",
        index_dir=temp_project / "data" / "02_clean_markdown" / "bad_index",
        collection_name="customer_claims",
        embedding_model="fake-embedding-model",
        batch_size=16,
        rebuild=True,
        permitted_root=temp_project.resolve(),
        embedding_provider_factory=_fake_provider_factory,
        vector_store_factory=_fake_store_factory,
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err
    assert source_file.stat().st_mtime == source_mtime_before
    assert not (temp_project / "data" / "02_clean_markdown" / "bad_index").exists()
