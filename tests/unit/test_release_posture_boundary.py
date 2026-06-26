"""Boundary repair tests for production release posture activation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from customer_claims_rag import env_bootstrap
from customer_claims_rag.application.settings import ApplicationSettings
from customer_claims_rag.cli import validate_release_posture as validate_cli
from customer_claims_rag.exceptions import ReleasePostureError, VectorStoreError
from customer_claims_rag.release.posture import (
    ReleasePostureDescriptor,
    load_release_posture_descriptor,
    resolve_production_release_posture,
    validate_production_release_posture,
)
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.chroma_storage import CHROMA_SQLITE_FILENAME
from tests.release_posture_helpers import (
    stage_test_production_posture,
    write_test_release_descriptor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_stale_rag_index_dir_does_not_block_active_startup(tmp_path: Path, monkeypatch) -> None:
    index_dir = stage_test_production_posture(tmp_path)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    env_bootstrap.reset_project_env()
    monkeypatch.setenv("RAG_INDEX_DIR", "data/04_index")
    monkeypatch.setenv("RAG_RELEASE_TARGET", "active")

    settings = ApplicationSettings.from_env()
    assert settings.release_target.target_name == "active"
    assert settings.retrieval.index_dir == index_dir


def test_arbitrary_rag_index_dir_does_not_redirect_production(tmp_path: Path, monkeypatch) -> None:
    index_dir = stage_test_production_posture(tmp_path)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    env_bootstrap.reset_project_env()
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "arbitrary-index"))
    monkeypatch.delenv("RAG_RELEASE_TARGET", raising=False)

    settings = ApplicationSettings.from_env()
    assert settings.retrieval.index_dir == index_dir


def test_default_production_target_is_active(tmp_path: Path, monkeypatch) -> None:
    stage_test_production_posture(tmp_path)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.delenv("RAG_RELEASE_TARGET", raising=False)

    context = resolve_production_release_posture(project_root=tmp_path)
    assert context.resolved_target.target_name == "active"


def test_rag_release_target_rollback_selects_rollback(tmp_path: Path, monkeypatch) -> None:
    stage_test_production_posture(tmp_path)
    rollback_dir = tmp_path / "rollback-index"
    rollback_dir.mkdir()
    (rollback_dir / CHROMA_SQLITE_FILENAME).write_bytes(b"rollback-db")

    descriptor_path = tmp_path / "configs" / "release" / "production_posture.json"
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["targets"]["rollback"]["index_path"] = "rollback-index"
    descriptor_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setenv("RAG_RELEASE_TARGET", "rollback")

    context = resolve_production_release_posture(project_root=tmp_path)
    assert context.resolved_target.target_name == "rollback"
    assert context.resolved_target.index_dir == rollback_dir.resolve()


def test_unknown_release_target_fails(tmp_path: Path, monkeypatch) -> None:
    stage_test_production_posture(tmp_path)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setenv("RAG_RELEASE_TARGET", "missing")

    with pytest.raises(ReleasePostureError, match="unknown release target"):
        resolve_production_release_posture(project_root=tmp_path)


def test_validator_and_startup_resolve_same_target(tmp_path: Path, monkeypatch) -> None:
    stage_test_production_posture(tmp_path)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setenv("RAG_INDEX_DIR", "data/04_index")

    startup_context = resolve_production_release_posture(project_root=tmp_path)
    validator_context = resolve_production_release_posture(project_root=tmp_path)
    assert (
        startup_context.resolved_target.index_dir
        == validator_context.resolved_target.index_dir
    )
    assert startup_context.resolved_target.target_name == validator_context.resolved_target.target_name


def test_validator_explicit_active_uses_shared_boundary(tmp_path: Path, monkeypatch) -> None:
    stage_test_production_posture(tmp_path)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    context = resolve_production_release_posture(project_root=tmp_path, target_name="active")
    diagnostics = validate_production_release_posture(context)
    assert diagnostics.selected_target == "active"


def test_validator_default_matches_application_settings(tmp_path: Path, monkeypatch) -> None:
    stage_test_production_posture(tmp_path)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setenv("RAG_INDEX_DIR", "data/04_index")
    settings = ApplicationSettings.from_env()
    context = resolve_production_release_posture(project_root=tmp_path)
    assert settings.release_target == context.resolved_target


def test_nonexistent_chroma_path_creates_no_storage(tmp_path: Path) -> None:
    missing = tmp_path / "missing-index"
    parent = tmp_path
    before = set(parent.iterdir())
    with pytest.raises(VectorStoreError, match="does not exist"):
        ChromaVectorStore(
            index_dir=missing,
            collection_name="customer_claims",
            open_existing=True,
        )
    after = set(parent.iterdir())
    assert before == after
    assert not missing.exists()


def test_empty_directory_creates_no_database(tmp_path: Path) -> None:
    empty = tmp_path / "empty-index"
    empty.mkdir()
    with pytest.raises(VectorStoreError, match=CHROMA_SQLITE_FILENAME):
        ChromaVectorStore(
            index_dir=empty,
            collection_name="customer_claims",
            open_existing=True,
        )
    assert list(empty.iterdir()) == []


def test_missing_chroma_sqlite_fails_before_client(tmp_path: Path) -> None:
    index_dir = tmp_path / "manifest-only"
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(VectorStoreError, match=CHROMA_SQLITE_FILENAME):
        ChromaVectorStore(
            index_dir=index_dir,
            collection_name="customer_claims",
            open_existing=True,
        )


def test_wrong_collection_fails_without_creation(tmp_path: Path) -> None:
    from customer_claims_rag.config import METADATA_SCHEMA_VERSION
    from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
    from customer_claims_rag.retrieval.manifest import build_manifest, write_manifest_atomic
    from tests.retrieval_helpers import make_chunk_record

    provider = FakeEmbeddingProvider(model_name="fake", vector_dimension=8)
    index_dir = tmp_path / "index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    chunk = make_chunk_record(
        chunk_id="01_service_overview::1",
        document_id="01_service_overview",
        content="hello",
        source_path="data/02_clean_markdown/01_service_overview.md",
    )
    store.recreate_collection(embedding_dimension=8)
    store.add_chunks([chunk], provider.embed_documents([chunk.content]))
    write_manifest_atomic(
        index_dir,
        build_manifest(
            collection_name="customer_claims",
            embedding_model="fake",
            corpus_fingerprint="fp",
            chunk_count=1,
            document_count=1,
            metadata_schema_version=METADATA_SCHEMA_VERSION,
            vector_dimension=8,
        ),
    )
    store.close()
    before = set(index_dir.iterdir())
    with pytest.raises(VectorStoreError, match="not found"):
        ChromaVectorStore(
            index_dir=index_dir,
            collection_name="wrong_collection",
            open_existing=True,
        )
    after = set(index_dir.iterdir())
    assert before == after


@pytest.mark.parametrize(
    "index_path",
    [
        "C:/abs/index",
        "/abs/index",
        "\\\\server\\share\\index",
        "../outside",
    ],
)
def test_descriptor_rejects_unsafe_index_paths(tmp_path: Path, index_path: str) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    payload = json.loads(
        (PROJECT_ROOT / "configs" / "release" / "production_posture.json").read_text(
            encoding="utf-8"
        )
    )
    payload["targets"]["active"]["index_path"] = index_path
    descriptor_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ReleasePostureError):
        load_release_posture_descriptor(descriptor_path)


def test_active_and_rollback_same_path_rejected(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative="shared-index",
        corpus_fingerprint="fp",
        chunk_count=1,
        document_count=1,
        supported_document_ids=["01_service_overview"],
    )
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["targets"]["rollback"]["index_path"] = "shared-index"
    descriptor_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    descriptor = ReleasePostureDescriptor.model_validate(payload)
    with pytest.raises(ReleasePostureError, match="same physical index path"):
        resolve_production_release_posture(
            descriptor_path=descriptor_path,
            project_root=tmp_path,
            target_name="active",
        )


def test_unknown_descriptor_fields_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        (PROJECT_ROOT / "configs" / "release" / "production_posture.json").read_text(
            encoding="utf-8"
        )
    )
    payload["targets"]["active"]["index_pathh"] = "data/04_index"
    with pytest.raises(Exception):
        ReleasePostureDescriptor.model_validate(payload)


def test_cli_validator_ignores_stale_rag_index_dir(tmp_path: Path, monkeypatch) -> None:
    stage_test_production_posture(tmp_path)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setenv("RAG_INDEX_DIR", "data/04_index")
    assert (
        validate_cli.main(
            [
                "--descriptor",
                str(tmp_path / "configs" / "release" / "production_posture.json"),
                "--skip-vector-store",
            ]
        )
        == 0
    )
