"""Unit tests for production release posture loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from customer_claims_rag.release.posture import (
    DEFAULT_DESCRIPTOR_PATH,
    ReleasePostureDescriptor,
    load_release_posture_descriptor,
    resolve_production_release_posture,
    resolve_release_target,
    resolve_release_target_name,
    validate_frozen_retrieval_contract,
    validate_release_posture_for_production,
)
from customer_claims_rag.application.settings import load_frozen_retrieval_config
from customer_claims_rag.evaluation.pool_expansion_metrics import compute_pool_expansion_config_hash
from customer_claims_rag.exceptions import ReleasePostureError
from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    load_canonical_corpus_manifest,
)
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.models import IndexManifest
from tests.release_posture_helpers import (
    PRODUCTION_FROZEN_CONFIG_HASH,
    resolved_release_target_for_index,
    stage_consistent_release,
    stage_test_production_posture,
    write_test_release_descriptor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"


def _validate(staged, *, vector_store=None):
    descriptor = load_release_posture_descriptor(staged.descriptor_path)
    resolved = resolve_release_target(descriptor, "active", project_root=staged.root)
    return validate_release_posture_for_production(
        descriptor,
        resolved,
        project_root=staged.root,
        vector_store=vector_store,
        frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
    )


def test_committed_descriptor_parses() -> None:
    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    assert descriptor.schema_version == "2.0.0"
    assert descriptor.release_posture_id == "foodflow-10doc-release-v2"
    assert descriptor.default_target == "active"


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "descriptor.json"
    payload = json.loads(DEFAULT_DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = "9.9.9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    descriptor = load_release_posture_descriptor(path)
    with pytest.raises(ReleasePostureError, match="unsupported release posture schema_version"):
        resolve_release_target_name(descriptor)


def test_previous_descriptor_schema_is_not_accepted(tmp_path: Path) -> None:
    """Schema 1.0.0 carried its own corpus pins; a descriptor that still does is refused."""
    path = tmp_path / "descriptor.json"
    payload = json.loads(DEFAULT_DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.0.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    descriptor = load_release_posture_descriptor(path)
    with pytest.raises(ReleasePostureError, match="unsupported release posture schema_version"):
        resolve_release_target_name(descriptor)
    payload["targets"]["active"]["expected_corpus_fingerprint"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleasePostureError, match="failed validation"):
        load_release_posture_descriptor(path)


def test_path_traversal_rejected(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative="../outside",
        corpus_fingerprint="fp",
        chunk_count=1,
        document_count=1,
        supported_document_ids=["01_service_overview"],
    )
    with pytest.raises(ReleasePostureError, match="parent-directory traversal"):
        load_release_posture_descriptor(descriptor_path)


def test_default_target_is_active() -> None:
    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    assert resolve_release_target_name(descriptor) == "active"


def test_active_target_identity_comes_from_the_canonical_corpus_manifest() -> None:
    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    target = descriptor.targets["active"]
    assert target.corpus_manifest == DEFAULT_CORPUS_MANIFEST_RELATIVE.as_posix()
    assert not hasattr(target, "expected_corpus_fingerprint")
    assert not hasattr(target, "supported_document_ids")

    resolved = resolve_release_target(descriptor, "active", project_root=PROJECT_ROOT)
    corpus = load_canonical_corpus_manifest(PROJECT_ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    assert resolved.corpus_id == corpus.corpus_id
    assert resolved.supported_document_ids == corpus.document_ids
    assert resolved.expected_document_count == len(corpus.documents) == 10
    assert resolved.expected_chunk_count == corpus.expected.chunk_count
    assert resolved.expected_corpus_fingerprint == corpus.expected.corpus_fingerprint
    assert resolved.expected_chunk_payload_digest == corpus.expected.chunk_payload_digest
    assert resolved.embedding_model == corpus.expected.embedding_model == "text-embedding-3-small"
    assert resolved.collection_name == "customer_claims"
    assert resolved.vector_dimension == 1536
    assert resolved.status == "selected_production_release"
    assert resolved.index_path_relative == "data/04_index_production"


def test_committed_posture_has_one_reproducible_target_and_no_archive() -> None:
    """No rollback copy, backup directory or archive stands behind the production posture."""
    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    assert set(descriptor.targets) == {"active"}
    for name, target in descriptor.targets.items():
        assert target.corpus_manifest == DEFAULT_CORPUS_MANIFEST_RELATIVE.as_posix(), name
        for text in (target.index_path, target.status):
            assert "backup" not in text and "archive" not in text and "rollback" not in text, name
    raw = DEFAULT_DESCRIPTOR_PATH.read_text(encoding="utf-8")
    for retired in ("10docs_215chunks", "emergency_rollback", "bf3df0d4", "b9526128"):
        assert retired not in raw


def test_unknown_target_rejected() -> None:
    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    with pytest.raises(ReleasePostureError, match="unknown release target"):
        resolve_release_target(descriptor, "missing", project_root=PROJECT_ROOT)


def test_missing_active_index_rejected(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative="missing-index",
        corpus_fingerprint="fp",
        chunk_count=1,
        document_count=1,
        supported_document_ids=["01_service_overview"],
    )
    descriptor = load_release_posture_descriptor(descriptor_path)
    resolved = resolve_release_target(descriptor, "active", project_root=tmp_path)
    with pytest.raises(ReleasePostureError, match="does not exist"):
        validate_release_posture_for_production(
            descriptor,
            resolved,
            project_root=tmp_path,
            frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
        )


def test_missing_manifest_rejected(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    descriptor_path = tmp_path / "descriptor.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative="index",
        corpus_fingerprint="fp",
        chunk_count=1,
        document_count=1,
        supported_document_ids=["01_service_overview"],
    )
    descriptor = load_release_posture_descriptor(descriptor_path)
    resolved = resolve_release_target(descriptor, "active", project_root=tmp_path)
    with pytest.raises(ReleasePostureError):
        validate_release_posture_for_production(
            descriptor,
            resolved,
            project_root=tmp_path,
            frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value", "match"),
    [
        ("corpus_fingerprint", "wrong-fingerprint", "corpus fingerprint mismatch"),
        ("chunk_count", 99, "chunk count mismatch"),
        ("document_count", 99, "document count mismatch"),
        ("collection_name", "wrong-collection", "collection name mismatch"),
        ("embedding_model", "wrong-model", "embedding model mismatch"),
        ("vector_dimension", 42, "vector dimension mismatch"),
        ("index_format_version", "0.0.1", "index format version mismatch"),
        ("metadata_schema_version", "0.0.1", "metadata schema version mismatch"),
        ("chunk_payload_digest", "0" * 64, "chunk payload digest mismatch"),
    ],
)
def test_manifest_mismatch_rejected(
    tmp_path: Path,
    field_name: str,
    field_value,
    match: str,
) -> None:
    staged = stage_consistent_release(tmp_path)
    manifest_path = staged.index_dir / "manifest.json"
    manifest_data = IndexManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    ).model_dump(mode="json")
    manifest_data[field_name] = field_value
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    with pytest.raises(ReleasePostureError, match=match):
        _validate(staged)


def test_consistent_release_validates_with_and_without_the_store(tmp_path: Path) -> None:
    staged = stage_consistent_release(tmp_path)
    static = _validate(staged)
    assert static.integrity == "manifest_self_attestation_only"
    store = ChromaVectorStore(
        index_dir=staged.index_dir,
        collection_name="customer_claims",
        open_existing=True,
    )
    try:
        recomputed = _validate(staged, vector_store=store)
    finally:
        store.close()
    assert recomputed.integrity == "store_recomputed"
    assert recomputed.corpus_fingerprint == staged.corpus_fingerprint
    assert recomputed.chunk_payload_digest == staged.chunk_payload_digest
    assert recomputed.corpus_id == "test-corpus"


def test_frozen_config_hash_mismatch_rejected(tmp_path: Path) -> None:
    staged = stage_consistent_release(tmp_path)
    payload = json.loads(staged.descriptor_path.read_text(encoding="utf-8"))
    payload["expected_frozen_config_hash"] = "0" * 64
    staged.descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleasePostureError, match="frozen retrieval config hash mismatch"):
        _validate(staged)


def test_retrieval_contract_mismatch_rejected() -> None:
    frozen = load_frozen_retrieval_config(FROZEN_CONFIG_PATH)
    with pytest.raises(ReleasePostureError, match="vector_fetch_k"):
        validate_frozen_retrieval_contract(
            frozen.model_copy(update={"vector_fetch_k": 12}),
            expected_hash=PRODUCTION_FROZEN_CONFIG_HASH,
            actual_hash=PRODUCTION_FROZEN_CONFIG_HASH,
        )


def test_production_factory_uses_release_target_not_raw_index_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from customer_claims_rag import env_bootstrap
    from customer_claims_rag.application.settings import ApplicationSettings

    release_index = tmp_path / "release-index"
    release_index.mkdir()
    legacy_index = tmp_path / "legacy-index"
    legacy_index.mkdir()
    monkeypatch.setenv("RAG_INDEX_DIR", str(legacy_index))
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: tmp_path)

    release_target = resolved_release_target_for_index(
        release_index,
        project_root=tmp_path,
        corpus_fingerprint="fp",
        chunk_count=1,
        document_count=1,
        supported_document_ids=("01_service_overview",),
    )
    with patch(
        "customer_claims_rag.release.posture.resolve_production_release_posture",
        return_value=MagicMock(
            descriptor=MagicMock(),
            resolved_target=release_target,
            descriptor_path=DEFAULT_DESCRIPTOR_PATH,
            project_root=tmp_path,
            frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
        ),
    ):
        settings = ApplicationSettings.from_env()
    assert settings.retrieval.index_dir == release_index
    assert settings.retrieval.index_dir != legacy_index


def test_stale_rag_index_dir_does_not_affect_descriptor_resolution(tmp_path: Path, monkeypatch) -> None:
    index_dir = stage_test_production_posture(tmp_path)
    monkeypatch.setattr(
        __import__("customer_claims_rag", fromlist=["env_bootstrap"]).env_bootstrap,
        "project_root",
        lambda: tmp_path,
    )
    monkeypatch.setenv("RAG_INDEX_DIR", "data/04_index")
    context = resolve_production_release_posture(project_root=tmp_path)
    assert context.resolved_target.index_dir == index_dir


def test_missing_collection_open_existing_does_not_create_collection(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    with pytest.raises(Exception, match="not found|collection"):
        ChromaVectorStore(
            index_dir=index_dir,
            collection_name="customer_claims",
            open_existing=True,
        )
    client = __import__("chromadb").PersistentClient(path=str(index_dir))
    assert client.list_collections() == []


def test_startup_diagnostics_expose_exact_fingerprints(tmp_path: Path) -> None:
    staged = stage_consistent_release(tmp_path)
    store = ChromaVectorStore(
        index_dir=staged.index_dir,
        collection_name="customer_claims",
        open_existing=True,
    )
    try:
        diagnostics = _validate(staged, vector_store=store)
    finally:
        store.close()
    rendered = "\n".join(diagnostics.format_lines())
    assert f"corpus_fingerprint={staged.corpus_fingerprint}" in rendered
    assert f"chunk_payload_digest={staged.chunk_payload_digest}" in rendered
    assert "canonical_corpus_id=test-corpus" in rendered
    assert "index_integrity=store_recomputed" in rendered
    assert PRODUCTION_FROZEN_CONFIG_HASH in rendered
    assert "retrieval_contract=24/24/12/0.0" in rendered
    assert "reranker_id=source-authority-v1" in rendered


def test_committed_descriptor_frozen_hash_matches_safety_oracle() -> None:
    from customer_claims_rag.evaluation.pool_expansion_metrics import load_pool_expansion_config

    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    config = load_pool_expansion_config(FROZEN_CONFIG_PATH)
    assert compute_pool_expansion_config_hash(config) == descriptor.expected_frozen_config_hash


def test_generic_build_index_still_creates_collection(tmp_path: Path) -> None:
    index_dir = tmp_path / "build-index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="build_test")
    assert store.count() == 0
    store.close()
    reopened = ChromaVectorStore(index_dir=index_dir, collection_name="build_test")
    assert reopened.collection_name == "build_test"
    reopened.close()


def test_load_production_release_context_default_target(tmp_path: Path, monkeypatch) -> None:
    from customer_claims_rag import env_bootstrap

    monkeypatch.delenv("RAG_RELEASE_TARGET", raising=False)
    monkeypatch.delenv("RAG_INDEX_DIR", raising=False)
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: PROJECT_ROOT)
    context = resolve_production_release_posture(project_root=PROJECT_ROOT)
    assert context.resolved_target.target_name == context.descriptor.default_target == "active"


def test_descriptor_model_must_match_the_model_the_corpus_is_fingerprinted_for(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "descriptor.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative="index",
        corpus_fingerprint="fp",
        chunk_count=1,
        document_count=1,
        supported_document_ids=["01_service_overview"],
    )
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    payload["targets"]["active"]["embedding_model"] = "some-other-model"
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    descriptor = ReleasePostureDescriptor.model_validate(payload)
    with pytest.raises(ReleasePostureError, match="does not match the model"):
        resolve_release_target(descriptor, "active", project_root=tmp_path)


@pytest.mark.parametrize("damage", ["missing", "not-json", "unknown-field"])
def test_unusable_canonical_corpus_manifest_fails_resolution(tmp_path: Path, damage: str) -> None:
    staged = stage_consistent_release(tmp_path)
    if damage == "missing":
        staged.corpus_manifest_path.unlink()
    elif damage == "not-json":
        staged.corpus_manifest_path.write_text("{", encoding="utf-8")
    else:
        payload = json.loads(staged.corpus_manifest_path.read_text(encoding="utf-8"))
        payload["surprise"] = 1
        staged.corpus_manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleasePostureError, match="canonical corpus manifest is unusable"):
        resolve_production_release_posture(
            descriptor_path=staged.descriptor_path,
            project_root=staged.root,
        )
