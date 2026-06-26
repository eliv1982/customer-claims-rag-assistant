"""Unit tests for production release posture loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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
from customer_claims_rag.config import METADATA_SCHEMA_VERSION
from customer_claims_rag.evaluation.pool_expansion_metrics import compute_pool_expansion_config_hash
from customer_claims_rag.exceptions import ReleasePostureError
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.manifest import build_manifest, write_manifest_atomic
from customer_claims_rag.retrieval.models import IndexManifest
from tests.release_posture_helpers import (
    PRODUCTION_FROZEN_CONFIG_HASH,
    resolved_release_target_for_index,
    stage_test_production_posture,
    write_test_release_descriptor,
)
from tests.retrieval_helpers import make_chunk_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
ACTIVE_FINGERPRINT = (
    "bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3"
)
ROLLBACK_FINGERPRINT = (
    "b9526128dad23e71e812fbd8f26452b8d6efa29e4faac898fa11f279b310e827"
)
ACTIVE_DOCUMENT_IDS = (
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
)


def _write_index_fixture(
    tmp_path: Path,
    *,
    corpus_fingerprint: str,
    chunk_count: int,
    document_ids: tuple[str, ...],
    collection_name: str = "customer_claims",
    embedding_model: str = "fake-embedding-model",
    vector_dimension: int = 8,
) -> Path:
    provider = FakeEmbeddingProvider(
        model_name=embedding_model,
        vector_dimension=vector_dimension,
    )
    index_dir = tmp_path / "index"
    chunks = [
        make_chunk_record(
            chunk_id=f"{document_id}::1",
            document_id=document_id,
            content=f"content for {document_id}",
            source_path=f"data/02_clean_markdown/{document_id}.md",
        )
        for document_id in document_ids
    ][:chunk_count]
    store = ChromaVectorStore(index_dir=index_dir, collection_name=collection_name)
    vectors = provider.embed_documents([chunk.content for chunk in chunks])
    store.recreate_collection(embedding_dimension=vector_dimension)
    store.add_chunks(chunks, vectors)
    write_manifest_atomic(
        index_dir,
        build_manifest(
            collection_name=collection_name,
            embedding_model=embedding_model,
            corpus_fingerprint=corpus_fingerprint,
            chunk_count=len(chunks),
            document_count=len(document_ids),
            metadata_schema_version=METADATA_SCHEMA_VERSION,
            vector_dimension=vector_dimension,
        ),
    )
    store.close()
    return index_dir


def test_committed_descriptor_parses() -> None:
    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    assert descriptor.schema_version == "1.0.0"
    assert descriptor.release_posture_id == "foodflow-10doc-release-v1"
    assert descriptor.default_target == "active"


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "descriptor.json"
    payload = json.loads(DEFAULT_DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = "9.9.9"
    path.write_text(json.dumps(payload), encoding="utf-8")
    descriptor = load_release_posture_descriptor(path)
    with pytest.raises(ReleasePostureError, match="unsupported release posture schema_version"):
        resolve_release_target_name(descriptor)


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


def test_active_target_identity_matches_posture_a() -> None:
    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    target = descriptor.targets["active"]
    assert target.expected_corpus_fingerprint == ACTIVE_FINGERPRINT
    assert target.expected_chunk_count == 215
    assert target.expected_document_count == 10
    assert tuple(target.supported_document_ids) == ACTIVE_DOCUMENT_IDS
    assert target.collection_name == "customer_claims"
    assert target.embedding_model == "text-embedding-3-small"
    assert target.vector_dimension == 1536
    assert target.status == "selected_production_release"


def test_rollback_target_is_emergency_archive_only() -> None:
    descriptor = load_release_posture_descriptor(DEFAULT_DESCRIPTOR_PATH)
    rollback = descriptor.targets["rollback"]
    assert rollback.status == "emergency_rollback_archive_only"
    assert rollback.index_path == "data/04_index"
    assert rollback.expected_corpus_fingerprint == ROLLBACK_FINGERPRINT
    assert rollback.expected_chunk_count == 333
    assert rollback.expected_document_count == 15


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
    ],
)
def test_manifest_mismatch_rejected(
    tmp_path: Path,
    field_name: str,
    field_value,
    match: str,
) -> None:
    index_dir = _write_index_fixture(
        tmp_path,
        corpus_fingerprint="fp",
        chunk_count=1,
        document_ids=("01_service_overview",),
    )
    manifest = IndexManifest.model_validate(
        json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    manifest_data = manifest.model_dump(mode="json")
    manifest_data[field_name] = field_value
    (index_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

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
    with pytest.raises(ReleasePostureError, match=match):
        validate_release_posture_for_production(
            descriptor,
            resolved,
            project_root=tmp_path,
            frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
        )


def test_frozen_config_hash_mismatch_rejected(tmp_path: Path) -> None:
    index_dir = _write_index_fixture(
        tmp_path,
        corpus_fingerprint="fp",
        chunk_count=1,
        document_ids=("01_service_overview",),
    )
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
    payload["expected_frozen_config_hash"] = "0" * 64
    descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    descriptor = load_release_posture_descriptor(descriptor_path)
    resolved = resolve_release_target(descriptor, "active", project_root=tmp_path)
    with pytest.raises(ReleasePostureError, match="frozen retrieval config hash mismatch"):
        validate_release_posture_for_production(
            descriptor,
            resolved,
            project_root=tmp_path,
            frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
        )


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
    index_dir = _write_index_fixture(
        tmp_path,
        corpus_fingerprint="diag-fingerprint",
        chunk_count=1,
        document_ids=("01_service_overview",),
    )
    descriptor_path = tmp_path / "descriptor.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative="index",
        corpus_fingerprint="diag-fingerprint",
        chunk_count=1,
        document_count=1,
        supported_document_ids=["01_service_overview"],
    )
    descriptor = load_release_posture_descriptor(descriptor_path)
    resolved = resolve_release_target(descriptor, "active", project_root=tmp_path)
    store = ChromaVectorStore(
        index_dir=index_dir,
        collection_name="customer_claims",
        open_existing=True,
    )
    diagnostics = validate_release_posture_for_production(
        descriptor,
        resolved,
        project_root=tmp_path,
        vector_store=store,
        frozen_retrieval_config_path=FROZEN_CONFIG_PATH,
    )
    store.close()
    rendered = "\n".join(diagnostics.format_lines())
    assert "diag-fingerprint" in rendered
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
