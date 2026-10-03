"""Tests for frozen embedding snapshot serialization and validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from customer_claims_rag.env_bootstrap import project_root
from customer_claims_rag.exceptions import IndexBuildError
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.embedding_snapshot import (
    SNAPSHOT_FORMAT_VERSION,
    SnapshotValidationError,
    canonical_snapshot_bytes,
    compute_snapshot_digest,
    decode_chunk_id_blob,
    encode_chunk_id_blob,
    load_embedding_snapshot_arrays,
    load_snapshot_manifest,
    resolve_snapshot_embeddings,
    validate_snapshot_against_chunks,
    write_embedding_snapshot,
)
from customer_claims_rag.retrieval.experiment_embedding_cache import experiment_cache_path
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint
from customer_claims_rag.retrieval.index_builder import IndexBuilder
from customer_claims_rag.retrieval.index_identity import compute_chunk_payload_digest
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.snapshot_embedding_provider import SnapshotEmbeddingProvider
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.token_counter import TiktokenCounter
from tests.retrieval_helpers import make_chunk_record

ROOT = project_root()


def _sample_chunks():
    return [
        make_chunk_record(chunk_id="doc::1", content="threat alpha"),
        make_chunk_record(chunk_id="doc::2", content="support operator"),
    ]


def _write_snapshot(tmp_path: Path, chunks=None):
    chunks = chunks or _sample_chunks()
    provider = FakeEmbeddingProvider(vector_dimension=8)
    embeddings = provider.embed_documents([chunk.content for chunk in chunks])
    ordered = sorted(chunks, key=lambda item: item.chunk_id)
    npz_path = tmp_path / "snapshot.npz"
    manifest_path = tmp_path / "snapshot.manifest.json"
    manifest = write_embedding_snapshot(
        npz_path=npz_path,
        manifest_path=manifest_path,
        chunks=ordered,
        embeddings=embeddings,
        embedding_model=provider.model_name,
        corpus_fingerprint=compute_corpus_fingerprint(ordered, embedding_model=provider.model_name),
        chunk_payload_digest=compute_chunk_payload_digest(ordered),
        candidate_experiment_id="test-exp",
        creation_source_commit="deadbeef",
        project_root=ROOT,
    )
    return npz_path, manifest_path, manifest, ordered, embeddings


def test_chunk_id_blob_round_trip() -> None:
    ids = ["a::1", "b::2", "z::last"]
    assert decode_chunk_id_blob(encode_chunk_id_blob(ids)) == ids


def test_snapshot_serialize_load_round_trip(tmp_path: Path) -> None:
    npz_path, manifest_path, manifest, chunks, _ = _write_snapshot(tmp_path)
    loaded_ids, matrix = load_embedding_snapshot_arrays(npz_path)
    assert loaded_ids == [chunk.chunk_id for chunk in sorted(chunks, key=lambda c: c.chunk_id)]
    assert matrix.dtype == np.dtype("<f4")
    assert matrix.shape == (2, 8)
    loaded_manifest = load_snapshot_manifest(manifest_path)
    assert loaded_manifest["snapshot_digest"] == manifest["snapshot_digest"]


def test_snapshot_bytes_deterministic_across_runs(tmp_path: Path) -> None:
    chunks = _sample_chunks()
    provider = FakeEmbeddingProvider(vector_dimension=8)
    embeddings = provider.embed_documents([chunk.content for chunk in chunks])
    ordered = sorted(chunks, key=lambda item: item.chunk_id)
    matrix = np.asarray(embeddings, dtype=np.dtype("<f4"))
    first = canonical_snapshot_bytes(
        chunk_ids=[chunk.chunk_id for chunk in ordered],
        embeddings=matrix,
        format_version=SNAPSHOT_FORMAT_VERSION,
        embedding_model=provider.model_name,
        dimension=8,
    )
    second = canonical_snapshot_bytes(
        chunk_ids=[chunk.chunk_id for chunk in ordered],
        embeddings=matrix,
        format_version=SNAPSHOT_FORMAT_VERSION,
        embedding_model=provider.model_name,
        dimension=8,
    )
    assert first == second


def test_snapshot_uses_no_pickle(tmp_path: Path) -> None:
    npz_path, _, _, _, _ = _write_snapshot(tmp_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        assert "embeddings" in archive.files
        assert "chunk_id_blob" in archive.files


def test_content_mutation_invalidates_snapshot(tmp_path: Path) -> None:
    npz_path, manifest_path, _, chunks, _ = _write_snapshot(tmp_path)
    mutated = [make_chunk_record(chunk_id="doc::1", content="threat beta"), chunks[1]]
    with pytest.raises(SnapshotValidationError, match="chunk_payload_digest"):
        validate_snapshot_against_chunks(
            manifest=load_snapshot_manifest(manifest_path),
            npz_path=npz_path,
            chunks=mutated,
            embedding_model="fake-embedding-model",
        )


def test_wrong_corpus_fingerprint_rejected(tmp_path: Path) -> None:
    npz_path, manifest_path, manifest, chunks, _ = _write_snapshot(tmp_path)
    manifest = dict(manifest)
    manifest["corpus_fingerprint"] = "0" * 64
    with pytest.raises(SnapshotValidationError, match="corpus_fingerprint"):
        validate_snapshot_against_chunks(
            manifest=manifest,
            npz_path=npz_path,
            chunks=chunks,
            embedding_model="fake-embedding-model",
        )


def test_missing_chunk_rejected(tmp_path: Path) -> None:
    npz_path, manifest_path, _, chunks, _ = _write_snapshot(tmp_path)
    with pytest.raises(SnapshotValidationError, match="chunk ID set mismatch"):
        validate_snapshot_against_chunks(
            manifest=load_snapshot_manifest(manifest_path),
            npz_path=npz_path,
            chunks=chunks[:1],
            embedding_model="fake-embedding-model",
        )


def test_extra_chunk_rejected(tmp_path: Path) -> None:
    npz_path, manifest_path, _, chunks, _ = _write_snapshot(tmp_path)
    extra = chunks + [make_chunk_record(chunk_id="doc::3", content="extra")]
    with pytest.raises(SnapshotValidationError, match="chunk ID set mismatch"):
        validate_snapshot_against_chunks(
            manifest=load_snapshot_manifest(manifest_path),
            npz_path=npz_path,
            chunks=extra,
            embedding_model="fake-embedding-model",
        )


def test_wrong_model_rejected(tmp_path: Path) -> None:
    npz_path, manifest_path, _, chunks, _ = _write_snapshot(tmp_path)
    with pytest.raises(SnapshotValidationError, match="embedding model mismatch"):
        validate_snapshot_against_chunks(
            manifest=load_snapshot_manifest(manifest_path),
            npz_path=npz_path,
            chunks=chunks,
            embedding_model="other-model",
        )


def test_wrong_dimension_rejected(tmp_path: Path) -> None:
    npz_path, manifest_path, manifest, chunks, _ = _write_snapshot(tmp_path)
    manifest = dict(manifest)
    manifest["embedding_dimension"] = 16
    with pytest.raises(SnapshotValidationError, match="dimension mismatch"):
        validate_snapshot_against_chunks(
            manifest=manifest,
            npz_path=npz_path,
            chunks=chunks,
            embedding_model="fake-embedding-model",
        )


def test_digest_mismatch_rejected(tmp_path: Path) -> None:
    npz_path, manifest_path, manifest, chunks, _ = _write_snapshot(tmp_path)
    manifest = dict(manifest)
    manifest["embedding_digest"] = "0" * 64
    with pytest.raises(SnapshotValidationError, match="embedding_digest mismatch"):
        validate_snapshot_against_chunks(
            manifest=manifest,
            npz_path=npz_path,
            chunks=chunks,
            embedding_model="fake-embedding-model",
        )


def test_snapshot_build_performs_no_document_api_calls(tmp_path: Path) -> None:
    chunks = _sample_chunks()
    npz_path, manifest_path, _, _, _ = _write_snapshot(tmp_path, chunks=chunks)
    provider = FakeEmbeddingProvider(vector_dimension=8)
    snapshot_provider = SnapshotEmbeddingProvider(query_provider=provider)
    with pytest.raises(IndexBuildError, match="disabled in snapshot build mode"):
        snapshot_provider.embed_documents(["must not call"])

    index_dir = tmp_path / "index"
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
    IndexBuilder(
        corpus_builder=CorpusBuilder(permitted_root=ROOT, token_counter=TiktokenCounter()),
        embedding_provider=snapshot_provider,
        vector_store=store,
        index_dir=index_dir,
        batch_size=2,
    ).build_from_chunks(
        documents=[],
        chunks=chunks,
        rebuild=True,
        embedding_snapshot_path=npz_path,
        embedding_snapshot_manifest=manifest_path,
    )
    store.close()


def test_three_temp_roots_same_collection_digest_from_snapshot(tmp_path: Path) -> None:
    chunks = _sample_chunks()
    npz_path, manifest_path, _, _, _ = _write_snapshot(tmp_path, chunks=chunks)
    provider = FakeEmbeddingProvider(vector_dimension=8)
    snapshot_provider = SnapshotEmbeddingProvider(query_provider=provider)
    digests: list[str] = []
    for label in ("A", "B", "C"):
        index_dir = tmp_path / f"BUILD_{label}"
        store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims")
        IndexBuilder(
            corpus_builder=CorpusBuilder(permitted_root=ROOT, token_counter=TiktokenCounter()),
            embedding_provider=snapshot_provider,
            vector_store=store,
            index_dir=index_dir,
            batch_size=2,
        ).build_from_chunks(
            documents=[],
            chunks=chunks,
            rebuild=True,
            embedding_snapshot_path=npz_path,
            embedding_snapshot_manifest=manifest_path,
        )
        manifest = load_manifest(index_dir)
        digests.append(manifest.collection_content_digest or "")
        store.close()
    assert len({digest for digest in digests if digest}) == 1


def test_manifest_paths_are_repo_relative(tmp_path: Path) -> None:
    project = tmp_path / "project"
    snapshot_dir = project / ".tmp" / "snapshot_manifest"
    snapshot_dir.mkdir(parents=True)
    npz_path = snapshot_dir / "snapshot.npz"
    manifest_path = snapshot_dir / "snapshot.manifest.json"
    chunks = _sample_chunks()
    provider = FakeEmbeddingProvider(vector_dimension=8)
    embeddings = provider.embed_documents([chunk.content for chunk in chunks])
    ordered = sorted(chunks, key=lambda item: item.chunk_id)
    manifest = write_embedding_snapshot(
        npz_path=npz_path,
        manifest_path=manifest_path,
        chunks=ordered,
        embeddings=embeddings,
        embedding_model=provider.model_name,
        corpus_fingerprint=compute_corpus_fingerprint(ordered, embedding_model=provider.model_name),
        chunk_payload_digest=compute_chunk_payload_digest(ordered),
        candidate_experiment_id="test-exp",
        creation_source_commit="deadbeef",
        project_root=project,
    )
    assert not manifest["snapshot_path"].startswith("C:")
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["snapshot_path"].startswith("data/") or manifest["snapshot_path"].startswith(".tmp/")
