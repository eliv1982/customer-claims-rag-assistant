"""Shared helpers for release posture tests.

Two kinds of fixtures live here:

* ``stage_consistent_release`` builds a small but *genuine* release: a Chroma index whose manifest
  and stored records agree, a canonical corpus manifest holding that index's true identity, and a
  descriptor pointing at both. The release validator recomputes identity from the stored records,
  so a test that wants it to pass needs data that really hashes to what the manifests say.
* ``write_test_release_descriptor`` / ``resolved_release_target_for_index`` build descriptors and
  resolved targets from *labels* (any string stands for a fingerprint) for tests that mock the
  validator or only exercise resolution; labels are hashed so they satisfy the digest format.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from customer_claims_rag.config import METADATA_SCHEMA_VERSION
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.release.posture import ResolvedReleaseTarget
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.fake_embeddings import FakeEmbeddingProvider
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint, sort_chunks_deterministic
from customer_claims_rag.retrieval.index_identity import (
    compute_chunk_payload_digest,
    compute_collection_content_digest,
    compute_embedding_digest,
)
from customer_claims_rag.retrieval.manifest import build_manifest, write_manifest_atomic
from tests.retrieval_helpers import make_chunk_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_FROZEN_CONFIG_HASH = (
    "ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048"
)
TEST_CORPUS_MANIFEST_RELATIVE = "configs/corpus/test_corpus.json"
TEST_CORPUS_ID = "test-corpus"
FAKE_MODEL = "fake-embedding-model"
FAKE_DIMENSION = 8


def hex64(label: str) -> str:
    """Stand-in digest for a label: tests name fingerprints, the schema wants 64 hex digits."""
    if len(label) == 64 and all(char in "0123456789abcdef" for char in label):
        return label
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def project_root_of_descriptor(descriptor_path: Path) -> Path:
    """``<root>/configs/release/x.json`` -> root; any other location -> its own directory."""
    if descriptor_path.parent.name == "release" and descriptor_path.parent.parent.name == "configs":
        return descriptor_path.parents[2]
    return descriptor_path.parent


def write_test_corpus_manifest(
    project_root: Path,
    *,
    document_ids: list[str] | tuple[str, ...],
    chunk_count: int,
    corpus_fingerprint: str,
    chunk_payload_digest: str | None = None,
    embedding_model: str = FAKE_MODEL,
) -> Path:
    ids = sorted(document_ids)
    source_dir = project_root / "data" / "02_clean_markdown"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    for doc in ids:
        content = f"# {doc}\n\nCanonical source used by the release-readiness test.\n"
        (source_dir / f"{doc}.md").write_text(content, encoding="utf-8")
        source_hashes[doc] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    payload = {
        "schema_version": "1.0.0",
        "corpus_id": TEST_CORPUS_ID,
        "description": "test corpus",
        "source_dir": "data/02_clean_markdown",
        "tokenizer_encoding": "cl100k_base",
        "documents": [
            {
                "document_id": doc,
                "file": f"{doc}.md",
                "source_sha256": source_hashes[doc],
            }
            for doc in ids
        ],
        "excluded_documents": [],
        "expected": {
            "document_count": len(ids),
            "chunk_count": max(chunk_count, len(ids)),
            "chunk_payload_digest": hex64(chunk_payload_digest or f"payload:{corpus_fingerprint}"),
            "embedding_model": embedding_model,
            "corpus_fingerprint": hex64(corpus_fingerprint),
        },
    }
    path = project_root / TEST_CORPUS_MANIFEST_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_test_release_descriptor(
    path: Path,
    *,
    index_path_relative: str,
    corpus_fingerprint: str,
    chunk_count: int,
    document_count: int,
    supported_document_ids: list[str],
    target_name: str = "active",
    status: str = "selected_production_release",
    expected_frozen_config_hash: str = PRODUCTION_FROZEN_CONFIG_HASH,
    chunk_payload_digest: str | None = None,
    embedding_model: str = FAKE_MODEL,
) -> None:
    """Write a v2 descriptor and the canonical corpus manifest it references."""
    _ = document_count  # derived from the corpus manifest, kept for call-site compatibility
    write_test_corpus_manifest(
        project_root_of_descriptor(path),
        document_ids=supported_document_ids,
        chunk_count=chunk_count,
        corpus_fingerprint=corpus_fingerprint,
        chunk_payload_digest=chunk_payload_digest,
        embedding_model=embedding_model,
    )
    payload = {
        "schema_version": "2.0.0",
        "release_posture_id": "test-release-posture",
        "default_target": target_name,
        "limitations_doc": "docs/06_release_posture.md",
        "frozen_retrieval_config_path": "configs/retrieval/vector_pool_expansion_v1.json",
        "expected_frozen_config_hash": expected_frozen_config_hash,
        "targets": {
            target_name: {
                "status": status,
                "corpus_manifest": TEST_CORPUS_MANIFEST_RELATIVE,
                "index_path": index_path_relative,
                "collection_name": "customer_claims",
                "embedding_model": embedding_model,
                "vector_dimension": FAKE_DIMENSION,
            },
            "secondary": {
                "status": "test_secondary_target",
                "corpus_manifest": TEST_CORPUS_MANIFEST_RELATIVE,
                "index_path": "data/04_index_secondary",
                "collection_name": "customer_claims",
                "embedding_model": embedding_model,
                "vector_dimension": FAKE_DIMENSION,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stage_project_configs(tmp_path: Path) -> tuple[Path, Path]:
    """Copy committed config trees into a temporary project root."""
    retrieval_dst = tmp_path / "configs" / "retrieval"
    reranker_dst = tmp_path / "configs" / "reranking"
    if not retrieval_dst.is_dir():
        shutil.copytree(PROJECT_ROOT / "configs" / "retrieval", retrieval_dst)
    if not reranker_dst.is_dir():
        shutil.copytree(PROJECT_ROOT / "configs" / "reranking", reranker_dst)
    return (
        retrieval_dst / "vector_pool_expansion_v1.json",
        reranker_dst / "source_authority_v1.json",
    )


@dataclass(frozen=True)
class StagedRelease:
    """A genuine small release under ``root`` and the identity of what was stored."""

    root: Path
    index_dir: Path
    descriptor_path: Path
    corpus_manifest_path: Path
    chunks: list[ChunkRecord]
    corpus_fingerprint: str
    chunk_payload_digest: str
    embedding_model: str
    vector_dimension: int


def make_test_chunks(
    document_ids: tuple[str, ...] = ("01_service_overview",),
    chunks_per_document: int = 1,
) -> list[ChunkRecord]:
    return [
        make_chunk_record(
            chunk_id=f"{document_id}::chunk-{index:03d}",
            document_id=document_id,
            content=f"content of {document_id} part {index}",
            source_path=f"data/02_clean_markdown/{document_id}.md",
            chunk_index=index,
        )
        for document_id in document_ids
        for index in range(1, chunks_per_document + 1)
    ]


def write_consistent_index(
    index_dir: Path,
    chunks: list[ChunkRecord],
    *,
    embedding_model: str = FAKE_MODEL,
    vector_dimension: int = FAKE_DIMENSION,
    collection_name: str = "customer_claims",
    with_digests: bool = True,
) -> None:
    """Chroma store plus manifest, written the way ``IndexBuilder`` writes them."""
    provider = FakeEmbeddingProvider(model_name=embedding_model, vector_dimension=vector_dimension)
    ordered = sort_chunks_deterministic(chunks)
    vectors = provider.embed_documents([chunk.content for chunk in ordered])
    store = ChromaVectorStore(index_dir=index_dir, collection_name=collection_name)
    store.recreate_collection(embedding_dimension=vector_dimension)
    store.add_chunks(ordered, vectors)
    records = store.export_collection_records()
    store.close()
    write_manifest_atomic(
        index_dir,
        build_manifest(
            collection_name=collection_name,
            embedding_model=embedding_model,
            corpus_fingerprint=compute_corpus_fingerprint(ordered, embedding_model=embedding_model),
            chunk_count=len(ordered),
            document_count=len({chunk.document_id for chunk in ordered}),
            metadata_schema_version=METADATA_SCHEMA_VERSION,
            vector_dimension=vector_dimension,
            chunk_payload_digest=compute_chunk_payload_digest(ordered) if with_digests else None,
            embedding_digest=compute_embedding_digest(ordered, vectors) if with_digests else None,
            collection_content_digest=(
                compute_collection_content_digest(records) if with_digests else None
            ),
        ),
    )


def stage_consistent_release(
    tmp_path: Path,
    *,
    document_ids: tuple[str, ...] = ("01_service_overview",),
    chunks_per_document: int = 1,
    index_relative: str = "index",
    embedding_model: str = FAKE_MODEL,
    vector_dimension: int = FAKE_DIMENSION,
) -> StagedRelease:
    """Index + canonical corpus manifest + descriptor that really agree with each other."""
    stage_project_configs(tmp_path)
    chunks = make_test_chunks(document_ids, chunks_per_document)
    index_dir = tmp_path / index_relative
    write_consistent_index(
        index_dir,
        chunks,
        embedding_model=embedding_model,
        vector_dimension=vector_dimension,
    )
    fingerprint = compute_corpus_fingerprint(chunks, embedding_model=embedding_model)
    payload_digest = compute_chunk_payload_digest(chunks)
    descriptor_path = tmp_path / "configs" / "release" / "production_posture.json"
    write_test_release_descriptor(
        descriptor_path,
        index_path_relative=index_relative,
        corpus_fingerprint=fingerprint,
        chunk_count=len(chunks),
        document_count=len(document_ids),
        supported_document_ids=list(document_ids),
        chunk_payload_digest=payload_digest,
        embedding_model=embedding_model,
    )
    return StagedRelease(
        root=tmp_path,
        index_dir=index_dir,
        descriptor_path=descriptor_path,
        corpus_manifest_path=tmp_path / TEST_CORPUS_MANIFEST_RELATIVE,
        chunks=chunks,
        corpus_fingerprint=fingerprint,
        chunk_payload_digest=payload_digest,
        embedding_model=embedding_model,
        vector_dimension=vector_dimension,
    )


def stage_test_production_posture(tmp_path: Path) -> Path:
    """Create a portable active release posture tree under tmp_path; returns the index dir."""
    return stage_consistent_release(tmp_path).index_dir


def resolved_release_target_for_index(
    index_dir: Path,
    *,
    project_root: Path,
    corpus_fingerprint: str,
    chunk_count: int,
    document_count: int,
    supported_document_ids: tuple[str, ...],
    target_name: str = "active",
    status: str = "selected_production_release",
    embedding_model: str = FAKE_MODEL,
    vector_dimension: int = 8,
    chunk_payload_digest: str | None = None,
) -> ResolvedReleaseTarget:
    """A resolved target equal to what ``write_test_release_descriptor`` would resolve to."""
    relative = index_dir.resolve().relative_to(project_root.resolve()).as_posix()
    return ResolvedReleaseTarget(
        target_name=target_name,
        status=status,
        index_dir=index_dir.resolve(),
        index_path_relative=relative,
        expected_corpus_fingerprint=hex64(corpus_fingerprint),
        expected_chunk_count=max(chunk_count, document_count),
        expected_document_count=document_count,
        supported_document_ids=tuple(sorted(supported_document_ids)),
        collection_name="customer_claims",
        embedding_model=embedding_model,
        vector_dimension=vector_dimension,
        corpus_id=TEST_CORPUS_ID,
        corpus_manifest_path=(project_root / TEST_CORPUS_MANIFEST_RELATIVE).resolve(),
        corpus_manifest_relative=TEST_CORPUS_MANIFEST_RELATIVE,
        expected_chunk_payload_digest=hex64(
            chunk_payload_digest or f"payload:{corpus_fingerprint}"
        ),
    )
