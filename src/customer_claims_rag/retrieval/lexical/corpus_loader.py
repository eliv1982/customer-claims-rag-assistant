"""Load lexical corpus from an existing Chroma collection without embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from customer_claims_rag.exceptions import DuplicateChunkIdError, IndexManifestError, VectorStoreError
from customer_claims_rag.retrieval.manifest import load_manifest

EXPECTED_CHUNK_COUNT = 215


@dataclass(frozen=True)
class LexicalChunk:
    """Single chunk record for lexical indexing."""

    chunk_id: str
    document_id: str
    content: str
    heading: str
    chunk_type: str
    source_path: str
    metadata: dict[str, str | int | float | bool]


def load_lexical_corpus_from_chroma(
    *,
    index_dir: Path,
    collection_name: str,
    expected_fingerprint: str | None = None,
) -> list[LexicalChunk]:
    """Load and validate all chunks from a persistent Chroma collection."""
    manifest = load_manifest(index_dir)
    if manifest.chunk_count != EXPECTED_CHUNK_COUNT:
        raise IndexManifestError(
            f"expected {EXPECTED_CHUNK_COUNT} chunks, manifest reports {manifest.chunk_count}"
        )
    if expected_fingerprint is not None and manifest.corpus_fingerprint != expected_fingerprint:
        raise IndexManifestError("corpus fingerprint mismatch between manifest and experiment")

    try:
        client = chromadb.PersistentClient(path=str(index_dir))
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise VectorStoreError(f"failed to open Chroma collection: {exc}") from exc

    count = int(collection.count())
    if count != EXPECTED_CHUNK_COUNT:
        raise IndexManifestError(f"expected {EXPECTED_CHUNK_COUNT} chunks, Chroma has {count}")

    try:
        result = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        raise VectorStoreError(f"failed to read Chroma collection: {exc}") from exc

    chunks: list[LexicalChunk] = []
    seen_ids: set[str] = set()
    for chunk_id, document, metadata in zip(
        result["ids"],
        result["documents"],
        result["metadatas"],
        strict=True,
    ):
        if chunk_id in seen_ids:
            raise DuplicateChunkIdError(f"duplicate chunk_id in Chroma collection: {chunk_id}")
        seen_ids.add(chunk_id)
        if document is None or metadata is None:
            raise VectorStoreError(f"missing document or metadata for chunk_id={chunk_id}")
        for field in ("chunk_id", "document_id", "heading", "chunk_type", "source_path"):
            if field not in metadata:
                raise VectorStoreError(f"missing metadata field {field!r} for chunk_id={chunk_id}")
        chunks.append(
            LexicalChunk(
                chunk_id=str(chunk_id),
                document_id=str(metadata["document_id"]),
                content=str(document),
                heading=str(metadata["heading"]),
                chunk_type=str(metadata["chunk_type"]),
                source_path=str(metadata["source_path"]),
                metadata={key: metadata[key] for key in metadata},
            )
        )

    chunks.sort(key=lambda item: item.chunk_id)
    return chunks
