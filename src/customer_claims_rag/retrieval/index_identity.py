"""Reproducible index-content digests for experiment integrity."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any, Sequence

from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.fingerprint import sort_chunks_deterministic
from customer_claims_rag.retrieval.metadata_mapper import canonical_metadata_for_fingerprint


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_chunk_payload_digest(chunks: list[ChunkRecord]) -> str:
    """Hash chunk payloads in deterministic chunk_id order."""
    ordered = sort_chunks_deterministic(chunks)
    entries = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.content,
            "metadata": canonical_metadata_for_fingerprint(chunk),
            "logical_source_path": chunk.source_path.replace("\\", "/"),
        }
        for chunk in ordered
    ]
    return hashlib.sha256(_canonical_json({"chunks": entries}).encode("utf-8")).hexdigest()


def _serialize_embedding_vector(vector: Sequence[float]) -> bytes:
    return b"".join(struct.pack("!e", float(value)) for value in vector)


def compute_embedding_digest(
    chunks: list[ChunkRecord],
    embeddings: Sequence[Sequence[float]],
) -> str:
    """Hash chunk_id + IEEE half-precision embedding bytes in chunk_id order."""
    ordered = sort_chunks_deterministic(chunks)
    by_id = {chunk.chunk_id: chunk for chunk in ordered}
    if len(by_id) != len(ordered):
        raise ValueError("duplicate chunk_id in embedding digest input")
    if len(embeddings) != len(ordered):
        raise ValueError("chunk/embedding count mismatch for embedding digest")

    digest = hashlib.sha256()
    for chunk, vector in zip(ordered, embeddings, strict=True):
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_serialize_embedding_vector(vector))
    return digest.hexdigest()


def compute_collection_record_digest(
    *,
    chunk_id: str,
    document: str,
    metadata: dict[str, Any],
    embedding: Sequence[float],
) -> bytes:
    canonical_metadata = {
        key: metadata[key]
        for key in sorted(metadata)
        if key in {
            "chunk_id",
            "document_id",
            "source_path",
            "chunk_type",
            "heading",
            "heading_path",
            "section",
            "subsection",
            "topic",
            "risk_level",
        }
    }
    payload = {
        "chunk_id": chunk_id,
        "document": document,
        "metadata": canonical_metadata,
        "embedding": [float(value) for value in embedding],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()


def compute_collection_content_digest(records: list[dict[str, Any]]) -> str:
    """Hash ordered collection records independent of storage paths."""
    ordered = sorted(records, key=lambda item: str(item["chunk_id"]))
    digest = hashlib.sha256()
    for record in ordered:
        digest.update(
            compute_collection_record_digest(
                chunk_id=str(record["chunk_id"]),
                document=str(record["document"]),
                metadata=dict(record["metadata"]),
                embedding=record["embedding"],
            )
        )
    return digest.hexdigest()
