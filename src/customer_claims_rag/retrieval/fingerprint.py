"""Deterministic corpus fingerprint for index manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from customer_claims_rag.config import INDEX_FORMAT_VERSION, METADATA_SCHEMA_VERSION
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.metadata_mapper import canonical_metadata_for_fingerprint


def sort_chunks_deterministic(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    return sorted(chunks, key=lambda chunk: chunk.chunk_id)


def build_fingerprint_payload(
    chunks: list[ChunkRecord],
    *,
    embedding_model: str,
    index_format_version: str = INDEX_FORMAT_VERSION,
    metadata_schema_version: str = METADATA_SCHEMA_VERSION,
) -> dict[str, Any]:
    ordered = sort_chunks_deterministic(chunks)
    return {
        "index_format_version": index_format_version,
        "metadata_schema_version": metadata_schema_version,
        "embedding_model": embedding_model,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": canonical_metadata_for_fingerprint(chunk),
            }
            for chunk in ordered
        ],
    }


def compute_corpus_fingerprint(
    chunks: list[ChunkRecord],
    *,
    embedding_model: str,
    index_format_version: str = INDEX_FORMAT_VERSION,
    metadata_schema_version: str = METADATA_SCHEMA_VERSION,
) -> str:
    payload = build_fingerprint_payload(
        chunks,
        embedding_model=embedding_model,
        index_format_version=index_format_version,
        metadata_schema_version=metadata_schema_version,
    )
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
