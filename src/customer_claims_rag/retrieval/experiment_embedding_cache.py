"""Deterministic experiment embedding cache for reproducible index rebuilds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from customer_claims_rag.exceptions import IndexBuildError
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.fingerprint import sort_chunks_deterministic
from customer_claims_rag.retrieval.index_identity import (
    compute_chunk_payload_digest,
    compute_embedding_digest,
)
from customer_claims_rag.retrieval.ports import EmbeddingProvider

CACHE_FILENAME = "experiment_embeddings_v1.json"
CACHE_FORMAT_VERSION = "1.0.0"


def experiment_cache_path(index_dir: Path) -> Path:
    return index_dir / CACHE_FILENAME


def is_experiment_index_dir(index_dir: Path, project_root: Path) -> bool:
    experiments_root = (project_root / "data/04_index_experiments").resolve()
    try:
        index_dir.resolve().relative_to(experiments_root)
        return True
    except ValueError:
        return False


def _load_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexBuildError(f"experiment embedding cache is corrupted: {path}") from exc


def _write_cache_atomic(
    path: Path,
    *,
    embedding_model: str,
    chunk_payload_digest: str,
    embedding_digest: str,
    vectors: dict[str, list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "embedding_model": embedding_model,
        "chunk_payload_digest": chunk_payload_digest,
        "embedding_digest": embedding_digest,
        "vectors": {
            chunk_id: vectors[chunk_id]
            for chunk_id in sorted(vectors)
        },
    }
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temp.replace(path)


def resolve_experiment_embeddings(
    chunks: list[ChunkRecord],
    *,
    embedding_provider: EmbeddingProvider,
    cache_path: Path,
) -> list[list[float]]:
    """Return embeddings from cache when chunk payloads match, else fetch and persist."""
    ordered = sort_chunks_deterministic(chunks)
    chunk_payload_digest = compute_chunk_payload_digest(ordered)
    cached = _load_cache(cache_path)

    if cached is not None:
        if cached.get("format_version") != CACHE_FORMAT_VERSION:
            raise IndexBuildError(
                f"unsupported experiment embedding cache version at {cache_path}"
            )
        if cached.get("embedding_model") != embedding_provider.model_name:
            raise IndexBuildError(
                "experiment embedding cache model mismatch; delete cache or rebuild"
            )
        if cached.get("chunk_payload_digest") != chunk_payload_digest:
            raise IndexBuildError(
                "experiment embedding cache chunk payload mismatch; delete cache or rebuild"
            )
        stored_vectors = cached.get("vectors") or {}
        missing = [chunk.chunk_id for chunk in ordered if chunk.chunk_id not in stored_vectors]
        if missing:
            raise IndexBuildError(
                f"experiment embedding cache missing vectors for: {missing[:5]}"
            )
        embeddings = [list(stored_vectors[chunk.chunk_id]) for chunk in ordered]
        expected_digest = cached.get("embedding_digest")
        actual_digest = compute_embedding_digest(ordered, embeddings)
        if expected_digest != actual_digest:
            raise IndexBuildError(
                "experiment embedding cache digest mismatch; delete cache or rebuild"
            )
        return embeddings

    texts = [chunk.content for chunk in ordered]
    embeddings: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch_vectors = embedding_provider.embed_documents(texts[start : start + batch_size])
        embeddings.extend(batch_vectors)

    embedding_digest = compute_embedding_digest(ordered, embeddings)
    _write_cache_atomic(
        cache_path,
        embedding_model=embedding_provider.model_name,
        chunk_payload_digest=chunk_payload_digest,
        embedding_digest=embedding_digest,
        vectors={chunk.chunk_id: vector for chunk, vector in zip(ordered, embeddings, strict=True)},
    )
    return embeddings
