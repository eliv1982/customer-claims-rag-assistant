"""Frozen embedding snapshot serialization for reproducible experiment replay."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from customer_claims_rag.exceptions import IndexBuildError
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint, sort_chunks_deterministic
from customer_claims_rag.retrieval.index_identity import (
    compute_chunk_payload_digest,
    compute_embedding_digest,
)

SNAPSHOT_FORMAT_VERSION = "1.0.0"
EMBEDDING_DTYPE = np.dtype("<f4")
NPZ_EMBEDDINGS_KEY = "embeddings"
NPZ_CHUNK_ID_BLOB_KEY = "chunk_id_blob"


class SnapshotValidationError(IndexBuildError):
    """Raised when a frozen embedding snapshot fails validation."""


def encode_chunk_id_blob(chunk_ids: Sequence[str]) -> bytes:
    """Serialize chunk IDs as length-prefixed UTF-8 strings (deterministic order)."""
    parts: list[bytes] = []
    for chunk_id in chunk_ids:
        encoded = chunk_id.encode("utf-8")
        parts.append(struct.pack("<I", len(encoded)))
        parts.append(encoded)
    return b"".join(parts)


def decode_chunk_id_blob(blob: bytes) -> list[str]:
    """Decode length-prefixed UTF-8 chunk IDs from canonical blob bytes."""
    chunk_ids: list[str] = []
    offset = 0
    while offset < len(blob):
        if offset + 4 > len(blob):
            raise SnapshotValidationError("truncated chunk_id_blob length prefix")
        (length,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        end = offset + length
        if end > len(blob):
            raise SnapshotValidationError("truncated chunk_id_blob payload")
        chunk_ids.append(blob[offset:end].decode("utf-8"))
        offset = end
    return chunk_ids


def canonical_snapshot_bytes(
    *,
    chunk_ids: Sequence[str],
    embeddings: np.ndarray,
    format_version: str,
    embedding_model: str,
    dimension: int,
) -> bytes:
    """Build canonical raw bytes used for snapshot_digest (cross-platform stable)."""
    if embeddings.dtype != EMBEDDING_DTYPE:
        matrix = embeddings.astype(EMBEDDING_DTYPE, copy=False)
    else:
        matrix = embeddings
    if matrix.ndim != 2:
        raise SnapshotValidationError("embeddings must be a 2-D matrix")
    if matrix.shape[0] != len(chunk_ids):
        raise SnapshotValidationError("chunk_id / embedding row count mismatch")
    if matrix.shape[1] != dimension:
        raise SnapshotValidationError("embedding dimension mismatch")

    parts = [
        format_version.encode("utf-8"),
        b"\x00",
        embedding_model.encode("utf-8"),
        b"\x00",
        struct.pack("<I", dimension),
        struct.pack("<I", len(chunk_ids)),
        encode_chunk_id_blob(chunk_ids),
    ]
    parts.append(matrix.tobytes(order="C"))
    return b"".join(parts)


def compute_snapshot_digest(
    *,
    chunk_ids: Sequence[str],
    embeddings: np.ndarray,
    format_version: str = SNAPSHOT_FORMAT_VERSION,
    embedding_model: str,
    dimension: int,
) -> str:
    payload = canonical_snapshot_bytes(
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        format_version=format_version,
        embedding_model=embedding_model,
        dimension=dimension,
    )
    return hashlib.sha256(payload).hexdigest()


def _repo_relative_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_embedding_snapshot(
    *,
    npz_path: Path,
    manifest_path: Path,
    chunks: list[ChunkRecord],
    embeddings: Sequence[Sequence[float]],
    embedding_model: str,
    corpus_fingerprint: str,
    chunk_payload_digest: str,
    candidate_experiment_id: str,
    creation_source_commit: str | None,
    project_root: Path,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize embeddings to NPZ and write manifest with digests."""
    ordered = sort_chunks_deterministic(chunks)
    ordered_ids = [chunk.chunk_id for chunk in ordered]
    matrix = np.asarray(embeddings, dtype=EMBEDDING_DTYPE)
    if matrix.ndim != 2:
        raise SnapshotValidationError("embeddings must be a 2-D matrix")
    dimension = int(matrix.shape[1])
    if len(ordered_ids) != matrix.shape[0]:
        raise SnapshotValidationError("chunk_ids and embeddings row count mismatch")

    embedding_digest = compute_embedding_digest(ordered, embeddings)
    snapshot_digest = compute_snapshot_digest(
        chunk_ids=ordered_ids,
        embeddings=matrix,
        embedding_model=embedding_model,
        dimension=dimension,
    )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_id_blob = np.frombuffer(encode_chunk_id_blob(ordered_ids), dtype=np.uint8)
    np.savez(
        npz_path,
        **{
            NPZ_EMBEDDINGS_KEY: matrix,
            NPZ_CHUNK_ID_BLOB_KEY: chunk_id_blob,
        },
    )

    manifest = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "snapshot_path": _repo_relative_path(npz_path, project_root),
        "candidate_experiment_id": candidate_experiment_id,
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "chunk_ids": ordered_ids,
        "corpus_fingerprint": corpus_fingerprint,
        "chunk_payload_digest": chunk_payload_digest,
        "embedding_digest": embedding_digest,
        "snapshot_digest": snapshot_digest,
        "creation_source_commit": creation_source_commit,
        "lineage": lineage or {},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_snapshot_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotValidationError(f"invalid snapshot manifest: {manifest_path}") from exc
    if manifest.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotValidationError(
            f"unsupported snapshot format version in {manifest_path}"
        )
    return manifest


def load_embedding_snapshot_arrays(npz_path: Path) -> tuple[list[str], np.ndarray]:
    """Load chunk IDs and embeddings from NPZ without pickle/object arrays."""
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if NPZ_EMBEDDINGS_KEY not in archive:
                raise SnapshotValidationError(f"missing {NPZ_EMBEDDINGS_KEY} in {npz_path}")
            if NPZ_CHUNK_ID_BLOB_KEY not in archive:
                raise SnapshotValidationError(f"missing {NPZ_CHUNK_ID_BLOB_KEY} in {npz_path}")
            embeddings = np.asarray(archive[NPZ_EMBEDDINGS_KEY], dtype=EMBEDDING_DTYPE)
            blob = archive[NPZ_CHUNK_ID_BLOB_KEY].tobytes()
    except SnapshotValidationError:
        raise
    except Exception as exc:
        raise SnapshotValidationError(f"failed to load snapshot NPZ: {npz_path}") from exc

    chunk_ids = decode_chunk_id_blob(blob)
    if embeddings.ndim != 2:
        raise SnapshotValidationError("snapshot embeddings must be 2-D")
    if embeddings.shape[0] != len(chunk_ids):
        raise SnapshotValidationError("snapshot chunk_id count does not match embedding rows")
    return chunk_ids, embeddings


def validate_snapshot_against_chunks(
    *,
    manifest: dict[str, Any],
    npz_path: Path,
    chunks: list[ChunkRecord],
    embedding_model: str,
) -> tuple[list[str], list[list[float]]]:
    """Validate snapshot identity and return ordered embeddings aligned to corpus chunks."""
    ordered_chunks = sort_chunks_deterministic(chunks)
    expected_ids = [chunk.chunk_id for chunk in ordered_chunks]
    expected_id_set = set(expected_ids)
    chunk_payload_digest = compute_chunk_payload_digest(ordered_chunks)
    corpus_fingerprint = compute_corpus_fingerprint(
        ordered_chunks,
        embedding_model=embedding_model,
    )

    manifest_ids = list(manifest.get("chunk_ids") or [])
    if manifest_ids != expected_ids:
        manifest_set = set(manifest_ids)
        if manifest_set != expected_id_set:
            missing = sorted(expected_id_set - manifest_set)
            extra = sorted(manifest_set - expected_id_set)
            raise SnapshotValidationError(
                f"snapshot chunk ID set mismatch: missing={missing[:3]} extra={extra[:3]}"
            )
        raise SnapshotValidationError(
            "snapshot chunk IDs are not in deterministic chunk_id order"
        )

    if manifest.get("chunk_payload_digest") != chunk_payload_digest:
        raise SnapshotValidationError(
            "snapshot chunk_payload_digest does not match current corpus chunks"
        )
    if manifest.get("embedding_model") != embedding_model:
        raise SnapshotValidationError(
            f"snapshot embedding model mismatch: expected {embedding_model}"
        )
    if manifest.get("corpus_fingerprint") != corpus_fingerprint:
        raise SnapshotValidationError(
            "snapshot corpus_fingerprint does not match current corpus fingerprint"
        )

    loaded_ids, embeddings = load_embedding_snapshot_arrays(npz_path)
    if loaded_ids != expected_ids:
        raise SnapshotValidationError("NPZ chunk_id order does not match manifest/corpus")

    dimension = int(manifest.get("embedding_dimension") or 0)
    if embeddings.shape[1] != dimension:
        raise SnapshotValidationError(
            f"snapshot dimension mismatch: expected {dimension}, got {embeddings.shape[1]}"
        )

    expected_snapshot_digest = manifest.get("snapshot_digest")
    actual_snapshot_digest = compute_snapshot_digest(
        chunk_ids=loaded_ids,
        embeddings=embeddings,
        embedding_model=embedding_model,
        dimension=dimension,
    )
    if expected_snapshot_digest != actual_snapshot_digest:
        raise SnapshotValidationError("snapshot_digest mismatch")

    expected_embedding_digest = manifest.get("embedding_digest")
    actual_embedding_digest = compute_embedding_digest(ordered_chunks, embeddings.tolist())
    if expected_embedding_digest != actual_embedding_digest:
        raise SnapshotValidationError("embedding_digest mismatch")

    return loaded_ids, [list(map(float, row)) for row in embeddings]


def resolve_snapshot_embeddings(
    chunks: list[ChunkRecord],
    *,
    npz_path: Path,
    manifest_path: Path,
    embedding_model: str,
) -> list[list[float]]:
    """Load validated snapshot vectors for index build (no provider API calls)."""
    manifest = load_snapshot_manifest(manifest_path)
    _, embeddings = validate_snapshot_against_chunks(
        manifest=manifest,
        npz_path=npz_path,
        chunks=chunks,
        embedding_model=embedding_model,
    )
    return embeddings


def export_snapshot_from_experiment_cache(
    *,
    chunks: list[ChunkRecord],
    cache_path: Path,
    npz_path: Path,
    manifest_path: Path,
    embedding_model: str,
    candidate_experiment_id: str,
    creation_source_commit: str | None,
    project_root: Path,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export NPZ snapshot from experiment_embeddings_v1.json cache."""
    from customer_claims_rag.retrieval.experiment_embedding_cache import (
        CACHE_FORMAT_VERSION,
        _load_cache,
    )

    ordered = sort_chunks_deterministic(chunks)
    chunk_payload_digest = compute_chunk_payload_digest(ordered)
    corpus_fingerprint = compute_corpus_fingerprint(
        ordered,
        embedding_model=embedding_model,
    )
    cached = _load_cache(cache_path)
    if cached is None:
        raise SnapshotValidationError(f"missing experiment embedding cache: {cache_path}")
    if cached.get("format_version") != CACHE_FORMAT_VERSION:
        raise SnapshotValidationError("unsupported experiment embedding cache version")
    if cached.get("embedding_model") != embedding_model:
        raise SnapshotValidationError("experiment cache embedding model mismatch")
    if cached.get("chunk_payload_digest") != chunk_payload_digest:
        raise SnapshotValidationError("experiment cache chunk_payload_digest mismatch")

    stored_vectors = cached.get("vectors") or {}
    missing = [chunk.chunk_id for chunk in ordered if chunk.chunk_id not in stored_vectors]
    if missing:
        raise SnapshotValidationError(f"experiment cache missing vectors: {missing[:5]}")

    embeddings = [list(stored_vectors[chunk.chunk_id]) for chunk in ordered]
    expected_digest = cached.get("embedding_digest")
    actual_digest = compute_embedding_digest(ordered, embeddings)
    if expected_digest != actual_digest:
        raise SnapshotValidationError("experiment cache embedding_digest mismatch")

    return write_embedding_snapshot(
        npz_path=npz_path,
        manifest_path=manifest_path,
        chunks=ordered,
        embeddings=embeddings,
        embedding_model=embedding_model,
        corpus_fingerprint=corpus_fingerprint,
        chunk_payload_digest=chunk_payload_digest,
        candidate_experiment_id=candidate_experiment_id,
        creation_source_commit=creation_source_commit,
        project_root=project_root,
        lineage=lineage,
    )


def export_snapshot_from_committed_index(
    *,
    chunks: list[ChunkRecord],
    index_dir: Path,
    collection_name: str,
    npz_path: Path,
    manifest_path: Path,
    embedding_model: str,
    candidate_experiment_id: str,
    creation_source_commit: str | None,
    project_root: Path,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export NPZ snapshot from vectors stored in a committed Chroma index (no API calls)."""
    from customer_claims_rag.retrieval.factory import create_vector_store

    ordered = sort_chunks_deterministic(chunks)
    chunk_payload_digest = compute_chunk_payload_digest(ordered)
    corpus_fingerprint = compute_corpus_fingerprint(
        ordered,
        embedding_model=embedding_model,
    )
    store = create_vector_store(index_dir=index_dir, collection_name=collection_name)
    try:
        exported = store.export_collection_records()
    finally:
        store.close()

    by_id = {str(record["chunk_id"]): record for record in exported}
    missing = [chunk.chunk_id for chunk in ordered if chunk.chunk_id not in by_id]
    if missing:
        raise SnapshotValidationError(
            f"committed index missing vectors for chunks: {missing[:5]}"
        )
    embeddings = [
        [float(value) for value in by_id[chunk.chunk_id]["embedding"]]
        for chunk in ordered
    ]
    return write_embedding_snapshot(
        npz_path=npz_path,
        manifest_path=manifest_path,
        chunks=ordered,
        embeddings=embeddings,
        embedding_model=embedding_model,
        corpus_fingerprint=corpus_fingerprint,
        chunk_payload_digest=chunk_payload_digest,
        candidate_experiment_id=candidate_experiment_id,
        creation_source_commit=creation_source_commit,
        project_root=project_root,
        lineage=lineage,
    )
