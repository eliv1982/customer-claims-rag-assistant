"""Load committed frozen embedding snapshots for exact evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from customer_claims_rag.evaluation.exact_vector_search import exact_vector_search
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.embedding_snapshot import (
    load_embedding_snapshot_arrays,
    load_snapshot_manifest,
    validate_snapshot_against_chunks,
)
from customer_claims_rag.retrieval.metadata_mapper import chunk_to_vector_metadata
from customer_claims_rag.retrieval.models import SearchResult


@dataclass(frozen=True)
class ExactFrozenArm:
    """Frozen snapshot vectors and corpus records for one experiment arm."""

    arm: str
    corpus_fingerprint: str
    chunk_payload_digest: str
    embedding_digest: str
    snapshot_digest: str
    chunk_ids: tuple[str, ...]
    chunk_records: tuple[dict[str, Any], ...]
    embedding_matrix: np.ndarray
    snapshot_path: str
    manifest_path: str

    def search(
        self,
        query_vector: list[float],
        *,
        fetch_k: int,
        threshold: float,
    ) -> list[SearchResult]:
        return exact_vector_search(
            query_vector=query_vector,
            chunk_records=self.chunk_records,
            embedding_matrix=self.embedding_matrix,
            fetch_k=fetch_k,
            threshold=threshold,
        )


def _chunk_to_record(chunk: ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document": chunk.content,
        "metadata": chunk_to_vector_metadata(chunk),
    }


def load_exact_frozen_arm(
    *,
    arm: str,
    chunks: list[ChunkRecord],
    npz_path: Path,
    manifest_path: Path,
    embedding_model: str,
    project_root: Path,
) -> ExactFrozenArm:
    manifest = load_snapshot_manifest(manifest_path)
    _, embeddings = validate_snapshot_against_chunks(
        manifest=manifest,
        npz_path=npz_path,
        chunks=chunks,
        embedding_model=embedding_model,
    )
    chunk_ids, matrix = load_embedding_snapshot_arrays(npz_path)
    ordered_chunks = {chunk.chunk_id: chunk for chunk in chunks}
    records = tuple(_chunk_to_record(ordered_chunks[chunk_id]) for chunk_id in chunk_ids)

    from customer_claims_rag.evaluation.doc12_threat_atomic_contract import repo_relative_path

    return ExactFrozenArm(
        arm=arm,
        corpus_fingerprint=str(manifest["corpus_fingerprint"]),
        chunk_payload_digest=str(manifest["chunk_payload_digest"]),
        embedding_digest=str(manifest["embedding_digest"]),
        snapshot_digest=str(manifest["snapshot_digest"]),
        chunk_ids=tuple(chunk_ids),
        chunk_records=records,
        embedding_matrix=matrix,
        snapshot_path=repo_relative_path(npz_path, project_root),
        manifest_path=repo_relative_path(manifest_path, project_root),
    )


def copy_snapshot_to_temp_root(
    *,
    npz_path: Path,
    manifest_path: Path,
    temp_root: Path,
) -> tuple[Path, Path]:
    temp_root.mkdir(parents=True, exist_ok=True)
    target_npz = temp_root / npz_path.name
    target_manifest = temp_root / manifest_path.name
    target_npz.write_bytes(npz_path.read_bytes())
    target_manifest.write_bytes(manifest_path.read_bytes())
    return target_npz, target_manifest


__all__ = [
    "ExactFrozenArm",
    "copy_snapshot_to_temp_root",
    "load_exact_frozen_arm",
]
