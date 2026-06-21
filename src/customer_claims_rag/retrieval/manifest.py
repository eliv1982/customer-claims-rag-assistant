"""Index manifest persistence and validation."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from customer_claims_rag.config import INDEX_FORMAT_VERSION, MANIFEST_FILENAME, METADATA_SCHEMA_VERSION
from customer_claims_rag.exceptions import IndexBuildError, IndexManifestError
from customer_claims_rag.retrieval.models import IndexManifest


def manifest_path(index_dir: Path) -> Path:
    return index_dir / MANIFEST_FILENAME


def invalidate_manifest(index_dir: Path) -> None:
    """Remove existing manifest before destructive index rebuild."""
    path = manifest_path(index_dir)
    if not path.is_file():
        return
    try:
        path.unlink()
    except OSError as exc:
        raise IndexBuildError(
            "failed to invalidate existing index manifest; rebuild aborted before "
            "store mutation"
        ) from exc


def write_manifest_atomic(index_dir: Path, manifest: IndexManifest) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_path(index_dir)
    fd, temp_name = tempfile.mkstemp(
        suffix=".json",
        dir=str(index_dir),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        import os

        os.close(fd)
        payload = manifest.model_dump(mode="json")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temp_path.replace(target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def load_manifest(index_dir: Path) -> IndexManifest:
    path = manifest_path(index_dir)
    if not path.is_file():
        raise IndexManifestError(
            f"index manifest not found at {path}; run index rebuild with "
            f"`python -m customer_claims_rag.cli.build_index --rebuild`"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexManifestError(
            f"index manifest at {path} is corrupted JSON; rebuild the index"
        ) from exc
    try:
        return IndexManifest.model_validate(raw)
    except Exception as exc:
        raise IndexManifestError(
            f"index manifest at {path} has invalid schema; rebuild the index"
        ) from exc


def validate_manifest_against_runtime(
    manifest: IndexManifest,
    *,
    collection_name: str,
    embedding_model: str,
    chunk_count: int,
    expected_index_format_version: str = INDEX_FORMAT_VERSION,
    expected_metadata_schema_version: str = METADATA_SCHEMA_VERSION,
    query_vector_dimension: int | None = None,
) -> None:
    if manifest.index_format_version != expected_index_format_version:
        raise IndexManifestError(
            f"index format version mismatch: manifest has "
            f"{manifest.index_format_version!r}, expected "
            f"{expected_index_format_version!r}; rebuild the index"
        )
    if manifest.metadata_schema_version != expected_metadata_schema_version:
        raise IndexManifestError(
            f"metadata schema version mismatch: manifest has "
            f"{manifest.metadata_schema_version!r}, expected "
            f"{expected_metadata_schema_version!r}; rebuild the index"
        )
    if manifest.collection_name != collection_name:
        raise IndexManifestError(
            f"collection name mismatch: manifest has {manifest.collection_name!r}, "
            f"runtime expects {collection_name!r}; rebuild the index"
        )
    if manifest.embedding_model != embedding_model:
        raise IndexManifestError(
            f"embedding model mismatch: manifest has {manifest.embedding_model!r}, "
            f"runtime expects {embedding_model!r}; rebuild the index"
        )
    if manifest.chunk_count != chunk_count:
        raise IndexManifestError(
            f"chunk count mismatch: manifest has {manifest.chunk_count}, "
            f"vector store has {chunk_count}; rebuild the index"
        )
    if query_vector_dimension is not None:
        validate_query_vector_dimension(manifest, query_vector_dimension)


def validate_query_vector_dimension(
    manifest: IndexManifest,
    query_vector_dimension: int,
) -> None:
    if manifest.vector_dimension is None:
        raise IndexManifestError(
            "manifest missing vector_dimension; rebuild the index"
        )
    if manifest.vector_dimension != query_vector_dimension:
        raise IndexManifestError(
            f"vector dimension mismatch: manifest has {manifest.vector_dimension}, "
            f"query embedding has {query_vector_dimension}; rebuild the index"
        )


def build_manifest(
    *,
    collection_name: str,
    embedding_model: str,
    corpus_fingerprint: str,
    chunk_count: int,
    document_count: int,
    metadata_schema_version: str,
    vector_dimension: int,
) -> IndexManifest:
    return IndexManifest(
        index_format_version=INDEX_FORMAT_VERSION,
        collection_name=collection_name,
        embedding_model=embedding_model,
        corpus_fingerprint=corpus_fingerprint,
        chunk_count=chunk_count,
        document_count=document_count,
        metadata_schema_version=metadata_schema_version,
        vector_dimension=vector_dimension,
        created_at=datetime.now(timezone.utc),
    )
