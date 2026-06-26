"""Chroma persistent storage layout validation."""

from __future__ import annotations

from pathlib import Path

from customer_claims_rag.exceptions import VectorStoreError

CHROMA_SQLITE_FILENAME = "chroma.sqlite3"


def chroma_sqlite_path(index_dir: Path) -> Path:
    return index_dir / CHROMA_SQLITE_FILENAME


def validate_open_existing_chroma_preconditions(
    index_dir: Path,
    *,
    collection_name: str,
) -> None:
    """Fail before PersistentClient when opening an existing local Chroma index."""
    if not collection_name or not collection_name.strip():
        raise VectorStoreError("collection name must not be empty for existing-only open")
    if not index_dir.exists():
        raise VectorStoreError(
            f"index directory does not exist: {index_dir}; "
            "existing-only mode must not create persistent storage"
        )
    if not index_dir.is_dir():
        raise VectorStoreError(f"index path is not a directory: {index_dir}")
    sqlite_path = chroma_sqlite_path(index_dir)
    if not sqlite_path.is_file():
        raise VectorStoreError(
            f"Chroma persistent database not found at {sqlite_path}; "
            f"existing-only mode requires {CHROMA_SQLITE_FILENAME}"
        )
