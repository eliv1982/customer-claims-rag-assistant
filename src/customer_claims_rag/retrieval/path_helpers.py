"""Path safety checks for retrieval CLI index directories."""

from __future__ import annotations

from pathlib import Path

from customer_claims_rag.exceptions import IngestionError, RetrievalError
from customer_claims_rag.ingestion.path_helpers import (
    CLEAN_MARKDOWN_DIRNAME,
    _is_under_or_equal,
    resolve_safe_path,
)

RAW_MARKDOWN_DIRNAME = "data/01_raw"


def _resolve_under_root(path: Path, root: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    return resolve_safe_path(candidate, root=root)


def validate_index_dir(
    index_dir: Path,
    *,
    project_root: Path,
    input_dir: Path | None = None,
) -> Path:
    """Validate CLI index directory before creating persistent store files."""
    root = project_root.resolve()
    try:
        resolved = _resolve_under_root(index_dir, root)
    except IngestionError as exc:
        raise RetrievalError(str(exc)) from None

    if input_dir is not None:
        try:
            input_resolved = _resolve_under_root(input_dir, root)
        except IngestionError as exc:
            raise RetrievalError(str(exc)) from None
        if resolved == input_resolved:
            raise RetrievalError("index_dir must not be the same as input_dir")
        if _is_under_or_equal(resolved, input_resolved):
            raise RetrievalError("index_dir must not be inside input directory")

    raw_dir = (root / RAW_MARKDOWN_DIRNAME).resolve()
    clean_dir = (root / CLEAN_MARKDOWN_DIRNAME).resolve()
    if _is_under_or_equal(resolved, raw_dir):
        raise RetrievalError("index_dir must not be inside raw source directory")
    if _is_under_or_equal(resolved, clean_dir):
        raise RetrievalError("index_dir must not be inside clean Markdown directory")

    return resolved
