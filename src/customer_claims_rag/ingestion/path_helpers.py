"""Path normalization and export safety checks."""

from __future__ import annotations

from pathlib import Path

from customer_claims_rag.exceptions import IngestionError

CLEAN_MARKDOWN_DIRNAME = "data/02_clean_markdown"


def resolve_safe_path(path: Path, *, root: Path | None = None) -> Path:
    """Resolve path and ensure it stays within root when provided."""
    resolved = path.resolve()
    if root is not None:
        root_resolved = root.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise IngestionError(
                f"path traversal rejected: {path} is outside permitted root {root}",
                file_path=str(path),
            ) from exc
    return resolved


def to_relative_posix_path(path: Path, root: Path) -> str:
    """Return path relative to root using POSIX separators."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise IngestionError(
            f"path {path} is outside permitted root {root}",
            file_path=str(path),
        ) from exc
    return relative.as_posix()


def validate_export_paths(
    *,
    permitted_root: Path,
    input_dir: Path,
    output_path: Path,
    stats_path: Path,
    source_paths: list[str],
) -> tuple[Path, Path]:
    """Validate output and stats paths before writing."""
    root = permitted_root.resolve()
    output_resolved = resolve_safe_path(output_path, root=root)
    stats_resolved = resolve_safe_path(stats_path, root=root)
    input_resolved = resolve_safe_path(input_dir, root=root)
    clean_dir = (root / CLEAN_MARKDOWN_DIRNAME).resolve()

    if output_resolved == stats_resolved:
        raise IngestionError(
            "output and stats paths must not refer to the same file",
            file_path=str(output_path),
        )

    if output_resolved.is_dir():
        raise IngestionError(
            f"output path is a directory: {output_path}",
            file_path=str(output_path),
        )

    if stats_resolved.is_dir():
        raise IngestionError(
            f"stats output path is a directory: {stats_path}",
            file_path=str(stats_path),
        )

    source_abs_paths = {(root / src).resolve() for src in source_paths}

    for label, candidate in (("output", output_resolved), ("stats", stats_resolved)):
        if candidate in source_abs_paths:
            raise IngestionError(
                f"{label} path must not overwrite a loaded source Markdown file",
                file_path=str(candidate),
            )

    if _is_under_or_equal(output_resolved, clean_dir):
        raise IngestionError(
            "output path must not be inside clean Markdown directory",
            file_path=str(output_path),
        )

    if _is_under_or_equal(stats_resolved, clean_dir):
        raise IngestionError(
            "stats output path must not be inside clean Markdown directory",
            file_path=str(stats_path),
        )

    if _is_under_or_equal(output_resolved, input_resolved):
        raise IngestionError(
            "output path must not be inside input directory",
            file_path=str(output_path),
        )

    if _is_under_or_equal(stats_resolved, input_resolved):
        raise IngestionError(
            "stats output path must not be inside input directory",
            file_path=str(stats_path),
        )

    return output_resolved, stats_resolved


def _is_under_or_equal(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return path == parent
