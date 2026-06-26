"""Atomic experiment index publish helpers."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from customer_claims_rag.exceptions import IndexBuildError
from customer_claims_rag.retrieval.experiment_embedding_cache import CACHE_FILENAME
from customer_claims_rag.retrieval.manifest import manifest_path

PRESERVED_INDEX_FILES = frozenset({CACHE_FILENAME})


def experiment_staging_dir(destination_dir: Path, *, run_id: str | None = None) -> tuple[Path, str]:
    run = run_id or uuid.uuid4().hex
    staging = destination_dir.parent / f"{destination_dir.name}.staging_{run[:8]}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    return staging, run


def publish_staged_index(staging_dir: Path, destination_dir: Path) -> None:
    """Atomically replace destination index files while preserving embedding cache."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for item in destination_dir.iterdir():
        if item.name in PRESERVED_INDEX_FILES:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in staging_dir.iterdir():
        target = destination_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    shutil.rmtree(staging_dir, ignore_errors=True)
    if not manifest_path(destination_dir).is_file():
        raise IndexBuildError("manifest missing after atomic index publish")
