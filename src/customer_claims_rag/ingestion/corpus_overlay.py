"""Build candidate corpus by overlaying a single document onto canonical clean markdown."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from customer_claims_rag.exceptions import IngestionError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.path_helpers import CLEAN_MARKDOWN_DIRNAME
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint

DOC08_DOCUMENT_ID = "08_escalation_and_risk_rules"
OVERLAY_LOGICAL_SOURCE_DIR = CLEAN_MARKDOWN_DIRNAME


@dataclass(frozen=True)
class CorpusOverlayBuildResult:
    """Result of building baseline vs overlay candidate corpora."""

    baseline_chunks: list[ChunkRecord]
    candidate_chunks: list[ChunkRecord]
    baseline_doc08_chunks: list[ChunkRecord]
    candidate_doc08_chunks: list[ChunkRecord]
    temp_input_dir: Path | None


@dataclass(frozen=True)
class Doc08ChunkDiff:
    """Machine-readable diff between baseline and candidate doc08 chunks."""

    baseline_chunk_ids: list[str]
    candidate_chunk_ids: list[str]
    added_chunk_ids: list[str]
    removed_chunk_ids: list[str]
    changed_chunk_ids: list[str]
    unchanged_non_doc08_chunk_ids: list[str]
    non_doc08_byte_identical: bool


def logical_source_path(document_id: str, *, source_dir: str = OVERLAY_LOGICAL_SOURCE_DIR) -> str:
    """Return stable repo-relative source identity for overlay corpus chunks."""
    return f"{source_dir}/{document_id}.md"


def normalize_overlay_corpus_source_paths(
    chunks: list[ChunkRecord],
    *,
    source_dir: str = OVERLAY_LOGICAL_SOURCE_DIR,
) -> list[ChunkRecord]:
    """Replace physical build paths with stable logical source identities."""
    return [
        chunk.model_copy(
            update={"source_path": logical_source_path(chunk.document_id, source_dir=source_dir)},
        )
        for chunk in chunks
    ]


def compute_overlay_corpus_fingerprint(
    chunks: list[ChunkRecord],
    *,
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """Fingerprint overlay corpus using normalized logical source paths."""
    normalized = normalize_overlay_corpus_source_paths(chunks)
    return compute_corpus_fingerprint(normalized, embedding_model=embedding_model)


def build_overlay_input_directory(
    *,
    canonical_dir: Path,
    overlay_document_path: Path,
    document_id: str,
    permitted_root: Path,
    staging_parent: Path | None = None,
) -> Path:
    """Create a temporary directory with canonical corpus and one replaced document."""
    canonical_resolved = canonical_dir.resolve()
    overlay_resolved = overlay_document_path.resolve()
    if not overlay_resolved.is_file():
        raise IngestionError(f"overlay document not found: {overlay_resolved}")

    expected_name = f"{document_id}.md"
    if overlay_resolved.name != expected_name:
        raise IngestionError(
            f"overlay file must be named {expected_name!r}, got {overlay_resolved.name!r}"
        )

    if staging_parent is not None:
        staging_root = staging_parent.resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
    else:
        staging_root = permitted_root / ".tmp" / "corpus_overlay"
        staging_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="build_", dir=str(staging_root)))
    for source in sorted(canonical_resolved.glob("*.md")):
        dest = temp_dir / source.name
        if source.stem == document_id:
            shutil.copy2(overlay_resolved, dest)
        else:
            shutil.copy2(source, dest)
    return temp_dir


def build_baseline_and_overlay_chunks(
    *,
    canonical_dir: Path,
    overlay_document_path: Path,
    document_id: str = DOC08_DOCUMENT_ID,
    permitted_root: Path,
    staging_parent: Path | None = None,
) -> CorpusOverlayBuildResult:
    """Build baseline chunks from canonical dir and candidate chunks with overlay."""
    builder = CorpusBuilder(permitted_root=permitted_root)
    _, baseline_chunks = builder.build_from_directory(canonical_dir)

    temp_input_dir = build_overlay_input_directory(
        canonical_dir=canonical_dir,
        overlay_document_path=overlay_document_path,
        document_id=document_id,
        permitted_root=permitted_root,
        staging_parent=staging_parent,
    )
    _, raw_candidate_chunks = builder.build_from_directory(temp_input_dir)
    candidate_chunks = normalize_overlay_corpus_source_paths(raw_candidate_chunks)

    baseline_doc08 = [c for c in baseline_chunks if c.document_id == document_id]
    candidate_doc08 = [c for c in candidate_chunks if c.document_id == document_id]

    return CorpusOverlayBuildResult(
        baseline_chunks=baseline_chunks,
        candidate_chunks=candidate_chunks,
        baseline_doc08_chunks=baseline_doc08,
        candidate_doc08_chunks=candidate_doc08,
        temp_input_dir=temp_input_dir,
    )


def compute_doc08_chunk_diff(
    baseline_chunks: list[ChunkRecord],
    candidate_chunks: list[ChunkRecord],
    *,
    document_id: str = DOC08_DOCUMENT_ID,
) -> Doc08ChunkDiff:
    """Compare doc08 chunks and verify non-doc08 chunks are byte-identical."""
    baseline_doc08 = {c.chunk_id: c for c in baseline_chunks if c.document_id == document_id}
    candidate_doc08 = {c.chunk_id: c for c in candidate_chunks if c.document_id == document_id}

    baseline_ids = sorted(baseline_doc08)
    candidate_ids = sorted(candidate_doc08)
    added = sorted(set(candidate_ids) - set(baseline_ids))
    removed = sorted(set(baseline_ids) - set(candidate_ids))
    common = set(baseline_ids) & set(candidate_ids)
    changed = sorted(
        chunk_id
        for chunk_id in common
        if baseline_doc08[chunk_id].content != candidate_doc08[chunk_id].content
    )

    baseline_non = {
        c.chunk_id: c.content
        for c in baseline_chunks
        if c.document_id != document_id
    }
    candidate_non = {
        c.chunk_id: c.content
        for c in candidate_chunks
        if c.document_id != document_id
    }
    non_doc08_identical = baseline_non == candidate_non

    return Doc08ChunkDiff(
        baseline_chunk_ids=baseline_ids,
        candidate_chunk_ids=candidate_ids,
        added_chunk_ids=added,
        removed_chunk_ids=removed,
        changed_chunk_ids=changed,
        unchanged_non_doc08_chunk_ids=sorted(baseline_non),
        non_doc08_byte_identical=non_doc08_identical,
    )


def compute_doc08_fingerprint(
    chunks: list[ChunkRecord],
    *,
    document_id: str = DOC08_DOCUMENT_ID,
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """SHA-256 fingerprint of doc08 chunk contents only."""
    doc_chunks = sorted(
        (c for c in chunks if c.document_id == document_id),
        key=lambda c: c.chunk_id,
    )
    payload = [
        {"chunk_id": c.chunk_id, "content": c.content}
        for c in doc_chunks
    ]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cleanup_overlay_temp_dir(temp_dir: Path | None) -> None:
    if temp_dir is not None and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = [
    "DOC08_DOCUMENT_ID",
    "OVERLAY_LOGICAL_SOURCE_DIR",
    "CorpusOverlayBuildResult",
    "Doc08ChunkDiff",
    "build_baseline_and_overlay_chunks",
    "build_overlay_input_directory",
    "cleanup_overlay_temp_dir",
    "compute_doc08_chunk_diff",
    "compute_doc08_fingerprint",
    "compute_overlay_corpus_fingerprint",
    "logical_source_path",
    "normalize_overlay_corpus_source_paths",
]
