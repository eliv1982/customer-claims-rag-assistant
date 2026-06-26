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
DOC12_DOCUMENT_ID = "12_staff_safety_and_threat_handling"
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
class DocumentChunkDiff:
    """Machine-readable diff for one overlay document between two corpora."""

    document_id: str
    baseline_chunk_ids: list[str]
    candidate_chunk_ids: list[str]
    added_chunk_ids: list[str]
    removed_chunk_ids: list[str]
    changed_chunk_ids: list[str]
    unchanged_non_target_chunk_ids: list[str]
    non_target_byte_identical: bool


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


@dataclass(frozen=True)
class MultiOverlayBuildResult:
    """Baseline experimental corpus vs candidate with stacked overlays."""

    baseline_chunks: list[ChunkRecord]
    candidate_chunks: list[ChunkRecord]
    baseline_doc12_chunks: list[ChunkRecord]
    candidate_doc12_chunks: list[ChunkRecord]
    baseline_doc08_fingerprint: str
    candidate_doc08_fingerprint: str
    temp_input_dir: Path | None


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
    return build_multi_overlay_input_directory(
        canonical_dir=canonical_dir,
        overlays={document_id: overlay_document_path},
        permitted_root=permitted_root,
        staging_parent=staging_parent,
    )


def build_multi_overlay_input_directory(
    *,
    canonical_dir: Path,
    overlays: dict[str, Path],
    permitted_root: Path,
    staging_parent: Path | None = None,
) -> Path:
    """Create a temp directory applying multiple document overlays onto canonical corpus."""
    canonical_resolved = canonical_dir.resolve()
    resolved_overlays: dict[str, Path] = {}
    for document_id, overlay_path in overlays.items():
        overlay_resolved = overlay_path.resolve()
        if not overlay_resolved.is_file():
            raise IngestionError(f"overlay document not found: {overlay_resolved}")
        expected_name = f"{document_id}.md"
        if overlay_resolved.name != expected_name:
            raise IngestionError(
                f"overlay file must be named {expected_name!r}, got {overlay_resolved.name!r}"
            )
        resolved_overlays[document_id] = overlay_resolved

    if staging_parent is not None:
        staging_root = staging_parent.resolve()
        staging_root.mkdir(parents=True, exist_ok=True)
    else:
        staging_root = permitted_root / ".tmp" / "corpus_overlay"
        staging_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="build_", dir=str(staging_root)))
    for source in sorted(canonical_resolved.glob("*.md")):
        dest = temp_dir / source.name
        if source.stem in resolved_overlays:
            shutil.copy2(resolved_overlays[source.stem], dest)
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


def compute_document_chunk_diff(
    baseline_chunks: list[ChunkRecord],
    candidate_chunks: list[ChunkRecord],
    *,
    document_id: str,
) -> DocumentChunkDiff:
    """Compare one document's chunks and verify all other documents are byte-identical."""
    baseline_target = {c.chunk_id: c for c in baseline_chunks if c.document_id == document_id}
    candidate_target = {c.chunk_id: c for c in candidate_chunks if c.document_id == document_id}

    baseline_ids = sorted(baseline_target)
    candidate_ids = sorted(candidate_target)
    added = sorted(set(candidate_ids) - set(baseline_ids))
    removed = sorted(set(baseline_ids) - set(candidate_ids))
    common = set(baseline_ids) & set(candidate_ids)
    changed = sorted(
        chunk_id
        for chunk_id in common
        if baseline_target[chunk_id].content != candidate_target[chunk_id].content
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
    non_target_identical = baseline_non == candidate_non

    return DocumentChunkDiff(
        document_id=document_id,
        baseline_chunk_ids=baseline_ids,
        candidate_chunk_ids=candidate_ids,
        added_chunk_ids=added,
        removed_chunk_ids=removed,
        changed_chunk_ids=changed,
        unchanged_non_target_chunk_ids=sorted(baseline_non),
        non_target_byte_identical=non_target_identical,
    )


def compute_doc08_chunk_diff(
    baseline_chunks: list[ChunkRecord],
    candidate_chunks: list[ChunkRecord],
    *,
    document_id: str = DOC08_DOCUMENT_ID,
) -> Doc08ChunkDiff:
    """Compare doc08 chunks and verify non-doc08 chunks are byte-identical."""
    diff = compute_document_chunk_diff(
        baseline_chunks,
        candidate_chunks,
        document_id=document_id,
    )
    return Doc08ChunkDiff(
        baseline_chunk_ids=diff.baseline_chunk_ids,
        candidate_chunk_ids=diff.candidate_chunk_ids,
        added_chunk_ids=diff.added_chunk_ids,
        removed_chunk_ids=diff.removed_chunk_ids,
        changed_chunk_ids=diff.changed_chunk_ids,
        unchanged_non_doc08_chunk_ids=diff.unchanged_non_target_chunk_ids,
        non_doc08_byte_identical=diff.non_target_byte_identical,
    )


def compute_doc12_chunk_diff(
    baseline_chunks: list[ChunkRecord],
    candidate_chunks: list[ChunkRecord],
    *,
    document_id: str = DOC12_DOCUMENT_ID,
) -> DocumentChunkDiff:
    """Compare doc12 chunks between experimental baseline and doc12-overlay candidate."""
    return compute_document_chunk_diff(
        baseline_chunks,
        candidate_chunks,
        document_id=document_id,
    )


def compute_document_fingerprint(
    chunks: list[ChunkRecord],
    *,
    document_id: str,
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """SHA-256 fingerprint of one document's chunk contents only."""
    _ = embedding_model
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


def compute_doc08_fingerprint(
    chunks: list[ChunkRecord],
    *,
    document_id: str = DOC08_DOCUMENT_ID,
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """SHA-256 fingerprint of doc08 chunk contents only."""
    return compute_document_fingerprint(
        chunks,
        document_id=document_id,
        embedding_model=embedding_model,
    )


def compute_doc12_fingerprint(
    chunks: list[ChunkRecord],
    *,
    document_id: str = DOC12_DOCUMENT_ID,
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """SHA-256 fingerprint of doc12 chunk contents only."""
    return compute_document_fingerprint(
        chunks,
        document_id=document_id,
        embedding_model=embedding_model,
    )


def build_doc12_experimental_corpora(
    *,
    canonical_dir: Path,
    doc08_overlay_path: Path,
    doc12_overlay_path: Path | None = None,
    permitted_root: Path,
    staging_parent: Path | None = None,
) -> MultiOverlayBuildResult:
    """Build doc08 experimental baseline corpus and optional doc12-overlay candidate."""
    builder = CorpusBuilder(permitted_root=permitted_root)

    baseline_temp = build_multi_overlay_input_directory(
        canonical_dir=canonical_dir,
        overlays={DOC08_DOCUMENT_ID: doc08_overlay_path},
        permitted_root=permitted_root,
        staging_parent=staging_parent,
    )
    _, raw_baseline = builder.build_from_directory(baseline_temp)
    baseline_chunks = normalize_overlay_corpus_source_paths(raw_baseline)
    cleanup_overlay_temp_dir(baseline_temp)

    candidate_overlays = {DOC08_DOCUMENT_ID: doc08_overlay_path}
    if doc12_overlay_path is not None:
        candidate_overlays[DOC12_DOCUMENT_ID] = doc12_overlay_path

    candidate_temp = build_multi_overlay_input_directory(
        canonical_dir=canonical_dir,
        overlays=candidate_overlays,
        permitted_root=permitted_root,
        staging_parent=staging_parent,
    )
    _, raw_candidate = builder.build_from_directory(candidate_temp)
    candidate_chunks = normalize_overlay_corpus_source_paths(raw_candidate)

    baseline_doc12 = [c for c in baseline_chunks if c.document_id == DOC12_DOCUMENT_ID]
    candidate_doc12 = [c for c in candidate_chunks if c.document_id == DOC12_DOCUMENT_ID]

    return MultiOverlayBuildResult(
        baseline_chunks=baseline_chunks,
        candidate_chunks=candidate_chunks,
        baseline_doc12_chunks=baseline_doc12,
        candidate_doc12_chunks=candidate_doc12,
        baseline_doc08_fingerprint=compute_doc08_fingerprint(baseline_chunks),
        candidate_doc08_fingerprint=compute_doc08_fingerprint(candidate_chunks),
        temp_input_dir=candidate_temp,
    )


def verify_doc08_overlay_unchanged(
    baseline_chunks: list[ChunkRecord],
    candidate_chunks: list[ChunkRecord],
    *,
    expected_doc08_fingerprint: str | None = None,
) -> None:
    """Ensure doc08 overlay bytes are identical between experimental arms."""
    baseline_fp = compute_doc08_fingerprint(baseline_chunks)
    candidate_fp = compute_doc08_fingerprint(candidate_chunks)
    if baseline_fp != candidate_fp:
        raise IngestionError("doc08 overlay fingerprint differs between baseline and candidate")
    if expected_doc08_fingerprint is not None and candidate_fp != expected_doc08_fingerprint:
        raise IngestionError(
            f"doc08 fingerprint {candidate_fp} != expected {expected_doc08_fingerprint}"
        )
    baseline_non_doc12 = {
        c.chunk_id: c.content
        for c in baseline_chunks
        if c.document_id != DOC12_DOCUMENT_ID
    }
    candidate_non_doc12 = {
        c.chunk_id: c.content
        for c in candidate_chunks
        if c.document_id != DOC12_DOCUMENT_ID
    }
    if baseline_non_doc12 != candidate_non_doc12:
        raise IngestionError("non-doc12 chunks differ between experimental baseline and candidate")


def cleanup_overlay_temp_dir(temp_dir: Path | None) -> None:
    if temp_dir is not None and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = [
    "DOC08_DOCUMENT_ID",
    "DOC12_DOCUMENT_ID",
    "OVERLAY_LOGICAL_SOURCE_DIR",
    "CorpusOverlayBuildResult",
    "Doc08ChunkDiff",
    "DocumentChunkDiff",
    "MultiOverlayBuildResult",
    "build_baseline_and_overlay_chunks",
    "build_doc12_experimental_corpora",
    "build_multi_overlay_input_directory",
    "build_overlay_input_directory",
    "cleanup_overlay_temp_dir",
    "compute_doc08_chunk_diff",
    "compute_doc08_fingerprint",
    "compute_doc12_chunk_diff",
    "compute_doc12_fingerprint",
    "compute_document_chunk_diff",
    "compute_document_fingerprint",
    "compute_overlay_corpus_fingerprint",
    "logical_source_path",
    "normalize_overlay_corpus_source_paths",
    "verify_doc08_overlay_unchanged",
]
