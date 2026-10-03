"""Canonical production corpus: an explicit, version-controlled document selection.

The repository holds more cleaned Markdown documents than the product uses. Which of them form the
production knowledge base is decided in one committed file (``configs/corpus/*.json``) and nowhere
else: the build tools read that file, the release posture points at it, and nothing depends on a
directory listing, a private archive or a remembered subset.

What the manifest pins, and what each part proves
-------------------------------------------------
``documents[].source_sha256``
    The exact text of each selected source file (line endings normalised to LF, so the value is the
    same on every platform). Needs no tokenizer; proves *the repository still contains the corpus
    that was approved*.

``expected.chunk_count`` / ``expected.chunk_payload_digest``
    The chunk topology: chunk ids, chunk text, per-chunk metadata and source path, in a form that
    does not depend on any embedding model (``retrieval/index_identity.compute_chunk_payload_digest``).
    Chunk boundaries come from the real ``cl100k_base`` tokenizer, so this is reproducible from the
    repository alone but only when that vocabulary is available (the real-tokenizer test lane and
    the production build tools).

``expected.corpus_fingerprint`` (for ``expected.embedding_model``)
    The existing index fingerprint: the chunk payload plus the *name* of the embedding model and the
    index/metadata format versions. Still a pure function of repository data and configuration, so
    it too is reproducible without OpenAI. It says nothing about vector values.

Embedding vectors and the Chroma representation are not part of this manifest and are not
reproducible from the repository (they come from the OpenAI API); the index manifest attests to
them (``embedding_digest``, ``collection_content_digest``) and the release validator re-checks the
stored vectors against that attestation (``release/index_integrity.py``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from customer_claims_rag.exceptions import CanonicalCorpusError, IngestionError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.path_helpers import resolve_safe_path
from customer_claims_rag.models import ChunkRecord, DocumentRecord
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint
from customer_claims_rag.retrieval.index_identity import compute_chunk_payload_digest
from customer_claims_rag.token_counter import TokenCounter

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0.0"})
DEFAULT_CORPUS_MANIFEST_RELATIVE = Path("configs") / "corpus" / "foodflow_production_v1.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_PATH = re.compile(r"^[a-zA-Z]:[/\\]")


def _require_repository_relative(path: str, *, field: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    if normalized.startswith("/") or _WINDOWS_DRIVE_PATH.match(normalized):
        raise ValueError(f"{field} must be repository-relative, not absolute")
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"{field} must not contain parent-directory traversal")
    return normalized


class CorpusDocument(BaseModel):
    """One document of the canonical corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    file: str
    source_sha256: str

    @field_validator("document_id", "file")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value

    @field_validator("source_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("source_sha256 must be 64 lowercase hex characters")
        return value

    @model_validator(mode="after")
    def validate_file_matches_id(self) -> Self:
        if self.file != f"{self.document_id}.md":
            raise ValueError(f"file must be '<document_id>.md', got {self.file!r}")
        return self


class ExcludedDocument(BaseModel):
    """A document that sits in the source directory but is deliberately not part of the corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: str
    file: str
    reason: str
    evidence: str

    @field_validator("document_id", "file", "reason", "evidence")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence_path(cls, value: str) -> str:
        return _require_repository_relative(value, field="evidence")

    @model_validator(mode="after")
    def validate_file_matches_id(self) -> Self:
        if self.file != f"{self.document_id}.md":
            raise ValueError(f"file must be '<document_id>.md', got {self.file!r}")
        return self


class ExpectedCorpusIdentity(BaseModel):
    """Deterministic identity of the corpus, reproducible without any embedding API."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_count: int
    chunk_count: int
    chunk_payload_digest: str
    embedding_model: str
    corpus_fingerprint: str

    @field_validator("chunk_payload_digest", "corpus_fingerprint")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if not _SHA256.match(value):
            raise ValueError("digest must be 64 lowercase hex characters")
        return value

    @field_validator("embedding_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding_model must be a non-empty string")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.document_count < 1:
            raise ValueError("document_count must be >= 1")
        if self.chunk_count < self.document_count:
            raise ValueError("chunk_count must be >= document_count")
        return self


class CanonicalCorpusManifest(BaseModel):
    """Machine-readable definition of the production corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str
    corpus_id: str
    description: str
    source_dir: str
    tokenizer_encoding: str
    documents: list[CorpusDocument]
    excluded_documents: list[ExcludedDocument]
    expected: ExpectedCorpusIdentity

    @field_validator("schema_version", "corpus_id", "description", "tokenizer_encoding")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value

    @field_validator("source_dir")
    @classmethod
    def validate_source_dir(cls, value: str) -> str:
        return _require_repository_relative(value, field="source_dir")

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported canonical corpus schema_version {self.schema_version!r}")
        if not self.documents:
            raise ValueError("documents must not be empty")
        ids = [document.document_id for document in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("documents must be unique")
        if ids != sorted(ids):
            raise ValueError("documents must be listed in ascending document_id order")
        excluded_ids = [document.document_id for document in self.excluded_documents]
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("excluded_documents must be unique")
        if excluded_ids != sorted(excluded_ids):
            raise ValueError("excluded_documents must be listed in ascending document_id order")
        overlap = sorted(set(ids) & set(excluded_ids))
        if overlap:
            raise ValueError(f"documents cannot be both included and excluded: {overlap}")
        if self.expected.document_count != len(self.documents):
            raise ValueError("expected.document_count must equal the number of documents")
        return self

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(document.document_id for document in self.documents)

    @property
    def file_names(self) -> tuple[str, ...]:
        return tuple(document.file for document in self.documents)

    def source_directory(self, project_root: Path) -> Path:
        return resolve_safe_path(project_root / self.source_dir, root=project_root)


@dataclass(frozen=True)
class CorpusIdentity:
    """What a built corpus is, as computed (not as claimed)."""

    document_ids: tuple[str, ...]
    chunk_count: int
    chunk_payload_digest: str
    embedding_model: str
    corpus_fingerprint: str

    @property
    def document_count(self) -> int:
        return len(self.document_ids)


@dataclass(frozen=True)
class CorpusSourceReport:
    """Comparison of the source directory with the manifest's selection."""

    missing: tuple[str, ...]
    changed: tuple[str, ...]
    undeclared: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.changed or self.undeclared)

    def describe(self) -> str:
        parts: list[str] = []
        if self.missing:
            parts.append(f"missing source files: {list(self.missing)}")
        if self.changed:
            parts.append(
                f"source files differ from the manifest (source_sha256): {list(self.changed)}"
            )
        if self.undeclared:
            parts.append(
                "markdown files that are neither included nor excluded by the manifest: "
                f"{list(self.undeclared)}"
            )
        return "; ".join(parts)


@dataclass(frozen=True)
class CanonicalCorpusBuild:
    """The canonical corpus as built from the repository."""

    manifest: CanonicalCorpusManifest
    documents: list[DocumentRecord]
    chunks: list[ChunkRecord]
    identity: CorpusIdentity
    source_dir: Path


def load_canonical_corpus_manifest(path: Path) -> CanonicalCorpusManifest:
    """Load and validate a canonical corpus manifest."""
    if not path.is_file():
        raise CanonicalCorpusError(f"canonical corpus manifest not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CanonicalCorpusError(f"canonical corpus manifest at {path} is invalid JSON") from exc
    try:
        return CanonicalCorpusManifest.model_validate(payload)
    except ValidationError as exc:
        raise CanonicalCorpusError(
            f"canonical corpus manifest at {path} failed validation: {exc}"
        ) from exc


def normalized_source_text(raw: str) -> str:
    """Line endings folded to LF: the same text on every platform and git configuration."""
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def source_sha256(path: Path) -> str:
    text = normalized_source_text(path.read_text(encoding="utf-8"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_corpus_sources(
    manifest: CanonicalCorpusManifest,
    *,
    project_root: Path,
) -> CorpusSourceReport:
    """Check the source directory against the manifest. Reads files, never writes."""
    source_dir = manifest.source_directory(project_root)
    missing: list[str] = []
    changed: list[str] = []
    for document in manifest.documents:
        candidate = source_dir / document.file
        if not candidate.is_file():
            missing.append(document.file)
        elif source_sha256(candidate) != document.source_sha256:
            changed.append(document.file)
    declared = {document.file for document in manifest.documents} | {
        document.file for document in manifest.excluded_documents
    }
    present = (
        sorted(path.name for path in source_dir.glob("*.md")) if source_dir.is_dir() else []
    )
    undeclared = [name for name in present if name not in declared]
    return CorpusSourceReport(
        missing=tuple(missing),
        changed=tuple(changed),
        undeclared=tuple(undeclared),
    )


def compute_corpus_identity(
    chunks: list[ChunkRecord],
    *,
    embedding_model: str,
) -> CorpusIdentity:
    """Deterministic identity of a chunk set (no embedding call involved)."""
    return CorpusIdentity(
        document_ids=tuple(sorted({chunk.document_id for chunk in chunks})),
        chunk_count=len(chunks),
        chunk_payload_digest=compute_chunk_payload_digest(chunks),
        embedding_model=embedding_model,
        corpus_fingerprint=compute_corpus_fingerprint(chunks, embedding_model=embedding_model),
    )


def identity_differences(identity: CorpusIdentity, expected: ExpectedCorpusIdentity) -> list[str]:
    """Every way ``identity`` departs from ``expected`` (empty when they agree)."""
    if identity.embedding_model != expected.embedding_model:
        return [
            f"embedding_model: manifest pins {expected.embedding_model!r}, "
            f"build used {identity.embedding_model!r}"
        ]
    differences: list[str] = []
    if identity.document_count != expected.document_count:
        differences.append(
            f"document_count: expected {expected.document_count}, got {identity.document_count}"
        )
    if identity.chunk_count != expected.chunk_count:
        differences.append(
            f"chunk_count: expected {expected.chunk_count}, got {identity.chunk_count}"
        )
    if identity.chunk_payload_digest != expected.chunk_payload_digest:
        differences.append(
            f"chunk_payload_digest: expected {expected.chunk_payload_digest}, "
            f"got {identity.chunk_payload_digest}"
        )
    if identity.corpus_fingerprint != expected.corpus_fingerprint:
        differences.append(
            f"corpus_fingerprint: expected {expected.corpus_fingerprint}, "
            f"got {identity.corpus_fingerprint}"
        )
    return differences


def verify_corpus_identity(identity: CorpusIdentity, expected: ExpectedCorpusIdentity) -> None:
    differences = identity_differences(identity, expected)
    if differences:
        raise CanonicalCorpusError(
            "canonical corpus identity mismatch; the corpus built from the repository is not "
            "the one the manifest approves ("
            + "; ".join(differences)
            + "). If the change is intended, review it and refresh the manifest with "
            "`build-chunks --corpus-manifest <manifest> --refresh-manifest`."
        )


def _require_pinned_tokenizer(manifest: CanonicalCorpusManifest, token_counter: TokenCounter) -> None:
    encoding = getattr(token_counter, "encoding_name", None)
    if encoding != manifest.tokenizer_encoding:
        raise CanonicalCorpusError(
            f"canonical corpus {manifest.corpus_id!r} is pinned to the {manifest.tokenizer_encoding!r} "
            f"tokenizer; the token counter in use is {encoding!r}"
        )


def _assemble_build(
    manifest: CanonicalCorpusManifest,
    *,
    project_root: Path,
    token_counter: TokenCounter,
    embedding_model: str,
) -> CanonicalCorpusBuild:
    _require_pinned_tokenizer(manifest, token_counter)
    root = project_root.resolve()
    source_dir = manifest.source_directory(root)
    builder = CorpusBuilder(token_counter=token_counter, permitted_root=root)
    try:
        documents, chunks = builder.build_from_selection(source_dir, manifest.file_names)
    except IngestionError as exc:
        raise CanonicalCorpusError(str(exc)) from exc
    loaded_ids = tuple(document.metadata.document_id for document in documents)
    if loaded_ids != manifest.document_ids:
        raise CanonicalCorpusError(
            f"document ids loaded from the source files {list(loaded_ids)} "
            f"differ from the manifest {list(manifest.document_ids)}"
        )
    return CanonicalCorpusBuild(
        manifest=manifest,
        documents=documents,
        chunks=chunks,
        identity=compute_corpus_identity(chunks, embedding_model=embedding_model),
        source_dir=source_dir,
    )


def build_canonical_corpus(
    manifest: CanonicalCorpusManifest,
    *,
    project_root: Path,
    token_counter: TokenCounter,
    embedding_model: str | None = None,
) -> CanonicalCorpusBuild:
    """Build the canonical corpus and verify it against the manifest before returning it.

    Fails closed, before anything is embedded or written: the sources must match their recorded
    hashes, every ``.md`` file must be declared, the tokenizer must be the pinned one, and the
    resulting chunk topology and fingerprint must equal the manifest's ``expected`` block.
    """
    model = embedding_model or manifest.expected.embedding_model
    if model != manifest.expected.embedding_model:
        raise CanonicalCorpusError(
            f"canonical corpus {manifest.corpus_id!r} is pinned to embedding model "
            f"{manifest.expected.embedding_model!r}; got {model!r}. Use --input-dir for "
            "experiments with other models."
        )
    report = verify_corpus_sources(manifest, project_root=project_root.resolve())
    if not report.ok:
        raise CanonicalCorpusError(
            "canonical corpus sources do not match the manifest: "
            + report.describe()
            + ". If the change is intended, review it and refresh the manifest with "
            "`build-chunks --corpus-manifest <manifest> --refresh-manifest`."
        )
    build = _assemble_build(
        manifest,
        project_root=project_root,
        token_counter=token_counter,
        embedding_model=model,
    )
    verify_corpus_identity(build.identity, manifest.expected)
    return build


def write_canonical_corpus_manifest(path: Path, manifest: CanonicalCorpusManifest) -> None:
    """Write a manifest atomically, in the canonical key order and with LF line endings."""
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    fd, temp_name = tempfile.mkstemp(suffix=".json", dir=str(path.parent), text=True)
    temp_path = Path(temp_name)
    try:
        os.close(fd)
        temp_path.write_text(payload, encoding="utf-8", newline="\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def refresh_canonical_corpus_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    token_counter: TokenCounter,
) -> CanonicalCorpusBuild:
    """Recompute ``source_sha256`` and ``expected`` from the repository (writes nothing).

    The returned build carries the refreshed manifest; persist it with
    ``write_canonical_corpus_manifest``. This is the deliberate, reviewable way to accept a change
    to the corpus: the document *selection* (which files are included or excluded) is never
    touched, only the values derived from it, and the resulting diff is part of the commit that
    changes the knowledge base.
    """
    manifest = load_canonical_corpus_manifest(manifest_path)
    root = project_root.resolve()
    source_dir = manifest.source_directory(root)
    refreshed_documents: list[CorpusDocument] = []
    for document in manifest.documents:
        candidate = source_dir / document.file
        if not candidate.is_file():
            raise CanonicalCorpusError(f"cannot refresh: source file {document.file!r} is missing")
        refreshed_documents.append(
            document.model_copy(update={"source_sha256": source_sha256(candidate)})
        )
    selection = manifest.model_copy(update={"documents": refreshed_documents})
    undeclared = verify_corpus_sources(selection, project_root=root).undeclared
    if undeclared:
        raise CanonicalCorpusError(
            "cannot refresh: markdown files that are neither included nor excluded by the "
            f"manifest: {list(undeclared)}; declare them first"
        )
    build = _assemble_build(
        selection,
        project_root=root,
        token_counter=token_counter,
        embedding_model=manifest.expected.embedding_model,
    )
    expected = ExpectedCorpusIdentity(
        document_count=build.identity.document_count,
        chunk_count=build.identity.chunk_count,
        chunk_payload_digest=build.identity.chunk_payload_digest,
        embedding_model=build.identity.embedding_model,
        corpus_fingerprint=build.identity.corpus_fingerprint,
    )
    refreshed = CanonicalCorpusManifest.model_validate(
        selection.model_copy(update={"expected": expected}).model_dump(mode="python")
    )
    return CanonicalCorpusBuild(
        manifest=refreshed,
        documents=build.documents,
        chunks=build.chunks,
        identity=build.identity,
        source_dir=build.source_dir,
    )
