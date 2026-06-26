"""Production release posture descriptor loading and fail-closed validation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from customer_claims_rag import env_bootstrap
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    compute_pool_expansion_config_hash,
    load_pool_expansion_config,
)
from customer_claims_rag.exceptions import IndexManifestError, ReleasePostureError
from customer_claims_rag.ingestion.path_helpers import resolve_safe_path, to_relative_posix_path
from customer_claims_rag.retrieval.chroma_storage import (
    CHROMA_SQLITE_FILENAME,
    chroma_sqlite_path,
    validate_open_existing_chroma_preconditions,
)
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.ports import VectorStore

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0.0"})
DEFAULT_DESCRIPTOR_RELATIVE = Path("configs") / "release" / "production_posture.json"
DEFAULT_DESCRIPTOR_PATH = env_bootstrap.project_root() / DEFAULT_DESCRIPTOR_RELATIVE
RELEASE_TARGET_ENV = "RAG_RELEASE_TARGET"

_WINDOWS_DRIVE_PATH = re.compile(r"^[a-zA-Z]:[/\\]")
_POSIX_ABSOLUTE_PATH = re.compile(r"^/")
_UNC_PATH = re.compile(r"^\\\\")

FROZEN_VECTOR_FETCH_K = 24
FROZEN_CANDIDATE_POOL_K = 24
FROZEN_FINAL_TOP_K = 12
FROZEN_SIMILARITY_THRESHOLD = 0.0
FROZEN_RERANKER_ID = "source-authority-v1"


class _ReleaseTargetDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: str
    index_path: str
    expected_corpus_fingerprint: str
    expected_chunk_count: int
    expected_document_count: int
    supported_document_ids: list[str]
    collection_name: str
    embedding_model: str
    vector_dimension: int

    @field_validator("index_path")
    @classmethod
    def validate_repository_relative_index_path(cls, value: str) -> str:
        return _validate_repository_relative_index_path(value)

    @field_validator("status", "expected_corpus_fingerprint", "collection_name", "embedding_model")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value

    @field_validator("supported_document_ids")
    @classmethod
    def validate_supported_document_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("supported_document_ids must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("supported_document_ids must be unique")
        for document_id in value:
            if not document_id.strip():
                raise ValueError("supported_document_ids must not contain empty values")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.expected_chunk_count < 1:
            raise ValueError("expected_chunk_count must be >= 1")
        if self.expected_document_count < 1:
            raise ValueError("expected_document_count must be >= 1")
        if self.vector_dimension < 1:
            raise ValueError("vector_dimension must be >= 1")
        if len(self.supported_document_ids) != self.expected_document_count:
            raise ValueError(
                "supported_document_ids length must equal expected_document_count"
            )
        return self


class ReleasePostureDescriptor(BaseModel):
    """Machine-readable production release posture."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: str
    release_posture_id: str
    default_target: str
    limitations_doc: str
    frozen_retrieval_config_path: str
    expected_frozen_config_hash: str
    targets: dict[str, _ReleaseTargetDescriptor]

    @field_validator(
        "schema_version",
        "release_posture_id",
        "default_target",
        "limitations_doc",
        "frozen_retrieval_config_path",
        "expected_frozen_config_hash",
    )
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        if not self.targets:
            raise ValueError("targets must not be empty")
        if self.default_target not in self.targets:
            raise ValueError("default_target must name a configured target")
        return self


@dataclass(frozen=True)
class ProductionReleaseContext:
    """Shared production release resolution used by startup and validator."""

    descriptor: ReleasePostureDescriptor
    resolved_target: ResolvedReleaseTarget
    descriptor_path: Path
    project_root: Path

    @property
    def frozen_retrieval_config_path(self) -> Path:
        return _resolve_repository_path(
            self.descriptor.frozen_retrieval_config_path,
            project_root=self.project_root,
        )


@dataclass(frozen=True)
class ResolvedReleaseTarget:
    """Validated release target ready for production wiring."""

    target_name: str
    status: str
    index_dir: Path
    index_path_relative: str
    expected_corpus_fingerprint: str
    expected_chunk_count: int
    expected_document_count: int
    supported_document_ids: tuple[str, ...]
    collection_name: str
    embedding_model: str
    vector_dimension: int


@dataclass(frozen=True)
class ReleasePostureDiagnostics:
    """Safe release identity for startup logging and validation CLI."""

    release_posture_id: str
    selected_target: str
    target_status: str
    index_path_relative: str
    corpus_fingerprint: str
    chunk_count: int
    document_count: int
    collection_name: str
    embedding_model: str
    vector_dimension: int
    frozen_retrieval_config_path: str
    frozen_config_hash: str
    vector_fetch_k: int
    candidate_pool_k: int
    final_top_k: int
    similarity_threshold: float
    reranker_id: str

    def format_lines(self) -> list[str]:
        return [
            f"release_posture_id={self.release_posture_id}",
            f"selected_target={self.selected_target}",
            f"target_status={self.target_status}",
            f"index_path={self.index_path_relative}",
            f"corpus_fingerprint={self.corpus_fingerprint}",
            f"chunk_count={self.chunk_count}",
            f"document_count={self.document_count}",
            f"collection={self.collection_name}",
            f"embedding_model={self.embedding_model}",
            f"vector_dimension={self.vector_dimension}",
            f"frozen_retrieval_config_path={self.frozen_retrieval_config_path}",
            f"frozen_config_hash={self.frozen_config_hash}",
            (
                "retrieval_contract="
                f"{self.vector_fetch_k}/{self.candidate_pool_k}/"
                f"{self.final_top_k}/{self.similarity_threshold}"
            ),
            f"reranker_id={self.reranker_id}",
        ]


def load_release_posture_descriptor(path: Path | None = None) -> ReleasePostureDescriptor:
    """Load and parse the committed release posture descriptor."""
    descriptor_path = path or DEFAULT_DESCRIPTOR_PATH
    if not descriptor_path.is_file():
        raise ReleasePostureError(
            f"release posture descriptor not found at {descriptor_path}"
        )
    try:
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleasePostureError(
            f"release posture descriptor at {descriptor_path} is invalid JSON"
        ) from exc
    try:
        descriptor = ReleasePostureDescriptor.model_validate(payload)
    except Exception as exc:
        raise ReleasePostureError(
            f"release posture descriptor at {descriptor_path} failed validation: {exc}"
        ) from exc
    return descriptor


def resolve_production_release_posture(
    *,
    descriptor_path: Path | None = None,
    project_root: Path | None = None,
    target_name: str | None = None,
) -> ProductionReleaseContext:
    """Resolve the production release target from descriptor and environment."""
    root = (project_root or env_bootstrap.project_root()).resolve()
    resolved_descriptor_path = descriptor_path or (root / DEFAULT_DESCRIPTOR_RELATIVE)
    descriptor = load_release_posture_descriptor(resolved_descriptor_path)
    _validate_descriptor_target_path_distinctness(descriptor, project_root=root)

    if target_name is None:
        selected_target = resolve_release_target_name(descriptor)
    else:
        selected_target = target_name.strip()
        if not selected_target:
            raise ReleasePostureError("release target name must not be empty")
        if selected_target not in descriptor.targets:
            known = ", ".join(sorted(descriptor.targets))
            raise ReleasePostureError(
                f"unknown release target {selected_target!r}; expected one of: {known}"
            )
        if descriptor.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ReleasePostureError(
                f"unsupported release posture schema_version {descriptor.schema_version!r}"
            )

    resolved_target = resolve_release_target(descriptor, selected_target, project_root=root)
    return ProductionReleaseContext(
        descriptor=descriptor,
        resolved_target=resolved_target,
        descriptor_path=resolved_descriptor_path,
        project_root=root,
    )


def load_production_release_context(
    *,
    descriptor_path: Path | None = None,
    project_root: Path | None = None,
) -> tuple[ReleasePostureDescriptor, ResolvedReleaseTarget]:
    """Load descriptor and resolve the production release target."""
    context = resolve_production_release_posture(
        descriptor_path=descriptor_path,
        project_root=project_root,
    )
    return context.descriptor, context.resolved_target


def _validate_repository_relative_index_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("index_path must be a non-empty repository-relative path")
    if _WINDOWS_DRIVE_PATH.match(normalized) or _POSIX_ABSOLUTE_PATH.match(normalized):
        raise ValueError("index_path must be repository-relative, not absolute")
    if _UNC_PATH.match(path.strip()):
        raise ValueError("index_path must not be a UNC path")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise ValueError("index_path must not contain parent-directory traversal")
    return normalized


def _validate_descriptor_target_path_distinctness(
    descriptor: ReleasePostureDescriptor,
    *,
    project_root: Path,
) -> None:
    canonical_by_target: dict[str, Path] = {}
    for target_name, target in descriptor.targets.items():
        resolved = _resolve_repository_path(target.index_path, project_root=project_root)
        canonical = resolved.resolve()
        for other_name, other_canonical in canonical_by_target.items():
            if canonical == other_canonical:
                raise ReleasePostureError(
                    f"release targets {target_name!r} and {other_name!r} resolve to the "
                    f"same physical index path: {to_relative_posix_path(canonical, project_root)}"
                )
        canonical_by_target[target_name] = canonical


def _resolve_repository_path(relative_path: str, *, project_root: Path) -> Path:
    _validate_repository_relative_index_path(relative_path)
    root = project_root.resolve()
    try:
        return resolve_safe_path(root / relative_path, root=root)
    except Exception as exc:
        raise ReleasePostureError(
            f"release posture path {relative_path!r} is outside repository root"
        ) from exc


def resolve_release_target_name(descriptor: ReleasePostureDescriptor) -> str:
    """Resolve explicit release target name from environment or descriptor default."""
    raw = os.environ.get(RELEASE_TARGET_ENV)
    target_name = descriptor.default_target if raw is None or not raw.strip() else raw.strip()
    if target_name not in descriptor.targets:
        known = ", ".join(sorted(descriptor.targets))
        raise ReleasePostureError(
            f"unknown release target {target_name!r}; expected one of: {known}"
        )
    if descriptor.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ReleasePostureError(
            f"unsupported release posture schema_version {descriptor.schema_version!r}"
        )
    return target_name


def resolve_release_target(
    descriptor: ReleasePostureDescriptor,
    target_name: str,
    *,
    project_root: Path,
) -> ResolvedReleaseTarget:
    """Resolve a named release target to validated absolute paths."""
    if descriptor.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ReleasePostureError(
            f"unsupported release posture schema_version {descriptor.schema_version!r}"
        )
    if target_name not in descriptor.targets:
        known = ", ".join(sorted(descriptor.targets))
        raise ReleasePostureError(
            f"unknown release target {target_name!r}; expected one of: {known}"
        )

    target = descriptor.targets[target_name]
    index_dir = _resolve_repository_path(target.index_path, project_root=project_root)
    return ResolvedReleaseTarget(
        target_name=target_name,
        status=target.status,
        index_dir=index_dir,
        index_path_relative=to_relative_posix_path(index_dir, project_root),
        expected_corpus_fingerprint=target.expected_corpus_fingerprint,
        expected_chunk_count=target.expected_chunk_count,
        expected_document_count=target.expected_document_count,
        supported_document_ids=tuple(target.supported_document_ids),
        collection_name=target.collection_name,
        embedding_model=target.embedding_model,
        vector_dimension=target.vector_dimension,
    )


def validate_frozen_retrieval_contract(
    frozen_config,
    *,
    expected_hash: str,
    actual_hash: str,
) -> None:
    if actual_hash != expected_hash:
        raise ReleasePostureError(
            f"frozen retrieval config hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if frozen_config.vector_fetch_k != FROZEN_VECTOR_FETCH_K:
        raise ReleasePostureError(
            f"frozen vector_fetch_k must be {FROZEN_VECTOR_FETCH_K}, "
            f"got {frozen_config.vector_fetch_k}"
        )
    if frozen_config.candidate_pool_k != FROZEN_CANDIDATE_POOL_K:
        raise ReleasePostureError(
            f"frozen candidate_pool_k must be {FROZEN_CANDIDATE_POOL_K}, "
            f"got {frozen_config.candidate_pool_k}"
        )
    if frozen_config.final_top_k != FROZEN_FINAL_TOP_K:
        raise ReleasePostureError(
            f"frozen final_top_k must be {FROZEN_FINAL_TOP_K}, got {frozen_config.final_top_k}"
        )
    if frozen_config.similarity_threshold != FROZEN_SIMILARITY_THRESHOLD:
        raise ReleasePostureError(
            f"frozen similarity_threshold must be {FROZEN_SIMILARITY_THRESHOLD}, "
            f"got {frozen_config.similarity_threshold}"
        )
    if frozen_config.reranker_id != FROZEN_RERANKER_ID:
        raise ReleasePostureError(
            f"frozen reranker_id must be {FROZEN_RERANKER_ID!r}, "
            f"got {frozen_config.reranker_id!r}"
        )


def _collect_document_ids(vector_store: VectorStore) -> set[str]:
    list_document_ids = getattr(vector_store, "list_document_ids", None)
    if callable(list_document_ids):
        return set(list_document_ids())
    raise ReleasePostureError(
        "vector store does not expose document IDs for release posture validation"
    )


def validate_release_posture_for_production(
    descriptor: ReleasePostureDescriptor,
    resolved_target: ResolvedReleaseTarget,
    *,
    project_root: Path,
    vector_store: VectorStore | None = None,
    frozen_retrieval_config_path: Path | None = None,
) -> ReleasePostureDiagnostics:
    """Fail-closed validation before production opens or queries the vector store."""
    if not resolved_target.index_dir.is_dir():
        raise ReleasePostureError(
            f"release target {resolved_target.target_name!r} index directory does not "
            f"exist: {resolved_target.index_path_relative}; provision the index locally"
        )
    try:
        validate_open_existing_chroma_preconditions(
            resolved_target.index_dir,
            collection_name=resolved_target.collection_name,
        )
    except Exception as exc:
        raise ReleasePostureError(str(exc)) from exc

    if frozen_retrieval_config_path is not None:
        frozen_config_path = frozen_retrieval_config_path.resolve()
        if not frozen_config_path.is_file():
            raise ReleasePostureError(
                f"frozen retrieval config not found at {frozen_config_path}"
            )
    else:
        frozen_config_path = _resolve_repository_path(
            descriptor.frozen_retrieval_config_path,
            project_root=project_root,
        )
        if not frozen_config_path.is_file():
            raise ReleasePostureError(
                f"frozen retrieval config not found at {descriptor.frozen_retrieval_config_path}"
            )

    pool_config = load_pool_expansion_config(frozen_config_path)
    frozen_hash = compute_pool_expansion_config_hash(pool_config)
    from customer_claims_rag.application.settings import load_frozen_retrieval_config

    frozen_config = load_frozen_retrieval_config(frozen_config_path)
    validate_frozen_retrieval_contract(
        frozen_config,
        expected_hash=descriptor.expected_frozen_config_hash,
        actual_hash=frozen_hash,
    )

    try:
        manifest = load_manifest(resolved_target.index_dir)
    except IndexManifestError as exc:
        raise ReleasePostureError(str(exc)) from exc
    if manifest.corpus_fingerprint != resolved_target.expected_corpus_fingerprint:
        raise ReleasePostureError(
            "corpus fingerprint mismatch for release target "
            f"{resolved_target.target_name!r}: manifest has "
            f"{manifest.corpus_fingerprint}, expected "
            f"{resolved_target.expected_corpus_fingerprint}"
        )
    if manifest.chunk_count != resolved_target.expected_chunk_count:
        raise ReleasePostureError(
            f"chunk count mismatch for release target {resolved_target.target_name!r}: "
            f"manifest has {manifest.chunk_count}, expected "
            f"{resolved_target.expected_chunk_count}"
        )
    if manifest.document_count != resolved_target.expected_document_count:
        raise ReleasePostureError(
            f"document count mismatch for release target {resolved_target.target_name!r}: "
            f"manifest has {manifest.document_count}, expected "
            f"{resolved_target.expected_document_count}"
        )
    if manifest.collection_name != resolved_target.collection_name:
        raise ReleasePostureError(
            f"collection name mismatch: manifest has {manifest.collection_name!r}, "
            f"expected {resolved_target.collection_name!r}"
        )
    if manifest.embedding_model != resolved_target.embedding_model:
        raise ReleasePostureError(
            f"embedding model mismatch: manifest has {manifest.embedding_model!r}, "
            f"expected {resolved_target.embedding_model!r}"
        )
    if manifest.vector_dimension != resolved_target.vector_dimension:
        raise ReleasePostureError(
            f"vector dimension mismatch: manifest has {manifest.vector_dimension}, "
            f"expected {resolved_target.vector_dimension}"
        )

    if vector_store is not None:
        if vector_store.collection_name != resolved_target.collection_name:
            raise ReleasePostureError(
                f"vector store collection mismatch: runtime has "
                f"{vector_store.collection_name!r}, expected "
                f"{resolved_target.collection_name!r}"
            )
        store_count = vector_store.count()
        if store_count != resolved_target.expected_chunk_count:
            raise ReleasePostureError(
                f"vector store chunk count mismatch: store has {store_count}, expected "
                f"{resolved_target.expected_chunk_count}"
            )
        actual_document_ids = _collect_document_ids(vector_store)
        expected_document_ids = set(resolved_target.supported_document_ids)
        if actual_document_ids != expected_document_ids:
            missing = sorted(expected_document_ids - actual_document_ids)
            extra = sorted(actual_document_ids - expected_document_ids)
            raise ReleasePostureError(
                "supported document IDs mismatch for release target "
                f"{resolved_target.target_name!r}: missing={missing!r}, extra={extra!r}"
            )

    return ReleasePostureDiagnostics(
        release_posture_id=descriptor.release_posture_id,
        selected_target=resolved_target.target_name,
        target_status=resolved_target.status,
        index_path_relative=resolved_target.index_path_relative,
        corpus_fingerprint=manifest.corpus_fingerprint,
        chunk_count=manifest.chunk_count,
        document_count=manifest.document_count,
        collection_name=manifest.collection_name,
        embedding_model=manifest.embedding_model,
        vector_dimension=manifest.vector_dimension or resolved_target.vector_dimension,
        frozen_retrieval_config_path=descriptor.frozen_retrieval_config_path,
        frozen_config_hash=frozen_hash,
        vector_fetch_k=frozen_config.vector_fetch_k,
        candidate_pool_k=frozen_config.candidate_pool_k,
        final_top_k=frozen_config.final_top_k,
        similarity_threshold=frozen_config.similarity_threshold,
        reranker_id=frozen_config.reranker_id,
    )


def validate_production_release_posture(
    context: ProductionReleaseContext,
    *,
    vector_store: VectorStore | None = None,
) -> ReleasePostureDiagnostics:
    """Validate a resolved production release context."""
    return validate_release_posture_for_production(
        context.descriptor,
        context.resolved_target,
        project_root=context.project_root,
        vector_store=vector_store,
        frozen_retrieval_config_path=context.frozen_retrieval_config_path,
    )
