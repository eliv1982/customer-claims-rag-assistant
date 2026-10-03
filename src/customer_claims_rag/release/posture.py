"""Production release posture descriptor loading and fail-closed validation.

A release target names *which canonical corpus* it serves (``corpus_manifest``), *where* its vector
index lives and *which embedding configuration* the index was built with. Everything that identifies
the corpus (document ids, chunk count, chunk payload digest, corpus fingerprint) is read from the
canonical corpus manifest and never repeated here, so there is one place to change it.

Validation separates what is recomputed from what is merely attested:

* the descriptor and the canonical corpus manifest are committed and checked against each other;
* the index ``manifest.json`` is self-attestation: its claims are compared with the committed
  expectations, but a claim alone never passes;
* when the vector store is open, the identity values are recomputed from the stored records
  (``release/index_integrity.py``) and compared with both the manifest and the committed
  expectations.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from customer_claims_rag import env_bootstrap
from customer_claims_rag.config import INDEX_FORMAT_VERSION, METADATA_SCHEMA_VERSION
from customer_claims_rag.evaluation.pool_expansion_metrics import (
    compute_pool_expansion_config_hash,
    load_pool_expansion_config,
)
from customer_claims_rag.exceptions import (
    CanonicalCorpusError,
    IndexManifestError,
    ReleasePostureError,
)
from customer_claims_rag.ingestion.canonical_corpus import (
    CanonicalCorpusManifest,
    load_canonical_corpus_manifest,
)
from customer_claims_rag.ingestion.path_helpers import resolve_safe_path, to_relative_posix_path
from customer_claims_rag.release.index_integrity import recompute_store_identity
from customer_claims_rag.retrieval.chroma_storage import (
    validate_open_existing_chroma_preconditions,
)
from customer_claims_rag.retrieval.manifest import load_manifest
from customer_claims_rag.retrieval.ports import VectorStore

SUPPORTED_SCHEMA_VERSIONS = frozenset({"2.0.0"})
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

INTEGRITY_STORE_RECOMPUTED = "store_recomputed"
INTEGRITY_MANIFEST_ONLY = "manifest_self_attestation_only"


class _ReleaseTargetDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: str
    corpus_manifest: str
    index_path: str
    collection_name: str
    embedding_model: str
    vector_dimension: int

    @field_validator("index_path", "corpus_manifest")
    @classmethod
    def validate_repository_relative_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)

    @field_validator("status", "collection_name", "embedding_model")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be a non-empty string")
        return value

    @model_validator(mode="after")
    def validate_dimension(self) -> Self:
        if self.vector_dimension < 1:
            raise ValueError("vector_dimension must be >= 1")
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
    """Validated release target ready for production wiring.

    ``expected_*`` and ``supported_document_ids`` come from the canonical corpus manifest the
    target references, not from the descriptor.
    """

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
    corpus_id: str
    corpus_manifest_path: Path
    corpus_manifest_relative: str
    expected_chunk_payload_digest: str


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
    corpus_id: str = ""
    chunk_payload_digest: str = ""
    integrity: str = INTEGRITY_MANIFEST_ONLY

    def format_lines(self) -> list[str]:
        return [
            f"release_posture_id={self.release_posture_id}",
            f"selected_target={self.selected_target}",
            f"target_status={self.target_status}",
            f"canonical_corpus_id={self.corpus_id}",
            f"index_path={self.index_path_relative}",
            f"corpus_fingerprint={self.corpus_fingerprint}",
            f"chunk_payload_digest={self.chunk_payload_digest}",
            f"chunk_count={self.chunk_count}",
            f"document_count={self.document_count}",
            f"collection={self.collection_name}",
            f"embedding_model={self.embedding_model}",
            f"vector_dimension={self.vector_dimension}",
            f"index_integrity={self.integrity}",
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


def _validate_repository_relative_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("path must be a non-empty repository-relative path")
    if _WINDOWS_DRIVE_PATH.match(normalized) or _POSIX_ABSOLUTE_PATH.match(normalized):
        raise ValueError("path must be repository-relative, not absolute")
    if _UNC_PATH.match(path.strip()):
        raise ValueError("path must not be a UNC path")
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        raise ValueError("path must not contain parent-directory traversal")
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
    try:
        _validate_repository_relative_path(relative_path)
    except ValueError as exc:
        raise ReleasePostureError(f"release posture path {relative_path!r}: {exc}") from exc
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


def load_target_corpus_manifest(
    corpus_manifest_path: Path,
    *,
    target_name: str,
    embedding_model: str,
) -> CanonicalCorpusManifest:
    """Load the canonical corpus a target serves and check it fits the target's embedding model."""
    try:
        manifest = load_canonical_corpus_manifest(corpus_manifest_path)
    except CanonicalCorpusError as exc:
        raise ReleasePostureError(
            f"release target {target_name!r}: canonical corpus manifest is unusable: {exc}"
        ) from exc
    if manifest.expected.embedding_model != embedding_model:
        raise ReleasePostureError(
            f"release target {target_name!r} embedding model {embedding_model!r} does not match "
            f"the model {manifest.expected.embedding_model!r} that the canonical corpus "
            f"{manifest.corpus_id!r} is fingerprinted for"
        )
    return manifest


def resolve_release_target(
    descriptor: ReleasePostureDescriptor,
    target_name: str,
    *,
    project_root: Path,
) -> ResolvedReleaseTarget:
    """Resolve a named release target to validated absolute paths and committed expectations."""
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
    corpus_manifest_path = _resolve_repository_path(
        target.corpus_manifest,
        project_root=project_root,
    )
    corpus = load_target_corpus_manifest(
        corpus_manifest_path,
        target_name=target_name,
        embedding_model=target.embedding_model,
    )
    return ResolvedReleaseTarget(
        target_name=target_name,
        status=target.status,
        index_dir=index_dir,
        index_path_relative=to_relative_posix_path(index_dir, project_root),
        expected_corpus_fingerprint=corpus.expected.corpus_fingerprint,
        expected_chunk_count=corpus.expected.chunk_count,
        expected_document_count=corpus.expected.document_count,
        supported_document_ids=corpus.document_ids,
        collection_name=target.collection_name,
        embedding_model=target.embedding_model,
        vector_dimension=target.vector_dimension,
        corpus_id=corpus.corpus_id,
        corpus_manifest_path=corpus_manifest_path,
        corpus_manifest_relative=to_relative_posix_path(corpus_manifest_path, project_root),
        expected_chunk_payload_digest=corpus.expected.chunk_payload_digest,
    )


def build_instruction(target: ResolvedReleaseTarget) -> str:
    """The command that builds this target's index from the repository.

    Every flag is spelled out so the command does not depend on environment defaults. It needs
    ``OPENAI_API_KEY`` (the only step of the whole path that calls the embedding API).
    """
    return (
        "python -m customer_claims_rag.cli.build_index "
        f"--corpus-manifest {target.corpus_manifest_relative} "
        f"--index-dir {target.index_path_relative} "
        f"--collection {target.collection_name} "
        f"--embedding-model {target.embedding_model} "
        "--rebuild"
    )


def build_provisioning_message(target: ResolvedReleaseTarget) -> str:
    return (
        f"the index is a build artifact, not a repository file. Build it from the repository "
        f"(needs OPENAI_API_KEY; embeds {target.expected_chunk_count} chunks of "
        f"{target.expected_document_count} documents with {target.embedding_model}): "
        f"{build_instruction(target)}"
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


def _validate_index_manifest_claims(manifest, target: ResolvedReleaseTarget, hint: str) -> None:
    """The index manifest is self-attestation: every claim must agree with the committed pins."""
    if manifest.corpus_fingerprint != target.expected_corpus_fingerprint:
        raise ReleasePostureError(
            "corpus fingerprint mismatch for release target "
            f"{target.target_name!r}: manifest has "
            f"{manifest.corpus_fingerprint}, expected "
            f"{target.expected_corpus_fingerprint}; the index was not built from the canonical "
            f"corpus {target.corpus_id!r} ({hint})"
        )
    if manifest.chunk_count != target.expected_chunk_count:
        raise ReleasePostureError(
            f"chunk count mismatch for release target {target.target_name!r}: "
            f"manifest has {manifest.chunk_count}, expected "
            f"{target.expected_chunk_count}"
        )
    if manifest.document_count != target.expected_document_count:
        raise ReleasePostureError(
            f"document count mismatch for release target {target.target_name!r}: "
            f"manifest has {manifest.document_count}, expected "
            f"{target.expected_document_count}"
        )
    if manifest.collection_name != target.collection_name:
        raise ReleasePostureError(
            f"collection name mismatch: manifest has {manifest.collection_name!r}, "
            f"expected {target.collection_name!r}"
        )
    if manifest.embedding_model != target.embedding_model:
        raise ReleasePostureError(
            f"embedding model mismatch: manifest has {manifest.embedding_model!r}, "
            f"expected {target.embedding_model!r}"
        )
    if manifest.vector_dimension != target.vector_dimension:
        raise ReleasePostureError(
            f"vector dimension mismatch: manifest has {manifest.vector_dimension}, "
            f"expected {target.vector_dimension}"
        )
    if manifest.index_format_version != INDEX_FORMAT_VERSION:
        raise ReleasePostureError(
            f"index format version mismatch: manifest has {manifest.index_format_version!r}, "
            f"expected {INDEX_FORMAT_VERSION!r}; rebuild the index ({hint})"
        )
    if manifest.metadata_schema_version != METADATA_SCHEMA_VERSION:
        raise ReleasePostureError(
            "metadata schema version mismatch: manifest has "
            f"{manifest.metadata_schema_version!r}, expected {METADATA_SCHEMA_VERSION!r}; "
            f"rebuild the index ({hint})"
        )
    for digest_name in ("chunk_payload_digest", "embedding_digest", "collection_content_digest"):
        if not getattr(manifest, digest_name):
            raise ReleasePostureError(
                f"index manifest lacks {digest_name}: it was not written by the supported "
                f"build path, so its content cannot be verified; rebuild the index ({hint})"
            )
    if manifest.chunk_payload_digest != target.expected_chunk_payload_digest:
        raise ReleasePostureError(
            f"chunk payload digest mismatch for release target {target.target_name!r}: "
            f"manifest has {manifest.chunk_payload_digest}, the canonical corpus "
            f"{target.corpus_id!r} expects {target.expected_chunk_payload_digest}; "
            f"the index was built from different chunks ({hint})"
        )


def _validate_store_content(
    vector_store: VectorStore,
    manifest,
    target: ResolvedReleaseTarget,
    hint: str,
) -> None:
    """Recompute the identity from the stored records and compare it with every claim."""
    if vector_store.collection_name != target.collection_name:
        raise ReleasePostureError(
            f"vector store collection mismatch: runtime has "
            f"{vector_store.collection_name!r}, expected "
            f"{target.collection_name!r}"
        )
    store_count = vector_store.count()
    if store_count != target.expected_chunk_count:
        raise ReleasePostureError(
            f"vector store chunk count mismatch: store has {store_count}, expected "
            f"{target.expected_chunk_count}"
        )
    actual_document_ids = _collect_document_ids(vector_store)
    expected_document_ids = set(target.supported_document_ids)
    if actual_document_ids != expected_document_ids:
        missing = sorted(expected_document_ids - actual_document_ids)
        extra = sorted(actual_document_ids - expected_document_ids)
        raise ReleasePostureError(
            "supported document IDs mismatch for release target "
            f"{target.target_name!r}: missing={missing!r}, extra={extra!r}"
        )
    identity = recompute_store_identity(
        vector_store,
        embedding_model=target.embedding_model,
        rebuild_hint=f"rebuild the index ({hint})",
    )
    if identity.chunk_count != store_count:
        raise ReleasePostureError(
            f"stored record count mismatch: the store reports {store_count} vectors but only "
            f"{identity.chunk_count} records are complete and readable; rebuild the index ({hint})"
        )
    if identity.chunk_payload_digest != target.expected_chunk_payload_digest:
        raise ReleasePostureError(
            f"stored chunk payload digest mismatch for release target {target.target_name!r}: "
            f"the stored chunks hash to {identity.chunk_payload_digest}, the canonical corpus "
            f"{target.corpus_id!r} expects {target.expected_chunk_payload_digest}; the stored "
            f"chunks are not the canonical corpus (stale or edited index; {hint})"
        )
    if identity.corpus_fingerprint != target.expected_corpus_fingerprint:
        raise ReleasePostureError(
            f"stored corpus fingerprint mismatch for release target {target.target_name!r}: "
            f"recomputed {identity.corpus_fingerprint}, expected "
            f"{target.expected_corpus_fingerprint} ({hint})"
        )
    if identity.collection_content_digest != manifest.collection_content_digest:
        raise ReleasePostureError(
            "stored collection content digest mismatch: the stored chunks or vectors differ from "
            "what the index manifest attests (changed after the manifest was written, or the "
            f"manifest was edited); rebuild the index ({hint})"
        )
    if identity.vector_dimensions != {target.vector_dimension}:
        raise ReleasePostureError(
            f"stored vector dimension mismatch: stored vectors have dimensions "
            f"{sorted(identity.vector_dimensions)}, expected {target.vector_dimension}"
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
    hint = f"build it with: {build_instruction(resolved_target)}"
    if not resolved_target.index_dir.is_dir():
        raise ReleasePostureError(
            f"release target {resolved_target.target_name!r} index directory does not "
            f"exist: {resolved_target.index_path_relative}; "
            + build_provisioning_message(resolved_target)
        )
    try:
        validate_open_existing_chroma_preconditions(
            resolved_target.index_dir,
            collection_name=resolved_target.collection_name,
        )
    except Exception as exc:
        raise ReleasePostureError(f"{exc}; {build_provisioning_message(resolved_target)}") from exc

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
        raise ReleasePostureError(f"{exc}; {build_provisioning_message(resolved_target)}") from exc
    _validate_index_manifest_claims(manifest, resolved_target, hint)

    if vector_store is not None:
        _validate_store_content(vector_store, manifest, resolved_target, hint)

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
        corpus_id=resolved_target.corpus_id,
        chunk_payload_digest=manifest.chunk_payload_digest or "",
        integrity=INTEGRITY_STORE_RECOMPUTED if vector_store is not None else INTEGRITY_MANIFEST_ONLY,
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
