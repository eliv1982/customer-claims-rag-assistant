"""Release readiness: which part of the release state is established, and what to do next.

``validate_production_release_posture`` answers one question (does this index satisfy the posture?)
by raising on the first violation. Readiness answers the questions an operator actually has on a
fresh clone or after a change, without raising:

* is the canonical corpus defined, and do the repository's source files still match it?
* is the vector index present at all?
* does the index match the canonical corpus (recomputed from the stored records)?
* can the release proceed, and if not, which command fixes it?
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from customer_claims_rag.exceptions import RetrievalError
from customer_claims_rag.ingestion.canonical_corpus import (
    load_canonical_corpus_manifest,
    verify_corpus_sources,
)
from customer_claims_rag.release.posture import (
    ProductionReleaseContext,
    ReleasePostureDiagnostics,
    build_instruction,
    validate_production_release_posture,
)
from customer_claims_rag.retrieval.chroma_storage import chroma_sqlite_path
from customer_claims_rag.retrieval.factory import create_vector_store
from customer_claims_rag.retrieval.ports import VectorStore

INDEX_MATCHES_YES = "yes"
INDEX_MATCHES_NO = "no"
INDEX_MATCHES_NOT_CHECKED = "not_checked"

# The release verdict is three-valued. ``not_established`` is not a soft "yes": the static checks
# (descriptor, manifests, frozen config) can all pass while nothing says the index content matches
# the canonical corpus, and only an established release may be treated as ready.
RELEASE_YES = "yes"
RELEASE_NO = "no"
RELEASE_NOT_ESTABLISHED = "not_established"


@dataclass(frozen=True)
class ReleaseReadiness:
    release_posture_id: str
    selected_target: str
    target_status: str
    corpus_id: str
    corpus_document_count: int
    corpus_chunk_count: int
    corpus_sources: str
    index_path: str
    index_present: bool
    index_matches_canonical_corpus: str
    release_can_proceed: bool
    problem: str | None
    next_step: str | None
    diagnostics: ReleasePostureDiagnostics | None

    @property
    def static_validation_passed(self) -> bool:
        """The descriptor, manifests and frozen config were validated without a failure.

        This says nothing about the index content, so it never implies ``release_can_proceed``.
        """
        return self.diagnostics is not None

    @property
    def release_status(self) -> str:
        """``yes`` only when readiness is positively established; ``not_established`` when the
        static checks passed but the index content was not verified; ``no`` for everything else."""
        if self.release_can_proceed:
            return RELEASE_YES
        content_unchecked = self.index_matches_canonical_corpus == INDEX_MATCHES_NOT_CHECKED
        if self.static_validation_passed and content_unchecked:
            return RELEASE_NOT_ESTABLISHED
        return RELEASE_NO

    def format_status_lines(self) -> list[str]:
        status = self.release_status
        proceed = status
        if status == RELEASE_NOT_ESTABLISHED:
            proceed += (
                " (index content was not verified; passing static validation is not "
                "release readiness)"
            )
        lines = [
            f"canonical_corpus_sources={self.corpus_sources}",
            f"index_present={'yes' if self.index_present else 'no'}",
            f"index_matches_canonical_corpus={self.index_matches_canonical_corpus}",
        ]
        if self.static_validation_passed:
            lines.append("static_validation=passed")
        lines.append(f"release_can_proceed={proceed}")
        return lines

    def format_lines(self) -> list[str]:
        lines = [
            f"release_posture_id={self.release_posture_id}",
            f"selected_target={self.selected_target}",
            f"target_status={self.target_status}",
            f"canonical_corpus_id={self.corpus_id}",
            f"canonical_corpus_documents={self.corpus_document_count}",
            f"canonical_corpus_chunks={self.corpus_chunk_count}",
            f"index_path={self.index_path}",
        ]
        lines.extend(self.format_status_lines())
        if self.next_step:
            lines.append(f"next_step={self.next_step}")
        return lines


def _describe_sources(context: ProductionReleaseContext) -> str:
    target = context.resolved_target
    try:
        manifest = load_canonical_corpus_manifest(target.corpus_manifest_path)
        source_dir = manifest.source_directory(context.project_root)
    except Exception as exc:  # the manifest was valid when the target resolved; report, not raise
        return f"unreadable ({exc})"
    if not source_dir.is_dir():
        return "not_present (source directory absent; this is a deployment without the corpus)"
    report = verify_corpus_sources(manifest, project_root=context.project_root)
    if report.ok:
        return f"verified ({len(manifest.documents)} source files match source_sha256)"
    return f"mismatch ({report.describe()})"


def assess_release_readiness(
    context: ProductionReleaseContext,
    *,
    check_index_content: bool = True,
    vector_store_factory: Callable[..., VectorStore] = create_vector_store,
) -> ReleaseReadiness:
    """Assess the release state of the resolved target. Never raises for release problems."""
    target = context.resolved_target
    corpus_sources = _describe_sources(context)
    instruction = build_instruction(target)
    common = {
        "release_posture_id": context.descriptor.release_posture_id,
        "selected_target": target.target_name,
        "target_status": target.status,
        "corpus_id": target.corpus_id,
        "corpus_document_count": target.expected_document_count,
        "corpus_chunk_count": target.expected_chunk_count,
        "corpus_sources": corpus_sources,
        "index_path": target.index_path_relative,
    }

    index_present = target.index_dir.is_dir() and chroma_sqlite_path(target.index_dir).is_file()
    if not index_present:
        return ReleaseReadiness(
            **common,
            index_present=False,
            index_matches_canonical_corpus=INDEX_MATCHES_NOT_CHECKED,
            release_can_proceed=False,
            problem=(
                f"no index at {target.index_path_relative}: the production index is a build "
                "artifact and has not been built here"
            ),
            next_step=instruction,
            diagnostics=None,
        )

    vector_store: VectorStore | None = None
    try:
        if check_index_content:
            vector_store = vector_store_factory(
                index_dir=target.index_dir,
                collection_name=target.collection_name,
                open_existing=True,
            )
        diagnostics = validate_production_release_posture(context, vector_store=vector_store)
    except RetrievalError as exc:  # ReleasePostureError and VectorStoreError both derive from it
        return ReleaseReadiness(
            **common,
            index_present=True,
            index_matches_canonical_corpus=INDEX_MATCHES_NO,
            release_can_proceed=False,
            problem=str(exc),
            next_step=instruction,
            diagnostics=None,
        )
    finally:
        if vector_store is not None:
            vector_store.close()

    return ReleaseReadiness(
        **common,
        index_present=True,
        index_matches_canonical_corpus=(
            INDEX_MATCHES_YES if check_index_content else INDEX_MATCHES_NOT_CHECKED
        ),
        release_can_proceed=check_index_content,
        problem=None,
        next_step=None,
        diagnostics=diagnostics,
    )
