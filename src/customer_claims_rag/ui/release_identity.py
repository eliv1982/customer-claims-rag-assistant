"""Safe release identity display for Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

from customer_claims_rag.release.posture import ReleasePostureDiagnostics


@dataclass(frozen=True)
class ReleaseIdentityView:
    """Customer-safe production release summary for UI display."""

    release_posture_id: str
    selected_target: str
    document_count: int
    chunk_count: int
    fingerprint_short: str
    retrieval_contract: str
    reranker_id: str
    service_ready: bool


def shorten_fingerprint(fingerprint: str, *, prefix_len: int = 8) -> str:
    """Return a shortened fingerprint prefix for display."""
    if len(fingerprint) <= prefix_len:
        return fingerprint
    return f"{fingerprint[:prefix_len]}…"


def map_diagnostics_to_release_view(
    diagnostics: ReleasePostureDiagnostics,
    *,
    service_ready: bool,
) -> ReleaseIdentityView:
    """Map validated runtime diagnostics to a safe UI view."""
    retrieval_contract = (
        f"{diagnostics.vector_fetch_k} / {diagnostics.candidate_pool_k} / "
        f"{diagnostics.final_top_k} / {diagnostics.similarity_threshold}"
    )
    return ReleaseIdentityView(
        release_posture_id=diagnostics.release_posture_id,
        selected_target=diagnostics.selected_target,
        document_count=diagnostics.document_count,
        chunk_count=diagnostics.chunk_count,
        fingerprint_short=shorten_fingerprint(diagnostics.corpus_fingerprint),
        retrieval_contract=retrieval_contract,
        reranker_id=diagnostics.reranker_id,
        service_ready=service_ready,
    )


def format_release_identity_lines(view: ReleaseIdentityView) -> list[str]:
    """Return sidebar-ready label lines without paths or secrets."""
    readiness = "готов" if view.service_ready else "не готов"
    return [
        f"Релиз: {view.release_posture_id}",
        f"Цель: {view.selected_target}",
        f"Документы: {view.document_count}",
        f"Чанки: {view.chunk_count}",
        f"Отпечаток: {view.fingerprint_short}",
        f"Контракт извлечения: {view.retrieval_contract}",
        f"Реранкер: {view.reranker_id}",
        f"Сервис: {readiness}",
    ]
