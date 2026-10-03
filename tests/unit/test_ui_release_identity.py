"""Tests for safe release identity display."""

from __future__ import annotations

from customer_claims_rag.release.posture import ReleasePostureDiagnostics
from customer_claims_rag.ui.release_identity import (
    format_release_identity_lines,
    map_diagnostics_to_release_view,
    shorten_fingerprint,
)

FINGERPRINT = "feedbeef" * 8


def _sample_diagnostics() -> ReleasePostureDiagnostics:
    return ReleasePostureDiagnostics(
        release_posture_id="foodflow-10doc-release-v2",
        selected_target="active",
        target_status="selected_production_release",
        index_path_relative="data/04_index_production",
        corpus_fingerprint=FINGERPRINT,
        chunk_count=215,
        document_count=10,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        vector_dimension=1536,
        frozen_retrieval_config_path="configs/retrieval/vector_pool_expansion_v1.json",
        frozen_config_hash="ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048",
        vector_fetch_k=24,
        candidate_pool_k=24,
        final_top_k=12,
        similarity_threshold=0.0,
        reranker_id="source-authority-v1",
    )


def test_map_diagnostics_uses_runtime_values_not_hardcoded_strings() -> None:
    diagnostics = _sample_diagnostics()
    view = map_diagnostics_to_release_view(diagnostics, service_ready=True)
    assert view.release_posture_id == diagnostics.release_posture_id
    assert view.selected_target == diagnostics.selected_target
    assert view.document_count == diagnostics.document_count
    assert view.chunk_count == diagnostics.chunk_count
    assert view.fingerprint_short == shorten_fingerprint(diagnostics.corpus_fingerprint)
    assert view.retrieval_contract == "24 / 24 / 12 / 0.0"
    assert view.reranker_id == diagnostics.reranker_id
    assert view.service_ready is True


def test_format_release_identity_lines_exclude_paths_and_secrets() -> None:
    view = map_diagnostics_to_release_view(_sample_diagnostics(), service_ready=True)
    rendered = "\n".join(format_release_identity_lines(view))
    assert "foodflow-10doc-release-v2" in rendered
    assert "feedbeef" in rendered
    assert "source-authority-v1" in rendered
    assert "OPENAI" not in rendered
    assert "sk-" not in rendered
    assert "data/" not in rendered
    assert "C:\\" not in rendered
    assert "/app/" not in rendered
    assert "API" not in rendered


def test_shorten_fingerprint() -> None:
    assert shorten_fingerprint(FINGERPRINT) == "feedbeef…"
