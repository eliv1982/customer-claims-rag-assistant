"""Validated release diagnostics for UI entrypoints."""

from __future__ import annotations

from customer_claims_rag import env_bootstrap
from customer_claims_rag.env_bootstrap import load_project_env
from customer_claims_rag.release.posture import (
    ReleasePostureDiagnostics,
    resolve_production_release_posture,
    validate_production_release_posture,
)
from customer_claims_rag.retrieval.factory import create_vector_store


def load_validated_release_diagnostics() -> ReleasePostureDiagnostics:
    """Load production release diagnostics using the shared validation boundary."""
    load_project_env()
    context = resolve_production_release_posture(project_root=env_bootstrap.project_root())
    vector_store = create_vector_store(
        index_dir=context.resolved_target.index_dir,
        collection_name=context.resolved_target.collection_name,
        open_existing=True,
    )
    try:
        return validate_production_release_posture(context, vector_store=vector_store)
    finally:
        vector_store.close()
