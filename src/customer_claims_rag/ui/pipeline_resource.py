"""Production pipeline construction for UI entrypoints."""

from __future__ import annotations

from collections.abc import Callable

from customer_claims_rag.application.factory import build_customer_claims_pipeline
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.application.settings import ApplicationSettings
from customer_claims_rag.env_bootstrap import load_project_env


def create_production_pipeline(
    *,
    settings_loader: Callable[[], ApplicationSettings] = ApplicationSettings.from_env,
    pipeline_factory: Callable[
        [ApplicationSettings],
        CustomerClaimsPipeline,
    ] = build_customer_claims_pipeline,
) -> CustomerClaimsPipeline:
    """Build the production customer claims pipeline from environment settings."""
    load_project_env()
    settings = settings_loader()
    return pipeline_factory(settings)
