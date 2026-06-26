"""Production composition root for the customer claims application."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from customer_claims_rag.application.frozen_retrieval import FrozenRetrievalService
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.release.posture import (
    DEFAULT_DESCRIPTOR_PATH,
    load_release_posture_descriptor,
    validate_release_posture_for_production,
)
from customer_claims_rag.application.settings import (
    ApplicationSettings,
    FrozenRetrievalConfig,
    load_frozen_retrieval_config,
)
from customer_claims_rag.generation.factory import build_grounded_generator
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.generation_config import GenerationSettings
from customer_claims_rag.retrieval.factory import create_embedding_provider, create_vector_store
from customer_claims_rag.retrieval.ports import EmbeddingProvider, VectorStore
from customer_claims_rag.retrieval.reranker import (
    RerankerConfig,
    SourceAuthorityV1Reranker,
    load_reranker_config,
)
from customer_claims_rag.retrieval.retriever import BaselineRetriever

logger = logging.getLogger(__name__)


def build_customer_claims_pipeline(
    settings: ApplicationSettings,
    *,
    embedding_provider_factory: Callable[..., EmbeddingProvider] = create_embedding_provider,
    vector_store_factory: Callable[..., VectorStore] = create_vector_store,
    grounded_generator_factory: Callable[[GenerationSettings], GroundedGenerator] = (
        build_grounded_generator
    ),
    frozen_config_loader: Callable[[Path], FrozenRetrievalConfig] = load_frozen_retrieval_config,
    reranker_config_loader: Callable[[Path], RerankerConfig] = load_reranker_config,
) -> CustomerClaimsPipeline:
    """Wire the production customer claims pipeline from application settings."""
    frozen_config = frozen_config_loader(settings.frozen_retrieval_config_path)
    reranker_config = reranker_config_loader(settings.reranker_config_path)
    reranker = SourceAuthorityV1Reranker(reranker_config)

    descriptor = load_release_posture_descriptor(
        settings.release_descriptor_path or DEFAULT_DESCRIPTOR_PATH,
    )
    resolved_target = settings.release_target
    from customer_claims_rag import env_bootstrap

    project_root = env_bootstrap.project_root()
    validate_release_posture_for_production(
        descriptor,
        resolved_target,
        project_root=project_root,
        frozen_retrieval_config_path=settings.frozen_retrieval_config_path,
    )

    embedding_provider = embedding_provider_factory(
        model_name=settings.retrieval.embedding_model,
        api_key=settings.retrieval.openai_api_key,
    )
    vector_store = vector_store_factory(
        index_dir=settings.retrieval.index_dir,
        collection_name=settings.retrieval.collection_name,
        open_existing=True,
    )
    diagnostics = validate_release_posture_for_production(
        descriptor,
        resolved_target,
        project_root=project_root,
        vector_store=vector_store,
        frozen_retrieval_config_path=settings.frozen_retrieval_config_path,
    )
    for line in diagnostics.format_lines():
        logger.info("release posture: %s", line)

    baseline_retriever = BaselineRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        index_dir=settings.retrieval.index_dir,
        top_k=frozen_config.vector_top_k,
        fetch_k=frozen_config.vector_fetch_k,
        similarity_threshold=frozen_config.similarity_threshold,
    )
    baseline_retriever.validate_index()

    frozen_retrieval = FrozenRetrievalService(
        retriever=baseline_retriever,
        reranker=reranker,
        config=frozen_config,
    )

    grounded_generator = grounded_generator_factory(settings.generation)
    risk_aware_generator = RiskAwareGroundedGenerator(
        grounded_generator=grounded_generator,
    )

    return CustomerClaimsPipeline(
        retrieval=frozen_retrieval,
        generator=risk_aware_generator,
    )
