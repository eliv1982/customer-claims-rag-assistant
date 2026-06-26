"""Factory helpers for retrieval CLI wiring."""

from __future__ import annotations

from pathlib import Path

from customer_claims_rag.exceptions import EmbeddingError
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.adapters.openai_embeddings import OpenAIEmbeddingProvider
from customer_claims_rag.retrieval.ports import EmbeddingProvider, VectorStore


def create_embedding_provider(
    *,
    model_name: str,
    api_key: str | None,
) -> EmbeddingProvider:
    return OpenAIEmbeddingProvider(model_name=model_name, api_key=api_key)


def create_vector_store(
    *,
    index_dir: Path,
    collection_name: str,
    open_existing: bool = False,
) -> VectorStore:
    return ChromaVectorStore(
        index_dir=index_dir,
        collection_name=collection_name,
        open_existing=open_existing,
    )


def require_openai_api_key(api_key: str | None) -> str:
    if not api_key:
        raise EmbeddingError(
            "OPENAI_API_KEY is required; set the environment variable before "
            "building or searching the index"
        )
    return api_key
