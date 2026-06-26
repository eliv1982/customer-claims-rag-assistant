"""Embedding provider that blocks document API calls during snapshot replay."""

from __future__ import annotations

from typing import Sequence

from customer_claims_rag.exceptions import IndexBuildError
from customer_claims_rag.retrieval.ports import EmbeddingProvider


class SnapshotEmbeddingProvider:
    """Query-only embedding provider used with frozen snapshot index builds."""

    def __init__(
        self,
        *,
        query_provider: EmbeddingProvider,
    ) -> None:
        self._query_provider = query_provider

    @property
    def model_name(self) -> str:
        return self._query_provider.model_name

    @property
    def vector_dimension(self) -> int | None:
        return self._query_provider.vector_dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if texts:
            raise IndexBuildError(
                "document embedding API calls are disabled in snapshot build mode"
            )
        return []

    def embed_query(self, text: str) -> list[float]:
        return self._query_provider.embed_query(text)
