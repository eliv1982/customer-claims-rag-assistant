"""Retrieval port interfaces."""

from __future__ import annotations

from typing import Protocol, Sequence

from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.models import VectorSearchHit


class EmbeddingProvider(Protocol):
    """Framework-independent embedding contract."""

    @property
    def model_name(self) -> str:
        """Embedding model identifier."""
        ...

    @property
    def vector_dimension(self) -> int | None:
        """Known vector dimension, if available before first embed."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...


class VectorStore(Protocol):
    """Framework-independent vector store contract."""

    @property
    def collection_name(self) -> str:
        """Active collection name."""
        ...

    def recreate_collection(self, *, embedding_dimension: int) -> None:
        """Drop and recreate the collection for a full rebuild."""
        ...

    def add_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Persist chunk vectors and scalar-safe metadata."""
        ...

    def count(self) -> int:
        """Return number of indexed vectors."""
        ...

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
    ) -> list[VectorSearchHit]:
        """Return up to k nearest neighbors ordered by increasing distance."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...
