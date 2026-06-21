"""OpenAI embeddings adapter via langchain-openai."""

from __future__ import annotations

from typing import Sequence

from langchain_openai import OpenAIEmbeddings

from customer_claims_rag.exceptions import EmbeddingError
from customer_claims_rag.retrieval.embedding_validation import (
    validate_embedding_batch,
    validate_embedding_vector,
    validate_query_text,
)


class OpenAIEmbeddingProvider:
    """LangChain-backed OpenAI embedding provider."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise EmbeddingError("embedding model name must not be empty")
        if not api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY is required for OpenAI embeddings; "
                "set the environment variable before building or searching"
            )
        self._model_name = model_name
        self._vector_dimension: int | None = None
        self._client = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vector_dimension(self) -> int | None:
        return self._vector_dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._client.embed_documents(list(texts))
        except Exception:
            raise EmbeddingError("OpenAI embedding request failed") from None
        validated = validate_embedding_batch(vectors, expected_count=len(texts))
        if self._vector_dimension is None and validated:
            self._vector_dimension = len(validated[0])
        if self._vector_dimension is not None:
            return validate_embedding_batch(
                validated,
                expected_count=len(texts),
                expected_dimension=self._vector_dimension,
            )
        return validated

    def embed_query(self, text: str) -> list[float]:
        cleaned = validate_query_text(text)
        try:
            vector = self._client.embed_query(cleaned)
        except Exception:
            raise EmbeddingError("OpenAI embedding request failed") from None
        validated = validate_embedding_vector(vector)
        if self._vector_dimension is None:
            self._vector_dimension = len(validated)
        return validate_embedding_vector(
            validated,
            expected_dimension=self._vector_dimension,
        )
