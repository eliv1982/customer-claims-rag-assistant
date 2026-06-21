"""Deterministic fake embedding provider for offline tests."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from customer_claims_rag.retrieval.embedding_validation import (
    validate_embedding_batch,
    validate_embedding_vector,
    validate_query_text,
)


class FakeEmbeddingProvider:
    """Hash-based deterministic embeddings without network access."""

    def __init__(
        self,
        *,
        model_name: str = "fake-embedding-model",
        vector_dimension: int = 8,
    ) -> None:
        if vector_dimension < 1:
            raise ValueError("vector_dimension must be >= 1")
        self._model_name = model_name
        self._vector_dimension = vector_dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vector_dimension(self) -> int | None:
        return self._vector_dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [self._vector_for_text(text) for text in texts]
        return validate_embedding_batch(
            vectors,
            expected_count=len(texts),
            expected_dimension=self._vector_dimension,
        )

    def embed_query(self, text: str) -> list[float]:
        cleaned = validate_query_text(text)
        vector = self._vector_for_text(cleaned)
        return validate_embedding_vector(vector, expected_dimension=self._vector_dimension)

    def _vector_for_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self._vector_dimension:
            for index in range(0, len(digest), 4):
                if len(values) >= self._vector_dimension:
                    break
                chunk = digest[index : index + 4]
                if len(chunk) < 4:
                    chunk = chunk.ljust(4, b"\x00")
                raw = int.from_bytes(chunk, byteorder="big", signed=False)
                values.append((raw / 2**32) * 2.0 - 1.0)
            digest = hashlib.sha256(digest).digest()
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            return [0.0] * self._vector_dimension
        return [value / norm for value in values]
