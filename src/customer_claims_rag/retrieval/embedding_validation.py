"""Embedding vector validation helpers."""

from __future__ import annotations

import math
from typing import Sequence

from customer_claims_rag.exceptions import EmbeddingError


def validate_embedding_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int | None = None,
) -> list[float]:
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise EmbeddingError(
            f"embedding dimension mismatch: expected {expected_dimension}, "
            f"got {len(vector)}"
        )
    result: list[float] = []
    for value in vector:
        if math.isnan(value) or math.isinf(value):
            raise EmbeddingError("embedding contains NaN or infinity")
        result.append(float(value))
    return result


def validate_embedding_batch(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int | None = None,
    expected_dimension: int | None = None,
) -> list[list[float]]:
    if expected_count is not None and len(vectors) != expected_count:
        raise EmbeddingError(
            f"embedding count mismatch: expected {expected_count}, got {len(vectors)}"
        )
    return [
        validate_embedding_vector(vector, expected_dimension=expected_dimension)
        for vector in vectors
    ]


def validate_query_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise EmbeddingError("query text must not be empty")
    return cleaned
