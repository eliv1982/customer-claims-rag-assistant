"""Framework-independent exact cosine search for evaluation oracle."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from customer_claims_rag.retrieval.metadata_mapper import vector_metadata_to_search_hit
from customer_claims_rag.retrieval.models import SearchResult


class ExactVectorSearchError(ValueError):
    """Raised when exact vector search inputs fail validation."""


def exact_cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def validate_embedding_vector(
    vector: Sequence[float],
    *,
    label: str,
    expected_dimension: int | None = None,
) -> None:
    if not vector:
        raise ExactVectorSearchError(f"{label}: empty vector")
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ExactVectorSearchError(
            f"{label}: dimension mismatch (expected {expected_dimension}, got {len(vector)})"
        )
    for index, value in enumerate(vector):
        if not math.isfinite(value):
            raise ExactVectorSearchError(f"{label}: non-finite value at index {index}")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ExactVectorSearchError(f"{label}: zero-norm vector")


def validate_embedding_matrix(
    matrix: np.ndarray,
    *,
    expected_rows: int,
    expected_dimension: int,
) -> None:
    if matrix.ndim != 2:
        raise ExactVectorSearchError("embedding matrix must be 2-D")
    if matrix.shape[0] != expected_rows:
        raise ExactVectorSearchError(
            f"embedding row count mismatch (expected {expected_rows}, got {matrix.shape[0]})"
        )
    if matrix.shape[1] != expected_dimension:
        raise ExactVectorSearchError(
            f"embedding dimension mismatch (expected {expected_dimension}, got {matrix.shape[1]})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ExactVectorSearchError("embedding matrix contains NaN or Inf")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ExactVectorSearchError("embedding matrix contains zero-norm row")


def exact_vector_search(
    *,
    query_vector: Sequence[float],
    chunk_records: Sequence[dict[str, Any]],
    embedding_matrix: np.ndarray,
    fetch_k: int,
    threshold: float = 0.0,
) -> list[SearchResult]:
    """Rank all chunks by exact cosine similarity (no ANN, no Chroma).

    Sort order: similarity descending, then chunk_id ascending.
    """
    if fetch_k < 1:
        raise ExactVectorSearchError("fetch_k must be >= 1")
    if len(chunk_records) != embedding_matrix.shape[0]:
        raise ExactVectorSearchError("chunk_records and embedding_matrix row count mismatch")

    expected_dimension = embedding_matrix.shape[1]
    validate_embedding_matrix(
        embedding_matrix,
        expected_rows=len(chunk_records),
        expected_dimension=expected_dimension,
    )
    validate_embedding_vector(
        query_vector,
        label="query_vector",
        expected_dimension=expected_dimension,
    )

    scored: list[tuple[float, str, int, dict[str, Any]]] = []
    for row_index, record in enumerate(chunk_records):
        chunk_id = str(record["chunk_id"])
        row = embedding_matrix[row_index]
        similarity = float(np.dot(row, query_vector) / (np.linalg.norm(row) * np.linalg.norm(query_vector)))
        scored.append((similarity, chunk_id, row_index, record))

    scored.sort(key=lambda item: (-item[0], item[1]))

    results: list[SearchResult] = []
    for rank, (similarity, _chunk_id, row_index, record) in enumerate(scored, start=1):
        if similarity < threshold:
            continue
        if len(results) >= fetch_k:
            break
        distance = max(0.0, 1.0 - similarity)
        hit = vector_metadata_to_search_hit(
            content=str(record["document"]),
            metadata=dict(record["metadata"]),
            distance=distance,
            backend_rank=row_index + 1,
        )
        results.append(
            SearchResult(
                rank=len(results) + 1,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                content=hit.content,
                source_path=hit.source_path,
                chunk_type=hit.chunk_type,
                topic=hit.topic,
                risk_level=hit.risk_level,
                heading=hit.heading,
                heading_path=hit.heading_path,
                section=hit.section,
                subsection=hit.subsection,
                similarity=similarity,
                distance=distance,
            )
        )
    return results


__all__ = [
    "ExactVectorSearchError",
    "exact_cosine_similarity",
    "exact_vector_search",
    "validate_embedding_matrix",
    "validate_embedding_vector",
]
