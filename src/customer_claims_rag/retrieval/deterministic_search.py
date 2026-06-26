"""Deterministic brute-force vector search for experiment integrity checks."""

from __future__ import annotations

import math
from typing import Any, Sequence

from customer_claims_rag.retrieval.metadata_mapper import vector_metadata_to_search_hit
from customer_claims_rag.retrieval.models import VectorSearchHit


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def deterministic_similarity_hits(
    records: list[dict[str, Any]],
    *,
    query_vector: Sequence[float],
    k: int,
) -> list[VectorSearchHit]:
    """Rank exported collection records by exact cosine similarity (no ANN)."""
    ordered_records = sorted(records, key=lambda item: str(item["chunk_id"]))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for backend_rank, record in enumerate(ordered_records, start=1):
        similarity = cosine_similarity(query_vector, record["embedding"])
        scored.append((similarity, backend_rank, record))

    scored.sort(key=lambda item: (-item[0], item[1], str(item[2]["chunk_id"])))
    hits: list[VectorSearchHit] = []
    for similarity, backend_rank, record in scored[:k]:
        distance = max(0.0, 1.0 - similarity)
        hits.append(
            vector_metadata_to_search_hit(
                content=str(record["document"]),
                metadata=dict(record["metadata"]),
                distance=distance,
                backend_rank=backend_rank,
            )
        )
    return hits
