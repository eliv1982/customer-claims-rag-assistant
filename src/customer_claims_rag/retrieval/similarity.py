"""Cosine distance to similarity conversion."""

from __future__ import annotations

from customer_claims_rag.exceptions import VectorStoreError


def distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine distance to similarity score.

    Chroma cosine space returns distance in [0, 2] where 0 is identical.
    Similarity is defined as ``1.0 - distance`` and clamped to [0.0, 1.0]
    only when floating-point noise produces values slightly outside range.
    """
    if distance < 0.0:
        # Chroma may return tiny negative distances for identical vectors.
        if distance >= -1e-6:
            distance = 0.0
        else:
            raise VectorStoreError(
                f"unexpected cosine distance {distance}; expected range [0.0, 2.0]"
            )
    if distance > 2.0:
        if distance <= 2.0 + 1e-6:
            distance = 2.0
        else:
            raise VectorStoreError(
                f"unexpected cosine distance {distance}; expected range [0.0, 2.0]"
            )
    similarity = 1.0 - distance
    if similarity < 0.0:
        return 0.0
    if similarity > 1.0:
        return 1.0
    return similarity
