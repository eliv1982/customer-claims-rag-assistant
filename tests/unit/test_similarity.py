"""Cosine similarity conversion tests."""

from __future__ import annotations

import pytest

from customer_claims_rag.exceptions import VectorStoreError
from customer_claims_rag.retrieval.similarity import distance_to_similarity


def test_distance_zero_is_similarity_one() -> None:
    assert distance_to_similarity(0.0) == 1.0


def test_distance_one_is_similarity_zero() -> None:
    assert distance_to_similarity(1.0) == 0.0


def test_threshold_boundary_exactly_070() -> None:
    assert distance_to_similarity(0.30) == pytest.approx(0.70)


def test_unexpected_distance_rejected() -> None:
    with pytest.raises(VectorStoreError, match="unexpected cosine distance"):
        distance_to_similarity(2.5)


def test_tiny_negative_distance_treated_as_identical() -> None:
    assert distance_to_similarity(-1e-9) == 1.0
