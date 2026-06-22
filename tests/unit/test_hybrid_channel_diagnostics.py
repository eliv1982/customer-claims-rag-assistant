"""Unit tests for hybrid channel reachability diagnostics."""

from __future__ import annotations

import pytest

from customer_claims_rag.evaluation.hybrid_metrics import (
    classify_primary_retrieval_channel,
    compute_channel_reachability,
    compute_primary_retrieval_channel,
)
from customer_claims_rag.retrieval.lexical.bm25 import LexicalHit
from customer_claims_rag.retrieval.models import SearchResult


def _result(chunk_id: str, document_id: str, rank: int) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        content="content",
        source_path=f"data/02_clean_markdown/{document_id}.md",
        chunk_type="policy",
        topic=None,
        risk_level=None,
        heading="Heading",
        heading_path=["Heading"],
        section=None,
        subsection=None,
        similarity=0.5,
        distance=0.5,
    )


def _lexical_hit(chunk_id: str, document_id: str, rank: int) -> LexicalHit:
    return LexicalHit(
        chunk_id=chunk_id,
        document_id=document_id,
        lexical_rank=rank,
        bm25_score=1.0,
        matched_tokens=["term"],
    )


@pytest.mark.parametrize(
    ("vector_reachable", "lexical_reachable", "expected"),
    [
        (True, False, "vector_only"),
        (False, True, "lexical_only"),
        (True, True, "both"),
        (False, False, "neither"),
    ],
)
def test_classify_primary_retrieval_channel(
    vector_reachable: bool,
    lexical_reachable: bool,
    expected: str,
) -> None:
    assert classify_primary_retrieval_channel(vector_reachable, lexical_reachable) == expected


def test_alternative_primary_docs_or_semantics() -> None:
    vector_pool = [
        _result("policy_a::chunk-001", "policy_a", 1),
    ]
    lexical_pool = [
        _lexical_hit("policy_b::chunk-002", "policy_b", 1),
    ]
    channel = compute_primary_retrieval_channel(
        vector_pool_chunk_ids=[item.chunk_id for item in vector_pool],
        lexical_pool_chunk_ids=[item.chunk_id for item in lexical_pool],
        expected_primary_documents=["policy_a", "policy_b"],
        fallback_expected=False,
    )
    assert channel == "both"


def test_fallback_without_expected_primary_is_not_neither_failure() -> None:
    channel = compute_primary_retrieval_channel(
        vector_pool_chunk_ids=[],
        lexical_pool_chunk_ids=[],
        expected_primary_documents=[],
        fallback_expected=True,
    )
    assert channel is None


def test_compute_channel_reachability_vector_only() -> None:
    vector_pool = [_result("doc::chunk-001", "doc", 1)]
    fusion_pool = vector_pool
    lexical_pool: list[LexicalHit] = []
    _, _, channel, vector_only, lexical_only, both = compute_channel_reachability(
        vector_pool=vector_pool,
        fusion_pool=fusion_pool,
        lexical_pool=lexical_pool,
        expected_primary_documents=["doc"],
        expected_supporting_documents=[],
        fallback_expected=False,
    )
    assert channel == "vector_only"
    assert vector_only is True
    assert lexical_only is False
    assert both is False


def test_compute_channel_reachability_lexical_only() -> None:
    vector_pool: list[SearchResult] = []
    lexical_pool = [_lexical_hit("doc::chunk-001", "doc", 1)]
    fusion_pool = [_result("doc::chunk-001", "doc", 1)]
    _, _, channel, vector_only, lexical_only, both = compute_channel_reachability(
        vector_pool=vector_pool,
        fusion_pool=fusion_pool,
        lexical_pool=lexical_pool,
        expected_primary_documents=["doc"],
        expected_supporting_documents=[],
        fallback_expected=False,
    )
    assert channel == "lexical_only"
    assert vector_only is False
    assert lexical_only is True
    assert both is False


def test_channel_regression_fails_if_vector_and_lexical_pools_are_swapped() -> None:
    """Guard against reintroducing vector_only_ids ↔ lexical_only_ids swap."""
    vector_pool = [_result("vector_doc::chunk-001", "vector_doc", 1)]
    lexical_pool = [_lexical_hit("lexical_doc::chunk-002", "lexical_doc", 1)]
    fusion_pool = vector_pool + [_result("lexical_doc::chunk-002", "lexical_doc", 2)]

    _, _, channel, vector_only, lexical_only, both = compute_channel_reachability(
        vector_pool=vector_pool,
        fusion_pool=fusion_pool,
        lexical_pool=lexical_pool,
        expected_primary_documents=["vector_doc"],
        expected_supporting_documents=[],
        fallback_expected=False,
    )
    assert channel == "vector_only"
    assert vector_only is True
    assert lexical_only is False
    assert both is False

    _, _, swapped_channel, swapped_vector_only, swapped_lexical_only, swapped_both = (
        compute_channel_reachability(
            vector_pool=vector_pool,
            fusion_pool=fusion_pool,
            lexical_pool=[_lexical_hit("vector_doc::chunk-001", "vector_doc", 1)],
            expected_primary_documents=["vector_doc"],
            expected_supporting_documents=[],
            fallback_expected=False,
        )
    )
    assert swapped_channel == "both"
    assert swapped_vector_only is False
    assert swapped_lexical_only is False
    assert swapped_both is True

    with pytest.raises(AssertionError):
        assert channel == swapped_channel
