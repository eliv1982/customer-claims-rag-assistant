"""Unit tests for reciprocal rank fusion."""

from __future__ import annotations

import pytest

from customer_claims_rag.retrieval.fusion import (
    compute_raw_rrf_score,
    compute_theoretical_max_rrf,
    fuse_rankings,
    normalize_rrf_score,
)
from customer_claims_rag.retrieval.lexical.bm25 import LexicalHit
from customer_claims_rag.retrieval.models import SearchResult


def _vector(chunk_id: str, rank: int, document_id: str = "doc") -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        content="body",
        source_path="data/02_clean_markdown/doc.md",
        chunk_type="policy",
        heading="H",
        heading_path=["H"],
        similarity=0.5,
        distance=0.5,
    )


def _lexical(chunk_id: str, rank: int, document_id: str = "doc") -> LexicalHit:
    return LexicalHit(
        chunk_id=chunk_id,
        document_id=document_id,
        lexical_rank=rank,
        bm25_score=1.0,
        matched_tokens=["term"],
    )


def test_raw_score_formula_both_channels() -> None:
    score = compute_raw_rrf_score(
        vector_rank=1,
        lexical_rank=2,
        rrf_k=60,
        vector_weight=1.0,
        lexical_weight=1.0,
    )
    assert score == pytest.approx(1 / 61 + 1 / 62)


def test_theoretical_max() -> None:
    assert compute_theoretical_max_rrf(rrf_k=60, vector_weight=1.0, lexical_weight=1.0) == pytest.approx(
        2 / 61
    )


def test_normalized_score_in_unit_interval() -> None:
    theoretical = compute_theoretical_max_rrf(rrf_k=60, vector_weight=1.0, lexical_weight=1.0)
    raw = compute_raw_rrf_score(
        vector_rank=3,
        lexical_rank=5,
        rrf_k=60,
        vector_weight=1.0,
        lexical_weight=1.0,
    )
    normalized = normalize_rrf_score(raw, theoretical_max=theoretical)
    assert 0.0 <= normalized <= 1.0


def test_vector_only_channel() -> None:
    fused = fuse_rankings(
        vector_results=[_vector("only_vector", 1)],
        lexical_results=[],
        fusion_k=5,
        rrf_k=60,
        vector_weight=1.0,
        lexical_weight=1.0,
        document_ids={"only_vector": "doc"},
    )
    assert len(fused) == 1
    assert fused[0].retrieval_channel == "vector_only"
    assert fused[0].lexical_rank is None


def test_lexical_only_channel() -> None:
    fused = fuse_rankings(
        vector_results=[],
        lexical_results=[_lexical("only_lexical", 1)],
        fusion_k=5,
        rrf_k=60,
        vector_weight=1.0,
        lexical_weight=1.0,
        document_ids={"only_lexical": "doc"},
    )
    assert fused[0].retrieval_channel == "lexical_only"
    assert fused[0].vector_rank is None


def test_both_channels() -> None:
    fused = fuse_rankings(
        vector_results=[_vector("both", 2)],
        lexical_results=[_lexical("both", 1)],
        fusion_k=5,
        rrf_k=60,
        vector_weight=1.0,
        lexical_weight=1.0,
        document_ids={"both": "doc"},
    )
    assert fused[0].retrieval_channel == "both"


def test_union_dedup_by_chunk_id() -> None:
    fused = fuse_rankings(
        vector_results=[_vector("shared", 1), _vector("vector_only", 2)],
        lexical_results=[_lexical("shared", 3), _lexical("lexical_only", 1)],
        fusion_k=10,
        rrf_k=60,
        vector_weight=1.0,
        lexical_weight=1.0,
        document_ids={
            "shared": "doc",
            "vector_only": "doc",
            "lexical_only": "doc",
        },
    )
    assert len(fused) == 3


def test_stable_tie_breaking_by_chunk_id() -> None:
    fused = fuse_rankings(
        vector_results=[],
        lexical_results=[_lexical("b_chunk", 5), _lexical("a_chunk", 5)],
        fusion_k=2,
        rrf_k=60,
        vector_weight=1.0,
        lexical_weight=1.0,
        document_ids={"a_chunk": "doc", "b_chunk": "doc"},
    )
    assert [item.chunk_id for item in fused] == ["a_chunk", "b_chunk"]
