"""Unit tests for pool expansion metrics."""

from __future__ import annotations

from customer_claims_rag.evaluation.pool_expansion_metrics import (
    FROZEN_RERANKER_CONFIG_HASH,
    compute_pool_expansion_config_hash,
    compute_pool_reachability,
    load_pool_expansion_config,
    prepare_pool,
)
from customer_claims_rag.retrieval.models import SearchResult

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"


def _result(rank: int, chunk_id: str, document_id: str, similarity: float = 0.5) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {document_id}",
        source_path=f"data/02_clean_markdown/{document_id}.md",
        chunk_type="policy",
        topic=None,
        risk_level=None,
        heading="Heading",
        heading_path=["Heading"],
        section=None,
        subsection=None,
        similarity=similarity,
        distance=1.0 - similarity,
    )


def test_config_hash_and_reranker_reference() -> None:
    config = load_pool_expansion_config(CONFIG)
    assert config.reranker_config_hash == FROZEN_RERANKER_CONFIG_HASH
    assert compute_pool_expansion_config_hash(config)


def test_or_semantics_primary_documents() -> None:
    pool = [
        _result(1, "a", "other"),
        _result(2, "b", "doc_primary_alt"),
    ]
    reachability = compute_pool_reachability(
        pool,
        expected_primary_documents=["doc_primary", "doc_primary_alt"],
        expected_supporting_documents=[],
        fallback_expected=False,
    )
    assert reachability.primary_reachable is True
    assert reachability.primary_best_pool_rank == 2


def test_no_primary_in_pool() -> None:
    pool = [_result(1, "a", "other")]
    reachability = compute_pool_reachability(
        pool,
        expected_primary_documents=["missing"],
        expected_supporting_documents=[],
        fallback_expected=False,
    )
    assert reachability.primary_reachable is False
    assert reachability.fully_unreachable is True


def test_fallback_case_excluded() -> None:
    reachability = compute_pool_reachability(
        [],
        expected_primary_documents=["doc"],
        expected_supporting_documents=[],
        fallback_expected=True,
    )
    assert reachability.fully_unreachable is False
    assert reachability.primary_reachable is False


def test_pool_prefix_assignment() -> None:
    deep = [_result(index, f"c{index}", f"d{index}", 1.0 - index * 0.01) for index in range(1, 25)]
    baseline = prepare_pool(deep, 12)
    candidate = prepare_pool(deep, 24)
    assert [item.chunk_id for item in baseline] == [f"c{index}" for index in range(1, 13)]
    assert [item.chunk_id for item in candidate][:12] == [item.chunk_id for item in baseline]
    assert len(candidate) == 24


def test_rank_13_absent_in_pool12_present_in_pool24() -> None:
    deep = [_result(index, f"c{index}", "other" if index < 13 else "expected_doc", 0.5) for index in range(1, 25)]
    baseline = prepare_pool(deep, 12)
    candidate = prepare_pool(deep, 24)
    base = compute_pool_reachability(
        baseline,
        expected_primary_documents=["expected_doc"],
        expected_supporting_documents=[],
        fallback_expected=False,
    )
    cand = compute_pool_reachability(
        candidate,
        expected_primary_documents=["expected_doc"],
        expected_supporting_documents=[],
        fallback_expected=False,
    )
    assert base.primary_reachable is False
    assert cand.primary_reachable is True
    assert cand.primary_best_pool_rank == 13
