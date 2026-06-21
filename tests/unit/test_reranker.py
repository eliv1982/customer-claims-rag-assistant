"""Unit tests for source-authority reranker."""

from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path

import pytest

from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.retrieval.reranker import (
    RerankerConfig,
    SourceAuthorityV1Reranker,
    compute_config_hash,
    load_reranker_config,
    source_authority_bonus,
)


def _result(
    *,
    rank: int,
    chunk_id: str,
    similarity: float,
    chunk_type: str = "policy",
) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id="doc",
        content="body",
        source_path="data/02_clean_markdown/doc.md",
        chunk_type=chunk_type,
        topic=None,
        risk_level=None,
        heading="Heading",
        heading_path=["Heading"],
        section="Section",
        subsection=None,
        similarity=similarity,
        distance=1.0 - similarity,
    )


def _config() -> RerankerConfig:
    path = Path("configs/reranking/source_authority_v1.json")
    return load_reranker_config(path)


def test_zero_bonuses_preserve_identity_order() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    candidates = [
        _result(rank=1, chunk_id="b::1", similarity=0.90, chunk_type="faq"),
        _result(rank=2, chunk_id="a::1", similarity=0.80, chunk_type="faq"),
    ]
    original = deepcopy(candidates)
    reranked = reranker.rerank("query", candidates)
    assert [item.result.chunk_id for item in reranked] == ["b::1", "a::1"]
    assert candidates == original


def test_deterministic_ordering() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    candidates = [
        _result(rank=1, chunk_id="faq::1", similarity=0.80, chunk_type="faq"),
        _result(rank=2, chunk_id="policy::1", similarity=0.79, chunk_type="policy"),
    ]
    first = reranker.rerank("query", deepcopy(candidates))
    second = reranker.rerank("query", deepcopy(candidates))
    assert [item.result.chunk_id for item in first] == [item.result.chunk_id for item in second]


def test_stable_ties_use_baseline_rank_then_chunk_id() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    candidates = [
        _result(rank=1, chunk_id="b::1", similarity=0.80, chunk_type="faq"),
        _result(rank=2, chunk_id="a::1", similarity=0.80, chunk_type="faq"),
    ]
    reranked = reranker.rerank("query", candidates)
    assert [item.result.chunk_id for item in reranked] == ["b::1", "a::1"]


def test_authority_types_receive_expected_bonuses() -> None:
    mapping = _config().source_authority_mapping
    assert source_authority_bonus("policy", mapping) == 0.03
    assert source_authority_bonus("escalation", mapping) == 0.03
    assert source_authority_bonus("procedure", mapping) == 0.02
    assert source_authority_bonus("reference", mapping) == 0.01
    assert source_authority_bonus("faq", mapping) == 0.0
    assert source_authority_bonus("templates", mapping) == 0.0


def test_missing_and_unknown_chunk_type_bonus_zero() -> None:
    mapping = _config().source_authority_mapping
    assert source_authority_bonus(None, mapping) == 0.0
    assert source_authority_bonus("unknown_type", mapping) == 0.0


def test_bonus_does_not_exceed_max() -> None:
    mapping = _config().source_authority_mapping
    for chunk_type in ("policy", "escalation", "procedure", "reference", "faq", "templates"):
        assert source_authority_bonus(chunk_type, mapping) <= 0.03


def test_similarity_remains_dominant_component() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    candidates = [
        _result(rank=1, chunk_id="faq::1", similarity=0.85, chunk_type="faq"),
        _result(rank=2, chunk_id="policy::1", similarity=0.80, chunk_type="policy"),
    ]
    reranked = reranker.rerank("query", candidates)
    assert reranked[0].result.chunk_id == "faq::1"


def test_policy_promoted_when_similarity_gap_within_bonus() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    candidates = [
        _result(rank=1, chunk_id="faq::1", similarity=0.805, chunk_type="faq"),
        _result(rank=2, chunk_id="policy::1", similarity=0.80, chunk_type="policy"),
    ]
    reranked = reranker.rerank("query", candidates)
    assert reranked[0].result.chunk_id == "policy::1"
    assert reranked[0].source_authority_bonus == 0.03
    assert reranked[0].rerank_score == pytest.approx(0.83)


def test_similarity_and_distance_not_mutated() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    candidates = [
        _result(rank=1, chunk_id="faq::1", similarity=0.70, chunk_type="faq"),
        _result(rank=2, chunk_id="policy::1", similarity=0.69, chunk_type="policy"),
    ]
    original_similarities = [item.similarity for item in candidates]
    original_distances = [item.distance for item in candidates]
    reranker.rerank("query", candidates)
    assert [item.similarity for item in candidates] == original_similarities
    assert [item.distance for item in candidates] == original_distances


def test_input_candidates_not_mutated_except_deepcopy_isolation() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    candidates = [
        _result(rank=1, chunk_id="faq::1", similarity=0.70, chunk_type="faq"),
    ]
    snapshot = deepcopy(candidates)
    reranker.rerank("query", candidates)
    assert candidates[0].rank == snapshot[0].rank


def test_fewer_than_twelve_candidates() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    candidates = [_result(rank=1, chunk_id="policy::1", similarity=0.5, chunk_type="policy")]
    reranked = reranker.rerank("query", candidates)
    assert len(reranked) == 1
    assert reranked[0].candidate_rank == 1


def test_empty_candidates() -> None:
    reranker = SourceAuthorityV1Reranker(_config())
    assert reranker.rerank("query", []) == []


def test_config_hash_reproducible() -> None:
    config = _config()
    assert compute_config_hash(config) == compute_config_hash(_config())


def test_reranker_contract_has_no_forbidden_evaluation_fields() -> None:
    signature = inspect.signature(SourceAuthorityV1Reranker.rerank)
    param_names = set(signature.parameters)
    forbidden = {
        "test_id",
        "expected_risk",
        "category",
        "expected_primary_documents",
        "expected_supporting_documents",
        "expected_sources",
        "case_id",
    }
    assert forbidden.isdisjoint(param_names)
