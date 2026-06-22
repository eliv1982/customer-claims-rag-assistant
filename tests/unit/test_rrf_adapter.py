"""Unit tests for RRF base score adapter."""

from __future__ import annotations

from customer_claims_rag.retrieval.fusion import FusionCandidate
from customer_claims_rag.retrieval.lexical.corpus_loader import LexicalChunk
from customer_claims_rag.retrieval.reranker import SourceAuthorityV1Reranker, compute_config_hash, load_reranker_config
from customer_claims_rag.retrieval.rrf_adapter import ADAPTER_ID, fusion_to_search_results, rerank_fused_candidates

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
RERANKER_CONFIG = PROJECT_ROOT / "configs" / "reranking" / "source_authority_v1.json"
FROZEN_HASH = "c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957"


def _chunk(chunk_id: str, *, chunk_type: str = "policy") -> LexicalChunk:
    return LexicalChunk(
        chunk_id=chunk_id,
        document_id="06_food_quality_and_packaging",
        content="металлическая осколка",
        heading="Качество",
        chunk_type=chunk_type,
        source_path="data/02_clean_markdown/06_food_quality_and_packaging.md",
        metadata={"heading_path": "[\"Качество\"]"},
    )


def _fusion(
    chunk_id: str,
    *,
    vector_rank: int | None,
    lexical_rank: int | None,
    normalized: float,
    channel: str,
) -> FusionCandidate:
    return FusionCandidate(
        chunk_id=chunk_id,
        document_id="06_food_quality_and_packaging",
        vector_rank=vector_rank,
        vector_similarity=0.42 if vector_rank is not None else None,
        vector_distance=0.58 if vector_rank is not None else None,
        lexical_rank=lexical_rank,
        bm25_score=2.5 if lexical_rank is not None else None,
        matched_tokens=["металл"] if lexical_rank is not None else [],
        retrieval_channel=channel,
        raw_rrf_score=normalized,
        normalized_rrf_score=normalized,
        fusion_rank=1,
    )


def test_normalized_rrf_used_as_reranker_base_score() -> None:
    fused = [
        _fusion("policy::1", vector_rank=2, lexical_rank=1, normalized=0.9, channel="both"),
        _fusion("faq::1", vector_rank=1, lexical_rank=None, normalized=0.4, channel="vector_only"),
    ]
    lookup = {item.chunk_id: _chunk(item.chunk_id, chunk_type="faq" if "faq" in item.chunk_id else "policy") for item in fused}
    lookup["policy::1"] = _chunk("policy::1", chunk_type="policy")
    pool = fusion_to_search_results(fused, chunk_lookup=lookup)
    assert pool[0].similarity == 0.9
    assert pool[0].similarity != 0.42


def test_lexical_only_vector_fields_remain_null_in_audit() -> None:
    fused = [
        _fusion("lex::1", vector_rank=None, lexical_rank=1, normalized=0.8, channel="lexical_only"),
    ]
    lookup = {"lex::1": _chunk("lex::1")}
    reranker = SourceAuthorityV1Reranker(load_reranker_config(RERANKER_CONFIG))
    _, audits = rerank_fused_candidates(
        reranker=reranker,
        query="металл",
        fused=fused,
        chunk_lookup=lookup,
        final_top_k=1,
    )
    assert audits[0].vector_rank is None
    assert audits[0].vector_similarity is None
    assert audits[0].vector_distance is None
    assert audits[0].normalized_rrf_score == 0.8


def test_source_authority_hash_unchanged() -> None:
    reranker = SourceAuthorityV1Reranker(load_reranker_config(RERANKER_CONFIG))
    assert compute_config_hash(reranker.config) == FROZEN_HASH


def test_adapter_id_constant() -> None:
    assert ADAPTER_ID == "rrf-base-score-adapter-v1"
