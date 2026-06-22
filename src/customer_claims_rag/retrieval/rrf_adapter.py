"""RRF base score adapter for frozen source-authority reranking."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from customer_claims_rag.retrieval.fusion import FusionCandidate
from customer_claims_rag.retrieval.lexical.corpus_loader import LexicalChunk
from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.retrieval.reranker import RerankedCandidate, SourceAuthorityV1Reranker

ADAPTER_ID = "rrf-base-score-adapter-v1"


@dataclass(frozen=True)
class HybridRerankAudit:
    """Audit record for one hybrid reranked candidate."""

    chunk_id: str
    document_id: str
    chunk_type: str
    vector_rank: int | None
    vector_similarity: float | None
    vector_distance: float | None
    lexical_rank: int | None
    bm25_score: float | None
    matched_tokens: list[str]
    retrieval_channel: str
    raw_rrf_score: float
    normalized_rrf_score: float
    fusion_rank: int
    source_authority_bonus: float
    final_rerank_score: float
    final_rank: int


def _deserialize_heading_path(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = json.loads(value)
    return [str(item) for item in parsed]


def fusion_to_search_results(
    fused: list[FusionCandidate],
    *,
    chunk_lookup: dict[str, LexicalChunk],
) -> list[SearchResult]:
    """Build reranker input using normalized RRF as the base score."""
    results: list[SearchResult] = []
    for candidate in fused:
        chunk = chunk_lookup[candidate.chunk_id]
        heading_path_raw = chunk.metadata.get("heading_path", "[]")
        if isinstance(heading_path_raw, str):
            heading_path = _deserialize_heading_path(heading_path_raw)
        else:
            heading_path = _deserialize_heading_path(str(heading_path_raw))
        results.append(
            SearchResult(
                rank=candidate.fusion_rank,
                chunk_id=candidate.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                source_path=chunk.source_path,
                chunk_type=chunk.chunk_type,
                topic=str(chunk.metadata["topic"]) if chunk.metadata.get("topic") is not None else None,
                risk_level=str(chunk.metadata["risk_level"])
                if chunk.metadata.get("risk_level") is not None
                else None,
                heading=chunk.heading,
                heading_path=heading_path,
                section=str(chunk.metadata["section"]) if chunk.metadata.get("section") is not None else None,
                subsection=str(chunk.metadata["subsection"])
                if chunk.metadata.get("subsection") is not None
                else None,
                similarity=candidate.normalized_rrf_score,
                distance=1.0 - candidate.normalized_rrf_score,
            )
        )
    return results


def rerank_fused_candidates(
    *,
    reranker: SourceAuthorityV1Reranker,
    query: str,
    fused: list[FusionCandidate],
    chunk_lookup: dict[str, LexicalChunk],
    final_top_k: int,
) -> tuple[list[SearchResult], list[HybridRerankAudit]]:
    """Apply frozen source-authority reranker to fused candidates."""
    pool = fusion_to_search_results(fused, chunk_lookup=chunk_lookup)
    fusion_by_id = {item.chunk_id: item for item in fused}
    reranked = reranker.rerank(query, pool)
    audits: list[HybridRerankAudit] = []
    for item in reranked[:final_top_k]:
        fusion_item = fusion_by_id[item.result.chunk_id]
        audits.append(
            HybridRerankAudit(
                chunk_id=item.result.chunk_id,
                document_id=item.result.document_id,
                chunk_type=item.result.chunk_type,
                vector_rank=fusion_item.vector_rank,
                vector_similarity=fusion_item.vector_similarity,
                vector_distance=fusion_item.vector_distance,
                lexical_rank=fusion_item.lexical_rank,
                bm25_score=fusion_item.bm25_score,
                matched_tokens=list(fusion_item.matched_tokens),
                retrieval_channel=fusion_item.retrieval_channel,
                raw_rrf_score=fusion_item.raw_rrf_score,
                normalized_rrf_score=fusion_item.normalized_rrf_score,
                fusion_rank=fusion_item.fusion_rank,
                source_authority_bonus=item.source_authority_bonus,
                final_rerank_score=item.rerank_score,
                final_rank=item.candidate_rank,
            )
        )
    final_results = [deepcopy(item.result) for item in reranked[:final_top_k]]
    return final_results, audits
