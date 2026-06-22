"""Reciprocal Rank Fusion for hybrid lexical + vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from customer_claims_rag.retrieval.lexical.bm25 import LexicalHit
from customer_claims_rag.retrieval.models import SearchResult

RetrievalChannel = Literal["both", "vector_only", "lexical_only"]


@dataclass(frozen=True)
class FusionCandidate:
    """Single fused candidate with channel diagnostics."""

    chunk_id: str
    document_id: str
    vector_rank: int | None
    vector_similarity: float | None
    vector_distance: float | None
    lexical_rank: int | None
    bm25_score: float | None
    matched_tokens: list[str]
    retrieval_channel: RetrievalChannel
    raw_rrf_score: float
    normalized_rrf_score: float
    fusion_rank: int


def compute_theoretical_max_rrf(
    *,
    rrf_k: int,
    vector_weight: float,
    lexical_weight: float,
) -> float:
    """Return the maximum possible raw RRF score when both ranks equal 1."""
    return vector_weight / (rrf_k + 1) + lexical_weight / (rrf_k + 1)


def compute_raw_rrf_score(
    *,
    vector_rank: int | None,
    lexical_rank: int | None,
    rrf_k: int,
    vector_weight: float,
    lexical_weight: float,
) -> float:
    """Compute raw reciprocal rank fusion score."""
    score = 0.0
    if vector_rank is not None:
        score += vector_weight / (rrf_k + vector_rank)
    if lexical_rank is not None:
        score += lexical_weight / (rrf_k + lexical_rank)
    return score


def normalize_rrf_score(raw_score: float, *, theoretical_max: float) -> float:
    """Normalize raw RRF into [0, 1] using the theoretical maximum."""
    if theoretical_max <= 0:
        return 0.0
    normalized = raw_score / theoretical_max
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def determine_retrieval_channel(
    *,
    vector_rank: int | None,
    lexical_rank: int | None,
) -> RetrievalChannel:
    if vector_rank is not None and lexical_rank is not None:
        return "both"
    if vector_rank is not None:
        return "vector_only"
    return "lexical_only"


def fuse_rankings(
    *,
    vector_results: list[SearchResult],
    lexical_results: list[LexicalHit],
    fusion_k: int,
    rrf_k: int,
    vector_weight: float,
    lexical_weight: float,
    document_ids: dict[str, str],
) -> list[FusionCandidate]:
    """Fuse vector and lexical rankings with equal-weight RRF."""
    vector_map = {item.chunk_id: (rank, item) for rank, item in enumerate(vector_results, start=1)}
    lexical_map = {
        item.chunk_id: (item.lexical_rank, item) for item in lexical_results
    }
    theoretical_max = compute_theoretical_max_rrf(
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        lexical_weight=lexical_weight,
    )

    fused_raw: list[tuple[float, str]] = []
    for chunk_id in set(vector_map) | set(lexical_map):
        vector_rank = vector_map[chunk_id][0] if chunk_id in vector_map else None
        lexical_rank = lexical_map[chunk_id][0] if chunk_id in lexical_map else None
        raw_score = compute_raw_rrf_score(
            vector_rank=vector_rank,
            lexical_rank=lexical_rank,
            rrf_k=rrf_k,
            vector_weight=vector_weight,
            lexical_weight=lexical_weight,
        )
        fused_raw.append((raw_score, chunk_id))

    fused_raw.sort(key=lambda item: (-item[0], item[1]))
    fused: list[FusionCandidate] = []
    for fusion_rank, (raw_score, chunk_id) in enumerate(fused_raw[:fusion_k], start=1):
        vector_rank, vector_item = vector_map.get(chunk_id, (None, None))
        lexical_rank, lexical_item = lexical_map.get(chunk_id, (None, None))
        fused.append(
            FusionCandidate(
                chunk_id=chunk_id,
                document_id=document_ids[chunk_id],
                vector_rank=vector_rank,
                vector_similarity=vector_item.similarity if vector_item is not None else None,
                vector_distance=vector_item.distance if vector_item is not None else None,
                lexical_rank=lexical_rank,
                bm25_score=lexical_item.bm25_score if lexical_item is not None else None,
                matched_tokens=list(lexical_item.matched_tokens) if lexical_item is not None else [],
                retrieval_channel=determine_retrieval_channel(
                    vector_rank=vector_rank,
                    lexical_rank=lexical_rank,
                ),
                raw_rrf_score=raw_score,
                normalized_rrf_score=normalize_rrf_score(raw_score, theoretical_max=theoretical_max),
                fusion_rank=fusion_rank,
            )
        )
    return fused
