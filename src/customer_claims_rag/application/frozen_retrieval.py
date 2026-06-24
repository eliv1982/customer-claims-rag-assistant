"""Production frozen retrieval service for the application layer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Protocol

from customer_claims_rag.application.settings import FrozenRetrievalConfig
from customer_claims_rag.exceptions import RetrievalError
from customer_claims_rag.retrieval.models import SearchResponse, SearchResult
from customer_claims_rag.retrieval.reranker import RerankedCandidate


class BaselineSearchPort(Protocol):
    """Minimal baseline vector search contract for frozen retrieval wiring."""

    def search(self, query: str) -> SearchResponse:
        ...


class RerankerPort(Protocol):
    """Minimal reranker contract for frozen retrieval wiring."""

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[RerankedCandidate]:
        ...


def _prepare_candidate_pool(
    results: Sequence[SearchResult],
    *,
    pool_k: int,
) -> list[SearchResult]:
    """Return a deep-copied pool prefix with ranks normalized to 1..N."""
    pool = deepcopy(list(results)[:pool_k])
    for index, item in enumerate(pool, start=1):
        item.rank = index
    return pool


def _validate_contiguous_ranks(results: Sequence[SearchResult], *, label: str) -> None:
    if not results:
        return
    expected_ranks = list(range(1, len(results) + 1))
    actual_ranks = [item.rank for item in results]
    if actual_ranks != expected_ranks:
        raise RetrievalError(
            f"{label} ranks must be contiguous 1..{len(results)}, "
            f"got {actual_ranks}",
        )


class FrozenRetrievalService:
    """Frozen vector-pool + source-authority rerank retrieval for production use."""

    def __init__(
        self,
        retriever: BaselineSearchPort,
        reranker: RerankerPort,
        config: FrozenRetrievalConfig,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._config = config
        self._reranker_config_id = self._validate_reranker_id(reranker)

    @property
    def retrieval_config_id(self) -> str:
        return self._config.config_id

    @property
    def retrieval_config_version(self) -> str:
        return self._config.version

    @property
    def reranker_config_id(self) -> str:
        return self._reranker_config_id

    def search(self, query: str) -> list[SearchResult]:
        response = self._retriever.search(query)
        self._validate_search_response(response)

        if not response.results:
            return []

        candidate_pool = _prepare_candidate_pool(
            response.results,
            pool_k=self._config.candidate_pool_k,
        )
        reranked = self._reranker.rerank(query, candidate_pool)
        final_results = [
            candidate.result for candidate in reranked[: self._config.final_top_k]
        ]
        _validate_contiguous_ranks(final_results, label="final retrieval")
        return final_results

    def _validate_reranker_id(self, reranker: RerankerPort) -> str:
        actual_id = reranker.config.reranker_id  # type: ignore[attr-defined]
        expected_id = self._config.reranker_id
        if actual_id != expected_id:
            raise ValueError(
                f"Frozen retrieval config requires reranker {expected_id!r}, "
                f"but received {actual_id!r}.",
            )
        return actual_id

    def _validate_search_response(self, response: SearchResponse) -> None:
        config = self._config
        if response.top_k != config.vector_top_k:
            raise RetrievalError(
                "retriever top_k mismatch: "
                f"expected {config.vector_top_k}, got {response.top_k}",
            )
        if response.fetch_k != config.vector_fetch_k:
            raise RetrievalError(
                "retriever fetch_k mismatch: "
                f"expected {config.vector_fetch_k}, got {response.fetch_k}",
            )
        if not math.isclose(
            response.similarity_threshold,
            config.similarity_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RetrievalError(
                "retriever similarity_threshold mismatch: "
                f"expected {config.similarity_threshold}, "
                f"got {response.similarity_threshold}",
            )
        if len(response.results) > response.top_k:
            raise RetrievalError(
                "retriever returned more results than top_k: "
                f"top_k={response.top_k}, results={len(response.results)}",
            )
