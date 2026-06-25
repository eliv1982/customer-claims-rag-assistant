"""Deterministic per-document cap helper for vector pool shaping."""

from __future__ import annotations

from copy import deepcopy

from customer_claims_rag.evaluation.diversity_models import CapPoolDiagnostics, CappedPoolEntry
from customer_claims_rag.evaluation.pool_expansion_metrics import prepare_pool
from customer_claims_rag.retrieval.models import SearchResult


class CapPoolValidationError(ValueError):
    """Raised when cap parameters are invalid."""


def validate_cap_params(
    *,
    fetch_k: int,
    pool_k: int,
    per_document_cap: int | None,
) -> None:
    """Validate positive fetch/pool sizes and logical cap constraints."""
    if fetch_k <= 0:
        raise CapPoolValidationError("fetch_k must be positive")
    if pool_k <= 0:
        raise CapPoolValidationError("pool_k must be positive")
    if pool_k > fetch_k:
        raise CapPoolValidationError("pool_k must not exceed fetch_k")
    if per_document_cap is not None and per_document_cap <= 0:
        raise CapPoolValidationError("per_document_cap must be positive when provided")


def apply_per_document_cap(
    vector_results: list[SearchResult],
    *,
    fetch_k: int,
    pool_k: int,
    per_document_cap: int | None,
) -> tuple[list[SearchResult], CapPoolDiagnostics]:
    """Shape a vector result list into a capped candidate pool.

    Preserves original vector similarity order, stores vector ranks in diagnostics,
    deep-copies ``SearchResult`` items before reassigning pool ranks, and never
    mutates the input list or its result objects.
    """
    validate_cap_params(fetch_k=fetch_k, pool_k=pool_k, per_document_cap=per_document_cap)

    if not vector_results:
        return [], _empty_diagnostics(pool_k=pool_k, cap_applied=per_document_cap is not None)

    deep_results = list(vector_results[:fetch_k])

    if per_document_cap is None:
        pool = prepare_pool(deep_results, pool_k)
        entries = [
            CappedPoolEntry(result=item, vector_rank=index)
            for index, item in enumerate(pool, start=1)
        ]
        document_counts = _count_documents(pool)
        return pool, CapPoolDiagnostics(
            pool_size=len(pool),
            removed_by_cap_count=0,
            cap_applied=False,
            unique_document_count=len(document_counts),
            max_chunks_from_one_document=max(document_counts.values(), default=0),
            document_counts=document_counts,
            pool_shorter_than_target=len(pool) < pool_k,
            entries=entries,
        )

    capped_pool: list[SearchResult] = []
    entries: list[CappedPoolEntry] = []
    document_counts: dict[str, int] = {}
    removed_by_cap = 0

    for vector_rank, result in enumerate(deep_results, start=1):
        document_id = result.document_id
        current_count = document_counts.get(document_id, 0)
        if current_count >= per_document_cap:
            removed_by_cap += 1
            continue

        copied = deepcopy(result)
        copied.rank = len(capped_pool) + 1
        capped_pool.append(copied)
        entries.append(CappedPoolEntry(result=copied, vector_rank=vector_rank))
        document_counts[document_id] = current_count + 1

        if len(capped_pool) == pool_k:
            break

    return capped_pool, CapPoolDiagnostics(
        pool_size=len(capped_pool),
        removed_by_cap_count=removed_by_cap,
        cap_applied=True,
        unique_document_count=len(document_counts),
        max_chunks_from_one_document=max(document_counts.values(), default=0),
        document_counts=document_counts,
        pool_shorter_than_target=len(capped_pool) < pool_k,
        entries=entries,
    )


def _count_documents(pool: list[SearchResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in pool:
        counts[item.document_id] = counts.get(item.document_id, 0) + 1
    return counts


def _empty_diagnostics(*, pool_k: int, cap_applied: bool) -> CapPoolDiagnostics:
    return CapPoolDiagnostics(
        pool_size=0,
        removed_by_cap_count=0,
        cap_applied=cap_applied,
        unique_document_count=0,
        max_chunks_from_one_document=0,
        document_counts={},
        pool_shorter_than_target=pool_k > 0,
        entries=[],
    )


__all__ = [
    "CapPoolValidationError",
    "apply_per_document_cap",
    "validate_cap_params",
]
