"""Build ContextPackage from ranked retrieval results."""

from __future__ import annotations

from collections.abc import Sequence

from customer_claims_rag.exceptions import GenerationValidationError
from customer_claims_rag.generation.models import ContextItem, ContextPackage
from customer_claims_rag.retrieval.models import SearchResult


def build_context_package(results: Sequence[SearchResult]) -> ContextPackage:
    """Map ranked SearchResult items to a deterministic ContextPackage."""
    if not results:
        return ContextPackage(items=[])

    ordered = sorted(results, key=lambda item: item.rank)
    _validate_unique_ranks_and_chunk_ids(ordered)

    items: list[ContextItem] = []
    for index, result in enumerate(ordered, start=1):
        items.append(
            ContextItem(
                citation_key=f"S{index}",
                rank=result.rank,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                heading=result.heading,
                source_path=result.source_path,
                content=result.content,
            )
        )
    return ContextPackage(items=items)


def _validate_unique_ranks_and_chunk_ids(results: Sequence[SearchResult]) -> None:
    seen_ranks: set[int] = set()
    seen_chunk_ids: set[str] = set()
    for result in results:
        if result.rank in seen_ranks:
            raise GenerationValidationError(f"duplicate rank in context input: {result.rank}")
        if result.chunk_id in seen_chunk_ids:
            raise GenerationValidationError(
                f"duplicate chunk_id in context input: {result.chunk_id}"
            )
        seen_ranks.add(result.rank)
        seen_chunk_ids.add(result.chunk_id)
