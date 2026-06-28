"""Structural validation for grounded generation drafts."""

from __future__ import annotations

from customer_claims_rag.exceptions import GenerationValidationError
from customer_claims_rag.generation.citation_markers import extract_citation_markers
from customer_claims_rag.generation.fallback import (
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
    OUT_OF_SCOPE_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.models import (
    Citation,
    ContextPackage,
    GroundedGenerationResult,
    RawGenerationDraft,
)


def insufficient_context_for_empty_package() -> GroundedGenerationResult:
    """Return deterministic insufficient_context output for an empty context package."""
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
        citations=[],
    )


def validate_generation_draft(
    draft: RawGenerationDraft,
    context_package: ContextPackage,
) -> GroundedGenerationResult:
    """Validate a raw draft and build a client-safe generation result."""
    if draft.response_mode == "insufficient_context":
        return _validate_insufficient_context(draft)
    if draft.response_mode == "grounded_answer":
        return _validate_grounded_answer(draft, context_package)
    if draft.response_mode == "out_of_scope":
        return _validate_out_of_scope(draft)
    raise GenerationValidationError(f"unsupported response_mode: {draft.response_mode}")


def _validate_insufficient_context(draft: RawGenerationDraft) -> GroundedGenerationResult:
    if draft.answer.strip():
        raise GenerationValidationError(
            "insufficient_context answer must be empty or whitespace-only"
        )
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
        citations=[],
    )


def _validate_out_of_scope(draft: RawGenerationDraft) -> GroundedGenerationResult:
    if draft.answer.strip():
        raise GenerationValidationError(
            "out_of_scope answer must be empty or whitespace-only"
        )
    return GroundedGenerationResult(
        response_mode="out_of_scope",
        customer_response=OUT_OF_SCOPE_CUSTOMER_RESPONSE,
        citations=[],
    )


def _validate_grounded_answer(
    draft: RawGenerationDraft,
    context_package: ContextPackage,
) -> GroundedGenerationResult:
    answer = draft.answer
    if not answer.strip():
        raise GenerationValidationError("grounded_answer requires a non-empty answer")

    markers = extract_citation_markers(answer)
    if not markers:
        raise GenerationValidationError("grounded_answer requires at least one citation marker")

    items_by_key = {item.citation_key: item for item in context_package.items}
    citations: list[Citation] = []
    for marker in markers:
        if marker not in items_by_key:
            raise GenerationValidationError(f"unknown citation marker in answer: {marker}")
        item = items_by_key[marker]
        citations.append(
            Citation(
                citation_key=item.citation_key,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                heading=item.heading,
                source_path=item.source_path,
            )
        )

    return GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=answer,
        citations=citations,
    )
