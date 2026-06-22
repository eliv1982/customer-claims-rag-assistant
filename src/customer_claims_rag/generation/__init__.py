"""Grounded generation contract layer."""

from customer_claims_rag.generation.context_builder import build_context_package
from customer_claims_rag.generation.fallback import INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.models import (
    Citation,
    ContextItem,
    ContextPackage,
    GroundedGenerationRequest,
    GroundedGenerationResult,
    RawGenerationDraft,
)
from customer_claims_rag.generation.parser import parse_generation_draft
from customer_claims_rag.generation.validator import (
    insufficient_context_for_empty_package,
    validate_generation_draft,
)

__all__ = [
    "Citation",
    "ContextItem",
    "ContextPackage",
    "GroundedGenerationRequest",
    "GroundedGenerationResult",
    "GroundedGenerator",
    "INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE",
    "RawGenerationDraft",
    "build_context_package",
    "insufficient_context_for_empty_package",
    "parse_generation_draft",
    "validate_generation_draft",
]
