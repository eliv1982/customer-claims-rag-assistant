"""Grounded generation contract layer."""

from customer_claims_rag.generation.context_builder import build_context_package
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
    generation_failure_result,
)
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.handoff import (
    CRITICAL_HANDOFF_NOTICE,
    HIGH_HANDOFF_NOTICE,
    build_handoff_notice,
)
from customer_claims_rag.generation.models import (
    Citation,
    ContextItem,
    ContextPackage,
    GroundedGenerationRequest,
    GroundedGenerationResult,
    RawGenerationDraft,
)
from customer_claims_rag.generation.parser import parse_generation_draft
from customer_claims_rag.generation.risk_aware_generator import (
    GroundedGenerationPort,
    RiskAssessor,
    RiskAwareGroundedGenerator,
)
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGenerationOutcome,
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.generation.validator import (
    insufficient_context_for_empty_package,
    validate_generation_draft,
)

__all__ = [
    "Citation",
    "ContextItem",
    "ContextPackage",
    "CRITICAL_HANDOFF_NOTICE",
    "GENERATION_FAILURE_CUSTOMER_RESPONSE",
    "GroundedGenerationPort",
    "GroundedGenerationRequest",
    "GroundedGenerationResult",
    "GroundedGenerator",
    "HIGH_HANDOFF_NOTICE",
    "INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE",
    "RawGenerationDraft",
    "RiskAssessor",
    "RiskAwareGenerationOutcome",
    "RiskAwareGroundedGenerationResult",
    "RiskAwareGroundedGenerator",
    "build_context_package",
    "build_handoff_notice",
    "generation_failure_result",
    "insufficient_context_for_empty_package",
    "parse_generation_draft",
    "validate_generation_draft",
]
