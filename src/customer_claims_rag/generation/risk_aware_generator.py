"""Risk-aware orchestration over grounded generation."""

from __future__ import annotations

import logging
from typing import Protocol

from customer_claims_rag.exceptions import GenerationError
from customer_claims_rag.generation.fallback import generation_failure_result
from customer_claims_rag.generation.handoff import build_handoff_notice
from customer_claims_rag.generation.models import (
    GroundedGenerationRequest,
    GroundedGenerationResult,
)
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGenerationOutcome,
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskAssessmentRequest

logger = logging.getLogger(__name__)


class RiskAssessor(Protocol):
    """Callable contract for deterministic risk floor assessment."""

    def __call__(
        self,
        request: RiskAssessmentRequest,
    ) -> DeterministicRiskResult:
        ...


class GroundedGenerationPort(Protocol):
    """Minimal contract for grounded answer generation."""

    def generate(
        self,
        request: GroundedGenerationRequest,
    ) -> GroundedGenerationResult:
        ...


class RiskAwareGroundedGenerator:
    """Orchestrate deterministic risk assessment and grounded generation.

    Model-proposed risk is intentionally absent in 3B.2.
    ``apply_risk_floor()`` will be used only when a real proposed level exists.
    """

    def __init__(
        self,
        grounded_generator: GroundedGenerationPort,
        risk_assessor: RiskAssessor = assess_deterministic_risk,
    ) -> None:
        self._grounded_generator = grounded_generator
        self._risk_assessor = risk_assessor

    def generate(
        self,
        request: GroundedGenerationRequest,
        risk_assessment: DeterministicRiskResult | None = None,
    ) -> RiskAwareGroundedGenerationResult:
        """Generate a grounded answer and attach the deterministic risk assessment.

        The application pipeline assesses risk once per request, before retrieval, and passes
        that result in; it is used as is and never recomputed. Only a standalone caller that
        passes no assessment gets one computed here from the request text.
        """
        if risk_assessment is None:
            risk_request = RiskAssessmentRequest(customer_query=request.customer_query)
            risk_assessment = self._risk_assessor(risk_request)

        try:
            generation = self._grounded_generator.generate(request)
        except GenerationError as exc:
            logger.warning(
                "Generation failed with %s: %s",
                type(exc).__name__,
                str(exc),
            )
            generation = generation_failure_result()
            generation_outcome: RiskAwareGenerationOutcome = "generation_error_fallback"
        else:
            generation_outcome = _generation_outcome_for_result(generation)

        handoff_notice = build_handoff_notice(risk_assessment)

        return RiskAwareGroundedGenerationResult(
            generation=generation,
            risk_assessment=risk_assessment,
            handoff_notice=handoff_notice,
            generation_outcome=generation_outcome,
        )


def _generation_outcome_for_result(
    generation: GroundedGenerationResult,
) -> RiskAwareGenerationOutcome:
    if generation.response_mode == "grounded_answer":
        return "grounded_answer"
    if generation.response_mode == "insufficient_context":
        return "insufficient_context"
    if generation.response_mode == "out_of_scope":
        return "out_of_scope"
    raise GenerationError(
        f"unsupported generation response_mode: {generation.response_mode}",
    )
