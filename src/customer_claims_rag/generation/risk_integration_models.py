"""Risk-aware grounded generation result contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    OUT_OF_SCOPE_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.handoff import build_handoff_notice
from customer_claims_rag.generation.models import GroundedGenerationResult
from customer_claims_rag.risk.models import DeterministicRiskResult

RiskAwareGenerationOutcome = Literal[
    "grounded_answer",
    "insufficient_context",
    "out_of_scope",
    "generation_error_fallback",
]


class RiskAwareGroundedGenerationResult(BaseModel):
    """Risk-aware output combining grounded generation and deterministic risk floor.

    Nested ``generation`` response-mode and citation semantics are enforced by
    ``GroundedGenerationResult``. This model adds outcome, handoff, and
    generation-error fallback invariants on top of that contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation: GroundedGenerationResult
    risk_assessment: DeterministicRiskResult
    handoff_notice: str | None
    generation_outcome: RiskAwareGenerationOutcome

    @model_validator(mode="after")
    def validate_cross_field_invariants(self) -> Self:
        expected_notice = build_handoff_notice(self.risk_assessment)
        if self.handoff_notice != expected_notice:
            raise ValueError(
                "handoff_notice must match build_handoff_notice(risk_assessment)",
            )

        mode = self.generation.response_mode
        outcome = self.generation_outcome

        if outcome == "grounded_answer":
            if mode != "grounded_answer":
                raise ValueError(
                    "generation_outcome=grounded_answer requires "
                    "generation.response_mode=grounded_answer",
                )
            return self

        if outcome == "out_of_scope":
            if mode != "out_of_scope":
                raise ValueError(
                    "generation_outcome=out_of_scope requires "
                    "generation.response_mode=out_of_scope",
                )
            if self.generation.customer_response != OUT_OF_SCOPE_CUSTOMER_RESPONSE:
                raise ValueError(
                    "generation_outcome=out_of_scope requires "
                    "canonical out-of-scope customer response",
                )
            return self

        if mode != "insufficient_context":
            raise ValueError(
                f"generation_outcome={outcome} requires "
                "generation.response_mode=insufficient_context",
            )

        if outcome == "insufficient_context":
            if self.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE:
                raise ValueError(
                    "generation_outcome=insufficient_context must not use "
                    "generation failure customer response",
                )
            return self

        if outcome == "generation_error_fallback":
            if self.generation.customer_response != GENERATION_FAILURE_CUSTOMER_RESPONSE:
                raise ValueError(
                    "generation_outcome=generation_error_fallback requires "
                    "canonical generation failure customer response",
                )
            return self

        raise ValueError(f"unsupported generation_outcome: {outcome}")
