"""Data models for deterministic risk floor (3B.1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_core import PydanticCustomError

from customer_claims_rag.exceptions import RiskValidationError
from customer_claims_rag.input_limits import MAX_CUSTOMER_QUERY_CHARS
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.invariants import (
    collect_result_invariant_errors,
    default_assessment_status,
)
from customer_claims_rag.risk.reason_codes import RiskReasonCode


class RiskLevel(StrEnum):
    """Canonical four-level risk taxonomy."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_RANK: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def risk_rank(level: RiskLevel) -> int:
    """Return deterministic ordering rank for a risk level."""
    return _RISK_RANK[level]


def max_risk_level(*levels: RiskLevel) -> RiskLevel:
    """Return the highest severity among the given levels."""
    if not levels:
        return RiskLevel.LOW
    return max(levels, key=risk_rank)


class RiskAssessmentRequest(BaseModel):
    """Production input for deterministic risk floor assessment."""

    model_config = ConfigDict(extra="forbid")

    customer_query: str

    @field_validator("customer_query")
    @classmethod
    def validate_customer_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("customer_query must not be empty or whitespace-only")
        # The safety rules run over the raw text, so the bound applies to it unstripped.
        if len(value) > MAX_CUSTOMER_QUERY_CHARS:
            raise PydanticCustomError(
                "customer_query_too_long",
                "customer_query must not exceed {max_chars} characters",
                {"max_chars": MAX_CUSTOMER_QUERY_CHARS},
            )
        return value


class RiskSignal(BaseModel):
    """Single deterministic risk signal raised by a rule match."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: RiskReasonCode
    level: RiskLevel
    rule_id: str

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("rule_id must not be empty or whitespace-only")
        return value


class DeterministicRiskResult(BaseModel):
    """Deterministic risk floor — not a final case classification.

    ``risk_floor`` is the lower bound of risk for the customer message.
    A later model-based assessment may raise the level but must never lower it.

    ``assessment_status`` says how the result came about and is what downstream code
    must use to tell the three situations apart:

    * ``RULE_MATCH`` - a deterministic rule matched (``explicit_match=True``). Only here can
      ``risk_floor=low`` be an affirmative rule outcome.
    * ``NO_SIGNAL`` - supported-language text and no rule matched. ``risk_floor=low`` is just
      the neutral lower bound; it does NOT mean the case was assessed as low risk.
    * ``UNSUPPORTED_LANGUAGE`` - the text cannot be assessed by the Russian rules. The case is
      not classified (``risk_floor=low`` is again only the neutral bound) and manual review is
      required (``handoff_required=True``).

    ``explicit_match=False`` therefore never guarantees the final case risk is ``low``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_floor: RiskLevel
    explicit_match: bool
    handoff_required: bool
    priority_handoff: bool
    reason_codes: tuple[RiskReasonCode, ...] = ()
    risk_signals: tuple[RiskSignal, ...] = ()
    explanation: str
    assessment_status: RiskAssessmentStatus

    @model_validator(mode="before")
    @classmethod
    def _derive_assessment_status(cls, data: Any) -> Any:
        """Keep historical constructors valid: status defaults from ``explicit_match``."""
        if isinstance(data, dict) and "assessment_status" not in data and "explicit_match" in data:
            explicit_match = data["explicit_match"]
            if isinstance(explicit_match, bool):
                return {**data, "assessment_status": default_assessment_status(explicit_match)}
        return data

    @model_validator(mode="after")
    def validate_cross_field_invariants(self) -> Self:
        errors = collect_result_invariant_errors(
            risk_floor=self.risk_floor,
            explicit_match=self.explicit_match,
            handoff_required=self.handoff_required,
            priority_handoff=self.priority_handoff,
            reason_codes=self.reason_codes,
            risk_signals=self.risk_signals,
            explanation=self.explanation,
            assessment_status=self.assessment_status,
        )
        if errors:
            raise RiskValidationError(errors[0])
        return self
