"""Data models for deterministic risk floor (3B.1)."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from customer_claims_rag.exceptions import RiskValidationError
from customer_claims_rag.risk.invariants import collect_result_invariant_errors
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

    ``explicit_match=False`` means deterministic rules found no elevating signal.
    That does not guarantee the final case risk is ``low``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_floor: RiskLevel
    explicit_match: bool
    handoff_required: bool
    priority_handoff: bool
    reason_codes: tuple[RiskReasonCode, ...] = ()
    risk_signals: tuple[RiskSignal, ...] = ()
    explanation: str

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
        )
        if errors:
            raise RiskValidationError(errors[0])
        return self
