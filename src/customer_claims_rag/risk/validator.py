"""Cross-field invariant validation for deterministic risk results."""

from __future__ import annotations

from customer_claims_rag.exceptions import RiskValidationError
from customer_claims_rag.risk.invariants import (
    build_risk_explanation,
    collect_result_invariant_errors,
    handoff_flags_for_level,
    order_reason_codes,
    order_risk_signals,
)
from customer_claims_rag.risk.models import (
    DeterministicRiskResult,
    RiskLevel,
    RiskSignal,
    max_risk_level,
    risk_rank,
)
from customer_claims_rag.risk.reason_codes import RiskReasonCode


def validate_deterministic_risk_result(result: DeterministicRiskResult) -> None:
    """Validate output invariants for a deterministic risk floor result."""
    errors = collect_result_invariant_errors(
        risk_floor=result.risk_floor,
        explicit_match=result.explicit_match,
        handoff_required=result.handoff_required,
        priority_handoff=result.priority_handoff,
        reason_codes=result.reason_codes,
        risk_signals=result.risk_signals,
        explanation=result.explanation,
    )
    if errors:
        raise RiskValidationError(errors[0])


def build_deterministic_risk_result(
    matched: list[RiskSignal],
) -> DeterministicRiskResult:
    """Build and validate a deterministic risk floor result from rule matches."""
    if not matched:
        explanation = build_risk_explanation(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            reason_codes=(),
            signal_count=0,
            handoff_required=False,
            priority_handoff=False,
        )
        return DeterministicRiskResult(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            handoff_required=False,
            priority_handoff=False,
            reason_codes=(),
            risk_signals=(),
            explanation=explanation,
        )

    seen: set[tuple[RiskReasonCode, RiskLevel, str]] = set()
    unique_signals: list[RiskSignal] = []
    for signal in matched:
        key = (signal.reason_code, signal.level, signal.rule_id)
        if key in seen:
            continue
        seen.add(key)
        unique_signals.append(signal)

    ordered_signals = order_risk_signals(unique_signals)

    unique_codes: dict[RiskReasonCode, RiskLevel] = {}
    for signal in ordered_signals:
        existing = unique_codes.get(signal.reason_code)
        if existing is None or risk_rank(signal.level) > risk_rank(existing):
            unique_codes[signal.reason_code] = signal.level

    floor = max_risk_level(*unique_codes.values())
    ordered_codes = order_reason_codes(unique_codes)
    handoff_required, priority_handoff = handoff_flags_for_level(floor)
    explanation = build_risk_explanation(
        risk_floor=floor,
        explicit_match=True,
        reason_codes=ordered_codes,
        signal_count=len(ordered_signals),
        handoff_required=handoff_required,
        priority_handoff=priority_handoff,
    )

    result = DeterministicRiskResult(
        risk_floor=floor,
        explicit_match=True,
        handoff_required=handoff_required,
        priority_handoff=priority_handoff,
        reason_codes=ordered_codes,
        risk_signals=ordered_signals,
        explanation=explanation,
    )
    validate_deterministic_risk_result(result)
    return result
