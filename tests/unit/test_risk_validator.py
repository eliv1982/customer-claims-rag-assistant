"""Unit tests for deterministic risk result validator."""

from __future__ import annotations

import pytest

from customer_claims_rag.exceptions import RiskValidationError
from customer_claims_rag.risk.invariants import build_risk_explanation
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskLevel, RiskSignal
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.validator import (
    build_deterministic_risk_result,
    handoff_flags_for_level,
    validate_deterministic_risk_result,
)


def _signal(
    reason_code: RiskReasonCode,
    level: RiskLevel,
    rule_id: str,
) -> RiskSignal:
    return RiskSignal(reason_code=reason_code, level=level, rule_id=rule_id)


def test_no_match_contract_via_builder() -> None:
    result = build_deterministic_risk_result([])
    assert result.explicit_match is False
    assert result.risk_floor is RiskLevel.LOW
    assert result.reason_codes == ()
    assert result.risk_signals == ()
    assert result.handoff_required is False
    assert result.priority_handoff is False
    assert result.explanation == (
        "risk_floor=low;explicit_match=false;reason_codes=none;"
        "signal_count=0;handoff_required=false;priority_handoff=false"
    )


def test_handoff_flags_by_level() -> None:
    assert handoff_flags_for_level(RiskLevel.LOW) == (False, False)
    assert handoff_flags_for_level(RiskLevel.MEDIUM) == (False, False)
    assert handoff_flags_for_level(RiskLevel.HIGH) == (True, False)
    assert handoff_flags_for_level(RiskLevel.CRITICAL) == (True, True)


def test_explicit_match_requires_reason_codes() -> None:
    with pytest.raises(RiskValidationError, match="non-empty reason_codes"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.HIGH,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=False,
            reason_codes=(),
            risk_signals=(_signal(RiskReasonCode.NON_DELIVERY, RiskLevel.HIGH, "non_delivery"),),
            explanation="ignored",
        )


def test_no_explicit_match_rejects_non_low_floor() -> None:
    with pytest.raises(RiskValidationError, match="risk_floor=low"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.HIGH,
            explicit_match=False,
            handoff_required=False,
            priority_handoff=False,
            risk_signals=(),
            explanation=(
                "risk_floor=low;explicit_match=false;reason_codes=none;"
                "signal_count=0;handoff_required=false;priority_handoff=false"
            ),
        )


def test_priority_handoff_requires_handoff_required() -> None:
    signal = _signal(RiskReasonCode.DIRECT_THREAT, RiskLevel.CRITICAL, "direct_threat")
    explanation = build_risk_explanation(
        risk_floor=RiskLevel.CRITICAL,
        explicit_match=True,
        reason_codes=(RiskReasonCode.DIRECT_THREAT,),
        signal_count=1,
        handoff_required=False,
        priority_handoff=True,
    )
    with pytest.raises(RiskValidationError, match="handoff_required must be True"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.CRITICAL,
            explicit_match=True,
            handoff_required=False,
            priority_handoff=True,
            reason_codes=(RiskReasonCode.DIRECT_THREAT,),
            risk_signals=(signal,),
            explanation=explanation,
        )


def test_builder_orders_reason_codes_severity_first() -> None:
    result = build_deterministic_risk_result(
        [
            _signal(RiskReasonCode.REFUND_REQUEST, RiskLevel.MEDIUM, "refund_request"),
            _signal(RiskReasonCode.DIRECT_THREAT, RiskLevel.CRITICAL, "direct_threat"),
            _signal(RiskReasonCode.DELAY_OVER_30_MINUTES, RiskLevel.MEDIUM, "delay_over_30_minutes"),
        ],
    )
    assert result.risk_floor is RiskLevel.CRITICAL
    assert result.reason_codes[0] is RiskReasonCode.DIRECT_THREAT
    assert result.risk_signals[0].reason_code is RiskReasonCode.DIRECT_THREAT


def test_builder_deduplicates_reason_codes_and_signals() -> None:
    signal = _signal(RiskReasonCode.DELAY_OVER_30_MINUTES, RiskLevel.MEDIUM, "delay_over_30_minutes")
    result = build_deterministic_risk_result([signal, signal])
    assert result.reason_codes == (RiskReasonCode.DELAY_OVER_30_MINUTES,)
    assert result.risk_signals == (signal,)


def test_builder_keeps_distinct_signals_for_same_code_different_levels() -> None:
    high_signal = _signal(
        RiskReasonCode.PERSONAL_DATA_EXPOSURE,
        RiskLevel.HIGH,
        "personal_data_exposure_high",
    )
    critical_signal = _signal(
        RiskReasonCode.PERSONAL_DATA_EXPOSURE,
        RiskLevel.CRITICAL,
        "personal_data_exposure_critical",
    )
    result = build_deterministic_risk_result([high_signal, critical_signal])
    assert result.risk_floor is RiskLevel.CRITICAL
    assert result.reason_codes == (RiskReasonCode.PERSONAL_DATA_EXPOSURE,)
    assert result.risk_signals == (critical_signal, high_signal)


def test_direct_construction_rejects_duplicate_reason_codes() -> None:
    signal = _signal(RiskReasonCode.FRAUD_INDICATORS, RiskLevel.CRITICAL, "fraud_indicators")
    with pytest.raises(RiskValidationError, match="duplicates"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.CRITICAL,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=True,
            reason_codes=(
                RiskReasonCode.FRAUD_INDICATORS,
                RiskReasonCode.FRAUD_INDICATORS,
            ),
            risk_signals=(signal,),
            explanation="ignored",
        )


def test_validate_deterministic_risk_result_accepts_builder_output() -> None:
    result = build_deterministic_risk_result(
        [_signal(RiskReasonCode.MISSING_ITEM, RiskLevel.MEDIUM, "missing_item")],
    )
    validate_deterministic_risk_result(result)
