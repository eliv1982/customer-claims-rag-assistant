"""Unit tests for deterministic risk models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from customer_claims_rag.exceptions import RiskValidationError
from customer_claims_rag.risk.invariants import build_risk_explanation
from customer_claims_rag.risk.models import (
    DeterministicRiskResult,
    RiskAssessmentRequest,
    RiskLevel,
    RiskSignal,
    max_risk_level,
    risk_rank,
)
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.validator import build_deterministic_risk_result


def _signal(
    reason_code: RiskReasonCode,
    level: RiskLevel,
    rule_id: str,
) -> RiskSignal:
    return RiskSignal(reason_code=reason_code, level=level, rule_id=rule_id)


def test_request_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        RiskAssessmentRequest(customer_query="hello", unexpected=True)  # type: ignore[call-arg]


def test_empty_customer_query_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskAssessmentRequest(customer_query="")


def test_whitespace_only_customer_query_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskAssessmentRequest(customer_query="   \n\t")


def test_valid_request_preserves_customer_query() -> None:
    original = "  FF-1, опоздание 31 минута  "
    request = RiskAssessmentRequest(customer_query=original)
    assert request.customer_query == original


def test_risk_level_values() -> None:
    assert RiskLevel.LOW.value == "low"
    assert RiskLevel.MEDIUM.value == "medium"
    assert RiskLevel.HIGH.value == "high"
    assert RiskLevel.CRITICAL.value == "critical"


def test_risk_rank_ordering() -> None:
    assert risk_rank(RiskLevel.LOW) < risk_rank(RiskLevel.MEDIUM)
    assert risk_rank(RiskLevel.MEDIUM) < risk_rank(RiskLevel.HIGH)
    assert risk_rank(RiskLevel.HIGH) < risk_rank(RiskLevel.CRITICAL)


def test_max_risk_level() -> None:
    assert max_risk_level(RiskLevel.LOW, RiskLevel.HIGH) is RiskLevel.HIGH
    assert max_risk_level(RiskLevel.MEDIUM, RiskLevel.CRITICAL) is RiskLevel.CRITICAL


def test_risk_signal_valid_construction() -> None:
    signal = _signal(
        RiskReasonCode.PACKAGE_TAMPERING,
        RiskLevel.HIGH,
        "package_tampering_explicit",
    )
    assert signal.reason_code is RiskReasonCode.PACKAGE_TAMPERING
    assert signal.level is RiskLevel.HIGH
    assert signal.rule_id == "package_tampering_explicit"


def test_risk_signal_empty_rule_id_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskSignal(
            reason_code=RiskReasonCode.MISSING_ITEM,
            level=RiskLevel.MEDIUM,
            rule_id="",
        )


def test_risk_signal_whitespace_rule_id_rejected() -> None:
    with pytest.raises(ValidationError):
        RiskSignal(
            reason_code=RiskReasonCode.MISSING_ITEM,
            level=RiskLevel.MEDIUM,
            rule_id="   ",
        )


def test_risk_signal_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        RiskSignal(
            reason_code=RiskReasonCode.MISSING_ITEM,
            level=RiskLevel.MEDIUM,
            rule_id="missing_item",
            unexpected=True,  # type: ignore[call-arg]
        )


def test_risk_signal_is_frozen() -> None:
    signal = _signal(RiskReasonCode.MISSING_ITEM, RiskLevel.MEDIUM, "missing_item")
    with pytest.raises(ValidationError):
        signal.rule_id = "other"  # type: ignore[misc]


def test_risk_signal_json_serialization() -> None:
    signal = _signal(
        RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION,
        RiskLevel.CRITICAL,
        "health_symptoms_after_consumption",
    )
    dumped = signal.model_dump(mode="json")
    assert dumped == {
        "reason_code": "health_symptoms_after_consumption",
        "level": "critical",
        "rule_id": "health_symptoms_after_consumption",
    }


def test_result_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        DeterministicRiskResult(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            handoff_required=False,
            priority_handoff=False,
            explanation="risk_floor=low",
            unexpected=True,  # type: ignore[call-arg]
        )


def test_result_is_frozen() -> None:
    result = build_deterministic_risk_result([])
    with pytest.raises(ValidationError):
        result.risk_floor = RiskLevel.HIGH  # type: ignore[misc]


def test_result_default_reason_codes_and_signals_empty_for_no_match() -> None:
    result = build_deterministic_risk_result([])
    assert result.reason_codes == ()
    assert result.risk_signals == ()


def test_result_has_no_customer_query_field() -> None:
    assert "customer_query" not in DeterministicRiskResult.model_fields


def test_result_rejects_duplicate_reason_codes() -> None:
    signal = _signal(
        RiskReasonCode.DELAY_OVER_30_MINUTES,
        RiskLevel.MEDIUM,
        "delay_over_30_minutes",
    )
    with pytest.raises(RiskValidationError, match="duplicates"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.MEDIUM,
            explicit_match=True,
            handoff_required=False,
            priority_handoff=False,
            reason_codes=(
                RiskReasonCode.DELAY_OVER_30_MINUTES,
                RiskReasonCode.DELAY_OVER_30_MINUTES,
            ),
            risk_signals=(signal,),
            explanation="ignored",
        )


def test_result_rejects_critical_without_priority_handoff() -> None:
    signal = _signal(RiskReasonCode.DIRECT_THREAT, RiskLevel.CRITICAL, "direct_threat")
    explanation = build_risk_explanation(
        risk_floor=RiskLevel.CRITICAL,
        explicit_match=True,
        reason_codes=(RiskReasonCode.DIRECT_THREAT,),
        signal_count=1,
        handoff_required=True,
        priority_handoff=False,
    )
    with pytest.raises(RiskValidationError, match="priority_handoff"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.CRITICAL,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=False,
            reason_codes=(RiskReasonCode.DIRECT_THREAT,),
            risk_signals=(signal,),
            explanation=explanation,
        )


def test_result_rejects_explicit_match_with_empty_reasons() -> None:
    with pytest.raises(RiskValidationError, match="non-empty reason_codes"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.LOW,
            explicit_match=True,
            handoff_required=False,
            priority_handoff=False,
            reason_codes=(),
            risk_signals=(),
            explanation="ignored",
        )


def test_result_explicit_match_requires_risk_signals() -> None:
    explanation = build_risk_explanation(
        risk_floor=RiskLevel.HIGH,
        explicit_match=True,
        reason_codes=(RiskReasonCode.NON_DELIVERY,),
        signal_count=0,
        handoff_required=True,
        priority_handoff=False,
    )
    with pytest.raises(RiskValidationError, match="non-empty risk_signals"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.HIGH,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=False,
            reason_codes=(RiskReasonCode.NON_DELIVERY,),
            risk_signals=(),
            explanation=explanation,
        )


def test_result_reason_codes_must_match_signals() -> None:
    signal = _signal(RiskReasonCode.MISSING_ITEM, RiskLevel.MEDIUM, "missing_item")
    explanation = build_risk_explanation(
        risk_floor=RiskLevel.HIGH,
        explicit_match=True,
        reason_codes=(RiskReasonCode.NON_DELIVERY,),
        signal_count=1,
        handoff_required=True,
        priority_handoff=False,
    )
    with pytest.raises(RiskValidationError, match="must be represented"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.HIGH,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=False,
            reason_codes=(RiskReasonCode.NON_DELIVERY,),
            risk_signals=(signal,),
            explanation=explanation,
        )


def test_result_risk_floor_must_equal_max_signal_level() -> None:
    signal = _signal(RiskReasonCode.DIRECT_THREAT, RiskLevel.CRITICAL, "direct_threat")
    explanation = build_risk_explanation(
        risk_floor=RiskLevel.HIGH,
        explicit_match=True,
        reason_codes=(RiskReasonCode.DIRECT_THREAT,),
        signal_count=1,
        handoff_required=True,
        priority_handoff=False,
    )
    with pytest.raises(RiskValidationError, match="maximum risk_signal level"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.HIGH,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=False,
            reason_codes=(RiskReasonCode.DIRECT_THREAT,),
            risk_signals=(signal,),
            explanation=explanation,
        )


def test_result_rejects_non_canonical_explanation() -> None:
    result = build_deterministic_risk_result(
        [_signal(RiskReasonCode.MISSING_ITEM, RiskLevel.MEDIUM, "missing_item")],
    )
    with pytest.raises(RiskValidationError, match="canonical"):
        DeterministicRiskResult(
            risk_floor=result.risk_floor,
            explicit_match=result.explicit_match,
            handoff_required=result.handoff_required,
            priority_handoff=result.priority_handoff,
            reason_codes=result.reason_codes,
            risk_signals=result.risk_signals,
            explanation="risk_floor=low;explicit_match=false",
        )


def test_result_explanation_is_canonical_and_stable() -> None:
    signals = [
        _signal(RiskReasonCode.PACKAGE_TAMPERING, RiskLevel.HIGH, "package_tampering_explicit"),
        _signal(
            RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE,
            RiskLevel.HIGH,
            "official_written_response_demand",
        ),
    ]
    first = build_deterministic_risk_result(signals)
    second = build_deterministic_risk_result(list(reversed(signals)))
    expected = (
        "risk_floor=high;explicit_match=true;"
        "reason_codes=package_tampering,official_written_response;"
        "signal_count=2;handoff_required=true;priority_handoff=false"
    )
    assert first.explanation == expected
    assert second.explanation == expected
    assert "претенз" not in first.explanation


def test_result_model_dump_json_round_trip() -> None:
    result = build_deterministic_risk_result(
        [
            _signal(
                RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION,
                RiskLevel.CRITICAL,
                "health_symptoms_after_consumption",
            ),
        ],
    )
    dumped = result.model_dump(mode="json")
    restored = DeterministicRiskResult.model_validate(dumped)
    assert restored == result


def test_result_signal_deduplication_via_builder() -> None:
    signal = _signal(RiskReasonCode.REFUND_REQUEST, RiskLevel.MEDIUM, "refund_request")
    result = build_deterministic_risk_result([signal, signal])
    assert result.risk_signals == (signal,)
    assert "signal_count=1" in result.explanation


def test_reason_code_registry_values() -> None:
    assert RiskReasonCode.DELAY_OVER_30_MINUTES.value == "delay_over_30_minutes"
    assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION.value == (
        "health_symptoms_after_consumption"
    )
    assert RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE.value == "official_written_response"


def test_direct_construction_positive_canonical_fields() -> None:
    built = build_deterministic_risk_result(
        [_signal(RiskReasonCode.MISSING_ITEM, RiskLevel.MEDIUM, "missing_item")],
    )
    direct = DeterministicRiskResult(
        risk_floor=built.risk_floor,
        explicit_match=built.explicit_match,
        handoff_required=built.handoff_required,
        priority_handoff=built.priority_handoff,
        reason_codes=built.reason_codes,
        risk_signals=built.risk_signals,
        explanation=built.explanation,
    )
    assert direct == built


def test_direct_construction_rejects_no_match_explanation_tampering() -> None:
    with pytest.raises(RiskValidationError, match="canonical"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            handoff_required=False,
            priority_handoff=False,
            reason_codes=(),
            risk_signals=(),
            explanation="anything",
        )


def test_direct_construction_rejects_extra_signal_code_absent_from_reason_codes() -> None:
    explanation = build_risk_explanation(
        risk_floor=RiskLevel.HIGH,
        explicit_match=True,
        reason_codes=(RiskReasonCode.PACKAGE_TAMPERING,),
        signal_count=2,
        handoff_required=True,
        priority_handoff=False,
    )
    with pytest.raises(RiskValidationError, match="must exist in reason_codes"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.HIGH,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=False,
            reason_codes=(RiskReasonCode.PACKAGE_TAMPERING,),
            risk_signals=(
                _signal(
                    RiskReasonCode.PACKAGE_TAMPERING,
                    RiskLevel.HIGH,
                    "package_tampering_explicit",
                ),
                _signal(
                    RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE,
                    RiskLevel.HIGH,
                    "official_written_response_demand",
                ),
            ),
            explanation=explanation,
        )


def test_direct_construction_rejects_non_canonical_reason_codes_order() -> None:
    tampering = _signal(
        RiskReasonCode.PACKAGE_TAMPERING,
        RiskLevel.HIGH,
        "package_tampering_explicit",
    )
    official = _signal(
        RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE,
        RiskLevel.HIGH,
        "official_written_response_demand",
    )
    canonical = build_deterministic_risk_result([tampering, official])
    with pytest.raises(RiskValidationError, match="canonical order"):
        DeterministicRiskResult(
            risk_floor=canonical.risk_floor,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=False,
            reason_codes=(
                RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE,
                RiskReasonCode.PACKAGE_TAMPERING,
            ),
            risk_signals=canonical.risk_signals,
            explanation=canonical.explanation,
        )


def test_direct_construction_rejects_non_canonical_risk_signals_order() -> None:
    tampering = _signal(
        RiskReasonCode.PACKAGE_TAMPERING,
        RiskLevel.HIGH,
        "package_tampering_explicit",
    )
    official = _signal(
        RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE,
        RiskLevel.HIGH,
        "official_written_response_demand",
    )
    canonical = build_deterministic_risk_result([tampering, official])
    with pytest.raises(RiskValidationError, match="canonical order"):
        DeterministicRiskResult(
            risk_floor=canonical.risk_floor,
            explicit_match=True,
            handoff_required=True,
            priority_handoff=False,
            reason_codes=canonical.reason_codes,
            risk_signals=(official, tampering),
            explanation=canonical.explanation,
        )
