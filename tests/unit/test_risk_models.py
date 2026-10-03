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


# ---------------------------------------------------------------------------
# Stage 2B: input bound and explicit assessment status
# ---------------------------------------------------------------------------

from customer_claims_rag.generation.handoff import (  # noqa: E402
    UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE,
    build_handoff_notice,
)
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus  # noqa: E402
from customer_claims_rag.risk.validator import build_unsupported_language_result  # noqa: E402


def test_risk_request_accepts_exactly_4000_characters() -> None:
    request = RiskAssessmentRequest(customer_query="а" * 4000)
    assert len(request.customer_query) == 4000


def test_risk_request_rejects_4001_characters() -> None:
    with pytest.raises(ValidationError, match="4000"):
        RiskAssessmentRequest(customer_query="а" * 4001)


def test_oversized_request_never_reaches_any_safety_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    from customer_claims_rag.risk import rules

    def _must_not_run(_: str) -> str:
        raise AssertionError("safety rules ran on an oversized request")

    monkeypatch.setattr(rules, "normalize_for_matching", _must_not_run)

    with pytest.raises(ValidationError):
        rules.assess_deterministic_risk(RiskAssessmentRequest(customer_query="а" * 4001))


def test_status_defaults_follow_explicit_match_for_historical_constructors() -> None:
    no_signal = build_deterministic_risk_result([])
    matched = build_deterministic_risk_result(
        [_signal(RiskReasonCode.MISSING_ITEM, RiskLevel.MEDIUM, "missing_item")],
    )
    assert no_signal.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert matched.assessment_status is RiskAssessmentStatus.RULE_MATCH

    rebuilt = DeterministicRiskResult(
        risk_floor=matched.risk_floor,
        explicit_match=matched.explicit_match,
        handoff_required=matched.handoff_required,
        priority_handoff=matched.priority_handoff,
        reason_codes=matched.reason_codes,
        risk_signals=matched.risk_signals,
        explanation=matched.explanation,
    )
    assert rebuilt.assessment_status is RiskAssessmentStatus.RULE_MATCH


def test_no_signal_is_not_an_affirmative_low_conclusion() -> None:
    """A LOW floor without a rule match is the absence of a signal, not a LOW verdict."""
    no_signal = build_deterministic_risk_result([])
    assert no_signal.risk_floor is RiskLevel.LOW
    assert no_signal.explicit_match is False
    assert no_signal.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert no_signal.risk_signals == ()
    assert no_signal.reason_codes == ()


def test_affirmative_low_rule_match_is_distinguishable_from_no_signal() -> None:
    low_signal = _signal(RiskReasonCode.MISSING_ITEM, RiskLevel.LOW, "hypothetical_low_rule")
    matched_low = build_deterministic_risk_result([low_signal])
    no_signal = build_deterministic_risk_result([])

    assert matched_low.risk_floor is RiskLevel.LOW
    assert no_signal.risk_floor is RiskLevel.LOW
    assert matched_low.explicit_match is True
    assert matched_low.assessment_status is RiskAssessmentStatus.RULE_MATCH
    assert no_signal.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert matched_low != no_signal


def test_no_signal_explanation_keeps_its_historical_canonical_form() -> None:
    assert build_deterministic_risk_result([]).explanation == (
        "risk_floor=low;explicit_match=false;reason_codes=none;"
        "signal_count=0;handoff_required=false;priority_handoff=false"
    )


def test_unsupported_language_result_is_unclassified_but_requires_manual_review() -> None:
    result = build_unsupported_language_result()

    assert result.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE
    assert result.explicit_match is False
    assert result.risk_signals == ()
    assert result.reason_codes == ()
    assert result.handoff_required is True
    assert result.priority_handoff is False
    assert result.explanation.endswith(";assessment_status=unsupported_language")
    assert build_handoff_notice(result) == UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE


def test_unsupported_language_result_json_round_trip() -> None:
    result = build_unsupported_language_result()
    restored = DeterministicRiskResult.model_validate(result.model_dump(mode="json"))
    assert restored == result
    assert restored.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE


def test_status_must_agree_with_explicit_match() -> None:
    matched = build_deterministic_risk_result(
        [_signal(RiskReasonCode.MISSING_ITEM, RiskLevel.MEDIUM, "missing_item")],
    )
    with pytest.raises(RiskValidationError, match="rule_match"):
        DeterministicRiskResult(
            risk_floor=matched.risk_floor,
            explicit_match=True,
            handoff_required=matched.handoff_required,
            priority_handoff=matched.priority_handoff,
            reason_codes=matched.reason_codes,
            risk_signals=matched.risk_signals,
            explanation=matched.explanation,
            assessment_status=RiskAssessmentStatus.NO_SIGNAL,
        )
    with pytest.raises(RiskValidationError, match="explicit match"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            handoff_required=False,
            priority_handoff=False,
            explanation=build_deterministic_risk_result([]).explanation,
            assessment_status=RiskAssessmentStatus.RULE_MATCH,
        )


def test_unsupported_language_requires_handoff_without_priority() -> None:
    template = build_unsupported_language_result()
    with pytest.raises(RiskValidationError, match="handoff_required=true"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            handoff_required=False,
            priority_handoff=False,
            explanation=template.explanation,
            assessment_status=RiskAssessmentStatus.UNSUPPORTED_LANGUAGE,
        )
    with pytest.raises(RiskValidationError, match="priority_handoff=false"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            handoff_required=True,
            priority_handoff=True,
            explanation=template.explanation,
            assessment_status=RiskAssessmentStatus.UNSUPPORTED_LANGUAGE,
        )


def test_no_signal_still_forbids_handoff_flags() -> None:
    with pytest.raises(RiskValidationError, match="no handoff flags"):
        DeterministicRiskResult(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            handoff_required=True,
            priority_handoff=False,
            explanation=build_deterministic_risk_result([]).explanation,
            assessment_status=RiskAssessmentStatus.NO_SIGNAL,
        )
