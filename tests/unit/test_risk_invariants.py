"""Unit tests for deterministic risk invariants."""

from __future__ import annotations

import itertools

from customer_claims_rag.risk.invariants import apply_risk_floor, build_risk_explanation
from customer_claims_rag.risk.models import RiskLevel, max_risk_level
from customer_claims_rag.risk.reason_codes import RiskReasonCode


_LEVELS = (
    RiskLevel.LOW,
    RiskLevel.MEDIUM,
    RiskLevel.HIGH,
    RiskLevel.CRITICAL,
)


def test_apply_risk_floor_all_sixteen_combinations() -> None:
    for deterministic_floor, proposed_level in itertools.product(_LEVELS, _LEVELS):
        assert apply_risk_floor(deterministic_floor, proposed_level) is max_risk_level(
            deterministic_floor,
            proposed_level,
        )


def test_apply_risk_floor_high_plus_low_is_high() -> None:
    assert apply_risk_floor(RiskLevel.HIGH, RiskLevel.LOW) is RiskLevel.HIGH


def test_apply_risk_floor_critical_plus_low_is_critical() -> None:
    assert apply_risk_floor(RiskLevel.CRITICAL, RiskLevel.LOW) is RiskLevel.CRITICAL


def test_apply_risk_floor_low_plus_critical_is_critical() -> None:
    assert apply_risk_floor(RiskLevel.LOW, RiskLevel.CRITICAL) is RiskLevel.CRITICAL


def test_apply_risk_floor_medium_plus_medium_is_medium() -> None:
    assert apply_risk_floor(RiskLevel.MEDIUM, RiskLevel.MEDIUM) is RiskLevel.MEDIUM


def test_build_risk_explanation_no_match_format() -> None:
    assert build_risk_explanation(
        risk_floor=RiskLevel.LOW,
        explicit_match=False,
        reason_codes=(),
        signal_count=0,
        handoff_required=False,
        priority_handoff=False,
    ) == (
        "risk_floor=low;explicit_match=false;reason_codes=none;"
        "signal_count=0;handoff_required=false;priority_handoff=false"
    )


def test_build_risk_explanation_reason_code_order_preserved() -> None:
    explanation = build_risk_explanation(
        risk_floor=RiskLevel.HIGH,
        explicit_match=True,
        reason_codes=(
            RiskReasonCode.PACKAGE_TAMPERING,
            RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE,
        ),
        signal_count=2,
        handoff_required=True,
        priority_handoff=False,
    )
    assert "reason_codes=package_tampering,official_written_response" in explanation
