"""Pure cross-field invariant checks for deterministic risk results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from customer_claims_rag.risk.reason_codes import RiskReasonCode

if TYPE_CHECKING:
    from customer_claims_rag.risk.models import RiskLevel, RiskSignal

_HANDOFF_BY_LEVEL: dict[str, tuple[bool, bool]] = {
    "low": (False, False),
    "medium": (False, False),
    "high": (True, False),
    "critical": (True, True),
}


def handoff_flags_for_level(level: RiskLevel) -> tuple[bool, bool]:
    """Return (handoff_required, priority_handoff) for a risk floor level."""
    return _HANDOFF_BY_LEVEL[level.value]


def apply_risk_floor(
    deterministic_floor: RiskLevel,
    proposed_level: RiskLevel,
) -> RiskLevel:
    """Return the higher of deterministic floor and proposed level.

    A later model-based assessment may increase the deterministic floor but
    must never lower it.
    """
    from customer_claims_rag.risk.models import max_risk_level

    return max_risk_level(deterministic_floor, proposed_level)


def build_risk_explanation(
    *,
    risk_floor: RiskLevel,
    explicit_match: bool,
    reason_codes: tuple[RiskReasonCode, ...],
    signal_count: int,
    handoff_required: bool,
    priority_handoff: bool,
) -> str:
    """Build a stable machine-readable explanation string."""
    codes_part = (
        ",".join(code.value for code in reason_codes) if reason_codes else "none"
    )
    return (
        f"risk_floor={risk_floor.value};"
        f"explicit_match={'true' if explicit_match else 'false'};"
        f"reason_codes={codes_part};"
        f"signal_count={signal_count};"
        f"handoff_required={'true' if handoff_required else 'false'};"
        f"priority_handoff={'true' if priority_handoff else 'false'}"
    )


def order_reason_codes(
    codes: dict[RiskReasonCode, RiskLevel],
) -> tuple[RiskReasonCode, ...]:
    """Order reason codes severity-first, then registry (enum definition) order."""
    from customer_claims_rag.risk.models import risk_rank

    registry_index = {member: index for index, member in enumerate(RiskReasonCode)}
    sorted_codes = sorted(
        codes,
        key=lambda code: (-risk_rank(codes[code]), registry_index[code]),
    )
    return tuple(sorted_codes)


def canonical_reason_codes_from_signals(
    risk_signals: tuple[RiskSignal, ...],
) -> tuple[RiskReasonCode, ...]:
    """Derive canonical ordered reason codes from risk signals."""
    from customer_claims_rag.risk.models import risk_rank

    unique_codes: dict[RiskReasonCode, RiskLevel] = {}
    for signal in risk_signals:
        existing = unique_codes.get(signal.reason_code)
        if existing is None or risk_rank(signal.level) > risk_rank(existing):
            unique_codes[signal.reason_code] = signal.level
    return order_reason_codes(unique_codes)


def order_risk_signals(signals: list[RiskSignal]) -> tuple[RiskSignal, ...]:
    """Order risk signals severity-first, then registry order, then rule_id."""
    from customer_claims_rag.risk.models import risk_rank

    registry_index = {member: index for index, member in enumerate(RiskReasonCode)}
    sorted_signals = sorted(
        signals,
        key=lambda signal: (
            -risk_rank(signal.level),
            registry_index[signal.reason_code],
            signal.rule_id,
        ),
    )
    return tuple(sorted_signals)


def collect_result_invariant_errors(
    *,
    risk_floor: RiskLevel,
    explicit_match: bool,
    handoff_required: bool,
    priority_handoff: bool,
    reason_codes: tuple[RiskReasonCode, ...],
    risk_signals: tuple[RiskSignal, ...],
    explanation: str,
) -> list[str]:
    """Return human-readable invariant violations; empty list means valid."""
    from customer_claims_rag.risk.models import RiskLevel, max_risk_level

    errors: list[str] = []

    if len(reason_codes) != len(set(reason_codes)):
        errors.append("reason_codes must not contain duplicates")

    signal_keys = [(s.reason_code, s.level, s.rule_id) for s in risk_signals]
    if len(signal_keys) != len(set(signal_keys)):
        errors.append("risk_signals must not contain duplicate (reason_code, level, rule_id)")

    if not explicit_match:
        if risk_floor is not RiskLevel.LOW:
            errors.append("no explicit match requires risk_floor=low")
        if reason_codes:
            errors.append("no explicit match requires empty reason_codes")
        if risk_signals:
            errors.append("no explicit match requires empty risk_signals")
        if handoff_required or priority_handoff:
            errors.append("no explicit match requires no handoff flags")
        expected_explanation = build_risk_explanation(
            risk_floor=RiskLevel.LOW,
            explicit_match=False,
            reason_codes=(),
            signal_count=0,
            handoff_required=False,
            priority_handoff=False,
        )
        if explanation != expected_explanation:
            errors.append("explanation must match canonical deterministic value")
        return errors

    if not reason_codes:
        errors.append("explicit match requires non-empty reason_codes")
    if not risk_signals:
        errors.append("explicit match requires non-empty risk_signals")

    signal_codes = {signal.reason_code for signal in risk_signals}
    reason_code_set = set(reason_codes)
    for code in reason_codes:
        if code not in signal_codes:
            errors.append(
                f"reason_code {code.value} must be represented by at least one risk_signal",
            )
    for signal in risk_signals:
        if signal.reason_code not in reason_code_set:
            errors.append(
                f"risk_signal reason_code {signal.reason_code.value} must exist in reason_codes",
            )

    canonical_codes = canonical_reason_codes_from_signals(risk_signals)
    if reason_codes != canonical_codes:
        errors.append("reason_codes must be in canonical order")

    canonical_signals = order_risk_signals(list(risk_signals))
    if risk_signals != canonical_signals:
        errors.append("risk_signals must be in canonical order")

    signal_floor = max_risk_level(*(signal.level for signal in risk_signals))
    if risk_floor is not signal_floor:
        errors.append("risk_floor must equal maximum risk_signal level")

    expected_handoff, expected_priority = handoff_flags_for_level(risk_floor)
    if handoff_required is not expected_handoff:
        errors.append(
            f"handoff_required must be {expected_handoff} for risk_floor={risk_floor.value}",
        )
    if priority_handoff is not expected_priority:
        errors.append(
            f"priority_handoff must be {expected_priority} for risk_floor={risk_floor.value}",
        )
    if priority_handoff and not handoff_required:
        errors.append("priority_handoff requires handoff_required=true")

    expected_explanation = build_risk_explanation(
        risk_floor=risk_floor,
        explicit_match=explicit_match,
        reason_codes=reason_codes,
        signal_count=len(risk_signals),
        handoff_required=handoff_required,
        priority_handoff=priority_handoff,
    )
    if explanation != expected_explanation:
        errors.append("explanation must match canonical deterministic value")

    return errors
