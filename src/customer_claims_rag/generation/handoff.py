"""Canonical handoff notices for risk-aware grounded generation."""

from __future__ import annotations

from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskLevel

HIGH_HANDOFF_NOTICE = (
    "Для решения этого вопроса требуется проверка сотрудником поддержки."
)

CRITICAL_HANDOFF_NOTICE = (
    "Обращение требует приоритетной проверки сотрудником поддержки."
)

UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE = (
    "Текст обращения не на русском языке: автоматическая оценка риска не выполнена. "
    "Требуется проверка сотрудником поддержки."
)


def build_handoff_notice(
    risk_assessment: DeterministicRiskResult,
) -> str | None:
    """Return the canonical customer-safe handoff notice for a risk assessment."""
    if risk_assessment.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE:
        if not risk_assessment.handoff_required or risk_assessment.priority_handoff:
            raise ValueError(
                "unsupported-language assessment requires handoff_required=true "
                "and priority_handoff=false",
            )
        return UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE

    level = risk_assessment.risk_floor

    if level is RiskLevel.LOW or level is RiskLevel.MEDIUM:
        if risk_assessment.handoff_required or risk_assessment.priority_handoff:
            raise ValueError(
                "low/medium risk assessment must not require handoff flags",
            )
        return None

    if level is RiskLevel.HIGH:
        if not risk_assessment.handoff_required or risk_assessment.priority_handoff:
            raise ValueError(
                "high risk assessment requires handoff_required=true "
                "and priority_handoff=false",
            )
        return HIGH_HANDOFF_NOTICE

    if level is RiskLevel.CRITICAL:
        if not risk_assessment.handoff_required or not risk_assessment.priority_handoff:
            raise ValueError(
                "critical risk assessment requires handoff_required=true "
                "and priority_handoff=true",
            )
        return CRITICAL_HANDOFF_NOTICE

    raise ValueError(f"unsupported risk floor level: {level}")
