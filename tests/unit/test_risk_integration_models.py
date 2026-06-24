"""Unit tests for risk-aware grounded generation contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
    generation_failure_result,
)
from customer_claims_rag.generation.handoff import (
    CRITICAL_HANDOFF_NOTICE,
    HIGH_HANDOFF_NOTICE,
    build_handoff_notice,
)
from customer_claims_rag.generation.models import Citation, GroundedGenerationResult
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import RiskAssessmentRequest, RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode


def _low_risk():
    return assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Где посмотреть правила доставки?"),
    )


def _medium_risk():
    return assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Не привезли одну позицию."),
    )


def _high_risk():
    return assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Упаковка была вскрыта."),
    )


def _critical_risk():
    return assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="После еды стало трудно дышать."),
    )


def _grounded_generation(answer: str = "Ответ [S1].") -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=answer,
        citations=[
            Citation(
                citation_key="S1",
                document_id="doc-a",
                chunk_id="doc-a::1",
                heading="Heading",
                source_path="path",
            ),
        ],
    )


def _insufficient_context_generation() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
        citations=[],
    )


def _valid_result(
    *,
    risk_assessment,
    generation: GroundedGenerationResult,
    generation_outcome: str,
) -> RiskAwareGroundedGenerationResult:
    return RiskAwareGroundedGenerationResult(
        generation=generation,
        risk_assessment=risk_assessment,
        handoff_notice=build_handoff_notice(risk_assessment),
        generation_outcome=generation_outcome,  # type: ignore[arg-type]
    )


def test_build_handoff_notice_low_and_medium_are_none() -> None:
    assert build_handoff_notice(_low_risk()) is None
    assert build_handoff_notice(_medium_risk()) is None


def test_build_handoff_notice_high_and_critical() -> None:
    assert build_handoff_notice(_high_risk()) == HIGH_HANDOFF_NOTICE
    assert build_handoff_notice(_critical_risk()) == CRITICAL_HANDOFF_NOTICE


def test_valid_grounded_answer_construction() -> None:
    result = _valid_result(
        risk_assessment=_low_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    assert result.generation_outcome == "grounded_answer"
    assert result.handoff_notice is None


def test_valid_insufficient_context_construction() -> None:
    result = _valid_result(
        risk_assessment=_high_risk(),
        generation=_insufficient_context_generation(),
        generation_outcome="insufficient_context",
    )
    assert result.generation_outcome == "insufficient_context"
    assert result.handoff_notice == HIGH_HANDOFF_NOTICE


def test_valid_generation_error_fallback_construction() -> None:
    result = _valid_result(
        risk_assessment=_critical_risk(),
        generation=generation_failure_result(),
        generation_outcome="generation_error_fallback",
    )
    assert result.generation_outcome == "generation_error_fallback"
    assert result.handoff_notice == CRITICAL_HANDOFF_NOTICE


def test_result_extra_forbid() -> None:
    low = _low_risk()
    with pytest.raises(ValidationError):
        RiskAwareGroundedGenerationResult(
            generation=_grounded_generation(),
            risk_assessment=low,
            handoff_notice=None,
            generation_outcome="grounded_answer",
            unexpected=True,  # type: ignore[call-arg]
        )


def test_result_is_frozen() -> None:
    result = _valid_result(
        risk_assessment=_low_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    with pytest.raises(ValidationError):
        result.generation_outcome = "insufficient_context"  # type: ignore[misc]


def test_result_json_round_trip() -> None:
    result = _valid_result(
        risk_assessment=_high_risk(),
        generation=_grounded_generation("Текст [S1]."),
        generation_outcome="grounded_answer",
    )
    restored = RiskAwareGroundedGenerationResult.model_validate_json(
        result.model_dump_json(),
    )
    assert restored == result


def test_arbitrary_handoff_notice_rejected() -> None:
    with pytest.raises(ValidationError, match="handoff_notice"):
        RiskAwareGroundedGenerationResult(
            generation=_grounded_generation(),
            risk_assessment=_high_risk(),
            handoff_notice="Произвольный текст.",
            generation_outcome="grounded_answer",
        )


def test_missing_high_handoff_notice_rejected() -> None:
    with pytest.raises(ValidationError, match="handoff_notice"):
        RiskAwareGroundedGenerationResult(
            generation=_grounded_generation(),
            risk_assessment=_high_risk(),
            handoff_notice=None,
            generation_outcome="grounded_answer",
        )


def test_handoff_notice_for_low_rejected() -> None:
    with pytest.raises(ValidationError, match="handoff_notice"):
        RiskAwareGroundedGenerationResult(
            generation=_grounded_generation(),
            risk_assessment=_low_risk(),
            handoff_notice=HIGH_HANDOFF_NOTICE,
            generation_outcome="grounded_answer",
        )


def test_high_notice_for_critical_rejected() -> None:
    with pytest.raises(ValidationError, match="handoff_notice"):
        RiskAwareGroundedGenerationResult(
            generation=_grounded_generation(),
            risk_assessment=_critical_risk(),
            handoff_notice=HIGH_HANDOFF_NOTICE,
            generation_outcome="grounded_answer",
        )


def test_grounded_outcome_with_insufficient_generation_rejected() -> None:
    with pytest.raises(ValidationError, match="generation_outcome=grounded_answer"):
        RiskAwareGroundedGenerationResult(
            generation=_insufficient_context_generation(),
            risk_assessment=_low_risk(),
            handoff_notice=None,
            generation_outcome="grounded_answer",
        )


def test_insufficient_outcome_with_grounded_generation_rejected() -> None:
    with pytest.raises(ValidationError, match="insufficient_context"):
        RiskAwareGroundedGenerationResult(
            generation=_grounded_generation(),
            risk_assessment=_low_risk(),
            handoff_notice=None,
            generation_outcome="insufficient_context",
        )


def test_generation_error_fallback_with_wrong_response_rejected() -> None:
    with pytest.raises(ValidationError, match="generation failure customer response"):
        RiskAwareGroundedGenerationResult(
            generation=_insufficient_context_generation(),
            risk_assessment=_high_risk(),
            handoff_notice=HIGH_HANDOFF_NOTICE,
            generation_outcome="generation_error_fallback",
        )


def test_generation_error_fallback_with_citations_rejected() -> None:
    high = _high_risk()
    with pytest.raises(ValidationError, match="empty citations"):
        RiskAwareGroundedGenerationResult(
            generation={
                "response_mode": "insufficient_context",
                "customer_response": GENERATION_FAILURE_CUSTOMER_RESPONSE,
                "citations": [
                    {
                        "citation_key": "S1",
                        "document_id": "doc-a",
                        "chunk_id": "doc-a::1",
                        "heading": "Heading",
                        "source_path": "path",
                    },
                ],
            },
            risk_assessment=high.model_dump(mode="python"),
            handoff_notice=HIGH_HANDOFF_NOTICE,
            generation_outcome="generation_error_fallback",
        )


def test_raw_grounded_answer_without_citations_rejected() -> None:
    low = _low_risk()
    with pytest.raises(ValidationError, match="at least one citation"):
        RiskAwareGroundedGenerationResult(
            generation={
                "response_mode": "grounded_answer",
                "customer_response": "Ответ без источников",
                "citations": [],
            },
            risk_assessment=low.model_dump(mode="python"),
            handoff_notice=None,
            generation_outcome="grounded_answer",
        )


def test_raw_insufficient_context_with_citations_rejected() -> None:
    high = _high_risk()
    with pytest.raises(ValidationError, match="empty citations"):
        RiskAwareGroundedGenerationResult(
            generation={
                "response_mode": "insufficient_context",
                "customer_response": "Недостаточно информации.",
                "citations": [
                    {
                        "citation_key": "S1",
                        "document_id": "doc-a",
                        "chunk_id": "doc-a::1",
                        "heading": "Heading",
                        "source_path": "path",
                    },
                ],
            },
            risk_assessment=high.model_dump(mode="python"),
            handoff_notice=HIGH_HANDOFF_NOTICE,
            generation_outcome="insufficient_context",
        )


def test_generation_failure_result_is_canonical() -> None:
    failure = generation_failure_result()
    assert failure.response_mode == "insufficient_context"
    assert failure.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert failure.citations == []
