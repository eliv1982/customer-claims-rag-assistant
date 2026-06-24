"""Unit tests for application-layer contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from customer_claims_rag.application.models import CustomerClaimsRequest, CustomerClaimsResult
from customer_claims_rag.generation.fallback import INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
from customer_claims_rag.generation.handoff import HIGH_HANDOFF_NOTICE
from customer_claims_rag.generation.models import Citation, GroundedGenerationResult
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import RiskAssessmentRequest


def _risk_aware_response() -> RiskAwareGroundedGenerationResult:
    risk = assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Упаковка была вскрыта."),
    )
    return RiskAwareGroundedGenerationResult(
        generation=GroundedGenerationResult(
            response_mode="grounded_answer",
            customer_response="По правилам вскрытая упаковка фиксируется [S1].",
            citations=[
                Citation(
                    citation_key="S1",
                    document_id="doc-a",
                    chunk_id="doc-a::1",
                    heading="Heading",
                    source_path="data/02_clean_markdown/doc-a.md",
                ),
            ],
        ),
        risk_assessment=risk,
        handoff_notice=HIGH_HANDOFF_NOTICE,
        generation_outcome="grounded_answer",
    )


def test_customer_claims_request_valid_query() -> None:
    request = CustomerClaimsRequest(customer_query="Где посмотреть правила доставки?")
    assert request.customer_query == "Где посмотреть правила доставки?"


def test_customer_claims_request_strips_leading_trailing_spaces() -> None:
    request = CustomerClaimsRequest(
        customer_query="  Упаковка была вскрыта.  ",
    )
    assert request.customer_query == "Упаковка была вскрыта."


def test_customer_claims_request_whitespace_only_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerClaimsRequest(customer_query="   ")


def test_customer_claims_request_empty_string_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerClaimsRequest(customer_query="")


def test_customer_claims_request_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerClaimsRequest(
            customer_query="hello",
            unexpected=True,  # type: ignore[call-arg]
        )


def test_customer_claims_request_is_frozen() -> None:
    request = CustomerClaimsRequest(customer_query="hello")
    with pytest.raises(ValidationError):
        request.customer_query = "other"  # type: ignore[misc]


def test_customer_claims_request_json_serialization() -> None:
    request = CustomerClaimsRequest(customer_query="  query  ")
    dumped = request.model_dump(mode="json")
    assert dumped == {"customer_query": "query"}


def test_customer_claims_request_json_round_trip() -> None:
    request = CustomerClaimsRequest(customer_query="  Не привезли одну позицию.  ")
    restored = CustomerClaimsRequest.model_validate_json(request.model_dump_json())
    assert restored == request
    assert restored.customer_query == "Не привезли одну позицию."


def test_customer_claims_result_valid_nested_response() -> None:
    response = _risk_aware_response()
    result = CustomerClaimsResult(response=response)
    assert result.response == response


def test_customer_claims_result_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        CustomerClaimsResult(
            response=_risk_aware_response(),
            unexpected=True,  # type: ignore[call-arg]
        )


def test_customer_claims_result_is_frozen() -> None:
    result = CustomerClaimsResult(response=_risk_aware_response())
    with pytest.raises(ValidationError):
        result.response = _risk_aware_response()  # type: ignore[misc]


def test_customer_claims_result_json_serialization() -> None:
    result = CustomerClaimsResult(response=_risk_aware_response())
    dumped = result.model_dump(mode="json")
    assert "response" in dumped
    assert dumped["response"]["generation_outcome"] == "grounded_answer"


def test_customer_claims_result_json_round_trip() -> None:
    original = CustomerClaimsResult(response=_risk_aware_response())
    restored = CustomerClaimsResult.model_validate_json(original.model_dump_json())
    assert restored == original
    assert (
        restored.response.generation.customer_response
        == original.response.generation.customer_response
    )
    assert restored.response.risk_assessment == original.response.risk_assessment
    assert restored.response.handoff_notice == original.response.handoff_notice
    assert restored.response.generation_outcome == original.response.generation_outcome


def test_customer_claims_result_preserves_insufficient_context_nested_response() -> None:
    risk = assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Упаковка была вскрыта."),
    )
    response = RiskAwareGroundedGenerationResult(
        generation=GroundedGenerationResult(
            response_mode="insufficient_context",
            customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
            citations=[],
        ),
        risk_assessment=risk,
        handoff_notice=HIGH_HANDOFF_NOTICE,
        generation_outcome="insufficient_context",
    )
    result = CustomerClaimsResult(response=response)
    assert result.response.generation.response_mode == "insufficient_context"
    assert result.response.generation.citations == []
