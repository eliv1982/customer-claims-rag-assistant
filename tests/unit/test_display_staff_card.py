"""Additional display tests for claim_category, staff_actions, and customer draft isolation."""
from __future__ import annotations

import re

import pytest

from customer_claims_rag.application.models import CustomerClaimsResult
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    OUT_OF_SCOPE_CUSTOMER_RESPONSE,
    out_of_scope_result,
)
from customer_claims_rag.generation.handoff import build_handoff_notice
from customer_claims_rag.generation.models import Citation, GroundedGenerationResult
from customer_claims_rag.generation.risk_integration_models import RiskAwareGroundedGenerationResult
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import RiskAssessmentRequest
from customer_claims_rag.ui.display import map_result_to_display


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


def _grounded(answer: str = "Ответ по правилам [S1].") -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=answer,
        citations=[
            Citation(
                citation_key="S1",
                document_id="policy_refund",
                chunk_id="policy_refund::1",
                heading="Возврат денежных средств",
                source_path="data/02_clean_markdown/policy.md",
            ),
        ],
    )


def _pipeline_result(*, risk_assessment, generation, generation_outcome):
    response = RiskAwareGroundedGenerationResult(
        generation=generation,
        risk_assessment=risk_assessment,
        handoff_notice=build_handoff_notice(risk_assessment),
        generation_outcome=generation_outcome,
    )
    return CustomerClaimsResult(response=response)


def test_claim_category_populated_for_high_risk():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded(),
        generation_outcome="grounded_answer",
    ))
    assert hasattr(view, "claim_category")
    assert isinstance(view.claim_category, str) and len(view.claim_category) > 0


def test_staff_actions_populated_for_high_risk():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded(),
        generation_outcome="grounded_answer",
    ))
    assert hasattr(view, "staff_actions")
    assert isinstance(view.staff_actions, tuple) and len(view.staff_actions) > 0


def test_critical_category_mentions_health():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_critical_risk(),
        generation=_grounded(),
        generation_outcome="grounded_answer",
    ))
    cat = view.claim_category.lower()
    assert any(w in cat for w in ("здоровь", "симптом", "жалоб", "потреблен"))


def test_critical_staff_actions_mention_specialist():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_critical_risk(),
        generation=_grounded(),
        generation_outcome="grounded_answer",
    ))
    combined = " ".join(view.staff_actions).lower()
    assert any(w in combined for w in ("старш", "специалист", "приоритет", "передайте"))


def test_high_category_mentions_packaging():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded(),
        generation_outcome="grounded_answer",
    ))
    cat = view.claim_category.lower()
    assert any(w in cat for w in ("упаковк", "целостност", "нарушен"))


def test_out_of_scope_category_mentions_oos():
    risk = _low_risk()
    view = map_result_to_display(CustomerClaimsResult(
        response=RiskAwareGroundedGenerationResult(
            generation=out_of_scope_result(),
            risk_assessment=risk,
            handoff_notice=build_handoff_notice(risk),
            generation_outcome="out_of_scope",
        )
    ))
    assert "FoodFlow" in view.claim_category or "вне" in view.claim_category.lower()


def test_low_risk_no_match_category_is_general():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_low_risk(),
        generation=_grounded(),
        generation_outcome="grounded_answer",
    ))
    cat = view.claim_category.lower()
    assert "общ" in cat or "foodflow" in cat or "запрос" in cat


def test_customer_draft_no_citation_markers():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded(answer="Ответ по правилам [S1]."),
        generation_outcome="grounded_answer",
    ))
    assert not re.search(r"\[S\d+\]", view.customer_draft)


def test_customer_draft_preserves_text():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded(answer="Обратитесь в FoodFlow [S1]."),
        generation_outcome="grounded_answer",
    ))
    assert "FoodFlow" in view.customer_draft
    assert "Обратитесь" in view.customer_draft


def test_customer_draft_no_false_refund_words():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_low_risk(),
        generation=_grounded(answer="Возврат возможен после проверки [S1]."),
        generation_outcome="grounded_answer",
    ))
    draft_lower = view.customer_draft.lower()
    assert "возврат оформлен" not in draft_lower
    assert "компенсация начислена" not in draft_lower


def test_customer_draft_fallback_no_risk_labels():
    risk = _critical_risk()
    view = map_result_to_display(CustomerClaimsResult(
        response=RiskAwareGroundedGenerationResult(
            generation=GroundedGenerationResult(
                response_mode="insufficient_context",
                customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
                citations=[],
            ),
            risk_assessment=risk,
            handoff_notice=build_handoff_notice(risk),
            generation_outcome="generation_error_fallback",
        )
    ))
    draft_lower = view.customer_draft.lower()
    assert "critical" not in draft_lower
    assert "критический" not in draft_lower


def test_out_of_scope_display_not_generation_error():
    risk = _low_risk()
    view = map_result_to_display(CustomerClaimsResult(
        response=RiskAwareGroundedGenerationResult(
            generation=out_of_scope_result(),
            risk_assessment=risk,
            handoff_notice=build_handoff_notice(risk),
            generation_outcome="out_of_scope",
        )
    ))
    assert view.generation_outcome == "out_of_scope"
    assert view.customer_draft == OUT_OF_SCOPE_CUSTOMER_RESPONSE
    assert view.customer_draft != GENERATION_FAILURE_CUSTOMER_RESPONSE


def test_requires_escalation_field_high_vs_low():
    high_view = map_result_to_display(_pipeline_result(
        risk_assessment=_high_risk(), generation=_grounded(), generation_outcome="grounded_answer",
    ))
    low_view = map_result_to_display(_pipeline_result(
        risk_assessment=_low_risk(), generation=_grounded(), generation_outcome="grounded_answer",
    ))
    assert high_view.requires_escalation is True
    assert low_view.requires_escalation is False


def test_routing_recommendation_is_present():
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_high_risk(), generation=_grounded(), generation_outcome="grounded_answer",
    ))
    assert isinstance(view.routing_recommendation, str) and len(view.routing_recommendation) > 0


def test_routing_recommendation_no_completed_action_verbs():
    """Routing must use future/conditional, not completed verbs."""
    view = map_result_to_display(_pipeline_result(
        risk_assessment=_critical_risk(), generation=_grounded(), generation_outcome="grounded_answer",
    ))
    rec_lower = view.routing_recommendation.lower()
    assert "передано" not in rec_lower
    assert "отправлен" not in rec_lower
    assert "оформлен" not in rec_lower
