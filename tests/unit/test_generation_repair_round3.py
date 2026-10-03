"""Regression tests for Manual Acceptance Round 3 observed defects.

These tests capture the specific bad outputs seen in Round 2 acceptance and
prove that the fixes prevent them from reaching the customer-facing draft.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from customer_claims_rag.application.models import CustomerClaimsResult, RetrievedItemMeta
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
    OUT_OF_SCOPE_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.handoff import build_handoff_notice
from customer_claims_rag.generation.models import Citation, GroundedGenerationResult
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.risk.models import (
    DeterministicRiskResult,
    RiskLevel,
    RiskSignal,
)
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.validator import build_deterministic_risk_result
from customer_claims_rag.application.claim_guidance import (
    OOS_STAFF_ACTIONS,
    STAFF_ACTIONS_BY_CATEGORY,
)
from customer_claims_rag.application.customer_templates import sanitize_customer_draft
from customer_claims_rag.application.customer_text_policy import detect_draft_violations
from customer_claims_rag.ui.display import map_result_to_display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_risk(level: RiskLevel, codes: list[RiskReasonCode] | None = None) -> DeterministicRiskResult:
    signals = [RiskSignal(reason_code=c, level=level, rule_id=c.value) for c in (codes or [])]
    return build_deterministic_risk_result(signals)


def _grounded_result(customer_text: str, citations: list[Citation] | None = None) -> GroundedGenerationResult:
    if citations is None:
        citations = [
            Citation(
                citation_key="S1",
                heading="Правила FoodFlow",
                document_id="doc-test",
                chunk_id="chunk-0",
                source_path="docs/rules.md",
            )
        ]
    return GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=customer_text,
        citations=tuple(citations),
    )


def _ic_result() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
        citations=(),
    )


def _oos_result() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="out_of_scope",
        customer_response=OUT_OF_SCOPE_CUSTOMER_RESPONSE,
        citations=(),
    )


def _error_result() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
        citations=(),
    )


def _pipeline_result(
    generation: GroundedGenerationResult,
    risk: DeterministicRiskResult,
    outcome: str,
    customer_query: str = "",
    retrieved_items: list[RetrievedItemMeta] | None = None,
) -> CustomerClaimsResult:
    response = RiskAwareGroundedGenerationResult(
        generation=generation,
        risk_assessment=risk,
        handoff_notice=build_handoff_notice(risk),
        generation_outcome=outcome,  # type: ignore[arg-type]
    )
    return CustomerClaimsResult(
        response=response,
        customer_query=customer_query,
        retrieved_items=tuple(retrieved_items or []),
    )


# ---------------------------------------------------------------------------
# Section 1: detect_draft_violations catches the bad outputs observed in Round 2
# ---------------------------------------------------------------------------

class TestDraftViolationDetection:
    """These verify that the observed contaminated strings are detected."""

    def test_detects_high_keis_pattern(self) -> None:
        """M03 defect: 'high-кейс' leaks into customer draft."""
        text = "Это high-кейс, требует немедленной проверки."
        violations = detect_draft_violations(text)
        assert violations, f"Expected violations but got none for: {text!r}"

    def test_detects_high_keis_with_space(self) -> None:
        text = "Данная ситуация — high кейс."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_critical_keis_pattern(self) -> None:
        text = "Обращение классифицировано как critical-кейс."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_raw_english_high(self) -> None:
        """Standalone 'high' risk label in otherwise Russian text."""
        text = "Это high риск для здоровья клиента."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_raw_english_critical(self) -> None:
        """Standalone 'critical' in Russian text."""
        text = "Обращение получило статус critical."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_backtick_risk_label(self) -> None:
        text = "Уровень риска: `high`. Требуется проверка."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_assistant_meta_phrase(self) -> None:
        """M04 defect: 'Ассистент не устанавливает причинную связь'."""
        text = "Ассистент не устанавливает причинную связь между едой и симптомами."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_assistant_classification_phrase(self) -> None:
        text = "Ассистент классифицирует данное обращение как приоритетное."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_false_completed_transfer(self) -> None:
        """M03/M04 defect: 'Ваше обращение будет передано'."""
        text = "Ваше обращение будет передано на проверку."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_false_was_transferred(self) -> None:
        text = "Ваше обращение уже было передано старшему специалисту."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_escalation_claim(self) -> None:
        text = "Обращение будет эскалировано согласно протоколу."
        violations = detect_draft_violations(text)
        assert violations

    def test_detects_technical_term_response_mode(self) -> None:
        text = "В данном response mode ответ не предусмотрен."
        violations = detect_draft_violations(text)
        assert violations

    def test_no_violations_in_clean_text(self) -> None:
        """A proper customer response should pass validation."""
        text = (
            "Приносим извинения за доставленные неудобства. "
            "Пожалуйста, сохраните упаковку и фото. "
            "Для проверки укажите номер заказа. "
            "После проверки сообщим возможный вариант решения."
        )
        violations = detect_draft_violations(text)
        assert violations == []

    def test_no_violations_for_health_safe_text(self) -> None:
        """Health-harm appropriate response has no violations."""
        text = (
            "Нам очень жаль слышать о вашем состоянии. "
            "Рекомендуем незамедлительно обратиться за медицинской помощью. "
            "Сохраните, пожалуйста, данные о заказе. "
            "Мы приоритетно проверим обращение."
        )
        violations = detect_draft_violations(text)
        assert violations == []

    def test_sanitize_returns_fallback_for_structural_violations(self) -> None:
        """Structural violations must not just strip text — a safe fallback is returned."""
        text = "Ассистент не устанавливает связь. Обращение будет передано."
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        violations = detect_draft_violations(text)
        sanitized, is_safe = sanitize_customer_draft(text, violations, risk)
        # is_safe=False means fallback was used, not raw model output
        assert not is_safe
        # Fallback must not contain the forbidden terms
        assert "ассистент" not in sanitized.lower()
        assert "передано" not in sanitized.lower()

    def test_sanitize_high_fallback_contains_safe_language(self) -> None:
        """High-risk structural fallback recommends staff review."""
        text = "Ваше обращение будет передано. high-кейс."
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        violations = detect_draft_violations(text)
        sanitized, _ = sanitize_customer_draft(text, violations, risk)
        # Fallback must contain staff review language
        assert "сотрудник" in sanitized.lower() or "проверк" in sanitized.lower()


# ---------------------------------------------------------------------------
# Section 2: map_result_to_display blocks contaminated drafts
# ---------------------------------------------------------------------------

class TestMapResultBlocksContamination:
    """These tests verify the display layer doesn't pass contaminated text to UI."""

    def test_m03_draft_with_high_keis_is_sanitized(self) -> None:
        """M03: 'high-кейс' in the model's answer must not reach customer_draft."""
        contaminated = (
            "Ваша ситуация относится к high-кейс. "
            "Не употребляйте продукт. Укажите номер заказа."
        )
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        result = _pipeline_result(
            _grounded_result(contaminated),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert "high" not in view.customer_draft.lower() or "high-кейс" not in view.customer_draft.lower()
        assert view.draft_sanitized is True

    def test_m04_draft_with_critical_is_sanitized(self) -> None:
        """M04: 'critical' enum in the model's answer must not reach customer_draft as-is."""
        contaminated = (
            "Данное обращение классифицировано как critical. "
            "Ассистент не устанавливает причинную связь. "
            "Обратитесь к врачу."
        )
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        result = _pipeline_result(
            _grounded_result(contaminated),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert view.draft_sanitized is True
        # Structural violations → fallback is used, ассистент phrase gone
        assert "ассистент не устанавл" not in view.customer_draft.lower()
        assert "critical" not in view.customer_draft.lower()

    def test_m04_fallback_recommends_medical_help(self) -> None:
        """After sanitization, critical-risk fallback must recommend medical help."""
        contaminated = "Ассистент классифицирует это как critical. Передает специалисту."
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        result = _pipeline_result(
            _grounded_result(contaminated),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        # The fallback for critical risk must mention medical help
        assert "медицинск" in view.customer_draft.lower() or "врач" in view.customer_draft.lower()

    def test_m01_draft_does_not_say_sluzhba_podderzhki(self) -> None:
        """M01: Customer draft must not say 'обратитесь в службу поддержки'
        — the agent IS the support, the phrasing sends them in circles.
        Clean text without the phrase passes through unchanged."""
        clean_text = (
            "Благодарим за обращение. "
            "Для проверки статуса заказа №12345 нам потребуются дополнительные данные. [S1] "
            "Статус оплаты и доставки будет проверен. "
            "Пожалуйста, укажите дату оформления заказа."
        )
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(
            _grounded_result(clean_text),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        # Must not direct agent to "support service" (circular)
        assert "в службу поддержки" not in view.customer_draft.lower()
        assert view.draft_sanitized is False

    def test_m03_false_transfer_claim_is_blocked(self) -> None:
        """M03: 'Ваше обращение будет передано' must not appear in customer draft."""
        contaminated = (
            "Не употребляйте продукт. "
            "Ваше обращение будет передано на проверку. "
            "Сохраните упаковку."
        )
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        result = _pipeline_result(
            _grounded_result(contaminated),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert "будет передано" not in view.customer_draft.lower()
        assert view.draft_sanitized is True

    def test_clean_draft_is_not_sanitized(self) -> None:
        """A clean draft should pass through unchanged, draft_sanitized=False."""
        clean = (
            "Благодарим за обращение. Пожалуйста, сохраните упаковку и укажите номер заказа. "
            "После проверки сообщим возможный вариант решения. [S1]"
        )
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        result = _pipeline_result(
            _grounded_result(clean),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert view.draft_sanitized is False
        assert "S1" not in view.customer_draft  # citation markers are stripped


# ---------------------------------------------------------------------------
# Section 3: Categories are case-specific
# ---------------------------------------------------------------------------

class TestCaseSpecificCategories:
    """Prove that all 7 manual scenarios get meaningful, distinct categories."""

    def test_m01_nondelivery_category_from_keyword(self) -> None:
        """M01: 'не был доставлен' doesn't trigger risk rules — keyword fallback fires."""
        risk = _make_risk(RiskLevel.LOW)  # No rule match → LOW
        result = _pipeline_result(
            _grounded_result("Мы проверим ваш заказ. [S1]"),
            risk,
            "grounded_answer",
            customer_query="Я оплатил заказ №12345, но он не был доставлен. Что делать?",
        )
        view = map_result_to_display(result)
        assert "Недоставка" in view.claim_category or "доставк" in view.claim_category.lower()

    def test_m02_incomplete_order_category_from_keyword(self) -> None:
        """M02: 'не хватало' doesn't trigger risk rule — keyword fallback fires."""
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(
            _grounded_result("Проверим состав заказа. [S1]"),
            risk,
            "grounded_answer",
            customer_query="В моем заказе не хватало двух позиций — суши и напитка.",
        )
        view = map_result_to_display(result)
        assert (
            "Неполный" in view.claim_category
            or "недокомплект" in view.claim_category.lower()
            or "частичн" in view.claim_category.lower()
        )

    def test_m03_package_tampering_category(self) -> None:
        """M03: PACKAGE_TAMPERING risk rule fires → specific category."""
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        result = _pipeline_result(
            _grounded_result("Не употребляйте продукт. [S1]"),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert "упаковк" in view.claim_category.lower()

    def test_m04_health_harm_category(self) -> None:
        """M04: HEALTH_SYMPTOMS rule fires → specific category."""
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        result = _pipeline_result(
            _grounded_result("Рекомендуем обратиться к врачу. [S1]"),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert "здоровь" in view.claim_category.lower() or "продукт" in view.claim_category.lower()

    def test_m05_refund_demand_category(self) -> None:
        """M05: REFUND_REQUEST rule fires → specific category."""
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
            customer_query="Требую полный возврат за все 5 заказов.",
        )
        view = map_result_to_display(result)
        assert "возврат" in view.claim_category.lower()

    def test_m06_m07_oos_category(self) -> None:
        """M06/M07: Out-of-scope gets dedicated category."""
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(
            _oos_result(),
            risk,
            "out_of_scope",
        )
        view = map_result_to_display(result)
        assert "вне области" in view.claim_category.lower() or "FoodFlow" in view.claim_category

    def test_all_categories_are_not_generic_for_coded_risk(self) -> None:
        """Any result with a reason code should NOT return the generic category."""
        for code, level in [
            (RiskReasonCode.NON_DELIVERY, RiskLevel.HIGH),
            (RiskReasonCode.MISSING_ITEM, RiskLevel.MEDIUM),
            (RiskReasonCode.PACKAGE_TAMPERING, RiskLevel.HIGH),
            (RiskReasonCode.REFUND_REQUEST, RiskLevel.MEDIUM),
            (RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION, RiskLevel.CRITICAL),
        ]:
            risk = _make_risk(level, [code])
            result = _pipeline_result(
                _grounded_result("Ответ клиенту. [S1]"),
                risk,
                "grounded_answer",
            )
            view = map_result_to_display(result)
            assert view.claim_category != "Общий запрос по сервису FoodFlow", (
                f"Expected specific category for {code!r} but got generic"
            )


# ---------------------------------------------------------------------------
# Section 4: Out-of-scope staff actions are specific
# ---------------------------------------------------------------------------

class TestOOSStaffActions:

    def test_oos_has_specific_staff_actions(self) -> None:
        """M06/M07: OOS staff actions must be specific (not generic 'обработайте стандартно')."""
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(_oos_result(), risk, "out_of_scope")
        view = map_result_to_display(result)
        assert view.staff_actions != ("Обработайте в стандартном режиме.",)
        assert view.staff_actions != ("Обработайте стандартно.",)

    def test_oos_staff_actions_match_oos_constant(self) -> None:
        """OOS staff actions should match the dedicated OOS constant."""
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(_oos_result(), risk, "out_of_scope")
        view = map_result_to_display(result)
        assert view.staff_actions == OOS_STAFF_ACTIONS

    def test_oos_staff_actions_mention_prepared_response(self) -> None:
        """OOS staff should be told to use a prepared/templated response."""
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(_oos_result(), risk, "out_of_scope")
        view = map_result_to_display(result)
        combined = " ".join(view.staff_actions).lower()
        assert "подготовл" in combined or "ограничени" in combined

    def test_oos_routing_is_not_escalation(self) -> None:
        """OOS should not recommend escalation."""
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(_oos_result(), risk, "out_of_scope")
        view = map_result_to_display(result)
        assert not view.requires_escalation


# ---------------------------------------------------------------------------
# Section 5: M05 - No contradiction in insufficient_context + medium risk
# ---------------------------------------------------------------------------

class TestM05Contradiction:

    def test_m05_refund_ic_routing_not_empty(self) -> None:
        """M05: routing must not say 'standard processing' for refund IC."""
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
            customer_query="Требую полный возврат за все 5 заказов.",
        )
        view = map_result_to_display(result)
        # The route should be more specific than just "standard"
        assert "ручная проверка" in view.routing_recommendation.lower() or "проверит" in view.routing_recommendation.lower()

    def test_m05_refund_ic_escalation_flag_consistent(self) -> None:
        """M05: MEDIUM risk REFUND_REQUEST should have requires_escalation=False (per rules)
        but routing should still recommend manual check."""
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
            customer_query="Требую полный возврат за все 5 заказов.",
        )
        view = map_result_to_display(result)
        # MEDIUM risk has handoff_required=False per the risk model (this is by design)
        assert view.requires_escalation is False
        # But routing should guide to manual check, not "no review needed"
        assert "стандартная обработка" not in view.routing_recommendation.lower() or "ручная" in view.routing_recommendation.lower()

    def test_m05_ic_shows_retrieved_materials(self) -> None:
        """M05: retrieved materials must be visible for insufficient_context even with no citations."""
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        retrieved = [
            RetrievedItemMeta(citation_key="S1", rank=1, heading="Условия возврата FoodFlow", document_id="doc-refund"),
            RetrievedItemMeta(citation_key="S2", rank=2, heading="Правила доставки", document_id="doc-delivery"),
        ]
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
            retrieved_items=retrieved,
        )
        view = map_result_to_display(result)
        assert len(view.retrieved_materials) == 2
        assert view.retrieved_materials[0].heading == "Условия возврата FoodFlow"
        # Confirmed citations should be empty for IC
        assert len(view.citations) == 0

    def test_ic_no_retrieved_items_gives_empty_materials(self) -> None:
        """IC with no retrieval results → retrieved_materials is empty."""
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
        )
        view = map_result_to_display(result)
        assert view.retrieved_materials == ()


# ---------------------------------------------------------------------------
# Section 6: Staff actions are case-specific (not generic for all risk levels)
# ---------------------------------------------------------------------------

class TestCaseSpecificStaffActions:

    def test_nondelivery_gets_specific_actions(self) -> None:
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(
            _grounded_result("Ответ. [S1]"),
            risk,
            "grounded_answer",
            customer_query="Заказ не был доставлен.",
        )
        view = map_result_to_display(result)
        combined = " ".join(view.staff_actions).lower()
        assert "историю заказа" in combined or "статус" in combined

    def test_incomplete_order_gets_specific_actions(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.MISSING_ITEM])
        result = _pipeline_result(
            _grounded_result("Ответ. [S1]"),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        combined = " ".join(view.staff_actions).lower()
        assert "состав" in combined or "накладн" in combined or "позиц" in combined

    def test_health_harm_gets_critical_actions(self) -> None:
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        result = _pipeline_result(
            _grounded_result("Рекомендуем обратиться к врачу. [S1]"),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        combined = " ".join(view.staff_actions).lower()
        assert "медицинск" in combined or "специалист" in combined or "старш" in combined

    def test_refund_demand_gets_specific_actions(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
        )
        view = map_result_to_display(result)
        combined = " ".join(view.staff_actions).lower()
        assert "историю заказ" in combined or "возврат" in combined


# ---------------------------------------------------------------------------
# Section 7: Escalation field semantics
# ---------------------------------------------------------------------------

class TestEscalationSemantics:

    def test_critical_risk_requires_escalation(self) -> None:
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        result = _pipeline_result(
            _grounded_result("Рекомендуем обратиться к врачу. [S1]"),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert view.requires_escalation is True

    def test_high_risk_requires_escalation(self) -> None:
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        result = _pipeline_result(
            _grounded_result("Ответ. [S1]"),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert view.requires_escalation is True

    def test_medium_risk_no_escalation(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
        )
        view = map_result_to_display(result)
        assert view.requires_escalation is False

    def test_low_risk_no_escalation(self) -> None:
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(
            _grounded_result("Ответ. [S1]"),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert view.requires_escalation is False

    def test_view_has_requires_escalation_not_requires_handoff(self) -> None:
        """ClaimSuccessView must expose requires_escalation, not the old requires_handoff name."""
        risk = _make_risk(RiskLevel.LOW)
        result = _pipeline_result(
            _grounded_result("Ответ. [S1]"),
            risk,
            "grounded_answer",
        )
        view = map_result_to_display(result)
        assert hasattr(view, "requires_escalation")
        assert not hasattr(view, "requires_handoff")


# ---------------------------------------------------------------------------
# Section 8: Retrieved materials are not shown as confirmed citations
# ---------------------------------------------------------------------------

class TestEvidenceHandlingDistinction:

    def test_grounded_answer_shows_confirmed_citations_not_retrieved(self) -> None:
        """For grounded_answer: confirmed citations shown, retrieved_materials empty."""
        risk = _make_risk(RiskLevel.LOW)
        cited = Citation(
            citation_key="S1",
            heading="Правила FoodFlow",
            document_id="doc1",
            chunk_id="chunk-1",
            source_path="docs/rules.md",
        )
        retrieved = [RetrievedItemMeta(citation_key="S1", rank=1, heading="Правила FoodFlow", document_id="doc1")]
        result = _pipeline_result(
            _grounded_result("Согласно правилам [S1].", citations=[cited]),
            risk,
            "grounded_answer",
            retrieved_items=retrieved,
        )
        view = map_result_to_display(result)
        assert len(view.citations) == 1
        assert view.citations[0].is_confirmed is True
        # Grounded answers should not show retrieved_materials (already in citations)
        assert view.retrieved_materials == ()

    def test_insufficient_context_shows_retrieved_not_confirmed(self) -> None:
        """For IC: retrieved_materials shown, citations empty."""
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        retrieved = [
            RetrievedItemMeta(citation_key="S1", rank=1, heading="Условия возврата", document_id="doc-r"),
        ]
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
            retrieved_items=retrieved,
        )
        view = map_result_to_display(result)
        assert len(view.citations) == 0
        assert len(view.retrieved_materials) == 1
        assert view.retrieved_materials[0].is_confirmed is False

    def test_retrieved_material_label_says_not_confirmed(self) -> None:
        """Retrieved materials must be marked as not confirmed sources."""
        risk = _make_risk(RiskLevel.LOW)
        retrieved = [RetrievedItemMeta(citation_key="S1", rank=1, heading="Политика", document_id="d1")]
        result = _pipeline_result(
            _ic_result(),
            risk,
            "insufficient_context",
            retrieved_items=retrieved,
        )
        view = map_result_to_display(result)
        assert view.retrieved_materials[0].is_confirmed is False
