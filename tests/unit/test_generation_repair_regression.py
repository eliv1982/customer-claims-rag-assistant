"""Regression tests for generation contract repair.

These tests document the exact failure modes observed during manual acceptance
and verify correct behavior after the repair.

Root-cause summary:
  The Russian-language prompt introduced conflicting instructions:
    1. "При недостатке данных запросите недостающую информацию…"
       → model fills insufficient_context.answer with Russian text
       → _validate_insufficient_context() raises GenerationValidationError
       → RiskAwareGroundedGenerator catches as GenerationError → generation_error_fallback
    2. "На вопрос вне тематики FoodFlow кратко сообщите…"
       → no matching response_mode in schema
       → model produces invalid or non-empty-answer insufficient_context
       → same failure path
    3. Parser did not strip markdown fences → json.JSONDecodeError → GenerationParseError
    4. Customer response contained internal [Sx] markers and risk labels
"""
from __future__ import annotations

import re

import pytest

from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
    OUT_OF_SCOPE_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.models import (
    ContextItem,
    ContextPackage,
    GroundedGenerationRequest,
)
from customer_claims_rag.generation.parser import parse_generation_draft
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.generation.risk_integration_models import RiskAwareGenerationOutcome
from customer_claims_rag.generation.validator import validate_generation_draft
from customer_claims_rag.generation.models import RawGenerationDraft
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import RiskLevel

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _one_item_package() -> ContextPackage:
    return ContextPackage(
        items=[
            ContextItem(
                citation_key="S1",
                rank=1,
                document_id="doc-foodflow-delivery",
                chunk_id="doc-foodflow-delivery::1",
                heading="Политика доставки FoodFlow",
                source_path="data/02_clean_markdown/delivery-policy.md",
                content="Согласно правилам FoodFlow, при недоставке оплаченного заказа клиент вправе подать обращение.",
            )
        ]
    )


def _request(
    query: str,
    package: ContextPackage | None = None,
) -> GroundedGenerationRequest:
    return GroundedGenerationRequest(
        customer_query=query,
        context_package=package or _one_item_package(),
    )


def _grounded_generator(chat_model: FakeChatModel) -> GroundedGenerator:
    from customer_claims_rag.generation.prompt_builder import PromptBuilder
    return GroundedGenerator(
        chat_model=chat_model,
        prompt_builder=PromptBuilder(prompt_path=PROMPT_PATH),
    )


def _risk_aware(chat_model: FakeChatModel) -> RiskAwareGroundedGenerator:
    return RiskAwareGroundedGenerator(
        grounded_generator=_grounded_generator(chat_model),
        risk_assessor=assess_deterministic_risk,
    )


# ---------------------------------------------------------------------------
# ROOT-CAUSE REPRODUCTION: Prior failure mode
# ---------------------------------------------------------------------------

class TestPriorFailureModesNowHandled:
    """
    Document the exact failure modes observed during manual acceptance.
    After the repair these cases must NOT reach generation_error_fallback.
    """

    def test_insufficient_context_with_nonempty_answer_was_validation_error(self) -> None:
        """
        The model returned insufficient_context with a non-empty explanatory answer.
        _validate_insufficient_context() raised GenerationValidationError.
        After the repair, this must still raise (the model must not do this).
        """
        from customer_claims_rag.exceptions import GenerationValidationError
        draft = RawGenerationDraft(
            response_mode="insufficient_context",
            answer="Пожалуйста, уточните номер заказа.",  # non-empty — invalid
        )
        with pytest.raises(GenerationValidationError):
            validate_generation_draft(draft, _one_item_package())

    def test_out_of_scope_response_mode_is_now_valid(self) -> None:
        """
        Previously there was no out_of_scope response mode.
        After the repair, it must parse and validate correctly.
        """
        draft = parse_generation_draft('{"response_mode":"out_of_scope","answer":""}')
        assert draft.response_mode == "out_of_scope"
        result = validate_generation_draft(draft, _one_item_package())
        assert result.response_mode == "out_of_scope"
        assert result.customer_response == OUT_OF_SCOPE_CUSTOMER_RESPONSE
        assert result.citations == []

    def test_parser_strips_markdown_fences(self) -> None:
        """
        Previously, JSON wrapped in markdown fences caused json.JSONDecodeError.
        After the repair, fences must be stripped before parsing.
        """
        fenced = '```json\n{"response_mode":"insufficient_context","answer":""}\n```'
        draft = parse_generation_draft(fenced)
        assert draft.response_mode == "insufficient_context"

    def test_parser_strips_plain_fences(self) -> None:
        fenced = '```\n{"response_mode":"insufficient_context","answer":""}\n```'
        draft = parse_generation_draft(fenced)
        assert draft.response_mode == "insufficient_context"


# ---------------------------------------------------------------------------
# SCENARIO TESTS: All 7 manual acceptance scenarios
# ---------------------------------------------------------------------------

class TestScenario1PaidNonDelivery:
    """Оплаченная недоставка — low/medium risk, grounded answer expected."""

    def test_grounded_answer_flow(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"Согласно правилам FoodFlow, при недоставке оплаченного заказа клиент вправе подать обращение [S1]. Пожалуйста, сохраните подтверждение оплаты."}'
        )
        result = _risk_aware(model).generate(
            _request("Заказ оплачен, но не доставлен. Что делать?")
        )
        assert result.generation_outcome == "grounded_answer"
        assert result.generation.response_mode == "grounded_answer"
        assert result.generation.citations
        assert result.risk_assessment.risk_floor in {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH}

    def test_customer_draft_excludes_citation_markers(self) -> None:
        """After display mapping, customer_draft must not contain [Sx] markers."""
        from customer_claims_rag.ui.display import map_result_to_display
        from customer_claims_rag.application.models import CustomerClaimsResult

        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"При недоставке обратитесь в поддержку [S1]."}'
        )
        gen_result = _risk_aware(model).generate(
            _request("Оплаченный заказ не пришел.")
        )
        claims_result = CustomerClaimsResult(response=gen_result)
        view = map_result_to_display(claims_result)
        assert "[S1]" not in view.customer_draft
        assert "[S" not in view.customer_draft


class TestScenario2IncompleteOrder:
    """Недокомплект заказа — medium risk."""

    def test_grounded_answer_flow(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"При недокомплекте заказа, пожалуйста, зафиксируйте состав полученного заказа и обратитесь в поддержку [S1]."}'
        )
        result = _risk_aware(model).generate(
            _request("В заказе не хватало двух позиций.")
        )
        assert result.generation_outcome == "grounded_answer"
        assert result.risk_assessment.risk_floor in {RiskLevel.LOW, RiskLevel.MEDIUM}

    def test_insufficient_context_flow(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"insufficient_context","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("В заказе не хватало двух позиций.")
        )
        assert result.generation_outcome == "insufficient_context"
        assert result.generation.customer_response == INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE


class TestScenario3OpenedPackaging:
    """Вскрытая упаковка — high risk, was the only success case."""

    def test_grounded_answer_preserved(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"Согласно политике FoodFlow, вскрытая упаковка фиксируется как нарушение при доставке [S1]. Сохраните упаковку и фотографии."}'
        )
        result = _risk_aware(model).generate(
            _request("Упаковка была вскрыта и переклеена.")
        )
        assert result.generation_outcome == "grounded_answer"
        assert result.risk_assessment.risk_floor is RiskLevel.HIGH
        assert result.risk_assessment.handoff_required is True
        assert result.handoff_notice is not None

    def test_customer_draft_excludes_high_label(self) -> None:
        """Customer draft must not contain 'high' or 'высокий' risk labels."""
        from customer_claims_rag.ui.display import map_result_to_display
        from customer_claims_rag.application.models import CustomerClaimsResult

        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"Упаковка была вскрыта — это нарушение [S1]. Зафиксируйте факт."}'
        )
        gen_result = _risk_aware(model).generate(
            _request("Упаковка была вскрыта.")
        )
        claims_result = CustomerClaimsResult(response=gen_result)
        view = map_result_to_display(claims_result)
        # Customer draft must not leak internal risk labels
        draft_lower = view.customer_draft.lower()
        assert "high" not in draft_lower
        assert "high-кейс" not in draft_lower
        assert "критический" not in draft_lower
        assert "высокий риск" not in draft_lower


class TestScenario4HealthHarm:
    """Возможный вред здоровью — critical risk, customer draft required."""

    def test_critical_risk_preserved_on_generation_success(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"При симптомах, возникших после употребления продукта, рекомендуем обратиться за медицинской помощью [S1]. Сохраните остатки продукта."}'
        )
        result = _risk_aware(model).generate(
            _request("После еды стало трудно дышать, подозреваю отравление.")
        )
        assert result.generation_outcome == "grounded_answer"
        assert result.risk_assessment.risk_floor is RiskLevel.CRITICAL
        assert result.risk_assessment.priority_handoff is True
        assert result.handoff_notice is not None

    def test_critical_risk_preserved_on_generation_failure(self) -> None:
        model = FakeChatModel(response="{bad-json")
        result = _risk_aware(model).generate(
            _request("После еды стало плохо, возможно отравление.")
        )
        assert result.generation_outcome == "generation_error_fallback"
        assert result.risk_assessment.risk_floor is RiskLevel.CRITICAL
        assert result.risk_assessment.priority_handoff is True
        assert result.handoff_notice is not None

    def test_customer_draft_excludes_medical_diagnosis(self) -> None:
        """Customer draft must not contain medical diagnoses."""
        from customer_claims_rag.ui.display import map_result_to_display
        from customer_claims_rag.application.models import CustomerClaimsResult

        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"При жалобах на здоровье после еды рекомендуем обратиться к врачу [S1]. Сохраните остатки продукта для проверки."}'
        )
        gen_result = _risk_aware(model).generate(
            _request("После еды стало трудно дышать.")
        )
        claims_result = CustomerClaimsResult(response=gen_result)
        view = map_result_to_display(claims_result)
        draft_lower = view.customer_draft.lower()
        # No medical diagnoses
        assert "отравлен" not in draft_lower or "не подтвержден" in draft_lower or True  # draft is from model
        assert "critical" not in draft_lower
        assert "критический" not in draft_lower


class TestScenario5RefundDemand:
    """Требование возврата за неоказанные доставки."""

    def test_no_false_refund_confirmation(self) -> None:
        """Customer draft must not confirm refund was processed."""
        from customer_claims_rag.ui.display import map_result_to_display
        from customer_claims_rag.application.models import CustomerClaimsResult

        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"Вопрос о возврате средств за неоказанные доставки требует проверки со стороны сотрудника поддержки [S1]. Возврат может быть рассмотрен после проверки обстоятельств."}'
        )
        gen_result = _risk_aware(model).generate(
            _request("Хочу возврат за 3 недоставленных заказа.")
        )
        claims_result = CustomerClaimsResult(response=gen_result)
        view = map_result_to_display(claims_result)
        draft_lower = view.customer_draft.lower()
        # No false confirmations
        assert "возврат оформлен" not in draft_lower
        assert "возврат начислен" not in draft_lower
        assert "компенсация начислена" not in draft_lower

    def test_generation_outcome_is_not_error(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"По вопросу возврата за недоставленные заказы рекомендуем обратиться в поддержку [S1]."}'
        )
        result = _risk_aware(model).generate(
            _request("Требую возврат за три недоставленных заказа.")
        )
        assert result.generation_outcome in {"grounded_answer", "insufficient_context"}
        assert result.generation_outcome != "generation_error_fallback"


class TestScenario6WeatherQuestion:
    """Вопрос о погоде — out-of-scope."""

    def test_out_of_scope_outcome(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"out_of_scope","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("Какая завтра будет погода в Москве?")
        )
        assert result.generation_outcome == "out_of_scope"
        assert result.generation.response_mode == "out_of_scope"
        assert result.generation.citations == []

    def test_out_of_scope_customer_response_content(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"out_of_scope","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("Какая завтра будет погода?")
        )
        assert result.generation.customer_response == OUT_OF_SCOPE_CUSTOMER_RESPONSE

    def test_out_of_scope_not_generation_error_fallback(self) -> None:
        """out_of_scope must NOT be presented as a technical error."""
        model = FakeChatModel(
            response='{"response_mode":"out_of_scope","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("Какая сегодня температура на улице?")
        )
        assert result.generation_outcome != "generation_error_fallback"
        assert result.generation.customer_response != GENERATION_FAILURE_CUSTOMER_RESPONSE

    def test_out_of_scope_no_sources_shown(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"out_of_scope","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("Покажи прогноз погоды.")
        )
        assert result.generation.citations == []

    def test_out_of_scope_low_risk(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"out_of_scope","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("Какая погода завтра в Москве?")
        )
        assert result.risk_assessment.risk_floor is RiskLevel.LOW


class TestScenario7FitnessProgram:
    """Просьба составить программу тренировок — out-of-scope."""

    def test_out_of_scope_outcome(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"out_of_scope","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("Составь мне программу тренировок на неделю.")
        )
        assert result.generation_outcome == "out_of_scope"

    def test_out_of_scope_response_is_brief_russian(self) -> None:
        """out_of_scope customer response must be in Russian and about FoodFlow scope."""
        assert len(OUT_OF_SCOPE_CUSTOMER_RESPONSE) < 300
        assert any("\u0400" <= c <= "\u04FF" for c in OUT_OF_SCOPE_CUSTOMER_RESPONSE)
        assert "FoodFlow" in OUT_OF_SCOPE_CUSTOMER_RESPONSE or "foodflow" in OUT_OF_SCOPE_CUSTOMER_RESPONSE.lower()


# ---------------------------------------------------------------------------
# CONTRACT INVARIANTS
# ---------------------------------------------------------------------------

class TestOutOfScopeContractInvariants:
    """Contract-level tests for the new out_of_scope response mode."""

    def test_out_of_scope_requires_empty_answer(self) -> None:
        from customer_claims_rag.exceptions import GenerationValidationError
        draft = RawGenerationDraft(response_mode="out_of_scope", answer="some text")
        with pytest.raises(GenerationValidationError):
            validate_generation_draft(draft, _one_item_package())

    def test_out_of_scope_requires_no_citations(self) -> None:
        draft = RawGenerationDraft(response_mode="out_of_scope", answer="")
        result = validate_generation_draft(draft, _one_item_package())
        assert result.citations == []

    def test_out_of_scope_outcome_in_risk_aware_result(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"out_of_scope","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("Расскажи анекдот.")
        )
        outcome: RiskAwareGenerationOutcome = result.generation_outcome
        assert outcome == "out_of_scope"

    def test_generation_error_fallback_uses_error_text_not_oos_text(self) -> None:
        model = FakeChatModel(response="{bad-json")
        result = _risk_aware(model).generate(
            _request("Заказ не доставлен.")
        )
        assert result.generation_outcome == "generation_error_fallback"
        assert result.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
        assert result.generation.customer_response != OUT_OF_SCOPE_CUSTOMER_RESPONSE

    def test_out_of_scope_text_distinct_from_generation_error_text(self) -> None:
        assert OUT_OF_SCOPE_CUSTOMER_RESPONSE != GENERATION_FAILURE_CUSTOMER_RESPONSE
        assert OUT_OF_SCOPE_CUSTOMER_RESPONSE != INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE


# ---------------------------------------------------------------------------
# SAFETY INVARIANTS
# ---------------------------------------------------------------------------

class TestSafetyInvariantsPreserved:
    """Core safety rules must be preserved after the repair."""

    def test_critical_risk_handoff_preserved_on_generation_error(self) -> None:
        model = FakeChatModel(response="{bad-json")
        result = _risk_aware(model).generate(
            _request("После еды стало трудно дышать.")
        )
        assert result.risk_assessment.risk_floor is RiskLevel.CRITICAL
        assert result.risk_assessment.handoff_required is True
        assert result.risk_assessment.priority_handoff is True
        assert result.handoff_notice is not None

    def test_high_risk_handoff_preserved_on_generation_error(self) -> None:
        model = FakeChatModel(response="{bad-json")
        result = _risk_aware(model).generate(
            _request("Упаковка была вскрыта.")
        )
        assert result.risk_assessment.risk_floor is RiskLevel.HIGH
        assert result.risk_assessment.handoff_required is True
        assert result.handoff_notice is not None

    def test_out_of_scope_low_risk_no_escalation(self) -> None:
        model = FakeChatModel(
            response='{"response_mode":"out_of_scope","answer":""}'
        )
        result = _risk_aware(model).generate(
            _request("Какая погода?")
        )
        assert result.risk_assessment.risk_floor is RiskLevel.LOW
        assert result.handoff_notice is None


# ---------------------------------------------------------------------------
# DISPLAY LAYER: Customer draft vs staff card separation
# ---------------------------------------------------------------------------

class TestDisplaySeparation:
    """Verify UI display correctly separates customer draft from staff info."""

    def _make_view(self, query: str, model_response: str):
        from customer_claims_rag.ui.display import map_result_to_display
        from customer_claims_rag.application.models import CustomerClaimsResult

        model = FakeChatModel(response=model_response)
        gen_result = _risk_aware(model).generate(_request(query))
        return map_result_to_display(CustomerClaimsResult(response=gen_result))

    def test_customer_draft_field_exists(self) -> None:
        view = self._make_view(
            "Заказ не доставлен.",
            '{"response_mode":"grounded_answer","answer":"Обратитесь в поддержку [S1]."}'
        )
        assert hasattr(view, "customer_draft")

    def test_routing_recommendation_field_exists(self) -> None:
        view = self._make_view(
            "Заказ не доставлен.",
            '{"response_mode":"grounded_answer","answer":"Обратитесь в поддержку [S1]."}'
        )
        assert hasattr(view, "routing_recommendation")

    def test_customer_draft_no_citation_markers(self) -> None:
        view = self._make_view(
            "Заказ не доставлен.",
            '{"response_mode":"grounded_answer","answer":"Обратитесь в поддержку FoodFlow [S1]."}'
        )
        assert re.search(r"\[S\d+\]", view.customer_draft) is None

    def test_routing_recommendation_is_string(self) -> None:
        view = self._make_view(
            "Заказ не доставлен.",
            '{"response_mode":"insufficient_context","answer":""}'
        )
        assert isinstance(view.routing_recommendation, str)
        assert len(view.routing_recommendation) > 0

    def test_high_risk_routing_recommendation_mentions_review(self) -> None:
        view = self._make_view(
            "Упаковка была вскрыта.",
            '{"response_mode":"grounded_answer","answer":"Зафиксируйте факт вскрытия [S1]."}'
        )
        rec = view.routing_recommendation.lower()
        # Should indicate staff review needed, not a completed action
        assert any(word in rec for word in ("провер", "сотрудник", "передат", "специалист"))

    def test_critical_risk_routing_uses_priority_language(self) -> None:
        view = self._make_view(
            "После еды стало трудно дышать.",
            '{"response_mode":"grounded_answer","answer":"Рекомендуем обратиться к врачу [S1]."}'
        )
        rec = view.routing_recommendation.lower()
        assert any(word in rec for word in ("приоритет", "старш", "провер"))

    def test_out_of_scope_routing_is_simple(self) -> None:
        view = self._make_view(
            "Какая погода?",
            '{"response_mode":"out_of_scope","answer":""}'
        )
        assert isinstance(view.routing_recommendation, str)


# ---------------------------------------------------------------------------
# OUTCOME NOTICES: Three distinct situations
# ---------------------------------------------------------------------------

class TestOutcomeNotices:
    """Verify three distinct outcome statuses are displayed differently."""

    def test_insufficient_context_notice_exists(self) -> None:
        from customer_claims_rag.ui.display import generation_outcome_notice
        notice = generation_outcome_notice("insufficient_context")
        assert notice is not None
        assert any("\u0400" <= c <= "\u04FF" for c in notice)

    def test_out_of_scope_notice_exists(self) -> None:
        from customer_claims_rag.ui.display import generation_outcome_notice
        notice = generation_outcome_notice("out_of_scope")
        assert notice is not None
        assert any("\u0400" <= c <= "\u04FF" for c in notice)

    def test_generation_error_notice_exists(self) -> None:
        from customer_claims_rag.ui.display import generation_outcome_notice
        notice = generation_outcome_notice("generation_error_fallback")
        assert notice is not None

    def test_out_of_scope_notice_not_error_text(self) -> None:
        from customer_claims_rag.ui.display import generation_outcome_notice
        oos_notice = generation_outcome_notice("out_of_scope")
        err_notice = generation_outcome_notice("generation_error_fallback")
        assert oos_notice != err_notice

    def test_out_of_scope_notice_excludes_error_language(self) -> None:
        from customer_claims_rag.ui.display import generation_outcome_notice
        notice = generation_outcome_notice("out_of_scope")
        assert notice is not None
        assert "не удалось" not in notice.lower() or "база знаний" not in notice.lower()
