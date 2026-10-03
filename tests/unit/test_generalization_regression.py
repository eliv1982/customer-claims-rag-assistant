"""Generalization regression matrix — demo-overfitting audit.

Covers:
- Different phrasings of the same problem category
- Different item counts, order counts, and symptom descriptions
- Primary path (grounded answer), IC path, and error path
- Unknown-but-thematic queries without a dedicated template
- Out-of-scope queries

No live LLM calls; all tests use constructed pipeline results.
"""

from __future__ import annotations

import pytest

from customer_claims_rag.application.models import CustomerClaimsResult, RetrievedItemMeta
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.handoff import build_handoff_notice
from customer_claims_rag.generation.models import Citation, GroundedGenerationResult
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskLevel, RiskSignal
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.rules import assess_deterministic_risk, normalize_for_matching
from customer_claims_rag.risk.models import RiskAssessmentRequest
from customer_claims_rag.risk.validator import build_deterministic_risk_result
from customer_claims_rag.application.customer_templates import (
    CATEGORY_DRAFT_TEMPLATES,
    GENERIC_RISK_DRAFTS,
    HARD_TEMPLATE_CATEGORIES,
)
from customer_claims_rag.application.customer_text_policy import detect_draft_violations
from customer_claims_rag.ui.display import map_result_to_display


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _risk_from_query(query: str) -> DeterministicRiskResult:
    return assess_deterministic_risk(RiskAssessmentRequest(customer_query=query))


def _make_risk(level: RiskLevel, codes: list[RiskReasonCode] | None = None) -> DeterministicRiskResult:
    signals = [RiskSignal(reason_code=c, level=level, rule_id=c.value) for c in (codes or [])]
    return build_deterministic_risk_result(signals)


def _fake_citation(key: str = "S1") -> Citation:
    return Citation(
        citation_key=key,
        heading="Правила FoodFlow",
        document_id="doc-test",
        chunk_id="chunk-0",
        source_path="docs/rules.md",
    )


def _grounded(text: str) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=text,
        citations=(_fake_citation(),),
    )


def _ic() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
        citations=(),
    )


def _error_gen() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
        citations=(),
    )


def _result(
    generation: GroundedGenerationResult,
    risk: DeterministicRiskResult,
    outcome: str,
    query: str = "",
    retrieved: list[RetrievedItemMeta] | None = None,
) -> CustomerClaimsResult:
    resp = RiskAwareGroundedGenerationResult(
        generation=generation,
        risk_assessment=risk,
        handoff_notice=build_handoff_notice(risk),
        generation_outcome=outcome,  # type: ignore[arg-type]
    )
    return CustomerClaimsResult(
        response=resp,
        customer_query=query,
        retrieved_items=tuple(retrieved or []),
    )


def _no_forbidden(draft: str) -> None:
    lower = draft.lower()
    assert "служб" not in lower or "поддержк" not in lower, f"Support redirect: {draft!r}"
    assert "insufficient_context" not in lower
    assert "generation_error" not in lower
    assert "не удалось подготовить" not in lower
    assert "в доступных материалах недостаточно" not in lower


# ---------------------------------------------------------------------------
# A. Risk classification — different phrasings of the same category
# ---------------------------------------------------------------------------

class TestRiskClassificationNonDelivery:
    """Non-delivery phrasings — at minimum one canonical phrase must be HIGH."""

    @pytest.mark.parametrize("query,expect_high", [
        ("Заказ так и не приехал", True),       # canonical NON_DELIVERY
        ("Вчера не доставили заказ", True),      # yesterday non-delivery
        ("Заказ вообще не привезли", True),      # canonical
        ("Оплаченный заказ не был доставлен", False),  # LLM-level; risk LOW by rules
        ("Курьер не привез заказ", False),       # LOW by rules (no canonical match)
        ("Доставка отменена, деньги не вернулись", False),  # LOW
    ])
    def test_non_delivery_risk(self, query: str, expect_high: bool) -> None:
        risk = _risk_from_query(query)
        if expect_high:
            assert risk.risk_floor in {RiskLevel.HIGH, RiskLevel.CRITICAL}, (
                f"Expected HIGH for: {query!r}, got {risk.risk_floor}"
            )
        # LOW-risk variants should not crash — just not high
        assert risk.risk_floor is not None
    def test_non_delivery_category_via_keyword_fallback(self) -> None:
        """Queries that don't trigger risk rules still get correct category via keyword."""
        risk = _make_risk(RiskLevel.LOW)
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context",
                    query="Оплаченный заказ не был доставлен")
        )
        assert view.claim_category == "Недоставка оплаченного заказа"
        assert "мы проверим" in view.customer_draft.lower()

    def test_non_delivery_template_no_order_number(self) -> None:
        """Template must not invent an order number."""
        risk = _make_risk(RiskLevel.LOW)
        for query in ["Заказ не доставили", "не привезли заказ"]:
            view = map_result_to_display(
                _result(_ic(), risk, "insufficient_context", query=query)
            )
            import re
            assert not re.search(r"№\s*\d+|\bзаказ\s+#\d+", view.customer_draft), (
                f"Template invented order number for query: {query!r}"
            )


class TestRiskClassificationHealth:
    """Health complaint phrasings — all must resolve to CRITICAL."""

    @pytest.mark.parametrize("query", [
        "Мне стало плохо после вашего заказа",
        "После доставки появилась тошнота",
        "После вашей еды рвота",
        "После ужина заболел живот",
        "Аллергическая реакция после заказа FoodFlow",
        "Слабость после еды",
        # NOTE: "Отравился после заказа" → LOW (gap: рефлексивная форма 'отравился'
        # не покрыта паттерном отравлен\w* в _HEALTH_SYMPTOM — известное ограничение)
    ])
    def test_is_critical(self, query: str) -> None:
        risk = _risk_from_query(query)
        assert risk.risk_floor == RiskLevel.CRITICAL, (
            f"Expected CRITICAL for health query: {query!r}, got {risk.risk_floor}"
        )
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes or \
               RiskReasonCode.MASS_INCIDENT in risk.reason_codes

    def test_otravils_ya_is_critical(self) -> None:
        # Gap fixed: 'отравился' (reflexive) now matches via 'отравил\w*' in _HEALTH_SYMPTOM.
        risk = _risk_from_query("Отравился после заказа")
        assert risk.risk_floor == RiskLevel.CRITICAL
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context",
                    query="Отравился после заказа")
        )
        _no_forbidden(view.customer_draft)
        expected = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]
        assert view.customer_draft == expected

    def test_mne_plokho_multi_sentence_is_critical(self) -> None:
        """Gap fixed: 'Мне плохо' now matches via 'мне плохо' in _HEALTH_SYMPTOM;
        consumption link 'заказывал' covered by bidirectional заказ.*мне плохо pattern."""
        risk = _risk_from_query("Мне плохо. Заказывал у вас вчера")
        assert risk.risk_floor == RiskLevel.CRITICAL
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context",
                    query="Мне плохо. Заказывал у вас вчера")
        )
        _no_forbidden(view.customer_draft)

    def test_belly_pain_is_critical(self) -> None:
        risk = _risk_from_query("Боль в животе после вашего заказа")
        assert risk.risk_floor == RiskLevel.CRITICAL

    def test_template_has_general_symptoms_not_specific(self) -> None:
        template = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]
        lower = template.lower()
        assert "тошнот" not in lower, "Demo-specific symptom 'тошнота'"
        assert "рвот" not in lower, "Demo-specific symptom 'рвота'"
        assert "выраженн" in lower or "усиливающ" in lower or "сохраняющ" in lower, \
            "Template must use general symptom language"

    @pytest.mark.parametrize("query", [
        "Мне стало плохо после вашего заказа",
        "Боль в животе после еды из FoodFlow",
        "Аллергическая реакция после заказа",
        "Слабость и недомогание после вашей доставки",
        "Мне плохо. Заказывал у вас вчера",
    ])
    def test_all_health_variants_get_safe_template(self, query: str) -> None:
        risk = _risk_from_query(query)
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context", query=query)
        )
        expected = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]
        assert view.customer_draft == expected, (
            f"Health query did not get safe template: {query!r}"
        )

    @pytest.mark.parametrize("query", [
        "Мне стало плохо после вашего заказа",
        "Боль в животе после еды из FoodFlow",
        "Аллергическая реакция после заказа",
    ])
    def test_health_grounded_also_uses_template(self, query: str) -> None:
        """Hard override: even a clean grounded answer gets the health template."""
        risk = _risk_from_query(query)
        clean = "Сожалеем. Обратитесь к врачу. [S1]"
        view = map_result_to_display(
            _result(_grounded(clean), risk, "grounded_answer", query=query)
        )
        expected = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]
        assert view.customer_draft == expected


class TestRiskClassificationMissingItem:
    """Missing item phrasings."""

    @pytest.mark.parametrize("query,expect_medium", [
        ("В заказе не хватало одной позиции", False),  # known gap: "не хватало" (past tense) vs "не хватил/не хватает"
        ("Мне не положили один товар", True),
        ("Часть заказа не доставили", True),
        ("Забыли положить соус", True),
        ("Привезли не весь заказ", True),
    ])
    def test_missing_item_classification(self, query: str, expect_medium: bool) -> None:
        risk = _risk_from_query(query)
        if expect_medium:
            assert RiskReasonCode.MISSING_ITEM in risk.reason_codes, (
                f"Expected MISSING_ITEM for: {query!r}"
            )

    def test_ne_khvatalo_is_medium(self) -> None:
        """Gap fixed: 'не хватало' (past tense) now covered by added _MISSING_ITEM pattern."""
        risk = _risk_from_query("В заказе не хватало одной позиции")
        assert risk.risk_floor == RiskLevel.MEDIUM
        assert RiskReasonCode.MISSING_ITEM in risk.reason_codes
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context",
                    query="В заказе не хватало одной позиции")
        )
        _no_forbidden(view.customer_draft)
        expected = CATEGORY_DRAFT_TEMPLATES["Неполный заказ (недокомплект)"]
        assert view.customer_draft == expected

    def test_single_item_missing_uses_template(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.MISSING_ITEM])
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context",
                    query="В заказе не было одного блюда")
        )
        expected = CATEGORY_DRAFT_TEMPLATES["Неполный заказ (недокомплект)"]
        assert view.customer_draft == expected

    def test_multiple_items_missing_same_template(self) -> None:
        """The template is generic — doesn't mention a specific count of items."""
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.MISSING_ITEM])
        for query in [
            "Не хватало трёх позиций",
            "Половина заказа не приехала",
            "Несколько блюд не положили",
        ]:
            view = map_result_to_display(
                _result(_ic(), risk, "insufficient_context", query=query)
            )
            lower = view.customer_draft.lower()
            # Template must not invent specific numbers
            assert "одн" not in lower or "одной позиц" not in lower


class TestRiskClassificationRefund:
    """Refund demand: different order counts must not leak demo-specific '5 заказов'."""

    EXPECTED_TEMPLATE = CATEGORY_DRAFT_TEMPLATES["Требование возврата средств"]

    def _ic_view(self, query: str) -> object:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        return map_result_to_display(
            _result(_ic(), risk, "insufficient_context", query=query)
        )

    def test_single_order_refund(self) -> None:
        view = self._ic_view("Требую возврат за заказ №99.")
        assert view.customer_draft == self.EXPECTED_TEMPLATE
        assert "пяти" not in view.customer_draft.lower()

    def test_three_orders_refund(self) -> None:
        view = self._ic_view("Прошу вернуть деньги за три заказа за последние недели.")
        assert view.customer_draft == self.EXPECTED_TEMPLATE
        assert "три" not in view.customer_draft.lower()

    def test_unspecified_count_refund(self) -> None:
        view = self._ic_view("Требую возврат средств за несколько доставок.")
        assert view.customer_draft == self.EXPECTED_TEMPLATE
        assert "пяти" not in view.customer_draft.lower()

    def test_partial_refund(self) -> None:
        view = self._ic_view("Хочу частичный возврат за недокомплект.")
        lower = view.customer_draft.lower()
        assert "пяти" not in lower

    def test_template_has_no_specific_count(self) -> None:
        t = self.EXPECTED_TEMPLATE.lower()
        assert "пяти" not in t
        assert "всех пяти" not in t
        assert "три заказа" not in t
        assert "одного заказа" not in t

    @pytest.mark.parametrize("query", [
        "Требую полный возврат за один заказ",
        "Верните деньги за три заказа",
        "Прошу компенсировать стоимость всех заказов",
        "Полный возврат за подписку с шестью доставками",
        "Возврат средств за недоставленный заказ из-за задержки",
    ])
    def test_no_demo_count_in_any_refund_case(self, query: str) -> None:
        view = self._ic_view(query)
        assert "пяти" not in view.customer_draft.lower()
        assert "всех пяти" not in view.customer_draft.lower()
        _no_forbidden(view.customer_draft)


# ---------------------------------------------------------------------------
# B. Packaging and food quality — different scenario descriptions
# ---------------------------------------------------------------------------

class TestPackagingVariants:

    @pytest.mark.parametrize("query", [
        "Пломба сорвана на контейнере",
        "Крышка треснула, всё вытекло",
        "Упаковка вскрыта при получении",
        # "Контейнер был открыт" → known gap: разделитель 'был' между существительным и прилагательным
        "Нарушена герметичность упаковки",
    ])
    def test_packaging_risk_detected(self, query: str) -> None:
        risk = _risk_from_query(query)
        assert RiskReasonCode.PACKAGE_TAMPERING in risk.reason_codes, (
            f"Expected PACKAGE_TAMPERING for: {query!r}"
        )

    def test_konteyner_byl_otkryt_is_high(self) -> None:
        # Gap fixed: 'Контейнер был открыт' now matches via added copula pattern
        # 'контейнер\w*\s+(?:был[аои]?\s+)?открыт\w*' in _PACKAGE_TAMPERING.
        risk = _risk_from_query("Контейнер был открыт")
        assert risk.risk_floor == RiskLevel.HIGH
        assert RiskReasonCode.PACKAGE_TAMPERING in risk.reason_codes
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context",
                    query="Контейнер был открыт")
        )
        _no_forbidden(view.customer_draft)
        expected = CATEGORY_DRAFT_TEMPLATES["Нарушение целостности упаковки"]
        assert view.customer_draft == expected

    def test_packaging_ic_uses_template(self) -> None:
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context")
        )
        assert view.customer_draft == CATEGORY_DRAFT_TEMPLATES["Нарушение целостности упаковки"]

    def test_packaging_template_says_do_not_consume(self) -> None:
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context")
        )
        assert "не употребляйте" in view.customer_draft.lower()

    def test_food_spoilage_category(self) -> None:
        """Странный запах / испорченный продукт → FOOD_SPOILAGE, dedicated template."""
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.FOOD_SPOILAGE])
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context")
        )
        assert view.claim_category == "Испорченный или некачественный продукт"
        assert "Испорченный или некачественный продукт" in CATEGORY_DRAFT_TEMPLATES
        assert view.customer_draft == CATEGORY_DRAFT_TEMPLATES["Испорченный или некачественный продукт"]

    def test_food_spoilage_not_mixed_with_health(self) -> None:
        """Spoiled product without consumption / health symptoms stays at HIGH, not CRITICAL."""
        risk = _risk_from_query("Еда испортилась, странный запах у блюда")
        assert risk.risk_floor == RiskLevel.HIGH
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION not in risk.reason_codes

    def test_packaging_not_auto_escalated_to_health(self) -> None:
        """Torn seal alone must not trigger health symptoms rule."""
        risk = _risk_from_query("Пломба сорвана, я только заметил при получении")
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION not in risk.reason_codes


# ---------------------------------------------------------------------------
# C. Primary vs fallback paths for known categories
# ---------------------------------------------------------------------------

class TestPrimaryAndFallbackPaths:
    """For each major category: grounded, IC, and error paths."""

    def test_non_delivery_grounded_clean_uses_llm_output(self) -> None:
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.NON_DELIVERY])
        clean = "Сожалеем о недоставке. Мы проверим статус заказа. [S1]"
        view = map_result_to_display(
            _result(_grounded(clean), risk, "grounded_answer",
                    query="заказ так и не приехал")
        )
        assert view.draft_sanitized is False
        assert "S1" not in view.customer_draft

    def test_non_delivery_grounded_violation_uses_template(self) -> None:
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.NON_DELIVERY])
        bad = "Служба поддержки проверит ваш заказ. [S1]"
        view = map_result_to_display(
            _result(_grounded(bad), risk, "grounded_answer",
                    query="заказ так и не приехал")
        )
        assert view.draft_sanitized is True
        assert "служб" not in view.customer_draft.lower() or \
               "поддержк" not in view.customer_draft.lower()

    def test_non_delivery_ic_uses_template(self) -> None:
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.NON_DELIVERY])
        view = map_result_to_display(_result(_ic(), risk, "insufficient_context"))
        assert view.customer_draft == CATEGORY_DRAFT_TEMPLATES["Недоставка оплаченного заказа"]
        assert view.draft_sanitized is True

    def test_non_delivery_error_uses_template(self) -> None:
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.NON_DELIVERY])
        view = map_result_to_display(_result(_error_gen(), risk, "generation_error_fallback"))
        assert view.customer_draft == CATEGORY_DRAFT_TEMPLATES["Недоставка оплаченного заказа"]
        assert view.draft_sanitized is True

    def test_missing_item_grounded_clean_uses_llm(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.MISSING_ITEM])
        clean = "Сожалеем, что позиция отсутствовала. Мы проверим состав. [S1]"
        view = map_result_to_display(
            _result(_grounded(clean), risk, "grounded_answer")
        )
        assert view.draft_sanitized is False

    def test_missing_item_ic_uses_template(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.MISSING_ITEM])
        view = map_result_to_display(_result(_ic(), risk, "insufficient_context"))
        assert view.customer_draft == CATEGORY_DRAFT_TEMPLATES["Неполный заказ (недокомплект)"]

    def test_health_grounded_always_template(self) -> None:
        """HARD override: even clean grounded answer uses health template."""
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        clean = "Сожалеем. Рекомендуем обратиться к врачу. [S1]"
        view = map_result_to_display(
            _result(_grounded(clean), risk, "grounded_answer")
        )
        assert view.customer_draft == CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]

    def test_refund_ic_uses_template(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        view = map_result_to_display(_result(_ic(), risk, "insufficient_context"))
        assert view.customer_draft == CATEGORY_DRAFT_TEMPLATES["Требование возврата средств"]

    def test_refund_grounded_clean_uses_llm(self) -> None:
        """Valid grounded answer for refund demand must not be blindly replaced."""
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        clean = (
            "Мы проверим заказы и применимые условия возврата. "
            "До завершения проверки подтвердить возврат нельзя. [S1]"
        )
        view = map_result_to_display(
            _result(_grounded(clean), risk, "grounded_answer")
        )
        assert view.draft_sanitized is False
        assert view.customer_draft != CATEGORY_DRAFT_TEMPLATES["Требование возврата средств"]


# ---------------------------------------------------------------------------
# D. Unknown thematic queries — no dedicated template
# ---------------------------------------------------------------------------

class TestUnknownThematicQueries:
    """Queries about FoodFlow that don't match any risk rule or template category."""

    @pytest.mark.parametrize("query", [
        "Можно ли изменить адрес доставки за день до заказа?",
        "Хочу оформить подписку на план питания, как это сделать?",
        "Привезли блюдо не из моего заказа — что делать?",
        "Статус показывает доставлено, но я ещё не получил ничего",
    ])
    def test_thematic_ic_no_crash_and_no_technical_text(self, query: str) -> None:
        risk = _risk_from_query(query)
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context", query=query)
        )
        _no_forbidden(view.customer_draft)
        assert len(view.customer_draft) > 10, "Draft must not be empty"

    @pytest.mark.parametrize("query", [
        "Можно ли изменить адрес доставки за день до заказа?",
        "Хочу оформить подписку на план питания",
    ])
    def test_thematic_grounded_clean_llm_passes_through(self, query: str) -> None:
        """Informational/thematic queries with clean LLM output show the LLM response."""
        risk = _risk_from_query(query)
        clean = "Согласно правилам FoodFlow, изменить адрес можно до начала сборки. [S1]"
        view = map_result_to_display(
            _result(_grounded(clean), risk, "grounded_answer", query=query)
        )
        assert view.draft_sanitized is False
        assert "S1" not in view.customer_draft

    def test_wrong_item_no_template_gets_generic_draft(self) -> None:
        """'Привезли другое блюдо' → no keyword match → generic FoodFlow clarification draft."""
        risk = _make_risk(RiskLevel.LOW)
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context",
                    query="Привезли совсем другое блюдо, не то что я заказывал")
        )
        _no_forbidden(view.customer_draft)
        # Generic draft is safe — doesn't echo back claim-specific details
        assert "другое блюдо" not in view.customer_draft
        # Now uses FoodFlow clarification template that asks for details
        assert any(
            phrase in view.customer_draft.lower()
            for phrase in ("уточните", "укажите", "сообщите")
        ), f"Expected clarification request; got: {view.customer_draft!r}"

    def test_wrong_item_grounded_clean_passes_through(self) -> None:
        risk = _make_risk(RiskLevel.LOW)
        clean = "Сожалеем, что привезли неверный заказ. Мы проверим детали. [S1]"
        view = map_result_to_display(
            _result(_grounded(clean), risk, "grounded_answer",
                    query="Привезли другое блюдо")
        )
        assert view.draft_sanitized is False


# ---------------------------------------------------------------------------
# E. Out-of-scope queries
# ---------------------------------------------------------------------------

class TestOutOfScopeGeneralization:

    @pytest.mark.parametrize("oos_query", [
        "Какая погода завтра?",
        "Переведи этот текст на английский",
        "Как накачать мышцы?",
    ])
    def test_oos_no_crash(self, oos_query: str) -> None:
        risk = _make_risk(RiskLevel.LOW)
        from customer_claims_rag.generation.fallback import OUT_OF_SCOPE_CUSTOMER_RESPONSE
        gen = GroundedGenerationResult(
            response_mode="out_of_scope",
            customer_response=OUT_OF_SCOPE_CUSTOMER_RESPONSE,
            citations=(),
        )
        view = map_result_to_display(_result(gen, risk, "out_of_scope", query=oos_query))
        assert view.generation_outcome == "out_of_scope"
        assert "служб" not in view.customer_draft.lower() or \
               "поддержк" not in view.customer_draft.lower()
        assert len(view.customer_draft) > 5

    def test_oos_staff_actions_are_oos_specific(self) -> None:
        risk = _make_risk(RiskLevel.LOW)
        from customer_claims_rag.generation.fallback import OUT_OF_SCOPE_CUSTOMER_RESPONSE
        gen = GroundedGenerationResult(
            response_mode="out_of_scope",
            customer_response=OUT_OF_SCOPE_CUSTOMER_RESPONSE,
            citations=(),
        )
        view = map_result_to_display(_result(gen, risk, "out_of_scope"))
        staff_text = " ".join(view.staff_actions).lower()
        assert "поиск" not in staff_text or "не выполняйте поиск" in staff_text


# ---------------------------------------------------------------------------
# F. Template integrity — no demo-specific facts in any template
# ---------------------------------------------------------------------------

class TestTemplateIntegrityNoDemoFacts:

    @pytest.mark.parametrize("category,template", list(CATEGORY_DRAFT_TEMPLATES.items()))
    def test_no_order_number_in_template(self, category: str, template: str) -> None:
        import re
        assert not re.search(r"№\s*\d+|\bзаказ\s*#\d+|\b\d{4,}\b", template), (
            f"Template '{category}' contains a specific order number"
        )

    @pytest.mark.parametrize("category,template", list(CATEGORY_DRAFT_TEMPLATES.items()))
    def test_no_specific_count_in_template(self, category: str, template: str) -> None:
        lower = template.lower()
        assert "пяти заказов" not in lower, f"Demo count in '{category}'"
        assert "всех пяти" not in lower
        assert "трёх заказов" not in lower
        assert "двух позиций" not in lower, f"Demo count in '{category}'"

    @pytest.mark.parametrize("category,template", list(CATEGORY_DRAFT_TEMPLATES.items()))
    def test_no_specific_product_name_in_template(self, category: str, template: str) -> None:
        lower = template.lower()
        assert "суши" not in lower
        assert "ролл" not in lower
        assert "пицц" not in lower

    def test_health_template_no_specific_symptoms(self) -> None:
        t = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"].lower()
        assert "тошнот" not in t, "Demo symptom тошнота"
        assert "рвот" not in t, "Demo symptom рвота"

    @pytest.mark.parametrize("category,template", list(CATEGORY_DRAFT_TEMPLATES.items()))
    def test_no_violations_detected_in_templates(self, category: str, template: str) -> None:
        """All deterministic templates must pass the safety validator themselves."""
        violations = detect_draft_violations(template)
        assert not violations, (
            f"Template '{category}' failed validator: {violations}"
        )

    @pytest.mark.parametrize("risk_level,draft", list(GENERIC_RISK_DRAFTS.items()))
    def test_no_violations_in_generic_risk_drafts(self, risk_level: RiskLevel, draft: str) -> None:
        violations = detect_draft_violations(draft)
        assert not violations, (
            f"Generic risk draft for {risk_level} failed validator: {violations}"
        )

    @pytest.mark.parametrize(
        "risk_level",
        [RiskLevel.LOW, RiskLevel.MEDIUM],
    )
    def test_generic_risk_drafts_request_clarification_not_vague_promises(
        self, risk_level: RiskLevel,
    ) -> None:
        """LOW/MEDIUM generic fallbacks must ask for details, not promise review/follow-up."""
        draft = GENERIC_RISK_DRAFTS[risk_level]
        lower = draft.lower()
        assert any(phrase in lower for phrase in ("уточните", "укажите")), (
            f"Generic draft for {risk_level} must request clarification: {draft!r}"
        )
        assert "мы проверим обращение" not in lower
        assert "мы проверим детали" not in lower
        assert "после проверки сообщим" not in lower
        assert "сообщим о результате" not in lower


# ---------------------------------------------------------------------------
# G. Evidence preservation during fallback
# ---------------------------------------------------------------------------

class TestEvidencePreservation:

    def test_ic_retrieved_items_preserved_for_refund(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        retrieved = [
            RetrievedItemMeta(citation_key="S1", rank=1, heading="Условия возврата", document_id="doc-04"),
            RetrievedItemMeta(citation_key="S2", rank=2, heading="Правила задержек", document_id="doc-02"),
        ]
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context", retrieved=retrieved)
        )
        assert len(view.retrieved_materials) == 2
        assert len(view.citations) == 0

    def test_error_retrieved_items_preserved(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.MISSING_ITEM])
        retrieved = [
            RetrievedItemMeta(citation_key="S1", rank=1, heading="Правила комплектации", document_id="doc-06"),
        ]
        view = map_result_to_display(
            _result(_error_gen(), risk, "generation_error_fallback", retrieved=retrieved)
        )
        assert len(view.retrieved_materials) == 1

    def test_grounded_citations_shown_as_confirmed(self) -> None:
        risk = _make_risk(RiskLevel.LOW)
        view = map_result_to_display(
            _result(
                GroundedGenerationResult(
                    response_mode="grounded_answer",
                    customer_response="Мы проверим заказ. [S1]",
                    citations=(_fake_citation("S1"), _fake_citation("S2")),
                ),
                risk, "grounded_answer",
            )
        )
        assert len(view.citations) == 2
        assert all(c.is_confirmed for c in view.citations)


# ---------------------------------------------------------------------------
# H. Ten-scenario risk coverage matrix (RISK COVERAGE REPAIR)
# ---------------------------------------------------------------------------

class TestTenScenarioRiskMatrix:
    """Exact ten-scenario regression matrix as required by repair specification.

    Covers all four coverage gaps fixed in this repair round:
      A. Health complaints (reflexive + skin + general malaise + multi-sentence)
      B. Missing-item past-tense forms
      C. Package tamper copula forms
      D. Wrong item (new category)
    """

    def _view(self, query: str) -> object:
        risk = _risk_from_query(query)
        gen = _ic()
        return map_result_to_display(_result(gen, risk, "insufficient_context", query=query))

    # Scenario 1: отравился после заказа
    def test_s1_otravils_ya(self) -> None:
        risk = _risk_from_query("Отравился после заказа.")
        assert risk.risk_floor == RiskLevel.CRITICAL
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
        assert risk.handoff_required is True
        assert risk.priority_handoff is True
        view = self._view("Отравился после заказа.")
        assert view.claim_category == "Жалоба на здоровье после употребления продукта"
        _no_forbidden(view.customer_draft)
        expected = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]
        assert view.customer_draft == expected

    # Scenario 2: мне плохо + заказывал (multi-sentence)
    def test_s2_mne_plokho(self) -> None:
        risk = _risk_from_query("Мне плохо. Заказывал у вас вчера.")
        assert risk.risk_floor == RiskLevel.CRITICAL
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
        view = self._view("Мне плохо. Заказывал у вас вчера.")
        assert view.claim_category == "Жалоба на здоровье после употребления продукта"
        _no_forbidden(view.customer_draft)

    # Scenario 3: кожная реакция (сыпь)
    def test_s3_sypj_after_order(self) -> None:
        risk = _risk_from_query("После заказа появилась сыпь.")
        assert risk.risk_floor == RiskLevel.CRITICAL
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
        view = self._view("После заказа появилась сыпь.")
        assert view.claim_category == "Жалоба на здоровье после употребления продукта"
        _no_forbidden(view.customer_draft)

    # Scenario 4: аллергическая реакция
    def test_s4_allergicheskaya_reakciya(self) -> None:
        risk = _risk_from_query("После еды началась аллергическая реакция.")
        assert risk.risk_floor == RiskLevel.CRITICAL
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
        view = self._view("После еды началась аллергическая реакция.")
        assert view.claim_category == "Жалоба на здоровье после употребления продукта"
        _no_forbidden(view.customer_draft)

    # Scenario 5: не хватало позиции (past tense)
    def test_s5_ne_khvatalo(self) -> None:
        risk = _risk_from_query("В заказе не хватало одной позиции.")
        assert risk.risk_floor == RiskLevel.MEDIUM
        assert RiskReasonCode.MISSING_ITEM in risk.reason_codes
        assert risk.handoff_required is False
        view = self._view("В заказе не хватало одной позиции.")
        assert view.claim_category == "Неполный заказ (недокомплект)"
        expected = CATEGORY_DRAFT_TEMPLATES["Неполный заказ (недокомплект)"]
        assert view.customer_draft == expected
        _no_forbidden(view.customer_draft)

    # Scenario 6: нам не положили напиток
    def test_s6_nam_ne_polozhili(self) -> None:
        risk = _risk_from_query("Нам не положили напиток.")
        assert risk.risk_floor == RiskLevel.MEDIUM
        assert RiskReasonCode.MISSING_ITEM in risk.reason_codes
        view = self._view("Нам не положили напиток.")
        assert view.claim_category == "Неполный заказ (недокомплект)"
        _no_forbidden(view.customer_draft)

    # Scenario 7: контейнер был открыт (copula form)
    def test_s7_konteyner_byl_otkryt(self) -> None:
        risk = _risk_from_query("Контейнер был открыт.")
        assert risk.risk_floor == RiskLevel.HIGH
        assert RiskReasonCode.PACKAGE_TAMPERING in risk.reason_codes
        assert risk.handoff_required is True
        view = self._view("Контейнер был открыт.")
        assert view.claim_category == "Нарушение целостности упаковки"
        expected = CATEGORY_DRAFT_TEMPLATES["Нарушение целостности упаковки"]
        assert view.customer_draft == expected
        _no_forbidden(view.customer_draft)

    # Scenario 8: упаковка была вскрыта + пломба сорвана
    def test_s8_upakovka_vskryta_plomba(self) -> None:
        risk = _risk_from_query("Упаковка была вскрыта, пломба сорвана.")
        assert risk.risk_floor == RiskLevel.HIGH
        assert RiskReasonCode.PACKAGE_TAMPERING in risk.reason_codes
        view = self._view("Упаковка была вскрыта, пломба сорвана.")
        assert view.claim_category == "Нарушение целостности упаковки"
        _no_forbidden(view.customer_draft)

    # Scenario 9: привезли другое блюдо (new WRONG_ITEM category)
    def test_s9_privezli_drugoe_blyudo(self) -> None:
        risk = _risk_from_query("Привезли другое блюдо, не то что я заказывал.")
        assert risk.risk_floor == RiskLevel.MEDIUM
        assert RiskReasonCode.WRONG_ITEM in risk.reason_codes
        assert risk.handoff_required is False
        view = self._view("Привезли другое блюдо, не то что я заказывал.")
        assert view.claim_category == "Неверная позиция в заказе"
        expected = CATEGORY_DRAFT_TEMPLATES["Неверная позиция в заказе"]
        assert view.customer_draft == expected
        _no_forbidden(view.customer_draft)

    # Scenario 10: перепутали позицию
    def test_s10_pereputa_li_poziciyu(self) -> None:
        risk = _risk_from_query("Перепутали одну позицию в заказе.")
        assert risk.risk_floor == RiskLevel.MEDIUM
        assert RiskReasonCode.WRONG_ITEM in risk.reason_codes
        view = self._view("Перепутали одну позицию в заказе.")
        assert view.claim_category == "Неверная позиция в заказе"
        _no_forbidden(view.customer_draft)


# ---------------------------------------------------------------------------
# I. Negative control tests (no false positives)
# ---------------------------------------------------------------------------

class TestNegativeControls:
    """Ensure specific safe phrases are NOT classified as incidents."""

    def test_customer_opened_container_themselves_no_tampering(self) -> None:
        """Client self-action: 'я сам открыл контейнер' must not trigger PACKAGE_TAMPERING."""
        risk = _risk_from_query("Я сам открыл контейнер после доставки")
        assert RiskReasonCode.PACKAGE_TAMPERING not in risk.reason_codes, (
            "Self-action 'открыл' incorrectly classified as package tampering"
        )

    def test_how_to_open_container_informational_no_tampering(self) -> None:
        """Informational question about opening container must not trigger PACKAGE_TAMPERING."""
        risk = _risk_from_query("Можно ли открыть контейнер и хранить еду в холодильнике?")
        assert RiskReasonCode.PACKAGE_TAMPERING not in risk.reason_codes, (
            "Informational 'можно ли открыть' incorrectly classified as package tampering"
        )

    def test_what_is_food_poisoning_informational_no_health(self) -> None:
        """Educational question 'что такое пищевое отравление' must not trigger health rule."""
        risk = _risk_from_query("Что такое пищевое отравление?")
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION not in risk.reason_codes, (
            "Informational 'что такое пищевое отравление' incorrectly triggered health rule"
        )

    def test_old_rash_unrelated_to_current_order_no_health(self) -> None:
        """Past health event explicitly scoped away from current order must not trigger health."""
        risk = _risk_from_query(
            "У меня была сыпь месяц назад, вопрос только о задержке заказа"
        )
        assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION not in risk.reason_codes, (
            "Old unrelated rash + explicit scope redirection incorrectly triggered health rule"
        )

    def test_order_had_all_items_no_missing_item(self) -> None:
        """Positive confirmation 'всего хватало' must not trigger MISSING_ITEM."""
        risk = _risk_from_query("В заказе всего хватало")
        assert RiskReasonCode.MISSING_ITEM not in risk.reason_codes, (
            "Positive 'всего хватало' incorrectly triggered missing item rule"
        )

    def test_correct_item_delivered_no_wrong_item(self) -> None:
        """Explicit confirmation of correct delivery must not trigger WRONG_ITEM."""
        risk = _risk_from_query("Привезли именно то блюдо, которое я заказывал")
        assert RiskReasonCode.WRONG_ITEM not in risk.reason_codes, (
            "Positive 'привезли именно то' incorrectly triggered wrong item rule"
        )


# ---------------------------------------------------------------------------
# J. Consistency repair: M01 non-delivery, delay disambiguation, draft safety
# ---------------------------------------------------------------------------

class TestConsistencyRepair:
    """Regression for UI/risk consistency repair round."""

    def _view(self, query: str) -> object:
        risk = _risk_from_query(query)
        return map_result_to_display(_result(_ic(), risk, "insufficient_context", query=query))

    def test_m01_passive_non_delivery_is_high(self) -> None:
        risk = _risk_from_query(
            "Я оплатил заказ №12345, но он не был доставлен. Деньги не вернули."
        )
        assert RiskReasonCode.NON_DELIVERY in risk.reason_codes
        assert risk.risk_floor == RiskLevel.HIGH
        assert risk.handoff_required is True

    def test_m01_category_is_non_delivery(self) -> None:
        view = self._view(
            "Я оплатил заказ №12345, но он не был доставлен. Деньги не вернули."
        )
        assert view.claim_category == "Недоставка оплаченного заказа"
        assert view.requires_escalation is True
        _no_forbidden(view.customer_draft)

    def test_non_delivery_vovremya_is_not_non_delivery(self) -> None:
        """не был доставлен вовремя = delay, not final non-delivery."""
        risk = _risk_from_query("Заказ не был доставлен вовремя.")
        assert RiskReasonCode.NON_DELIVERY not in risk.reason_codes

    def test_courier_not_in_time_is_delay_category(self) -> None:
        view = self._view(
            "Курьер не приехал с заказом в срок, не могу до него дозвониться."
        )
        assert view.claim_category == "Задержка доставки"

    def test_courier_not_in_time_no_non_delivery_code(self) -> None:
        risk = _risk_from_query(
            "Курьер не приехал с заказом в срок, не могу до него дозвониться."
        )
        assert RiskReasonCode.NON_DELIVERY not in risk.reason_codes
        assert RiskReasonCode.DELAY_OVER_120_MINUTES not in risk.reason_codes

    def test_delay_draft_no_forbidden(self) -> None:
        view = self._view(
            "Курьер не приехал с заказом в срок, не могу до него дозвониться."
        )
        _no_forbidden(view.customer_draft)

    def test_prioritetnoy_proverki_sotrudnikom_blocked(self) -> None:
        bad = (
            "Если с конца интервала прошло более 120 минут, "
            "это также требует приоритетной проверки сотрудником."
        )
        assert detect_draft_violations(bad), (
            "Internal routing phrase 'приоритетной проверки сотрудником' not detected as violation"
        )

    def test_starshiy_specialist_blocked(self) -> None:
        bad = "Ваше обращение будет передано старшему специалисту для проверки."
        assert detect_draft_violations(bad), "'старший специалист' not detected"

    def test_category_templates_no_sotrudnik(self) -> None:
        for cat, tpl in CATEGORY_DRAFT_TEMPLATES.items():
            lower = tpl.lower()
            assert "сотрудник" not in lower, f"Template {cat!r} contains сотрудник"
            assert "старш" not in lower, f"Template {cat!r} contains старш"

    def test_health_ic_uses_safe_template(self) -> None:
        risk = _make_risk(
            RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION]
        )
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context", query="Отравился после заказа")
        )
        expected = CATEGORY_DRAFT_TEMPLATES[
            "Жалоба на здоровье после употребления продукта"
        ]
        assert view.customer_draft == expected
        _no_forbidden(view.customer_draft)

    def test_refund_ic_uses_safe_template(self) -> None:
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        view = map_result_to_display(
            _result(_ic(), risk, "insufficient_context", query="Верните деньги")
        )
        expected = CATEGORY_DRAFT_TEMPLATES["Требование возврата средств"]
        assert view.customer_draft == expected
        _no_forbidden(view.customer_draft)
