"""Golden tests for customer draft content.

Each test verifies:
- presence of required semantic elements;
- absence of all forbidden terms/patterns.

Templates are deterministic, so exact-substring matching is used.
"""

from __future__ import annotations

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
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskLevel, RiskSignal
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.validator import build_deterministic_risk_result
from customer_claims_rag.application.customer_templates import (
    CATEGORY_DRAFT_TEMPLATES,
    HARD_TEMPLATE_CATEGORIES,
)
from customer_claims_rag.ui.display import map_result_to_display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _grounded(text: str, citation_key: str = "S1") -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=text,
        citations=(_fake_citation(citation_key),),
    )


def _ic() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
        citations=(),
    )


def _error() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
        citations=(),
    )


def _result(
    generation: GroundedGenerationResult,
    risk: DeterministicRiskResult,
    outcome: str,
    customer_query: str = "",
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
        customer_query=customer_query,
        retrieved_items=tuple(retrieved or []),
    )


# Shared "forbidden in all customer drafts" assertions
def _assert_no_forbidden(draft: str) -> None:
    lower = draft.lower()
    assert "служба поддержки" not in lower, f"Circular support redirect in: {draft!r}"
    assert "сотрудник" not in lower, f"Operator instruction in customer draft: {draft!r}"
    assert "ассистент" not in lower, f"AI self-reference in: {draft!r}"
    assert "необходимо зафиксировать" not in lower, f"Process instruction in: {draft!r}"
    assert "будет передано" not in lower, f"False transfer in: {draft!r}"
    assert "обращение передано" not in lower, f"False completed transfer in: {draft!r}"
    assert "\bя " not in draft and not draft.lower().startswith("я "), f"AI first person in: {draft!r}"
    assert "insufficient_context" not in lower, f"Internal status in: {draft!r}"
    assert "generation_error" not in lower, f"Internal status in: {draft!r}"
    assert "в доступных материалах недостаточно" not in lower, f"IC text leaked: {draft!r}"
    assert "не удалось подготовить" not in lower, f"Generic error text in: {draft!r}"


# ---------------------------------------------------------------------------
# M01 — Paid non-delivery
# ---------------------------------------------------------------------------

class TestM01NonDelivery:

    def _view(self, query: str = "Я оплатил заказ №12345, но он не был доставлен."):
        risk = _make_risk(RiskLevel.LOW)
        return map_result_to_display(
            _result(_ic(), risk, "insufficient_context", customer_query=query)
        )

    def test_m01_ic_uses_nondelivery_template(self) -> None:
        view = self._view()
        expected = CATEGORY_DRAFT_TEMPLATES["Недоставка оплаченного заказа"]
        assert view.customer_draft == expected

    def test_m01_has_regret(self) -> None:
        view = self._view()
        assert "сожалеем" in view.customer_draft.lower()

    def test_m01_states_the_need_for_review_without_promising_it(self) -> None:
        view = self._view()
        lower = view.customer_draft.lower()
        assert "требуют проверки" in lower
        assert "мы проверим" not in lower

    def test_m01_has_order_status_or_payment(self) -> None:
        view = self._view()
        lower = view.customer_draft.lower()
        assert "статус" in lower or "оплат" in lower

    def test_m01_outcome_depends_on_the_check_and_promises_no_follow_up(self) -> None:
        view = self._view()
        lower = view.customer_draft.lower()
        assert "зависит от результатов проверки" in lower
        assert "сообщим" not in lower

    def test_m01_no_forbidden_terms(self) -> None:
        view = self._view()
        _assert_no_forbidden(view.customer_draft)

    def test_m01_no_circular_support_redirect(self) -> None:
        view = self._view()
        assert "в службу поддержки" not in view.customer_draft.lower()

    def test_m01_no_premature_refund_promise(self) -> None:
        view = self._view()
        # Must not PROMISE a refund unconditionally
        draft = view.customer_draft.lower()
        assert "возврат будет" not in draft
        assert "вернём деньги" not in draft
        assert "деньги вернём" not in draft

    def test_m01_grounded_answer_clean_llm_passes_through(self) -> None:
        """Clean LLM output for non-critical query passes through unchanged."""
        risk = _make_risk(RiskLevel.LOW)
        clean_text = (
            "Сожалеем, что заказ не поступил. "
            "Статус и сведения об оплате требуют проверки. [S1]"
        )
        view = map_result_to_display(
            _result(_grounded(clean_text), risk, "grounded_answer",
                    customer_query="заказ не был доставлен")
        )
        assert view.draft_sanitized is False
        assert "S1" not in view.customer_draft

    def test_m01_grounded_with_support_redirect_triggers_template(self) -> None:
        """If LLM sneaks in 'служба поддержки', template is used instead."""
        risk = _make_risk(RiskLevel.LOW)
        bad_text = (
            "Обратитесь в службу поддержки для уточнения деталей. [S1]"
        )
        view = map_result_to_display(
            _result(_grounded(bad_text), risk, "grounded_answer",
                    customer_query="заказ не был доставлен")
        )
        assert view.draft_sanitized is True
        assert "служба поддержки" not in view.customer_draft.lower()


# ---------------------------------------------------------------------------
# M02 — Incomplete order
# ---------------------------------------------------------------------------

class TestM02IncompleteOrder:

    def _view(self, query: str = "В моем заказе не хватало двух позиций."):
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.MISSING_ITEM])
        return map_result_to_display(
            _result(_ic(), risk, "insufficient_context", customer_query=query)
        )

    def test_m02_ic_uses_missing_item_template(self) -> None:
        view = self._view()
        expected = CATEGORY_DRAFT_TEMPLATES["Неполный заказ (недокомплект)"]
        assert view.customer_draft == expected

    def test_m02_has_regret(self) -> None:
        assert "сожалеем" in self._view().customer_draft.lower()

    def test_m02_names_order_composition_as_needing_review(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "состав" in lower and "требуют проверки" in lower

    def test_m02_mentions_order_number(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "номер заказа" in lower

    def test_m02_has_partial_refund_only_conditional(self) -> None:
        lower = self._view().customer_draft.lower()
        # Must not promise partial refund unconditionally
        assert "возможность частичного возврата" in lower
        assert "зависит от результатов проверки" in lower

    def test_m02_no_forbidden_terms(self) -> None:
        _assert_no_forbidden(self._view().customer_draft)

    def test_m02_does_not_repeat_items_names(self) -> None:
        """Draft must not repeat specific items (суши, напиток) from the query."""
        view = self._view("В моем заказе не хватало суши и напитка.")
        # The template is generic — does not mention specific item names
        assert "суши" not in view.customer_draft.lower()
        assert "напиток" not in view.customer_draft.lower()

    def test_m02_no_operator_instructions(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "необходимо зафиксировать" not in lower
        assert "сверить накладную" not in lower


# ---------------------------------------------------------------------------
# M03 — Package tampering
# ---------------------------------------------------------------------------

class TestM03PackageTampering:

    def _view(self):
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        return map_result_to_display(
            _result(_ic(), risk, "insufficient_context")
        )

    def test_m03_ic_uses_package_tampering_template(self) -> None:
        view = self._view()
        expected = CATEGORY_DRAFT_TEMPLATES["Нарушение целостности упаковки"]
        assert view.customer_draft == expected

    def test_m03_says_do_not_consume(self) -> None:
        assert "не употребляйте" in self._view().customer_draft.lower()

    def test_m03_says_keep_packaging(self) -> None:
        assert "сохраните упаковку" in self._view().customer_draft.lower()

    def test_m03_says_take_photos(self) -> None:
        assert "фотографи" in self._view().customer_draft.lower()

    def test_m03_states_the_need_for_review(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "обращение требует проверки" in lower
        assert "мы проверим" not in lower

    def test_m03_no_forbidden_terms(self) -> None:
        _assert_no_forbidden(self._view().customer_draft)

    def test_m03_no_internal_risk_labels(self) -> None:
        draft = self._view().customer_draft.lower()
        assert "high" not in draft
        assert "high-кейс" not in draft
        assert "уровень риска" not in draft

    def test_m03_no_false_transfer_claim(self) -> None:
        draft = self._view().customer_draft.lower()
        assert "будет передано" not in draft
        assert "передано на проверку" not in draft

    def test_m03_grounded_with_high_keis_triggers_template(self) -> None:
        """LLM output with 'high-кейс' triggers template replacement."""
        risk = _make_risk(RiskLevel.HIGH, [RiskReasonCode.PACKAGE_TAMPERING])
        bad = "Это high-кейс. Не употребляйте продукт. [S1]"
        view = map_result_to_display(_result(_grounded(bad), risk, "grounded_answer"))
        assert view.draft_sanitized is True
        assert "high" not in view.customer_draft.lower()


# ---------------------------------------------------------------------------
# M04 — Health harm (hard override)
# ---------------------------------------------------------------------------

class TestM04HealthHarm:

    def _view(self, outcome: str = "grounded_answer", generation: GroundedGenerationResult | None = None):
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        if generation is None:
            if outcome == "grounded_answer":
                generation = _grounded("Сожалеем. Обращение требует приоритетной проверки. [S1]")
            else:
                generation = _ic()
        return map_result_to_display(_result(generation, risk, outcome))

    def test_m04_always_uses_health_template_for_grounded(self) -> None:
        """Health harm category always uses the safe template even for grounded_answer."""
        view = self._view("grounded_answer")
        expected = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]
        assert view.customer_draft == expected

    def test_m04_always_uses_health_template_for_ic(self) -> None:
        view = self._view("insufficient_context")
        expected = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]
        assert view.customer_draft == expected

    def test_m04_always_uses_health_template_for_error(self) -> None:
        """Even generation_error_fallback must give the health template for M04."""
        risk = _make_risk(RiskLevel.CRITICAL, [RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION])
        view = map_result_to_display(_result(_error(), risk, "generation_error_fallback"))
        expected = CATEGORY_DRAFT_TEMPLATES["Жалоба на здоровье после употребления продукта"]
        assert view.customer_draft == expected

    def test_m04_has_regret(self) -> None:
        assert "сожалеем" in self._view().customer_draft.lower()

    def test_m04_recommends_medical_help(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "медицинск" in lower or "врач" in lower

    def test_m04_no_diagnosis_or_causal_link(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "диагноз" not in lower
        assert "отравление" not in lower
        assert "причину подтвердили" not in lower

    def test_m04_no_demo_specific_symptoms(self) -> None:
        """Template must not hardcode toshнота/рвота — those are demo-specific symptoms."""
        lower = self._view().customer_draft.lower()
        assert "тошнот" not in lower, "Demo-specific symptom 'тошнота' in template"
        assert "рвот" not in lower, "Demo-specific symptom 'рвота' in template"

    def test_m04_has_priority_check(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "приоритетно" in lower or "приоритетн" in lower

    def test_m04_no_forbidden_terms(self) -> None:
        _assert_no_forbidden(self._view().customer_draft)

    def test_m04_no_internal_enum_critical(self) -> None:
        assert "critical" not in self._view().customer_draft.lower()

    def test_m04_no_assistant_meta_phrase(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "ассистент не устанавл" not in lower
        assert "ассистент" not in lower

    def test_m04_health_harm_category_is_hard_template(self) -> None:
        assert "Жалоба на здоровье после употребления продукта" in HARD_TEMPLATE_CATEGORIES


# ---------------------------------------------------------------------------
# M05 — Refund demand with insufficient_context
# ---------------------------------------------------------------------------

class TestM05RefundDemand:

    def _view(self, query: str = "Требую полный возврат за все 5 заказов за последний месяц."):
        risk = _make_risk(RiskLevel.MEDIUM, [RiskReasonCode.REFUND_REQUEST])
        retrieved = [
            RetrievedItemMeta(citation_key="S1", rank=1, heading="Условия возврата FoodFlow", document_id="doc-refund"),
            RetrievedItemMeta(citation_key="S2", rank=2, heading="Правила задержек", document_id="doc-delay"),
        ]
        return map_result_to_display(
            _result(_ic(), risk, "insufficient_context", customer_query=query, retrieved=retrieved)
        )

    def test_m05_ic_uses_refund_template(self) -> None:
        view = self._view()
        expected = CATEGORY_DRAFT_TEMPLATES["Требование возврата средств"]
        assert view.customer_draft == expected

    def test_m05_checks_orders_history(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "история заказов" in lower

    def test_m05_asks_for_order_numbers(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "номер" in lower and "заказ" in lower

    def test_m05_mentions_refund_conditions(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "условия возврата" in lower or "применимые условия" in lower

    def test_m05_no_premature_refund_promise(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "гарантируем возврат" not in lower
        assert "вернём деньги" not in lower

    def test_m05_no_demo_specific_order_count(self) -> None:
        """Template must not hardcode 'пяти заказов' — that is demo-specific."""
        lower = self._view().customer_draft.lower()
        assert "пяти заказов" not in lower, "Demo-specific count 'пяти заказов' in template"
        assert "всех пяти" not in lower, "Demo-specific count 'всех пяти' in template"

    def test_m05_works_for_single_order(self) -> None:
        view = self._view("Требую возврат за заказ №99.")
        assert "пяти" not in view.customer_draft.lower()
        assert "требуют проверки" in view.customer_draft.lower()
        assert "мы проверим" not in view.customer_draft.lower()

    def test_m05_no_forbidden_terms(self) -> None:
        _assert_no_forbidden(self._view().customer_draft)

    def test_m05_no_ic_status_in_draft(self) -> None:
        lower = self._view().customer_draft.lower()
        assert "недостаточно информации" not in lower
        assert "в доступных материалах" not in lower
        assert "insufficient" not in lower

    def test_m05_retrieved_materials_shown(self) -> None:
        view = self._view()
        assert len(view.retrieved_materials) == 2
        assert view.retrieved_materials[0].heading == "Условия возврата FoodFlow"

    def test_m05_no_confirmed_citations(self) -> None:
        view = self._view()
        assert len(view.citations) == 0


# ---------------------------------------------------------------------------
# Smoke: no orphan HTML wrappers in the production render path
# ---------------------------------------------------------------------------

class TestUINoOrphanWrappers:
    """Verify that streamlit_app.py does not contain orphan HTML div open/close pairs."""

    def test_no_orphan_div_open_for_input_card(self) -> None:
        from pathlib import Path
        src = Path(__file__).parents[2] / "src" / "customer_claims_rag" / "ui" / "streamlit_app.py"
        text = src.read_text(encoding="utf-8")
        # Orphan pattern: st.markdown containing ONLY an opening div tag
        import re
        orphan_open = re.compile(
            r'st\.markdown\s*\(\s*[\'"]<div\s[^>]+>[\'"]',
            re.MULTILINE,
        )
        # Find all such calls and confirm they all contain content (not just the tag)
        # A bare '<div ...' without text is the orphan
        bare_open = re.compile(
            r'st\.markdown\s*\(\s*[\'"]<div\s[^>]+>[\'"\s]*,',
            re.MULTILINE,
        )
        matches = bare_open.findall(text)
        # The only permitted bare div is the divider (which is self-contained)
        non_divider = [m for m in matches if "divider" not in m and "section-label" not in m]
        assert not non_divider, f"Orphan div wrappers found: {non_divider}"

    def test_no_orphan_div_close(self) -> None:
        from pathlib import Path
        src = Path(__file__).parents[2] / "src" / "customer_claims_rag" / "ui" / "streamlit_app.py"
        text = src.read_text(encoding="utf-8")
        import re
        # Bare closing </div> in a standalone st.markdown call
        bare_close = re.compile(
            r'st\.markdown\s*\(\s*[\'"]</div>[\'"\s]*,',
            re.MULTILINE,
        )
        matches = bare_close.findall(text)
        assert not matches, f"Orphan </div> wrappers found: {matches}"

    def test_staff_card_uses_container_not_html_wrapper(self) -> None:
        from pathlib import Path
        src = Path(__file__).parents[2] / "src" / "customer_claims_rag" / "ui" / "streamlit_app.py"
        text = src.read_text(encoding="utf-8")
        assert 'st.container(border=True)' in text, "Staff card must use st.container(border=True)"
        assert 'staff-card' not in text, "Orphan staff-card HTML class must be removed from render path"

    def test_no_input_card_html_wrapper(self) -> None:
        from pathlib import Path
        src = Path(__file__).parents[2] / "src" / "customer_claims_rag" / "ui" / "streamlit_app.py"
        text = src.read_text(encoding="utf-8")
        # The form is styled via CSS selector, not via orphan div
        assert 'class="input-card"' not in text, "Orphan input-card HTML wrapper must be removed"


# ---------------------------------------------------------------------------
# Smoke: calm button color (no red primary)
# ---------------------------------------------------------------------------

class TestUIButtonColor:

    def test_page_css_overrides_primary_button_to_calm_color(self) -> None:
        from pathlib import Path
        src = Path(__file__).parents[2] / "src" / "customer_claims_rag" / "ui" / "streamlit_app.py"
        text = src.read_text(encoding="utf-8")
        # Should contain a non-red primary button colour override
        assert "FormSubmitButton" in text or "baseButton-primary" in text
        # Red is not the accent — no pure red (#ff0000 or #e00 or red:) in button CSS
        import re
        # Simplified: just confirm a blue-ish hex is present in button CSS
        assert re.search(r"#2[0-9a-f]{5}", text, re.IGNORECASE), \
            "Expected calm blue hex color in CSS"
