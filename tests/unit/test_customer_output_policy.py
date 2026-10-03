"""Unified customer-output boundary (Stage 2C): provenance, fallbacks, risk status, citations, UI/CLI parity."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from customer_claims_rag.application.claim_guidance import (
    CATEGORY_FOREIGN_OBJECT,
    CATEGORY_FRAUD,
    CATEGORY_HEALTH,
    CATEGORY_MASS_INCIDENT,
    CATEGORY_PERSONAL_DATA,
    CATEGORY_THREAT,
    CATEGORY_UNSUPPORTED_LANGUAGE,
    STAFF_ACTIONS_BY_CATEGORY,
    STAFF_ACTIONS_BY_LEVEL,
    derive_claim_category,
)
from customer_claims_rag.application.customer_output import (
    NO_SIGNAL_RISK_LABEL,
    UNSUPPORTED_LANGUAGE_RISK_LABEL,
    CustomerOutput,
    SourceReference,
    build_customer_output,
)
from customer_claims_rag.application.customer_templates import (
    CATEGORY_DRAFT_TEMPLATES,
    GENERIC_RISK_DRAFTS,
    HARD_TEMPLATE_CATEGORIES,
    UNSUPPORTED_LANGUAGE_CUSTOMER_TEXT,
)
from customer_claims_rag.application.customer_text_policy import detect_draft_violations
from customer_claims_rag.application.models import (
    CustomerClaimsRequest,
    CustomerClaimsResult,
    RetrievedItemMeta,
)
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.cli import answer_claim
from customer_claims_rag.exceptions import GenerationValidationError, LLMCallError, SearchError
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
    OUT_OF_SCOPE_CUSTOMER_RESPONSE,
    out_of_scope_result,
)
from customer_claims_rag.generation.handoff import (
    CRITICAL_HANDOFF_NOTICE,
    HIGH_HANDOFF_NOTICE,
    UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE,
    build_handoff_notice,
)
from customer_claims_rag.generation.models import Citation, GroundedGenerationResult
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.retrieval.models import SearchResult
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.models import (
    DeterministicRiskResult,
    RiskAssessmentRequest,
    RiskLevel,
    RiskSignal,
)
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.validator import (
    build_deterministic_risk_result,
    build_unsupported_language_result,
)
from customer_claims_rag.ui.display import ClaimSuccessView, map_result_to_display

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEALTH_Q = "После еды стало трудно дышать, состояние ухудшается."
_OBJECT_Q = "FF-40404, в супе металлическая осколка, острая. Не ел, сохранил."
_MASS_Q = (
    "После одной партии супа из FF-BATCH-12 плохо стало нескольким людям в офисе — "
    "у всех похожие симптомы."
)
_FRAUD_Q = "FF-91919, подозреваю, что с моего аккаунта оформляли заказы мошенники — странные списания."
_THREAT_Q = "Если этот курьер FF-00001 снова приедет, я ему физически покажу — предупреждаю."
_DATA_CRITICAL_Q = "Сообщаю о массовой утечке персональных данных клиентов FoodFlow."
_DATA_HIGH_Q = "В письме чужой адрес и имя другого клиента."
_ENGLISH_Q = "My order is late and I want my money back."
_VAGUE_Q = "Где посмотреть правила доставки?"
_REFUND_Q = "Хочу вернуть деньги за заказ."

_UNSAFE_LLM_TEXT = "Мы вернем вам деньги, это наша вина. У вас пищевое отравление. <script>alert(1)</script> [S1]"


def _assess(query: str) -> DeterministicRiskResult:
    return assess_deterministic_risk(RiskAssessmentRequest(customer_query=query))


def _risk(level: RiskLevel, *codes: RiskReasonCode) -> DeterministicRiskResult:
    signals = [RiskSignal(reason_code=c, level=level, rule_id=c.value) for c in codes]
    return build_deterministic_risk_result(signals)


def _citation(key: str = "S1") -> Citation:
    return Citation(
        citation_key=key,
        heading=f"Раздел {key}",
        document_id=f"doc-{key}",
        chunk_id=f"chunk-{key}",
        source_path=f"docs/{key}.md",
    )


def _grounded(text: str, *keys: str) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=text,
        citations=[_citation(k) for k in (keys or ("S1",))],
    )


def _insufficient() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
        citations=[],
    )


def _failure() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
        citations=[],
    )


def _retrieved(*keys: str) -> tuple[RetrievedItemMeta, ...]:
    return tuple(
        RetrievedItemMeta(
            citation_key=key,
            rank=index,
            heading=f"Найдено {key}",
            document_id=f"retrieved-{key}",
        )
        for index, key in enumerate(keys or ("S1", "S2"), start=1)
    )


def _result(
    generation: GroundedGenerationResult,
    risk: DeterministicRiskResult,
    outcome: str,
    *,
    query: str = "",
    retrieved: tuple[RetrievedItemMeta, ...] = (),
    retrieval_failed: bool = False,
    context_build_failed: bool = False,
) -> CustomerClaimsResult:
    response = RiskAwareGroundedGenerationResult(
        generation=generation,
        risk_assessment=risk,
        handoff_notice=build_handoff_notice(risk),
        generation_outcome=outcome,  # type: ignore[arg-type]
    )
    return CustomerClaimsResult(
        response=response,
        customer_query=query,
        retrieved_items=retrieved,
        retrieval_failed=retrieval_failed,
        context_build_failed=context_build_failed,
    )


def _grounded_result(text: str, risk: DeterministicRiskResult, query: str = "") -> CustomerClaimsResult:
    return _result(_grounded(text), risk, "grounded_answer", query=query, retrieved=_retrieved())


# ---------------------------------------------------------------------------
# Provenance: why does the displayed customer text exist?
# ---------------------------------------------------------------------------

_CLEAN_DRAFT = "Сожалеем, что в заказе не хватало позиции. Просим указать номер заказа [S1]."


def test_accepted_grounded_draft_is_llm_draft_with_validated_sources() -> None:
    risk = _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM)
    output = build_customer_output(_grounded_result(_CLEAN_DRAFT, risk))
    assert output.provenance == "llm_draft"
    assert output.customer_text == "Сожалеем, что в заказе не хватало позиции. Просим указать номер заказа."
    assert output.is_llm_authored
    assert output.answer_sources == (SourceReference(key="S1", heading="Раздел S1", document_id="doc-S1"),)
    assert output.retrieved_materials == ()
    assert output.policy_violations == ()
    assert output.outcome_notice is None


def test_critical_category_always_uses_its_deterministic_template() -> None:
    risk = _assess(_HEALTH_Q)
    output = build_customer_output(_grounded_result("Всё будет хорошо, отдохните [S1].", risk, _HEALTH_Q))
    assert output.provenance == "category_template"
    assert output.customer_text == CATEGORY_DRAFT_TEMPLATES[CATEGORY_HEALTH]
    assert output.answer_sources == ()
    assert output.policy_violations == ()
    assert output.outcome_notice is not None


@pytest.mark.parametrize(
    ("draft", "expected_code"),
    [
        ("Мы вернем вам деньги [S1].", "refund-promise"),
        ("Мы выплатим компенсацию [S1].", "compensation-promise"),
        ("Возврат уже оформлен [S1].", "completed-refund"),
        ("Это наша вина [S1].", "fault-admission"),
        ("У вас пищевое отравление [S1].", "medical-diagnosis"),
        ("Примите таблетку [S1].", "medical-treatment"),
        ("Ваше обращение зарегистрировано [S1].", "false-registration"),
        ("Мы ответим в течение 24 часов [S1].", "deadline-commitment"),
    ],
)
def test_policy_violation_replaces_the_model_draft(draft: str, expected_code: str) -> None:
    risk = _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM)
    output = build_customer_output(_grounded_result(draft, risk, "В заказе не хватало позиции"))
    assert output.provenance == "safety_replacement"
    assert expected_code in output.policy_violations
    assert output.customer_text == CATEGORY_DRAFT_TEMPLATES["Неполный заказ (недокомплект)"]
    assert output.answer_sources == ()
    assert draft.replace(" [S1]", "") not in output.customer_text


def test_empty_draft_after_marker_stripping_is_replaced() -> None:
    risk = _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM)
    output = build_customer_output(_grounded_result("[S1]", risk))
    assert output.provenance == "safety_replacement"
    assert output.policy_violations == ("empty-draft",)
    assert output.customer_text.strip()


def test_insufficient_context_uses_deterministic_template() -> None:
    risk = _risk(RiskLevel.HIGH, RiskReasonCode.NON_DELIVERY)
    output = build_customer_output(
        _result(_insufficient(), risk, "insufficient_context", query="Заказ не был доставлен", retrieved=_retrieved())
    )
    assert output.provenance == "insufficient_context"
    assert output.customer_text == CATEGORY_DRAFT_TEMPLATES["Недоставка оплаченного заказа"]
    assert INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE not in output.customer_text


def test_generation_failure_uses_deterministic_template_and_names_its_source() -> None:
    risk = _risk(RiskLevel.HIGH, RiskReasonCode.NON_DELIVERY)
    output = build_customer_output(_result(_failure(), risk, "generation_error_fallback", retrieved=_retrieved()))
    assert output.provenance == "failure_fallback"
    assert output.failure_source == "generation"
    assert GENERATION_FAILURE_CUSTOMER_RESPONSE not in output.customer_text
    assert "обратитесь в поддержку" not in output.customer_text.lower()


def test_out_of_scope_keeps_the_canonical_reply_and_no_sources() -> None:
    output = build_customer_output(
        _result(out_of_scope_result(), _assess("Какая погода завтра?"), "out_of_scope", retrieved=_retrieved())
    )
    assert output.provenance == "out_of_scope"
    assert output.customer_text == OUT_OF_SCOPE_CUSTOMER_RESPONSE
    assert output.answer_sources == ()
    assert output.retrieved_materials == ()


def test_unsupported_language_gets_manual_review_text_never_model_text() -> None:
    risk = build_unsupported_language_result()
    output = build_customer_output(
        _grounded_result("We will refund you right away [S1].", risk, _ENGLISH_Q)
    )
    assert output.provenance == "unsupported_language"
    assert output.customer_text == UNSUPPORTED_LANGUAGE_CUSTOMER_TEXT
    assert output.claim_category == CATEGORY_UNSUPPORTED_LANGUAGE
    assert output.answer_sources == ()
    assert output.handoff_notice == UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE
    assert output.handoff_required is True
    assert output.priority_handoff is False
    assert "вручную" in " ".join(output.staff_actions).lower()
    assert "ручной" in output.routing_recommendation or "вручную" in output.routing_recommendation


def test_unsupported_language_beats_a_model_out_of_scope_verdict() -> None:
    risk = build_unsupported_language_result()
    output = build_customer_output(_result(out_of_scope_result(), risk, "out_of_scope", query=_ENGLISH_Q))
    assert output.provenance == "unsupported_language"
    assert "дальнейшая обработка не требуется" not in output.routing_recommendation


@pytest.mark.parametrize(
    ("query", "expected_category"),
    [(_THREAT_Q, CATEGORY_THREAT), (_DATA_CRITICAL_Q, CATEGORY_PERSONAL_DATA)],
)
def test_model_out_of_scope_never_overrides_a_high_or_critical_floor(
    query: str,
    expected_category: str,
) -> None:
    risk = _assess(query)
    output = build_customer_output(_result(out_of_scope_result(), risk, "out_of_scope", query=query))
    assert output.provenance == "category_template"
    assert output.customer_text == CATEGORY_DRAFT_TEMPLATES[expected_category]
    assert output.claim_category == expected_category
    assert "дальнейшая обработка не требуется" not in output.routing_recommendation
    assert "не эскалируйте" not in " ".join(output.staff_actions).lower()
    assert output.priority_handoff is True


def test_model_out_of_scope_stands_for_a_low_or_medium_floor() -> None:
    for risk in (_assess(_VAGUE_Q), _assess(_REFUND_Q)):
        output = build_customer_output(_result(out_of_scope_result(), risk, "out_of_scope"))
        assert output.provenance == "out_of_scope"


# ---------------------------------------------------------------------------
# One boundary for UI and CLI
# ---------------------------------------------------------------------------

def _scenarios() -> dict[str, CustomerClaimsResult]:
    medium = _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM)
    return {
        "llm_draft": _grounded_result(_CLEAN_DRAFT, medium),
        "category_template": _grounded_result(_UNSAFE_LLM_TEXT, _assess(_HEALTH_Q), _HEALTH_Q),
        "safety_replacement": _grounded_result(_UNSAFE_LLM_TEXT, medium, "Не хватало позиции"),
        "insufficient_context": _result(
            _insufficient(), _assess(_FRAUD_Q), "insufficient_context", query=_FRAUD_Q, retrieved=_retrieved()
        ),
        "failure_fallback": _result(
            _failure(), _assess(_THREAT_Q), "generation_error_fallback", query=_THREAT_Q
        ),
        "unsupported_language": _grounded_result(
            "Order refunded [S1].", build_unsupported_language_result(), _ENGLISH_Q
        ),
        "out_of_scope": _result(out_of_scope_result(), _assess(_VAGUE_Q), "out_of_scope"),
    }


@pytest.mark.parametrize("name", list(_scenarios()))
def test_ui_and_cli_show_the_same_final_text_and_semantics(name: str) -> None:
    result = _scenarios()[name]
    view = map_result_to_display(result)
    payload = answer_claim.format_answer_payload(result)
    output = build_customer_output(result)

    assert view.customer_draft == payload["answer"] == output.customer_text
    assert view.provenance == payload["answer_provenance"] == output.provenance == name
    assert view.assessment_status.value == payload["assessment_status"]
    assert view.risk_floor == payload["risk_floor"]
    assert view.risk_label == payload["risk_label"]
    assert view.risk_note == payload["risk_note"]
    assert view.requires_escalation == payload["handoff_required"]
    assert view.priority_handoff == payload["priority_handoff"]
    assert view.handoff_notice == payload["handoff_notice"]
    assert view.failure_source == payload["failure_source"]
    assert [c.key for c in view.citations] == [c["key"] for c in payload["citations"]]
    assert [m.key for m in view.retrieved_materials] == [m["key"] for m in payload["retrieved_materials"]]
    json.dumps(payload, ensure_ascii=False)


def test_raw_unsafe_model_text_cannot_bypass_the_policy_through_the_cli() -> None:
    result = _grounded_result(
        _UNSAFE_LLM_TEXT,
        _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM),
        "Не хватало позиции",
    )
    # The raw model text is exactly what the CLI used to print as "answer".
    assert result.response.generation.customer_response == _UNSAFE_LLM_TEXT

    payload = answer_claim.format_answer_payload(result)
    for fragment in ("вернем", "наша вина", "отравление", "<script", "[S1]"):
        assert fragment not in payload["answer"]
    assert payload["answer"] == CATEGORY_DRAFT_TEMPLATES["Неполный заказ (недокомплект)"]
    assert payload["answer_provenance"] == "safety_replacement"
    assert "risk_level" not in payload


def test_cli_answer_has_citation_markers_stripped() -> None:
    result = _grounded_result(_CLEAN_DRAFT, _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM))
    assert "[S1]" in result.response.generation.customer_response
    assert "[S1]" not in answer_claim.format_answer_payload(result)["answer"]


def test_projection_rejects_inconsistent_provenance() -> None:
    base = build_customer_output(_grounded_result(_CLEAN_DRAFT, _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM)))
    data = base.model_dump()
    source = SourceReference(key="S1", heading="h", document_id="d")

    with pytest.raises(ValidationError, match="answer sources are only allowed for llm_draft"):
        CustomerOutput(**{**data, "provenance": "category_template", "answer_sources": (source,)})
    with pytest.raises(ValidationError, match="llm_draft requires validated answer sources"):
        CustomerOutput(**{**data, "answer_sources": ()})
    with pytest.raises(ValidationError, match="must not list retrieved materials"):
        CustomerOutput(**{**data, "retrieved_materials": (source,)})
    with pytest.raises(ValidationError, match="policy_violations"):
        CustomerOutput(**{**data, "provenance": "category_template", "answer_sources": (), "policy_violations": ("x",)})
    with pytest.raises(ValidationError, match="unsupported_language provenance"):
        CustomerOutput(**{**data, "provenance": "unsupported_language", "answer_sources": ()})
    with pytest.raises(ValidationError, match="customer_text"):
        CustomerOutput(**{**data, "customer_text": "  "})
    with pytest.raises(ValidationError, match="risk_tone"):
        CustomerOutput(**{**data, "risk_tone": "undetermined"})


# ---------------------------------------------------------------------------
# H3: category-specific deterministic fallbacks for CRITICAL categories
# ---------------------------------------------------------------------------

_CRITICAL_CASES = {
    "health": (_HEALTH_Q, CATEGORY_HEALTH),
    "foreign_object": (_OBJECT_Q, CATEGORY_FOREIGN_OBJECT),
    "mass_incident": (_MASS_Q, CATEGORY_MASS_INCIDENT),
    "fraud": (_FRAUD_Q, CATEGORY_FRAUD),
    "threat": (_THREAT_Q, CATEGORY_THREAT),
    "personal_data_critical": (_DATA_CRITICAL_Q, CATEGORY_PERSONAL_DATA),
    "personal_data_high": (_DATA_HIGH_Q, CATEGORY_PERSONAL_DATA),
}
_OUTCOMES = {
    "insufficient_context": (_insufficient, "insufficient_context"),
    "generation_error_fallback": (_failure, "generation_error_fallback"),
    "grounded_unsafe_draft": (lambda: _grounded(_UNSAFE_LLM_TEXT), "grounded_answer"),
}


def _critical_output(case: str, outcome: str) -> CustomerOutput:
    query, _ = _CRITICAL_CASES[case]
    make_generation, outcome_name = _OUTCOMES[outcome]
    return build_customer_output(
        _result(make_generation(), _assess(query), outcome_name, query=query, retrieved=_retrieved())
    )


@pytest.mark.parametrize("outcome", list(_OUTCOMES))
@pytest.mark.parametrize("case", list(_CRITICAL_CASES))
def test_each_critical_category_gets_its_own_fallback(case: str, outcome: str) -> None:
    _, expected_category = _CRITICAL_CASES[case]
    output = _critical_output(case, outcome)
    assert output.claim_category == expected_category
    assert output.customer_text == CATEGORY_DRAFT_TEMPLATES[expected_category]
    assert output.answer_sources == ()


@pytest.mark.parametrize("case", list(_CRITICAL_CASES))
def test_no_critical_category_receives_another_categorys_template(case: str) -> None:
    _, own_category = _CRITICAL_CASES[case]
    text = _critical_output(case, "insufficient_context").customer_text
    for category, template in CATEGORY_DRAFT_TEMPLATES.items():
        if category != own_category:
            assert text != template, f"{case} received the template of {category!r}"
    for generic in GENERIC_RISK_DRAFTS.values():
        assert text != generic
    if own_category != CATEGORY_HEALTH:
        assert "вам стало плохо" not in text
        assert "симптом" not in text.lower()


def test_every_critical_reason_code_maps_to_a_hard_template_category() -> None:
    critical_codes = [
        RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION,
        RiskReasonCode.DANGEROUS_FOREIGN_OBJECT,
        RiskReasonCode.MASS_INCIDENT,
        RiskReasonCode.FRAUD_INDICATORS,
        RiskReasonCode.DIRECT_THREAT,
        RiskReasonCode.PERSONAL_DATA_EXPOSURE,
    ]
    for code in critical_codes:
        category = derive_claim_category(_risk(RiskLevel.CRITICAL, code), "insufficient_context")
        assert category in HARD_TEMPLATE_CATEGORIES
        assert category in CATEGORY_DRAFT_TEMPLATES
        assert category in STAFF_ACTIONS_BY_CATEGORY


def test_generic_critical_draft_is_neutral_and_not_the_health_text() -> None:
    generic = GENERIC_RISK_DRAFTS[RiskLevel.CRITICAL]
    assert generic != CATEGORY_DRAFT_TEMPLATES[CATEGORY_HEALTH]
    for word in ("медицин", "симптом", "продукт", "плохо"):
        assert word not in generic.lower()
    assert detect_draft_violations(generic) == []
    staff = " ".join(STAFF_ACTIONS_BY_LEVEL[RiskLevel.CRITICAL]).lower()
    assert "медицин" not in staff and "симптом" not in staff


def test_critical_priority_puts_immediate_safety_before_fraud() -> None:
    both = _risk(RiskLevel.CRITICAL, RiskReasonCode.FRAUD_INDICATORS, RiskReasonCode.DIRECT_THREAT)
    assert derive_claim_category(both, "insufficient_context") == CATEGORY_THREAT
    health_fraud = _risk(RiskLevel.CRITICAL, RiskReasonCode.FRAUD_INDICATORS, RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION)
    assert derive_claim_category(health_fraud, "insufficient_context") == CATEGORY_HEALTH


_CRITICAL_TEXT_FORBIDDEN = (
    "возврат оформлен", "возврат будет", "выплат", "компенсац", "зарегистрирован", "передан",
    "уже уведомлен", "в течение", "гарантир", "наша вина", "диагноз", "отравлен",
)


@pytest.mark.parametrize("category", sorted(HARD_TEMPLATE_CATEGORIES))
def test_critical_templates_claim_no_operational_step_outcome_or_deadline(category: str) -> None:
    text = CATEGORY_DRAFT_TEMPLATES[category]
    assert detect_draft_violations(text) == []
    lower = text.lower()
    for phrase in _CRITICAL_TEXT_FORBIDDEN:
        assert phrase not in lower, f"{category!r} contains {phrase!r}"
    assert not any(ch.isdigit() for ch in text), "no numeric deadline or order number"
    assert len(text) < 600, "customer wording must stay concise"


def test_fraud_template_never_asks_for_payment_secrets() -> None:
    text = CATEGORY_DRAFT_TEMPLATES[CATEGORY_FRAUD]
    assert "не нужны" in text and "CVV" in text
    assert detect_draft_violations(text) == []


def test_personal_data_template_neither_confirms_nor_denies_the_leak() -> None:
    text = CATEGORY_DRAFT_TEMPLATES[CATEGORY_PERSONAL_DATA]
    assert "не можем подтвердить или опровергнуть" in text


def test_threat_template_does_not_discuss_refund_or_promise_punishment() -> None:
    lower = CATEGORY_DRAFT_TEMPLATES[CATEGORY_THREAT].lower()
    for word in ("возврат", "компенсац", "наказ", "полици", "охран"):
        assert word not in lower


_STAFF_ACTION_EXPECTATIONS = {
    CATEGORY_FRAUD: ("cvv", "мошенничество доказано"),
    CATEGORY_THREAT: ("не раскрывайте", "приоритетной проверки"),
    CATEGORY_PERSONAL_DATA: ("утечк", "уполномоченному"),
    CATEGORY_MASS_INCIDENT: ("номера заказов", "масштаб"),
    CATEGORY_HEALTH: ("медицинской помощью",),
}


@pytest.mark.parametrize("category", list(_STAFF_ACTION_EXPECTATIONS))
def test_staff_actions_are_category_specific(category: str) -> None:
    joined = " ".join(STAFF_ACTIONS_BY_CATEGORY[category]).lower()
    for fragment in _STAFF_ACTION_EXPECTATIONS[category]:
        assert fragment in joined
    if category not in {CATEGORY_HEALTH, CATEGORY_MASS_INCIDENT}:
        assert "медицин" not in joined


def test_staff_view_of_a_fraud_case_carries_fraud_guidance_not_health_guidance() -> None:
    output = _critical_output("fraud", "insufficient_context")
    joined = " ".join(output.staff_actions).lower()
    assert "cvv" in joined
    assert "медицин" not in joined
    assert output.priority_handoff is True


# ---------------------------------------------------------------------------
# Assessment semantics: rule_match / no_signal / unsupported_language
# ---------------------------------------------------------------------------

_SEMANTIC_CASES = {
    "rule_match_low": (_risk(RiskLevel.LOW, RiskReasonCode.REFUND_REQUEST), "Низкий", "neutral", False),
    "rule_match_medium": (_assess(_REFUND_Q), "Средний", "neutral", False),
    "rule_match_high": (_assess(_DATA_HIGH_Q), "Высокий", "warning", False),
    "rule_match_critical": (_assess(_FRAUD_Q), "Критический", "critical", False),
    "no_signal": (_assess(_VAGUE_Q), NO_SIGNAL_RISK_LABEL, "undetermined", True),
    "unsupported_language": (
        build_unsupported_language_result(), UNSUPPORTED_LANGUAGE_RISK_LABEL, "undetermined", True,
    ),
}


@pytest.mark.parametrize("name", list(_SEMANTIC_CASES))
def test_ui_and_cli_present_the_same_assessment_semantics(name: str) -> None:
    risk, label, tone, undetermined = _SEMANTIC_CASES[name]
    result = _grounded_result(_CLEAN_DRAFT, risk)
    view = map_result_to_display(result)
    payload = answer_claim.format_answer_payload(result)

    assert view.risk_label == payload["risk_label"] == label
    assert view.risk_tone == tone
    assert view.assessment_status is risk.assessment_status
    assert payload["assessment_status"] == risk.assessment_status.value
    assert view.risk_floor == payload["risk_floor"] == risk.risk_floor.value
    assert payload["handoff_required"] is risk.handoff_required
    assert payload["priority_handoff"] is risk.priority_handoff
    if undetermined:
        # The neutral LOW floor must never read as an affirmative conclusion.
        assert view.risk_floor == "low"
        assert view.risk_note
        assert payload["risk_note"] == view.risk_note
        assert "Низкий" not in view.risk_label
    else:
        assert view.risk_note is None


def test_no_signal_is_not_a_proof_of_safety() -> None:
    view = map_result_to_display(_grounded_result(_CLEAN_DRAFT, _assess(_VAGUE_Q)))
    assert view.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert view.requires_escalation is False
    assert "не означает" in (view.risk_note or "")
    assert view.risk_tone == "undetermined"


def test_unsupported_language_requires_manual_review_without_priority_escalation() -> None:
    result = _grounded_result(_CLEAN_DRAFT, _assess(_ENGLISH_Q), _ENGLISH_Q)
    view = map_result_to_display(result)
    payload = answer_claim.format_answer_payload(result)
    assert view.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE
    assert view.risk_label == UNSUPPORTED_LANGUAGE_RISK_LABEL
    assert "ручная проверка" in (view.risk_note or "").lower()
    assert view.requires_escalation is True
    assert view.priority_handoff is False
    assert view.handoff_notice == UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE
    assert view.risk_tone != "critical"
    assert payload["assessment_status"] == "unsupported_language"
    assert payload["risk_floor"] == "low" and payload["risk_label"] != "Низкий"


def test_high_and_critical_keep_their_handoff_notices() -> None:
    high = answer_claim.format_answer_payload(_grounded_result(_CLEAN_DRAFT, _assess(_DATA_HIGH_Q)))
    critical = answer_claim.format_answer_payload(_grounded_result(_CLEAN_DRAFT, _assess(_FRAUD_Q)))
    assert high["handoff_notice"] == HIGH_HANDOFF_NOTICE and high["priority_handoff"] is False
    assert critical["handoff_notice"] == CRITICAL_HANDOFF_NOTICE and critical["priority_handoff"] is True


# ---------------------------------------------------------------------------
# Citations: sources shown as support must support the final text
# ---------------------------------------------------------------------------

def test_accepted_draft_exposes_its_validated_citations_only() -> None:
    risk = _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM)
    generation = _grounded("Сожалеем. Просим указать номер заказа [S1][S2].", "S1", "S2")
    result = _result(generation, risk, "grounded_answer", retrieved=_retrieved("S1", "S2", "S3"))
    view = map_result_to_display(result)
    payload = answer_claim.format_answer_payload(result)

    assert [c.key for c in view.citations] == ["S1", "S2"]
    assert all(c.is_confirmed for c in view.citations)
    assert [c["key"] for c in payload["citations"]] == ["S1", "S2"]
    assert view.retrieved_materials == ()
    assert payload["retrieved_materials"] == []


_REPLACEMENT_RESULTS = {
    "safety_replacement": lambda: _grounded_result(
        "Мы вернем вам деньги [S1].", _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM), "Не хватало позиции"
    ),
    "category_template": lambda: _grounded_result(_CLEAN_DRAFT, _assess(_FRAUD_Q), _FRAUD_Q),
    "unsupported_language": lambda: _grounded_result(
        _CLEAN_DRAFT, build_unsupported_language_result(), _ENGLISH_Q
    ),
    "insufficient_context": lambda: _result(
        _insufficient(), _assess(_REFUND_Q), "insufficient_context", retrieved=_retrieved("S1", "S2")
    ),
    "failure_fallback": lambda: _result(
        _failure(), _assess(_REFUND_Q), "generation_error_fallback", retrieved=_retrieved("S1", "S2")
    ),
}


@pytest.mark.parametrize("name", list(_REPLACEMENT_RESULTS))
def test_deterministic_text_never_presents_model_citations_as_its_support(name: str) -> None:
    result = _REPLACEMENT_RESULTS[name]()
    view = map_result_to_display(result)
    payload = answer_claim.format_answer_payload(result)

    assert view.provenance == name
    assert view.citations == ()
    assert payload["citations"] == []
    # The model's own citations exist on the raw result but are not shown as grounds.
    if name in {"safety_replacement", "category_template", "unsupported_language"}:
        assert result.response.generation.citations
    # Retrieved fragments stay available to staff, separately labelled as not supporting.
    assert [m.key for m in view.retrieved_materials] == ["S1", "S2"]
    assert all(m.is_confirmed is False for m in view.retrieved_materials)
    assert [m["key"] for m in payload["retrieved_materials"]] == ["S1", "S2"]


def test_out_of_scope_lists_neither_citations_nor_materials() -> None:
    result = _result(out_of_scope_result(), _assess(_VAGUE_Q), "out_of_scope", retrieved=_retrieved())
    view = map_result_to_display(result)
    assert view.citations == () and view.retrieved_materials == ()


# ---------------------------------------------------------------------------
# Degraded paths through the real pipeline: safe text, preserved assessment, no leaks
# ---------------------------------------------------------------------------

class _Retrieval:
    def __init__(self, results: list[SearchResult] | None = None, error: Exception | None = None) -> None:
        self._results = results or []
        self._error = error

    def search(self, query: str) -> list[SearchResult]:
        if self._error is not None:
            raise self._error
        return list(self._results)


class _Generator:
    """A generator that must not be reached when retrieval / context building fails."""

    def generate(self, request, risk_assessment):  # noqa: ANN001
        raise AssertionError("generation must not run")


def _search_result(rank: int = 1) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=f"doc::{rank}",
        document_id="06_food_quality_and_packaging",
        content="Policy text.",
        source_path="data/02_clean_markdown/06_food_quality_and_packaging.md",
        chunk_type="policy",
        topic="packaging",
        risk_level="high",
        heading="Упаковка",
        heading_path=["Упаковка"],
        section="S",
        subsection="s",
        similarity=0.9,
        distance=0.1,
    )


_PROVIDER_SECRET = "sk-LEAK-provider-internal-detail"


@pytest.mark.parametrize(
    ("query", "category"),
    [(_HEALTH_Q, CATEGORY_HEALTH), (_FRAUD_Q, CATEGORY_FRAUD), (_THREAT_Q, CATEGORY_THREAT)],
)
def test_retrieval_failure_preserves_risk_and_gets_the_safe_category_text(query: str, category: str) -> None:
    pipeline = CustomerClaimsPipeline(
        retrieval=_Retrieval(error=SearchError(f"vector store down {_PROVIDER_SECRET}")),
        generator=_Generator(),
    )
    result = pipeline.handle(CustomerClaimsRequest(customer_query=query))
    output = build_customer_output(result)
    view = map_result_to_display(result)

    assert result.retrieval_failed is True
    assert output.provenance == "failure_fallback"
    assert output.failure_source == "retrieval"
    assert output.assessment_status is RiskAssessmentStatus.RULE_MATCH
    assert output.risk_floor is RiskLevel.CRITICAL
    assert output.priority_handoff is True and output.handoff_notice == CRITICAL_HANDOFF_NOTICE
    assert output.claim_category == category
    assert output.customer_text == CATEGORY_DRAFT_TEMPLATES[category]
    assert output.answer_sources == () and output.retrieved_materials == ()
    assert _PROVIDER_SECRET not in json.dumps(answer_claim.format_answer_payload(result), ensure_ascii=False)
    assert view.failure_source == "retrieval"
    assert view.risk_label == "Критический"


@pytest.mark.parametrize(
    ("query", "category"),
    [(_HEALTH_Q, CATEGORY_HEALTH), (_DATA_CRITICAL_Q, CATEGORY_PERSONAL_DATA)],
)
def test_context_build_failure_preserves_risk_and_gets_the_safe_category_text(query: str, category: str) -> None:
    def broken_builder(_results):  # noqa: ANN001, ANN202
        raise GenerationValidationError(f"duplicate rank in context input: 1 {_PROVIDER_SECRET}")

    pipeline = CustomerClaimsPipeline(
        retrieval=_Retrieval([_search_result()]),
        generator=_Generator(),
        context_builder=broken_builder,
    )
    result = pipeline.handle(CustomerClaimsRequest(customer_query=query))
    output = build_customer_output(result)

    assert result.context_build_failed is True and result.retrieval_failed is False
    assert output.provenance == "failure_fallback"
    assert output.failure_source == "context_build"
    assert output.risk_floor is RiskLevel.CRITICAL
    assert output.assessment_status is RiskAssessmentStatus.RULE_MATCH
    assert output.claim_category == category
    assert output.customer_text == CATEGORY_DRAFT_TEMPLATES[category]
    assert _PROVIDER_SECRET not in json.dumps(answer_claim.format_answer_payload(result), ensure_ascii=False)
    assert _PROVIDER_SECRET not in repr(map_result_to_display(result))


def test_staff_can_tell_the_three_failure_sources_apart() -> None:
    risk_query = _HEALTH_Q

    def broken_builder(_results):  # noqa: ANN001, ANN202
        raise GenerationValidationError("duplicate rank")

    class _FailingGrounded:
        def generate(self, request):  # noqa: ANN001, ANN202
            raise LLMCallError(f"provider outage {_PROVIDER_SECRET}")

    request = CustomerClaimsRequest(customer_query=risk_query)
    retrieval_out = build_customer_output(
        CustomerClaimsPipeline(retrieval=_Retrieval(error=SearchError("x")), generator=_Generator()).handle(request)
    )
    context_out = build_customer_output(
        CustomerClaimsPipeline(
            retrieval=_Retrieval([_search_result()]), generator=_Generator(), context_builder=broken_builder
        ).handle(request)
    )
    generation_out = build_customer_output(
        CustomerClaimsPipeline(
            retrieval=_Retrieval([_search_result()]),
            generator=RiskAwareGroundedGenerator(_FailingGrounded()),
        ).handle(request)
    )

    assert [o.failure_source for o in (retrieval_out, context_out, generation_out)] == [
        "retrieval", "context_build", "generation",
    ]
    notices = {o.outcome_notice for o in (retrieval_out, context_out, generation_out)}
    assert len(notices) == 3 and None not in notices
    # Same deterministic assessment and the same safe customer text in every degraded state.
    assert {o.risk_floor for o in (retrieval_out, context_out, generation_out)} == {RiskLevel.CRITICAL}
    assert {o.customer_text for o in (retrieval_out, context_out, generation_out)} == {
        CATEGORY_DRAFT_TEMPLATES[CATEGORY_HEALTH]
    }
    for out in (retrieval_out, context_out, generation_out):
        assert _PROVIDER_SECRET not in out.model_dump_json()


def test_degraded_state_does_not_contradict_a_no_signal_assessment() -> None:
    request = CustomerClaimsRequest(customer_query=_VAGUE_Q)
    output = build_customer_output(
        CustomerClaimsPipeline(retrieval=_Retrieval(error=SearchError("x")), generator=_Generator()).handle(request)
    )
    assert output.failure_source == "retrieval"
    assert output.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert output.risk_label == NO_SIGNAL_RISK_LABEL
    assert output.handoff_required is False


# ---------------------------------------------------------------------------
# The boundary fails closed: an inconsistent result never leaks internals through an adapter
# ---------------------------------------------------------------------------

def _inconsistent_result() -> CustomerClaimsResult:
    # A grounded draft that also claims a retrieval failure cannot be projected consistently.
    return _result(
        _grounded(_CLEAN_DRAFT),
        _risk(RiskLevel.MEDIUM, RiskReasonCode.MISSING_ITEM),
        "grounded_answer",
        retrieval_failed=True,
    )


def test_inconsistent_result_is_rejected_by_the_projection() -> None:
    with pytest.raises(ValidationError):
        build_customer_output(_inconsistent_result())


def test_ui_adapter_turns_a_projection_failure_into_the_generic_error_view() -> None:
    from unittest.mock import MagicMock

    from customer_claims_rag.ui.display import UNEXPECTED_ERROR_MESSAGE, ClaimErrorView, process_claim

    pipeline = MagicMock()
    pipeline.handle.return_value = _inconsistent_result()
    view = process_claim(pipeline, "В заказе не хватало позиции")
    assert isinstance(view, ClaimErrorView)
    assert view.category == "unexpected"
    assert view.message == UNEXPECTED_ERROR_MESSAGE


def test_cli_adapter_turns_a_projection_failure_into_the_generic_runtime_error(capsys) -> None:
    from unittest.mock import MagicMock

    pipeline = MagicMock()
    pipeline.handle.return_value = _inconsistent_result()
    code, payload = answer_claim.run_answer(
        message="В заказе не хватало позиции",
        settings_loader=MagicMock(),
        pipeline_factory=lambda _settings: pipeline,
    )
    captured = capsys.readouterr()
    assert (code, payload) == (1, None)
    assert captured.out == ""
    assert captured.err.strip() == "Error: claim handling failed"
