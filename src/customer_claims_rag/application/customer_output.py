"""Authoritative customer-visible output of one handled claim.

``build_customer_output`` is the single place where the raw generation result becomes what a
customer-facing draft may say, *why* it says it, and which sources may be shown beside it. The
Streamlit UI and the CLI are thin adapters over ``CustomerOutput``: neither re-applies nor bypasses
the policy, and neither sees the raw model text.

Provenance (``CustomerOutput.provenance``) states why the displayed text exists:

``llm_draft``             the grounded model draft passed the text policy; only here are the
                          validated citations presented as answer sources.
``category_template``     deterministic category text (critical categories always use it, and so does
                          a case whose HIGH/CRITICAL floor overrides a model out-of-scope verdict).
``safety_replacement``    the model draft broke the text policy and was replaced.
``insufficient_context``  deterministic text for an insufficient-context outcome.
``failure_fallback``      deterministic text after a retrieval / context-build / generation failure
                          (``failure_source`` says which).
``unsupported_language``  neutral manual-review text; the input language cannot be assessed.
``out_of_scope``          the canonical out-of-scope reply.

Anything but ``llm_draft`` never presents model citations as support for the text. Retrieved
materials may still be listed for staff, labelled as not supporting the reply.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from customer_claims_rag.application.claim_guidance import (
    derive_claim_category,
    is_out_of_scope_presentation,
    routing_recommendation,
    staff_actions,
)
from customer_claims_rag.application.customer_templates import (
    CATEGORY_DRAFT_TEMPLATES,
    HARD_TEMPLATE_CATEGORIES,
    UNSUPPORTED_LANGUAGE_CUSTOMER_TEXT,
    safe_customer_draft,
    sanitize_customer_draft,
)
from customer_claims_rag.application.customer_text_policy import (
    detect_draft_violations,
    strip_citation_markers,
)
from customer_claims_rag.application.models import CustomerClaimsResult
from customer_claims_rag.generation.models import GroundedGenerationResult, ResponseMode
from customer_claims_rag.generation.risk_integration_models import RiskAwareGenerationOutcome
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskLevel

AnswerProvenance = Literal[
    "llm_draft",
    "category_template",
    "safety_replacement",
    "insufficient_context",
    "failure_fallback",
    "unsupported_language",
    "out_of_scope",
]
FailureSource = Literal["retrieval", "context_build", "generation"]
RiskTone = Literal["neutral", "warning", "critical", "undetermined"]

EMPTY_DRAFT_VIOLATION = "empty-draft"

RISK_LEVEL_LABELS: dict[RiskLevel, str] = {
    RiskLevel.LOW: "Низкий",
    RiskLevel.MEDIUM: "Средний",
    RiskLevel.HIGH: "Высокий",
    RiskLevel.CRITICAL: "Критический",
}
NO_SIGNAL_RISK_LABEL = "Риск не определён правилами"
NO_SIGNAL_RISK_NOTE = (
    "Автоматические правила не выявили признаков риска. Это не означает, что обращение безопасно: "
    "при необходимости оцените его вручную."
)
UNSUPPORTED_LANGUAGE_RISK_LABEL = "Оценка недоступна для языка обращения"
UNSUPPORTED_LANGUAGE_RISK_NOTE = (
    "Автоматическая оценка риска для этого языка не выполнялась. Требуется ручная проверка."
)

OUTCOME_NOTICES: dict[RiskAwareGenerationOutcome, str | None] = {
    "grounded_answer": None,
    "insufficient_context": (
        "В базе знаний недостаточно подтверждённой информации для полного ответа."
    ),
    "out_of_scope": (
        "Вопрос находится за пределами области поддержки FoodFlow."
    ),
    "generation_error_fallback": (
        "Ответ подготовлен в режиме отказа от генерации."
    ),
}
_FAILURE_NOTICES: dict[FailureSource, str] = {
    "retrieval": (
        "Поиск по базе знаний недоступен: ответ подготовлен по проверенному шаблону, "
        "оценка риска выполнена."
    ),
    "context_build": (
        "Материалы базы знаний не удалось подготовить для генерации: ответ подготовлен "
        "по проверенному шаблону, оценка риска выполнена."
    ),
    "generation": OUTCOME_NOTICES["generation_error_fallback"] or "",
}
_PROVENANCE_NOTICES: dict[str, str] = {
    "category_template": "Для данной категории используется проверенный шаблон ответа.",
    "safety_replacement": (
        "Черновик модели не прошёл проверку текста для клиента; используется проверенный шаблон ответа."
    ),
    "unsupported_language": (
        "Автоматическая оценка риска для языка обращения недоступна; используется нейтральный ответ."
    ),
}
_OUT_OF_SCOPE_OVERRIDDEN_NOTICE = (
    "Модель сочла вопрос внепредметным, но детерминированная оценка риска требует обработки: "
    "используется проверенный шаблон ответа."
)

_NON_SUPPORTING_PROVENANCE = frozenset({"llm_draft", "out_of_scope"})


class SourceReference(BaseModel):
    """Display-safe reference to a knowledge-base fragment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    heading: str
    document_id: str


class CustomerOutput(BaseModel):
    """Customer-safe projection of a pipeline result; the only input of UI and CLI rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    customer_text: str
    provenance: AnswerProvenance
    policy_violations: tuple[str, ...] = ()
    # Raw generation facts, kept for staff diagnostics. ``provenance`` is the authority on the text.
    response_mode: ResponseMode
    generation_outcome: RiskAwareGenerationOutcome
    failure_source: FailureSource | None = None
    outcome_notice: str | None = None
    # Deterministic assessment: status and raw floor, plus how to present them.
    assessment_status: RiskAssessmentStatus
    risk_floor: RiskLevel
    risk_label: str
    risk_note: str | None = None
    risk_tone: RiskTone
    handoff_required: bool
    priority_handoff: bool
    handoff_notice: str | None
    claim_category: str
    routing_recommendation: str
    staff_actions: tuple[str, ...]
    # Sources that support ``customer_text`` (only for ``llm_draft``) ...
    answer_sources: tuple[SourceReference, ...] = ()
    # ... and retrieved fragments that do NOT: shown to staff, never as grounds for the reply.
    retrieved_materials: tuple[SourceReference, ...] = ()

    @property
    def is_llm_authored(self) -> bool:
        return self.provenance == "llm_draft"

    @model_validator(mode="after")
    def validate_projection_invariants(self) -> Self:
        if not self.customer_text.strip():
            raise ValueError("customer_text must not be empty")
        if self.is_llm_authored:
            if not self.answer_sources:
                raise ValueError("llm_draft requires validated answer sources")
            if self.policy_violations:
                raise ValueError("llm_draft must not carry policy violations")
        elif self.answer_sources:
            raise ValueError("answer sources are only allowed for llm_draft")
        if self.retrieved_materials and self.provenance in _NON_SUPPORTING_PROVENANCE:
            raise ValueError(f"{self.provenance} must not list retrieved materials")
        if (self.provenance == "safety_replacement") != bool(self.policy_violations):
            raise ValueError("policy_violations are set exactly for safety_replacement")
        unsupported = self.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE
        if (self.provenance == "unsupported_language") != unsupported:
            raise ValueError("unsupported_language provenance requires unsupported_language status")
        if self.provenance == "failure_fallback" and self.failure_source is None:
            raise ValueError("failure_fallback requires failure_source")
        if self.failure_source is not None and self.provenance not in {
            "failure_fallback",
            "unsupported_language",
        }:
            raise ValueError("failure_source requires a deterministic failure or manual-review text")
        determined = self.assessment_status is RiskAssessmentStatus.RULE_MATCH
        if (self.risk_tone == "undetermined") == determined:
            raise ValueError("risk_tone=undetermined exactly when no rule matched")
        return self


def describe_risk(risk: DeterministicRiskResult) -> tuple[str, str | None, RiskTone]:
    """Return ``(label, note, tone)`` for a deterministic assessment.

    Only a rule match is shown as a risk level. ``NO_SIGNAL`` and ``UNSUPPORTED_LANGUAGE`` carry a
    neutral ``LOW`` floor that is merely the lower bound, so they never get an affirmative level.
    """
    if risk.assessment_status is RiskAssessmentStatus.NO_SIGNAL:
        return NO_SIGNAL_RISK_LABEL, NO_SIGNAL_RISK_NOTE, "undetermined"
    if risk.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE:
        return UNSUPPORTED_LANGUAGE_RISK_LABEL, UNSUPPORTED_LANGUAGE_RISK_NOTE, "undetermined"
    label = RISK_LEVEL_LABELS[risk.risk_floor]
    if risk.risk_floor is RiskLevel.CRITICAL or risk.priority_handoff:
        return label, None, "critical"
    if risk.risk_floor is RiskLevel.HIGH:
        return label, None, "warning"
    return label, None, "neutral"


def _failure_source(result: CustomerClaimsResult) -> FailureSource | None:
    if result.retrieval_failed:
        return "retrieval"
    if result.context_build_failed:
        return "context_build"
    if result.response.generation_outcome == "generation_error_fallback":
        return "generation"
    return None


def _select_customer_text(
    generation: GroundedGenerationResult,
    risk: DeterministicRiskResult,
    outcome: RiskAwareGenerationOutcome,
    category: str,
) -> tuple[str, AnswerProvenance, tuple[str, ...]]:
    """Choose the customer text and its provenance. Precedence is deliberate:

    1. an unassessable language is never answered with model text;
    2. a model out-of-scope verdict stands only while the deterministic floor is below HIGH;
    3. failures and insufficient context get the deterministic category text;
    4. a grounded draft is used only for non-critical categories and only if the policy accepts it.
    """
    if risk.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE:
        return UNSUPPORTED_LANGUAGE_CUSTOMER_TEXT, "unsupported_language", ()
    if outcome == "out_of_scope":
        if is_out_of_scope_presentation(risk, outcome):
            return generation.customer_response, "out_of_scope", ()
        return safe_customer_draft(category, risk), "category_template", ()
    if outcome == "generation_error_fallback":
        return safe_customer_draft(category, risk), "failure_fallback", ()
    if outcome == "insufficient_context":
        return safe_customer_draft(category, risk), "insufficient_context", ()

    if category in HARD_TEMPLATE_CATEGORIES:
        return CATEGORY_DRAFT_TEMPLATES[category], "category_template", ()
    candidate = strip_citation_markers(generation.customer_response)
    violations = tuple(dict.fromkeys(detect_draft_violations(candidate)))
    if not candidate:
        violations = (EMPTY_DRAFT_VIOLATION,)
    text, is_safe = sanitize_customer_draft(candidate, violations, risk, category)
    if not is_safe:
        return text, "safety_replacement", violations
    return text, "llm_draft", ()


def _outcome_notice(
    outcome: RiskAwareGenerationOutcome,
    provenance: AnswerProvenance,
    failure_source: FailureSource | None,
) -> str | None:
    if provenance == "unsupported_language":
        return _PROVENANCE_NOTICES[provenance]
    if failure_source is not None:
        return _FAILURE_NOTICES[failure_source]
    if outcome == "out_of_scope" and provenance == "category_template":
        return _OUT_OF_SCOPE_OVERRIDDEN_NOTICE
    if provenance in {"category_template", "safety_replacement"}:
        return _PROVENANCE_NOTICES[provenance]
    return OUTCOME_NOTICES[outcome]


def build_customer_output(result: CustomerClaimsResult) -> CustomerOutput:
    """Apply the customer-output policy to a pipeline result."""
    response = result.response
    generation = response.generation
    risk = response.risk_assessment
    outcome = response.generation_outcome

    category = derive_claim_category(risk, outcome, result.customer_query)
    text, provenance, violations = _select_customer_text(generation, risk, outcome, category)
    failure_source = _failure_source(result)
    risk_label, risk_note, risk_tone = describe_risk(risk)

    answer_sources: tuple[SourceReference, ...] = ()
    if provenance == "llm_draft":
        answer_sources = tuple(
            SourceReference(
                key=citation.citation_key,
                heading=citation.heading,
                document_id=citation.document_id,
            )
            for citation in generation.citations
        )
    retrieved_materials: tuple[SourceReference, ...] = ()
    if provenance not in _NON_SUPPORTING_PROVENANCE:
        retrieved_materials = tuple(
            SourceReference(key=item.citation_key, heading=item.heading, document_id=item.document_id)
            for item in result.retrieved_items
        )

    return CustomerOutput(
        customer_text=text,
        provenance=provenance,
        policy_violations=violations,
        response_mode=generation.response_mode,
        generation_outcome=outcome,
        failure_source=failure_source,
        outcome_notice=_outcome_notice(outcome, provenance, failure_source),
        assessment_status=risk.assessment_status,
        risk_floor=risk.risk_floor,
        risk_label=risk_label,
        risk_note=risk_note,
        risk_tone=risk_tone,
        handoff_required=risk.handoff_required,
        priority_handoff=risk.priority_handoff,
        handoff_notice=response.handoff_notice,
        claim_category=category,
        routing_recommendation=routing_recommendation(risk, outcome),
        staff_actions=staff_actions(risk, outcome, category),
        answer_sources=answer_sources,
        retrieved_materials=retrieved_materials,
    )
