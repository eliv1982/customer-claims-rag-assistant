"""Pure display helpers for customer claim results.

Thin adapter: every customer-visible decision (text, provenance, sources, risk presentation) is made
by ``customer_claims_rag.application.customer_output`` and only mapped to a view here. Nothing in
this module chooses or filters what a customer may be told.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from customer_claims_rag.application.customer_output import (
    OUTCOME_NOTICES,
    AnswerProvenance,
    CustomerOutput,
    FailureSource,
    RiskTone,
    SourceReference,
    build_customer_output,
)
from customer_claims_rag.application.models import (
    CustomerClaimsRequest,
    CustomerClaimsResult,
    is_query_too_long_error,
)
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.exceptions import (
    GenerationError,
    IndexManifestError,
    ReleasePostureError,
    RetrievalError,
)
from customer_claims_rag.generation.models import Citation
from customer_claims_rag.generation.risk_integration_models import RiskAwareGenerationOutcome
from customer_claims_rag.input_limits import MAX_CUSTOMER_QUERY_CHARS
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus

INPUT_ERROR_MESSAGE = "Введите текст обращения."
INPUT_TOO_LONG_MESSAGE = (
    f"Текст обращения слишком длинный. Максимум — {MAX_CUSTOMER_QUERY_CHARS} символов."
)
LOADING_MESSAGE = "Анализируем обращение и проверяем правила FoodFlow…"
STARTUP_CONFIG_ERROR_MESSAGE = (
    "Конфигурация сервиса недоступна. Проверьте настройки окружения."
)
STARTUP_INDEX_ERROR_MESSAGE = (
    "Рабочая база знаний недоступна. Убедитесь, что индекс установлен "
    "и прошёл проверку рабочего релиза."
)
STARTUP_ERROR_MESSAGE = (
    "Сервис не готов к запуску. Проверьте настройки окружения и индекс."
)
SERVICE_ERROR_MESSAGE = "Не удалось обработать обращение. Попробуйте еще раз."
UNEXPECTED_ERROR_MESSAGE = "Произошла непредвиденная ошибка. Попробуйте еще раз."

PROVENANCE_LABELS: dict[AnswerProvenance, str] = {
    "llm_draft": "Черновик модели на основе базы знаний",
    "category_template": "Проверенный шаблон для категории обращения",
    "safety_replacement": "Проверенный шаблон вместо черновика модели",
    "insufficient_context": "Проверенный шаблон: в базе знаний недостаточно данных",
    "failure_fallback": "Проверенный шаблон: ответ модели получить не удалось",
    "unsupported_language": "Нейтральный ответ: язык обращения не поддерживается правилами",
    "out_of_scope": "Стандартный ответ: вопрос вне области поддержки",
}
FAILURE_SOURCE_LABELS: dict[FailureSource, str] = {
    "retrieval": "сбой поиска по базе знаний",
    "context_build": "сбой подготовки материалов базы знаний",
    "generation": "сбой генерации ответа",
}

_MARKDOWN_PUNCTUATION_RE = re.compile(r"([!-/:-@\[-`{-~])")


def escape_markdown_text(text: str) -> str:
    """Return ``text`` so that Markdown renders it literally.

    Customer/model/retrieved text is untrusted. Every ASCII punctuation character is
    backslash-escaped (valid CommonMark), which disables links, emphasis, headings, autolinks and
    ``:color[...]``-style directives; HTML stays disabled because it is never rendered with
    ``unsafe_allow_html``. Line breaks are kept as hard breaks.
    """
    escaped = _MARKDOWN_PUNCTUATION_RE.sub(r"\\\1", text.replace("\r\n", "\n").replace("\r", "\n"))
    return escaped.replace("\n", "  \n")


@dataclass(frozen=True)
class DisplayCitation:
    key: str
    heading: str
    document_id: str
    label: str
    is_confirmed: bool


@dataclass(frozen=True)
class ClaimSuccessView:
    customer_draft: str
    provenance: AnswerProvenance
    response_mode: str
    generation_outcome: RiskAwareGenerationOutcome
    failure_source: FailureSource | None
    assessment_status: RiskAssessmentStatus
    risk_floor: str
    risk_label: str
    risk_note: str | None
    risk_tone: RiskTone
    handoff_notice: str | None
    priority_handoff: bool
    requires_escalation: bool
    outcome_notice: str | None
    routing_recommendation: str
    claim_category: str
    staff_actions: tuple[str, ...]
    # Sources that support the draft: non-empty only when the draft is the accepted model answer.
    citations: tuple[DisplayCitation, ...]
    # Retrieved fragments that do NOT support the draft (staff reference only).
    retrieved_materials: tuple[DisplayCitation, ...]
    policy_violations: tuple[str, ...] = ()

    @property
    def draft_sanitized(self) -> bool:
        """True when the shown text is a deterministic template, not the model's own draft."""
        return self.provenance not in {"llm_draft", "out_of_scope"}


@dataclass(frozen=True)
class ClaimErrorView:
    message: str
    category: Literal["input", "startup", "startup_config", "startup_index", "service", "unexpected"]


ClaimView = ClaimSuccessView | ClaimErrorView


def format_citation_label(citation: Citation) -> str:
    return f"[{citation.citation_key}] {citation.heading} — {citation.document_id}"


def generation_outcome_notice(outcome: RiskAwareGenerationOutcome) -> str | None:
    return OUTCOME_NOTICES[outcome]


def _display_source(source: SourceReference, *, is_confirmed: bool) -> DisplayCitation:
    return DisplayCitation(
        key=source.key,
        heading=source.heading,
        document_id=source.document_id,
        label=f"[{source.key}] {source.heading} — {source.document_id}",
        is_confirmed=is_confirmed,
    )


def map_output_to_display(output: CustomerOutput) -> ClaimSuccessView:
    """Map the customer-safe projection to view data."""
    return ClaimSuccessView(
        customer_draft=output.customer_text,
        provenance=output.provenance,
        response_mode=output.response_mode,
        generation_outcome=output.generation_outcome,
        failure_source=output.failure_source,
        assessment_status=output.assessment_status,
        risk_floor=output.risk_floor.value,
        risk_label=output.risk_label,
        risk_note=output.risk_note,
        risk_tone=output.risk_tone,
        handoff_notice=output.handoff_notice,
        priority_handoff=output.priority_handoff,
        requires_escalation=output.handoff_required,
        outcome_notice=output.outcome_notice,
        routing_recommendation=output.routing_recommendation,
        claim_category=output.claim_category,
        staff_actions=output.staff_actions,
        citations=tuple(_display_source(s, is_confirmed=True) for s in output.answer_sources),
        retrieved_materials=tuple(
            _display_source(s, is_confirmed=False) for s in output.retrieved_materials
        ),
        policy_violations=output.policy_violations,
    )


def map_result_to_display(result: CustomerClaimsResult) -> ClaimSuccessView:
    """Map a pipeline result to display-safe view data."""
    return map_output_to_display(build_customer_output(result))


def process_claim(
    pipeline: CustomerClaimsPipeline,
    message: str,
) -> ClaimView:
    try:
        request = CustomerClaimsRequest(customer_query=message)
    except ValidationError as exc:
        if is_query_too_long_error(exc):
            return ClaimErrorView(INPUT_TOO_LONG_MESSAGE, "input")
        return ClaimErrorView(INPUT_ERROR_MESSAGE, "input")

    try:
        result = pipeline.handle(request)
        # Mapping applies the customer-output policy; a failure there must not surface raw
        # exception details in the browser either.
        return map_result_to_display(result)
    except (RetrievalError, GenerationError):
        return ClaimErrorView(SERVICE_ERROR_MESSAGE, "service")
    except Exception:
        return ClaimErrorView(UNEXPECTED_ERROR_MESSAGE, "unexpected")


def startup_error_view() -> ClaimErrorView:
    return ClaimErrorView(STARTUP_ERROR_MESSAGE, "startup")


def startup_error_view_for_exception(exc: Exception) -> ClaimErrorView:
    if isinstance(exc, (ReleasePostureError, IndexManifestError)):
        return ClaimErrorView(STARTUP_INDEX_ERROR_MESSAGE, "startup_index")
    if isinstance(exc, (GenerationError, ValidationError, ValueError)):
        return ClaimErrorView(STARTUP_CONFIG_ERROR_MESSAGE, "startup_config")
    if isinstance(exc, RetrievalError):
        return ClaimErrorView(STARTUP_INDEX_ERROR_MESSAGE, "startup_index")
    return startup_error_view()
