"""Pure display helpers for customer claim results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from customer_claims_rag.application.models import CustomerClaimsRequest, CustomerClaimsResult
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.exceptions import (
    GenerationError,
    IndexManifestError,
    ReleasePostureError,
    RetrievalError,
)
from customer_claims_rag.generation.models import Citation
from customer_claims_rag.generation.risk_integration_models import RiskAwareGenerationOutcome
from customer_claims_rag.risk.models import RiskLevel

INPUT_ERROR_MESSAGE = "Введите текст обращения."
LOADING_MESSAGE = "Анализируем обращение и проверяем правила FoodFlow…"
STARTUP_CONFIG_ERROR_MESSAGE = (
    "Конфигурация сервиса недоступна. Проверьте настройки окружения."
)
STARTUP_INDEX_ERROR_MESSAGE = (
    "База знаний production недоступна. Убедитесь, что индекс установлен "
    "и прошёл проверку release posture."
)
STARTUP_ERROR_MESSAGE = (
    "Сервис не готов к запуску. Проверьте настройки окружения и индекс."
)
SERVICE_ERROR_MESSAGE = "Не удалось обработать обращение. Попробуйте еще раз."
UNEXPECTED_ERROR_MESSAGE = "Произошла непредвиденная ошибка. Попробуйте еще раз."

RiskTone = Literal["neutral", "warning", "critical"]

_OUTCOME_NOTICES: dict[RiskAwareGenerationOutcome, str | None] = {
    "grounded_answer": None,
    "insufficient_context": (
        "В базе знаний недостаточно подтверждённой информации для полного ответа."
    ),
    "generation_error_fallback": (
        "Ответ подготовлен в режиме отказа от генерации."
    ),
}

_RISK_LABELS: dict[RiskLevel, str] = {
    RiskLevel.LOW: "Низкий",
    RiskLevel.MEDIUM: "Средний",
    RiskLevel.HIGH: "Высокий",
    RiskLevel.CRITICAL: "Критический",
}


@dataclass(frozen=True)
class DisplayCitation:
    key: str
    heading: str
    document_id: str
    label: str


@dataclass(frozen=True)
class ClaimSuccessView:
    answer: str
    response_mode: str
    generation_outcome: RiskAwareGenerationOutcome
    risk_level: str
    risk_label: str
    risk_tone: RiskTone
    handoff_notice: str | None
    priority_handoff: bool
    outcome_notice: str | None
    citations: tuple[DisplayCitation, ...]


@dataclass(frozen=True)
class ClaimErrorView:
    message: str
    category: Literal["input", "startup", "startup_config", "startup_index", "service", "unexpected"]


ClaimView = ClaimSuccessView | ClaimErrorView


def format_citation_label(citation: Citation) -> str:
    """Return a safe user-facing citation label."""
    return f"[{citation.citation_key}] {citation.heading} — {citation.document_id}"


def generation_outcome_notice(outcome: RiskAwareGenerationOutcome) -> str | None:
    """Return an optional user-facing notice for a generation outcome."""
    return _OUTCOME_NOTICES[outcome]


def risk_display_metadata(
    risk_level: RiskLevel,
    *,
    priority_handoff: bool,
) -> tuple[str, RiskTone]:
    """Return a Russian risk label and display tone."""
    label = _RISK_LABELS[risk_level]
    if risk_level is RiskLevel.CRITICAL or priority_handoff:
        return label, "critical"
    if risk_level is RiskLevel.HIGH:
        return label, "warning"
    return label, "neutral"


def map_result_to_display(result: CustomerClaimsResult) -> ClaimSuccessView:
    """Map a pipeline result to display-safe view data."""
    response = result.response
    generation = response.generation
    risk = response.risk_assessment
    risk_label, risk_tone = risk_display_metadata(
        risk.risk_floor,
        priority_handoff=risk.priority_handoff,
    )
    citations = tuple(
        DisplayCitation(
            key=citation.citation_key,
            heading=citation.heading,
            document_id=citation.document_id,
            label=format_citation_label(citation),
        )
        for citation in generation.citations
    )
    return ClaimSuccessView(
        answer=generation.customer_response,
        response_mode=generation.response_mode,
        generation_outcome=response.generation_outcome,
        risk_level=risk.risk_floor.value,
        risk_label=risk_label,
        risk_tone=risk_tone,
        handoff_notice=response.handoff_notice,
        priority_handoff=risk.priority_handoff,
        outcome_notice=generation_outcome_notice(response.generation_outcome),
        citations=citations,
    )


def process_claim(
    pipeline: CustomerClaimsPipeline,
    message: str,
) -> ClaimView:
    """Validate input, invoke the pipeline once, and map to a display view."""
    try:
        request = CustomerClaimsRequest(customer_query=message)
    except ValidationError:
        return ClaimErrorView(INPUT_ERROR_MESSAGE, "input")

    try:
        result = pipeline.handle(request)
    except (RetrievalError, GenerationError):
        return ClaimErrorView(SERVICE_ERROR_MESSAGE, "service")
    except Exception:
        return ClaimErrorView(UNEXPECTED_ERROR_MESSAGE, "unexpected")

    return map_result_to_display(result)


def startup_error_view() -> ClaimErrorView:
    """Return the canonical startup/configuration error view."""
    return ClaimErrorView(STARTUP_ERROR_MESSAGE, "startup")


def startup_error_view_for_exception(exc: Exception) -> ClaimErrorView:
    """Map a startup exception to a safe user-facing category."""
    if isinstance(exc, (ReleasePostureError, IndexManifestError)):
        return ClaimErrorView(STARTUP_INDEX_ERROR_MESSAGE, "startup_index")
    if isinstance(exc, (GenerationError, ValidationError, ValueError)):
        return ClaimErrorView(STARTUP_CONFIG_ERROR_MESSAGE, "startup_config")
    if isinstance(exc, RetrievalError):
        return ClaimErrorView(STARTUP_INDEX_ERROR_MESSAGE, "startup_index")
    return startup_error_view()
