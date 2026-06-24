"""Deterministic fallback text for grounded generation."""

from customer_claims_rag.generation.models import GroundedGenerationResult

INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE = (
    "В доступных материалах недостаточно информации, чтобы дать подтвержденный ответ. "
    "Вопрос требует дополнительной проверки."
)

GENERATION_FAILURE_CUSTOMER_RESPONSE = (
    "Сейчас не удалось подготовить подтвержденный ответ по базе знаний. "
    "Попробуйте повторить запрос или обратитесь в поддержку."
)


def generation_failure_result() -> GroundedGenerationResult:
    """Return canonical fallback when grounded generation fails operationally."""
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
        citations=[],
    )
