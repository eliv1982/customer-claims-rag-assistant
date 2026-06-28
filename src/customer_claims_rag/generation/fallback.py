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

OUT_OF_SCOPE_CUSTOMER_RESPONSE = (
    "Я могу помочь только с обращениями по заказам и сервису FoodFlow. "
    "По этому вопросу рекомендуем воспользоваться подходящим профильным сервисом."
)


def generation_failure_result() -> GroundedGenerationResult:
    """Return canonical fallback when grounded generation fails operationally."""
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
        citations=[],
    )


def out_of_scope_result() -> GroundedGenerationResult:
    """Return canonical response when the request is outside FoodFlow scope."""
    return GroundedGenerationResult(
        response_mode="out_of_scope",
        customer_response=OUT_OF_SCOPE_CUSTOMER_RESPONSE,
        citations=[],
    )
