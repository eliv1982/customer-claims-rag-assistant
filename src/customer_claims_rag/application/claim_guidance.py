"""Claim category, staff actions and routing guidance derived from the deterministic assessment.

Everything here is staff-facing guidance plus the category key that selects a deterministic
customer text (see ``customer_templates``). Wording follows the project policy documents
(08 escalation, 11 payment security, 12 staff safety) and never describes an operational step as
already performed: it tells staff what to do, and tells nobody that a transfer happened.
"""

from __future__ import annotations

import re

from customer_claims_rag.generation.risk_integration_models import RiskAwareGenerationOutcome
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.rules import normalize_for_matching

CATEGORY_HEALTH = "Жалоба на здоровье после употребления продукта"
CATEGORY_FOREIGN_OBJECT = "Обнаружение постороннего предмета в еде"
CATEGORY_MASS_INCIDENT = "Массовое обращение"
CATEGORY_FRAUD = "Признаки мошенничества"
CATEGORY_THREAT = "Прямая угроза"
CATEGORY_PERSONAL_DATA = "Утечка персональных данных"
CATEGORY_UNSUPPORTED_LANGUAGE = "Обращение не на русском языке"
CATEGORY_OUT_OF_SCOPE = "Вопрос вне области поддержки FoodFlow"
CATEGORY_GENERAL = "Общий запрос по сервису FoodFlow"

REASON_CODE_CATEGORIES: dict[RiskReasonCode, str] = {
    RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION: CATEGORY_HEALTH,
    RiskReasonCode.DANGEROUS_FOREIGN_OBJECT: CATEGORY_FOREIGN_OBJECT,
    RiskReasonCode.MASS_INCIDENT: CATEGORY_MASS_INCIDENT,
    RiskReasonCode.FRAUD_INDICATORS: CATEGORY_FRAUD,
    RiskReasonCode.DIRECT_THREAT: CATEGORY_THREAT,
    RiskReasonCode.DELAY_OVER_120_MINUTES: "Задержка доставки более 120 минут",
    RiskReasonCode.NON_DELIVERY: "Недоставка оплаченного заказа",
    RiskReasonCode.FALSE_DELIVERY_STATUS: "Неверный статус доставки",
    RiskReasonCode.PACKAGE_TAMPERING: "Нарушение целостности упаковки",
    RiskReasonCode.FOOD_SPOILAGE: "Испорченный или некачественный продукт",
    # Not confirmed dangerous (hair, 'foreign object'): the same HIGH quality category and text as
    # spoilage, not the critical hazard category.
    RiskReasonCode.UNCONFIRMED_FOREIGN_OBJECT: "Испорченный или некачественный продукт",
    RiskReasonCode.LEGAL_OR_REGULATORY_ESCALATION: "Юридическая или регуляторная эскалация",
    RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE: "Требование официального письменного ответа",
    RiskReasonCode.PERSONAL_DATA_EXPOSURE: CATEGORY_PERSONAL_DATA,
    RiskReasonCode.DELAY_OVER_30_MINUTES: "Задержка доставки более 30 минут",
    RiskReasonCode.MISSING_ITEM: "Неполный заказ (недокомплект)",
    RiskReasonCode.REFUND_REQUEST: "Требование возврата средств",
    RiskReasonCode.WRONG_ITEM: "Неверная позиция в заказе",
}

# When several critical signals are present the customer text and staff guidance follow the most
# safety-relevant one. Immediate physical safety (health, hazard, mass incident, threat to staff)
# precedes payment fraud; the remaining reason codes keep their canonical severity order.
_CRITICAL_CATEGORY_PRIORITY: tuple[RiskReasonCode, ...] = (
    RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION,
    RiskReasonCode.DANGEROUS_FOREIGN_OBJECT,
    RiskReasonCode.MASS_INCIDENT,
    RiskReasonCode.DIRECT_THREAT,
    RiskReasonCode.FRAUD_INDICATORS,
)

_KEYWORD_CATEGORY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # delay signals must precede non-delivery to prevent ambiguous "не приехал" → non-delivery
    (
        re.compile(
            r"не\s+приехал\w*\s+в\s+срок"
            r"|не\s+приехал\s+с\s+заказ"
            r"|курьер\s+(?:ещё|еще|все\s+ещё|все\s+еще)?\s*не\s+приехал"
            r"|опаздывает"
            r"|задерживается"
            r"|не\s+могу\s+до\s+(?:него|неё|курьер)",
            re.IGNORECASE,
        ),
        "Задержка доставки",
    ),
    (
        re.compile(
            r"не\s+(?:был[аи]?\s+)?доставл\w+|не\s+привез\w*|не\s+дошел\w*|не\s+пришел\w*",
            re.IGNORECASE,
        ),
        "Недоставка оплаченного заказа",
    ),
    (
        re.compile(r"не\s+хватал\w*|недокомплект\w*|не\s+довез\w*|не\s+положил\w*|не\s+хватает\s+\w+позиц", re.IGNORECASE),
        "Неполный заказ (недокомплект)",
    ),
    (
        re.compile(r"возврат\w*|верн\w+\s+деньг|компенсац\w*\s+за", re.IGNORECASE),
        "Требование возврата средств",
    ),
    (
        re.compile(r"задержк\w*|опоздал\w*|не\s+в\s+срок|долго\s+ждал", re.IGNORECASE),
        "Задержка доставки",
    ),
    (
        re.compile(r"плохое\s+качеств\w*|некачественн\w*|вкус\s+плох\w*|не\s+свежий\w*|несвежий\w*", re.IGNORECASE),
        "Жалоба на качество продукта",
    ),
    (
        re.compile(
            r"привезли\s+другое\s+блюд\w*|не\s+то\s*,?\s*что\s+заказ\w*|перепутали\s+позиц\w*|положили\s+другой\s+товар|заказ\s+не\s+соответствует\s+составу",
            re.IGNORECASE,
        ),
        "Неверная позиция в заказе",
    ),
]


def is_out_of_scope_presentation(
    risk: DeterministicRiskResult,
    generation_outcome: RiskAwareGenerationOutcome,
) -> bool:
    """True when the out-of-scope reply may be shown.

    The model's out-of-scope judgement never overrides the deterministic assessment: a HIGH/CRITICAL
    floor keeps its category, staff guidance and routing instead of "no further handling needed",
    and text in a language the rules cannot assess stays a manual-review case.
    """
    return (
        generation_outcome == "out_of_scope"
        and risk.assessment_status is not RiskAssessmentStatus.UNSUPPORTED_LANGUAGE
        and risk.risk_floor not in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    )


def derive_claim_category(
    risk: DeterministicRiskResult,
    generation_outcome: RiskAwareGenerationOutcome,
    customer_query: str = "",
) -> str:
    """Derive a human-readable claim category from risk signals and query keywords."""
    if is_out_of_scope_presentation(risk, generation_outcome):
        return CATEGORY_OUT_OF_SCOPE
    if risk.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE:
        return CATEGORY_UNSUPPORTED_LANGUAGE
    for code in _CRITICAL_CATEGORY_PRIORITY:
        if code in risk.reason_codes:
            return REASON_CODE_CATEGORIES[code]
    for code in risk.reason_codes:
        if code in REASON_CODE_CATEGORIES:
            return REASON_CODE_CATEGORIES[code]
    if customer_query:
        normalized = normalize_for_matching(customer_query)
        for pattern, category in _KEYWORD_CATEGORY_PATTERNS:
            if pattern.search(normalized):
                return category
    return CATEGORY_GENERAL


# ---------------------------------------------------------------------------
# Staff actions
# ---------------------------------------------------------------------------

OOS_STAFF_ACTIONS = (
    "Убедитесь, что вопрос действительно не относится к заказам или сервису FoodFlow.",
    "Направьте клиенту подготовленный ответ об ограничении области поддержки.",
    "Не выполняйте поиск по заказам и не эскалируйте без дополнительных оснований.",
)

UNSUPPORTED_LANGUAGE_STAFF_ACTIONS = (
    "Прочитайте обращение вручную: автоматическая оценка риска для этого языка не выполнялась.",
    "Определите категорию и срочность самостоятельно; при признаках угрозы здоровью "
    "или безопасности действуйте по процедуре приоритетной проверки.",
    "Подготовьте ответ вручную или попросите клиента описать ситуацию на русском языке.",
)

STAFF_ACTIONS_BY_LEVEL: dict[RiskLevel, tuple[str, ...]] = {
    RiskLevel.CRITICAL: (
        "Немедленно передайте обращение старшему специалисту для приоритетной проверки.",
        "Зафиксируйте детали: номер заказа, суть обращения, время обращения.",
        "Не формулируйте собственных заключений о причинах и не обещайте результат до проверки.",
    ),
    RiskLevel.HIGH: (
        "Проверьте детали обращения и историю заказа.",
        "При необходимости запросите у клиента подтверждающие материалы (фото, чек).",
        "Передайте обращение на проверку согласно процедуре.",
        "Зафиксируйте факт обращения и принятые меры.",
    ),
    RiskLevel.MEDIUM: (
        "Проверьте историю заказа и состав доставки.",
        "Уточните недостающие детали у клиента при необходимости.",
        "Проверьте применимые правила FoodFlow по данной категории.",
        "Обработайте согласно стандартному процессу.",
    ),
    RiskLevel.LOW: (
        "Проверьте историю заказа и статус доставки в системе.",
        "Уточните детали у клиента при необходимости.",
        "Обработайте в стандартном порядке.",
    ),
}

STAFF_ACTIONS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "Недоставка оплаченного заказа": (
        "Проверьте статус и историю заказа в системе.",
        "Установите, была ли доставка совершена и получена ли клиентом.",
        "Проверьте возможность возврата средств согласно правилам FoodFlow.",
        "При подтверждении недоставки — инициируйте процедуру возврата или повторной доставки.",
    ),
    "Неполный заказ (недокомплект)": (
        "Проверьте состав доставленного заказа по накладной.",
        "Сравните с заказанными позициями.",
        "При подтверждении недокомплекта — проверьте возможность частичного возврата.",
        "Запросите фото при необходимости.",
    ),
    "Требование возврата средств": (
        "Проверьте историю заказов и причины задержки или неисполнения.",
        "Установите, выполнялась ли каждая из перечисленных доставок.",
        "Проверьте применимые условия возврата для каждого заказа.",
        "Не подтверждайте и не отклоняйте возврат до полной проверки.",
    ),
    "Нарушение целостности упаковки": (
        "Зафиксируйте факт вскрытия упаковки и запросите фото.",
        "Проверьте, была ли нарушена пломба или защитная плёнка.",
        "Не рекомендуйте употреблять продукт до выяснения обстоятельств.",
        "Передайте обращение на проверку согласно процедуре.",
    ),
    CATEGORY_HEALTH: (
        "Убедитесь, что клиент получил рекомендацию обратиться за медицинской помощью.",
        "Немедленно передайте обращение старшему специалисту.",
        "Зафиксируйте полные детали: номер заказа, симптомы, время.",
        "Попросите сохранить остатки продукта, если это безопасно.",
    ),
    CATEGORY_FOREIGN_OBJECT: (
        "Убедитесь, что клиент не употребляет продукт.",
        "Зафиксируйте детали и запросите фотографии предмета и упаковки.",
        "Немедленно передайте обращение старшему специалисту.",
    ),
    CATEGORY_MASS_INCIDENT: (
        "Зафиксируйте номера заказов, даты доставки и описание повторяющегося признака.",
        "Убедитесь, что клиент получил рекомендацию обратиться за медицинской помощью, "
        "если кому-то стало плохо.",
        "Не подтверждайте масштаб и причину инцидента.",
        "Немедленно передайте обращение старшему специалисту для приоритетной проверки.",
    ),
    CATEGORY_FRAUD: (
        "Не запрашивайте и не принимайте в чате CVV/CVC, PIN, полный номер карты, "
        "коды подтверждения и пароли.",
        "Зафиксируйте безопасные сведения: номер заказа, дату и сумму списания, описание клиента.",
        "Не утверждайте, что мошенничество доказано или опровергнуто, и не обещайте возврат до проверки.",
        "Передайте обращение для приоритетной проверки по маршруту платежей и безопасности.",
    ),
    CATEGORY_THREAT: (
        "Не спорьте с клиентом, не провоцируйте его и не обесценивайте угрозу.",
        "Зафиксируйте номер заказа, канал и время обращения, суть угрозы без избыточного цитирования.",
        "Не раскрывайте клиенту личные данные, контакты и местоположение курьера или сотрудника.",
        "Немедленно передайте обращение для приоритетной проверки по правилам безопасности персонала; "
        "вопросы доставки и возврата — после, без обещания результата.",
    ),
    CATEGORY_PERSONAL_DATA: (
        "Зафиксируйте сообщение клиента: что, где и когда он увидел. Не просите повторно присылать "
        "чужие персональные данные.",
        "Не подтверждайте и не опровергайте факт и масштаб утечки.",
        "Не запрашивайте в чате пароли и коды подтверждения.",
        "Передайте обращение уполномоченному сотруднику для проверки; "
        "при признаках массового или продолжающегося раскрытия — приоритетно.",
    ),
    "Неверная позиция в заказе": (
        "Проверьте состав доставленного заказа по накладной и заказанным позициям.",
        "Установите, какая позиция была доставлена вместо заказанной.",
        "Запросите фотографию полученной позиции при необходимости.",
        "Проверьте возможность компенсации или повторной доставки согласно правилам.",
    ),
}


def staff_actions(
    risk: DeterministicRiskResult,
    generation_outcome: RiskAwareGenerationOutcome,
    claim_category: str,
) -> tuple[str, ...]:
    if is_out_of_scope_presentation(risk, generation_outcome):
        return OOS_STAFF_ACTIONS
    if risk.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE:
        return UNSUPPORTED_LANGUAGE_STAFF_ACTIONS
    if claim_category in STAFF_ACTIONS_BY_CATEGORY:
        return STAFF_ACTIONS_BY_CATEGORY[claim_category]
    return STAFF_ACTIONS_BY_LEVEL.get(risk.risk_floor, STAFF_ACTIONS_BY_LEVEL[RiskLevel.LOW])


# ---------------------------------------------------------------------------
# Routing recommendation
# ---------------------------------------------------------------------------

def routing_recommendation(
    risk: DeterministicRiskResult,
    generation_outcome: RiskAwareGenerationOutcome,
) -> str:
    if is_out_of_scope_presentation(risk, generation_outcome):
        return (
            "Направить клиенту сообщение об ограничении области поддержки; "
            "дальнейшая обработка не требуется."
        )
    if risk.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE:
        return "Обращение следует проверить вручную: автоматическая оценка риска не выполнена."
    if risk.priority_handoff:
        return (
            "Обращение следует передать старшему специалисту. "
            "Требуется приоритетная проверка."
        )
    if risk.handoff_required:
        return "Обращение следует проверить согласно процедуре."
    if generation_outcome == "insufficient_context" and risk.risk_floor in {RiskLevel.MEDIUM, RiskLevel.LOW}:
        return "Стандартная обработка. Рекомендуется ручная проверка условий и истории заказа."
    return "Стандартная обработка."
