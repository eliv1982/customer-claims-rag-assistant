"""Pure display helpers for customer claim results."""

from __future__ import annotations

import re
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
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.rules import normalize_for_matching

INPUT_ERROR_MESSAGE = "Введите текст обращения."
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

RiskTone = Literal["neutral", "warning", "critical"]

_OUTCOME_NOTICES: dict[RiskAwareGenerationOutcome, str | None] = {
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

_RISK_LABELS: dict[RiskLevel, str] = {
    RiskLevel.LOW: "Низкий",
    RiskLevel.MEDIUM: "Средний",
    RiskLevel.HIGH: "Высокий",
    RiskLevel.CRITICAL: "Критический",
}

# ---------------------------------------------------------------------------
# Category-specific deterministic customer draft templates
# Company voice: first-person plural ("мы"), official, 2–5 sentences.
# No operator instructions, no internal terminology.
# ---------------------------------------------------------------------------

_CATEGORY_DRAFT_TEMPLATES: dict[str, str] = {
    "Недоставка оплаченного заказа": (
        "Сожалеем, что заказ не был доставлен. "
        "Мы проверим его статус, сведения об оплате и возможные варианты решения. "
        "Если потребуется дополнительная информация, уточним её отдельно. "
        "После проверки сообщим, возможны ли повторная доставка или возврат средств."
    ),
    "Неполный заказ (недокомплект)": (
        "Сожалеем, что в заказе не оказалось всех позиций. "
        "Мы проверим состав заказа и сведения о доставке. "
        "Просим указать номер заказа и, если возможно, приложить фотографию полученного заказа. "
        "После проверки сообщим, возможен ли частичный возврат или другой вариант решения."
    ),
    "Нарушение целостности упаковки": (
        "Сожалеем, что упаковка оказалась вскрыта. "
        "Не употребляйте продукт, сохраните упаковку и сделайте фотографии "
        "в том виде, в котором получили заказ. "
        "Просим указать номер заказа. "
        "Мы проверим обращение и сообщим дальнейшие действия."
    ),
    "Жалоба на здоровье после употребления продукта": (
        "Сожалеем, что вам стало плохо. "
        "По одному обращению невозможно установить причину симптомов. "
        "При выраженных, усиливающихся или сохраняющихся симптомах обратитесь "
        "за профессиональной медицинской помощью. "
        "Сохраните сведения о заказе, упаковку и остатки продукта, если это безопасно. "
        "Мы приоритетно проверим обращение."
    ),
    "Обнаружение постороннего предмета в еде": (
        "Сожалеем о произошедшем. "
        "Не употребляйте продукт, сохраните его и упаковку для возможной проверки. "
        "Просим указать номер заказа и, если возможно, приложить фотографии. "
        "Мы приоритетно проверим обращение и сообщим дальнейшие действия."
    ),
    "Требование возврата средств": (
        "Мы проверим историю заказов, сроки доставки и применимые условия возврата по каждому из них. "
        "Просим указать номера заказов, если они ещё не указаны в обращении. "
        "Возможность возврата будет определена после проверки каждого заказа. "
        "О результатах сообщим отдельно."
    ),
    "Задержка доставки": (
        "Сожалеем за доставленные неудобства. "
        "Мы проверим статус и историю доставки по вашему заказу. "
        "Просим указать номер заказа. "
        "После проверки сообщим возможный вариант решения."
    ),
    "Задержка доставки более 30 минут": (
        "Сожалеем за доставленные неудобства. "
        "Мы проверим статус и историю доставки по вашему заказу. "
        "Просим указать номер заказа. "
        "После проверки сообщим возможный вариант решения."
    ),
    "Задержка доставки более 120 минут": (
        "Сожалеем за длительное ожидание. "
        "Мы проверим статус и историю доставки по вашему заказу. "
        "Просим указать номер заказа. "
        "После проверки сообщим возможный вариант решения, включая возможную компенсацию."
    ),
    "Испорченный или некачественный продукт": (
        "Сожалеем о произошедшем. "
        "Не употребляйте продукт. "
        "Просим указать номер заказа и, если возможно, приложить фотографии продукта и упаковки. "
        "Мы проверим обращение и сообщим дальнейшие действия."
    ),
    "Неверный статус доставки": (
        "Сожалеем, что статус доставки не совпал с фактическим. "
        "Мы проверим статус заказа и сведения о доставке. "
        "После проверки сообщим фактический результат и возможные варианты решения."
    ),
    "Жалоба на качество продукта": (
        "Сожалеем, что качество продукта вас не устроило. "
        "Мы проверим обращение и детали заказа. "
        "Просим указать номер заказа и, если возможно, приложить фотографии. "
        "После проверки сообщим возможный вариант решения."
    ),
    "Неверная позиция в заказе": (
        "Сожалеем, что в заказе оказалась другая позиция. "
        "Мы проверим состав заказа и сведения о доставке. "
        "Просим указать номер заказа и, если возможно, приложить фотографию полученной позиции. "
        "После проверки сообщим возможный вариант решения."
    ),
    "Общий запрос по сервису FoodFlow": (
        "Пожалуйста, уточните, что именно произошло с заказом: "
        "доставка задерживается, заказ не получен, не хватает позиций "
        "или возникла проблема с качеством. "
        "Также укажите номер заказа и дату доставки, если они доступны. "
        "Это поможет быстрее рассмотреть обращение."
    ),
}

# These categories always use the safe template regardless of LLM output
_HARD_TEMPLATE_CATEGORIES: frozenset[str] = frozenset({
    "Жалоба на здоровье после употребления продукта",
    "Обнаружение постороннего предмета в еде",
    "Массовое обращение",
})

# Generic risk-level drafts as final fallback when category has no template
_GENERIC_RISK_DRAFTS: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: (
        "Сожалеем, что вам стало плохо. "
        "При выраженных или продолжающихся симптомах обратитесь за профессиональной медицинской помощью. "
        "Сохраните сведения о заказе и остатки продукта, если это безопасно. "
        "Мы приоритетно проверим обращение."
    ),
    RiskLevel.HIGH: (
        "Благодарим за обращение. "
        "Мы проверим детали и историю заказа. "
        "Просим указать номер заказа и при наличии приложить фотографии. "
        "После проверки сообщим возможный вариант решения."
    ),
    RiskLevel.MEDIUM: (
        "Благодарим за обращение. "
        "Ситуация требует дополнительного рассмотрения. "
        "Пожалуйста, уточните детали: укажите номер заказа, дату доставки "
        "и кратко опишите, что произошло."
    ),
    RiskLevel.LOW: (
        "Благодарим за обращение. "
        "Пожалуйста, уточните детали ситуации. "
        "Если вопрос связан с заказом, укажите номер заказа, дату доставки "
        "и кратко опишите, что произошло."
    ),
}


def _get_safe_draft(category: str, risk: DeterministicRiskResult) -> str:
    """Return the safest available deterministic draft for the given category/risk."""
    if category in _CATEGORY_DRAFT_TEMPLATES:
        return _CATEGORY_DRAFT_TEMPLATES[category]
    return _GENERIC_RISK_DRAFTS[risk.risk_floor]


# ---------------------------------------------------------------------------
# Citation-marker stripping
# ---------------------------------------------------------------------------

_CITATION_MARKER_RE = re.compile(r"\s*\[S\d+\]")


def _strip_citation_markers(text: str) -> str:
    """Remove inline [Sx] citation markers from customer-facing text."""
    cleaned = _CITATION_MARKER_RE.sub("", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# Client draft safety validation (pattern + semantic)
# ---------------------------------------------------------------------------

_DRAFT_FORBIDDEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # ── Risk enum labels ────────────────────────────────────────────────────
    (re.compile(r"\bhigh[\s\-–—]кейс\b", re.IGNORECASE), "risk-label-high-case"),
    (re.compile(r"\bcritical[\s\-–—]кейс\b", re.IGNORECASE), "risk-label-critical-case"),
    (re.compile(r"`\s*(?:high|critical|low|medium)\s*`", re.IGNORECASE), "backtick-risk-label"),
    (re.compile(r"\bhigh\b", re.IGNORECASE), "raw-english-risk-label-high"),
    (re.compile(r"\bcritical\b", re.IGNORECASE), "raw-english-risk-label-critical"),
    (re.compile(r"\bуровень\s+риска\b", re.IGNORECASE), "risk-level-label"),
    # ── System / AI meta-phrases ────────────────────────────────────────────
    (re.compile(r"\bассистент\b.{0,60}(?:устанавл|класс|передаё|отправл|обрабат|определ|не\s+может)", re.IGNORECASE | re.DOTALL), "system-meta-phrase"),
    (re.compile(r"\b(?:нейросет\w*|алгоритм|модел\w+)\b.{0,40}(?:класс|определ|оцен)", re.IGNORECASE | re.DOTALL), "ai-system-description"),
    # ── AI first-person voice (must not appear in company response) ─────────
    (re.compile(r"\bя\s+(?:не\s+могу|не\s+имею|проверю|уточню|направлю|передаю|передам|сообщу|не\s+вправе)\b", re.IGNORECASE), "ai-first-person"),
    # ── False completed-action claims ───────────────────────────────────────
    (re.compile(r"(?:ваше\s+)?обращение\s+(?:уже\s+)?(?:было\s+|будет\s+)?передано", re.IGNORECASE), "false-completed-transfer"),
    (re.compile(r"обращение\s+(?:уже\s+)?будет\s+передано", re.IGNORECASE), "false-completed-transfer"),
    (re.compile(r"обращение\s+(?:будет\s+)?эскалировано", re.IGNORECASE), "false-escalation"),
    (re.compile(r"запрос\s+(?:уже\s+)?(?:был\s+)?отправлен\s+(?:юрист|специалист|старш)", re.IGNORECASE), "false-specialist-transfer"),
    # ── Circular support redirect ────────────────────────────────────────────
    (re.compile(r"\bслуж\w+\s+поддержк\w*\b", re.IGNORECASE), "circular-support-reference"),
    (re.compile(r"\bобратитесь\s+в\s+(?:службу?\s+)?поддержк\w*\b", re.IGNORECASE), "circular-support-reference"),
    # ── Operator instructions in client text ────────────────────────────────
    (re.compile(r"\bсотрудник\b.{0,50}(?:должен|обязан|проверит|проверяет|уточнит|свяжется|обработает|выполнит|передаст)", re.IGNORECASE | re.DOTALL), "operator-instruction"),
    # 'требует (приоритетной) проверки сотрудником' — escalation directive, not a service promise
    (re.compile(r"требует\w*\s+(?:\w+\s+){0,2}проверк\w+\s+сотрудник\w*", re.IGNORECASE), "operator-instruction"),
    (re.compile(r"\bстарш\w+\s+специалист\w*\b", re.IGNORECASE), "operator-instruction"),
    (re.compile(r"необходимо\s+(?:зафиксировать|зарегистрировать|документировать)\b", re.IGNORECASE), "process-instruction"),
    (re.compile(r"необходимо\s+(?:зарегистрировать|передать)\s+обращение\b", re.IGNORECASE), "process-instruction"),
    # ── Internal technical terms ─────────────────────────────────────────────
    (re.compile(r"\b(?:response[\s_]mode|generation[\s_]outcome|insufficient[\s_]context|response_mode)\b", re.IGNORECASE), "internal-technical-term"),
    # ── Generic IC/error fallback phrases that must not reach customer ───────
    (re.compile(r"(?:в\s+доступных\s+материалах|в\s+базе\s+знаний).{0,30}недостаточно", re.IGNORECASE), "ic-text-leaked"),
    (re.compile(r"(?:сейчас\s+)?не\s+удалось\s+подготовить\s+(?:подтвержд\w+\s+)?ответ", re.IGNORECASE), "generic-error-text"),
    (re.compile(r"повторите\s+запрос\s+или\s+обратитесь", re.IGNORECASE), "generic-error-text"),
    (re.compile(r"вопрос\s+требует\s+дополнительной\s+проверки\s*\.$", re.IGNORECASE), "ic-text-leaked"),
    # ── Vague generic promise phrases: no specific facts, no clarification ────
    (re.compile(r"мы\s+проверим\s+информацию", re.IGNORECASE), "vague-review-promise"),
    (re.compile(r"сообщим\s+о\s+результате", re.IGNORECASE), "vague-result-promise"),
]


def detect_draft_violations(text: str) -> list[str]:
    """Return list of violation type codes found in the customer draft."""
    violations: list[str] = []
    for pattern, violation_type in _DRAFT_FORBIDDEN_PATTERNS:
        if pattern.search(text):
            violations.append(violation_type)
    return violations


def sanitize_customer_draft(
    text: str,
    violations: list[str],
    risk: DeterministicRiskResult,
    category: str = "",
) -> tuple[str, bool]:
    """
    Return a safe customer draft given detected violations.

    Any violation triggers the safe template for the category (or generic risk
    fallback). We do not silently strip fragments that could leave the response
    incoherent — is_safe=False always means a deterministic replacement was used.
    """
    if not violations:
        return text, True
    return _get_safe_draft(category, risk), False


# ---------------------------------------------------------------------------
# Category derivation with keyword fallback
# ---------------------------------------------------------------------------

_REASON_CODE_CATEGORIES: dict[RiskReasonCode, str] = {
    RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION: "Жалоба на здоровье после употребления продукта",
    RiskReasonCode.DANGEROUS_FOREIGN_OBJECT: "Обнаружение постороннего предмета в еде",
    RiskReasonCode.MASS_INCIDENT: "Массовое обращение",
    RiskReasonCode.FRAUD_INDICATORS: "Признаки мошенничества",
    RiskReasonCode.DIRECT_THREAT: "Прямая угроза",
    RiskReasonCode.DELAY_OVER_120_MINUTES: "Задержка доставки более 120 минут",
    RiskReasonCode.NON_DELIVERY: "Недоставка оплаченного заказа",
    RiskReasonCode.FALSE_DELIVERY_STATUS: "Неверный статус доставки",
    RiskReasonCode.PACKAGE_TAMPERING: "Нарушение целостности упаковки",
    RiskReasonCode.FOOD_SPOILAGE: "Испорченный или некачественный продукт",
    RiskReasonCode.LEGAL_OR_REGULATORY_ESCALATION: "Юридическая или регуляторная эскалация",
    RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE: "Требование официального письменного ответа",
    RiskReasonCode.PERSONAL_DATA_EXPOSURE: "Утечка персональных данных",
    RiskReasonCode.DELAY_OVER_30_MINUTES: "Задержка доставки более 30 минут",
    RiskReasonCode.MISSING_ITEM: "Неполный заказ (недокомплект)",
    RiskReasonCode.REFUND_REQUEST: "Требование возврата средств",
    RiskReasonCode.WRONG_ITEM: "Неверная позиция в заказе",
}

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


def _claim_category(
    risk: DeterministicRiskResult,
    generation_outcome: RiskAwareGenerationOutcome,
    customer_query: str = "",
) -> str:
    """Derive a human-readable claim category from risk signals and query keywords."""
    if generation_outcome == "out_of_scope":
        return "Вопрос вне области поддержки FoodFlow"
    if risk.reason_codes:
        for code in risk.reason_codes:
            if code in _REASON_CODE_CATEGORIES:
                return _REASON_CODE_CATEGORIES[code]
    if customer_query:
        normalized = normalize_for_matching(customer_query)
        for pattern, category in _KEYWORD_CATEGORY_PATTERNS:
            if pattern.search(normalized):
                return category
    return "Общий запрос по сервису FoodFlow"


# ---------------------------------------------------------------------------
# Staff actions
# ---------------------------------------------------------------------------

_OOS_STAFF_ACTIONS = (
    "Убедитесь, что вопрос действительно не относится к заказам или сервису FoodFlow.",
    "Направьте клиенту подготовленный ответ об ограничении области поддержки.",
    "Не выполняйте поиск по заказам и не эскалируйте без дополнительных оснований.",
)

_STAFF_ACTIONS_BY_LEVEL: dict[RiskLevel, tuple[str, ...]] = {
    RiskLevel.CRITICAL: (
        "Убедитесь, что клиент получил рекомендацию обратиться за медицинской помощью.",
        "Немедленно передайте обращение старшему специалисту для приоритетной проверки.",
        "Зафиксируйте детали: номер заказа, описание симптомов, время обращения.",
        "Попросите клиента сохранить остатки продукта, если это безопасно.",
        "Не формулируйте собственных заключений о причинах жалобы.",
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

_STAFF_ACTIONS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
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
    "Жалоба на здоровье после употребления продукта": (
        "Убедитесь, что клиент получил рекомендацию обратиться за медицинской помощью.",
        "Немедленно передайте обращение старшему специалисту.",
        "Зафиксируйте полные детали: номер заказа, симптомы, время.",
        "Попросите сохранить остатки продукта, если это безопасно.",
    ),
    "Обнаружение постороннего предмета в еде": (
        "Убедитесь, что клиент не употребляет продукт.",
        "Зафиксируйте детали и запросите фотографии предмета и упаковки.",
        "Немедленно передайте обращение старшему специалисту.",
    ),
    "Неверная позиция в заказе": (
        "Проверьте состав доставленного заказа по накладной и заказанным позициям.",
        "Установите, какая позиция была доставлена вместо заказанной.",
        "Запросите фотографию полученной позиции при необходимости.",
        "Проверьте возможность компенсации или повторной доставки согласно правилам.",
    ),
}


def _staff_actions(
    risk: DeterministicRiskResult,
    generation_outcome: RiskAwareGenerationOutcome,
    claim_category: str,
) -> tuple[str, ...]:
    if generation_outcome == "out_of_scope":
        return _OOS_STAFF_ACTIONS
    if claim_category in _STAFF_ACTIONS_BY_CATEGORY:
        return _STAFF_ACTIONS_BY_CATEGORY[claim_category]
    return _STAFF_ACTIONS_BY_LEVEL.get(risk.risk_floor, _STAFF_ACTIONS_BY_LEVEL[RiskLevel.LOW])


# ---------------------------------------------------------------------------
# Routing recommendation
# ---------------------------------------------------------------------------

def _routing_recommendation(
    risk: DeterministicRiskResult,
    generation_outcome: RiskAwareGenerationOutcome,
) -> str:
    if generation_outcome == "out_of_scope":
        return (
            "Направить клиенту сообщение об ограничении области поддержки; "
            "дальнейшая обработка не требуется."
        )
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


# ---------------------------------------------------------------------------
# Display models
# ---------------------------------------------------------------------------

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
    draft_sanitized: bool
    response_mode: str
    generation_outcome: RiskAwareGenerationOutcome
    risk_level: str
    risk_label: str
    risk_tone: RiskTone
    handoff_notice: str | None
    priority_handoff: bool
    requires_escalation: bool
    outcome_notice: str | None
    routing_recommendation: str
    claim_category: str
    staff_actions: tuple[str, ...]
    citations: tuple[DisplayCitation, ...]
    retrieved_materials: tuple[DisplayCitation, ...]


@dataclass(frozen=True)
class ClaimErrorView:
    message: str
    category: Literal["input", "startup", "startup_config", "startup_index", "service", "unexpected"]


ClaimView = ClaimSuccessView | ClaimErrorView


def format_citation_label(citation: Citation) -> str:
    return f"[{citation.citation_key}] {citation.heading} — {citation.document_id}"


def generation_outcome_notice(outcome: RiskAwareGenerationOutcome) -> str | None:
    return _OUTCOME_NOTICES[outcome]


def risk_display_metadata(
    risk_level: RiskLevel,
    *,
    priority_handoff: bool,
) -> tuple[str, RiskTone]:
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
    outcome = response.generation_outcome

    risk_label, risk_tone = risk_display_metadata(
        risk.risk_floor,
        priority_handoff=risk.priority_handoff,
    )

    # Confirmed citations from grounded answer
    confirmed_citations = tuple(
        DisplayCitation(
            key=citation.citation_key,
            heading=citation.heading,
            document_id=citation.document_id,
            label=format_citation_label(citation),
            is_confirmed=True,
        )
        for citation in generation.citations
    )

    # Retrieved materials for insufficient_context display
    confirmed_keys = {c.key for c in confirmed_citations}
    retrieved_materials: tuple[DisplayCitation, ...] = ()
    if outcome in {"insufficient_context", "generation_error_fallback"}:
        retrieved_materials = tuple(
            DisplayCitation(
                key=item.citation_key,
                heading=item.heading,
                document_id=item.document_id,
                label=f"[{item.citation_key}] {item.heading} — {item.document_id}",
                is_confirmed=False,
            )
            for item in result.retrieved_items
            if item.citation_key not in confirmed_keys
        )

    # Derive category from risk reason codes + query keywords
    category = _claim_category(risk, outcome, result.customer_query)

    # ── Customer draft selection logic ────────────────────────────────────
    #
    # Priority:
    # 1. Hard template: health-harm and dangerous-object categories always
    #    get the safe deterministic draft — no LLM output shown.
    # 2. Non-grounded outcomes (IC, error): always use category template.
    # 3. Grounded answer: validate the LLM text; use template if violated.
    # 4. Clean grounded answer: show LLM output (citations already stripped).
    #
    # OOS is a special case — its fixed constant goes through unchanged.

    draft_sanitized = False

    if outcome == "out_of_scope":
        customer_draft = generation.customer_response

    elif category in _HARD_TEMPLATE_CATEGORIES:
        # Hard override — always safe template for critical health/safety categories
        customer_draft = _CATEGORY_DRAFT_TEMPLATES.get(category, _GENERIC_RISK_DRAFTS[risk.risk_floor])
        raw_was_template = customer_draft != generation.customer_response
        draft_sanitized = raw_was_template

    elif outcome in {"insufficient_context", "generation_error_fallback"}:
        # IC and error outcomes: always use category-specific or generic template
        customer_draft = _get_safe_draft(category, risk)
        draft_sanitized = True

    else:
        # grounded_answer — validate LLM output
        raw_response = _strip_citation_markers(generation.customer_response)
        violations = detect_draft_violations(raw_response)
        if violations:
            customer_draft, _ = sanitize_customer_draft(raw_response, violations, risk, category)
            draft_sanitized = True
        else:
            customer_draft = raw_response
            draft_sanitized = False

    return ClaimSuccessView(
        customer_draft=customer_draft,
        draft_sanitized=draft_sanitized,
        response_mode=generation.response_mode,
        generation_outcome=outcome,
        risk_level=risk.risk_floor.value,
        risk_label=risk_label,
        risk_tone=risk_tone,
        handoff_notice=response.handoff_notice,
        priority_handoff=risk.priority_handoff,
        requires_escalation=risk.handoff_required,
        outcome_notice=generation_outcome_notice(outcome),
        routing_recommendation=_routing_recommendation(risk, outcome),
        claim_category=category,
        staff_actions=_staff_actions(risk, outcome, category),
        citations=confirmed_citations,
        retrieved_materials=retrieved_materials,
    )


def process_claim(
    pipeline: CustomerClaimsPipeline,
    message: str,
) -> ClaimView:
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
    return ClaimErrorView(STARTUP_ERROR_MESSAGE, "startup")


def startup_error_view_for_exception(exc: Exception) -> ClaimErrorView:
    if isinstance(exc, (ReleasePostureError, IndexManifestError)):
        return ClaimErrorView(STARTUP_INDEX_ERROR_MESSAGE, "startup_index")
    if isinstance(exc, (GenerationError, ValidationError, ValueError)):
        return ClaimErrorView(STARTUP_CONFIG_ERROR_MESSAGE, "startup_config")
    if isinstance(exc, RetrievalError):
        return ClaimErrorView(STARTUP_INDEX_ERROR_MESSAGE, "startup_index")
    return startup_error_view()
