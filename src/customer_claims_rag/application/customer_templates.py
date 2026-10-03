"""Deterministic customer texts: category templates and generic fallbacks.

Company voice: first-person plural ("мы"), official, 2–5 sentences. No operator instructions and
no internal terminology. These texts are the only customer-visible wording the application
authors itself, so none of them may claim an operational step (registration, transfer,
notification), a refund / compensation outcome, an admission of fault or a deadline;
``tests/unit/test_customer_output_policy.py`` runs every text through ``customer_text_policy``.

Critical categories (health, hazard, mass incident, threat, fraud, personal data) each have their
own text: no critical case may receive another category's wording.
"""

from __future__ import annotations

from customer_claims_rag.application.claim_guidance import (
    CATEGORY_FOREIGN_OBJECT,
    CATEGORY_FRAUD,
    CATEGORY_HEALTH,
    CATEGORY_MASS_INCIDENT,
    CATEGORY_PERSONAL_DATA,
    CATEGORY_THREAT,
    CATEGORY_UNSUPPORTED_LANGUAGE,
)
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskLevel

UNSUPPORTED_LANGUAGE_CUSTOMER_TEXT = (
    "Благодарим за обращение. "
    "Сейчас мы можем обработать обращение автоматически только на русском языке. "
    "Пожалуйста, опишите ситуацию на русском языке и, если вопрос связан с заказом, "
    "укажите номер заказа. "
    "Обращение потребует ручной проверки."
)

CATEGORY_DRAFT_TEMPLATES: dict[str, str] = {
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
    CATEGORY_HEALTH: (
        "Сожалеем, что вам стало плохо. "
        "По одному обращению невозможно установить причину симптомов. "
        "При выраженных, усиливающихся или сохраняющихся симптомах обратитесь "
        "за профессиональной медицинской помощью. "
        "Сохраните сведения о заказе, упаковку и остатки продукта, если это безопасно. "
        "Мы приоритетно проверим обращение."
    ),
    CATEGORY_FOREIGN_OBJECT: (
        "Сожалеем о произошедшем. "
        "Не употребляйте продукт, сохраните его и упаковку для возможной проверки. "
        "Просим указать номер заказа и, если возможно, приложить фотографии. "
        "Мы приоритетно проверим обращение и сообщим дальнейшие действия."
    ),
    CATEGORY_MASS_INCIDENT: (
        "Сожалеем о случившемся. "
        "Если кому-либо стало плохо, обратитесь за профессиональной медицинской помощью. "
        "Не употребляйте оставшиеся продукты из заказа и сохраните упаковку, если это безопасно. "
        "Просим указать номера заказов, даты доставки и, если возможно, сколько человек затронуто. "
        "Мы приоритетно проверим обращение."
    ),
    CATEGORY_FRAUD: (
        "Сожалеем о возникшей ситуации. "
        "Не отправляйте в чате CVV/CVC, PIN, полный номер карты, коды из SMS и пароли: "
        "для рассмотрения обращения они не нужны. "
        "Если вы не совершали эту операцию, при необходимости обратитесь в свой банк. "
        "Просим указать номер заказа, дату и сумму списания. "
        "Мы приоритетно проверим обращение."
    ),
    CATEGORY_THREAT: (
        "Мы получили ваше сообщение. "
        "Безопасность курьеров и персонала для нас важна, угрозы в их адрес недопустимы. "
        "Если ваш вопрос связан с заказом, укажите номер заказа и кратко опишите ситуацию. "
        "Мы приоритетно проверим обращение."
    ),
    CATEGORY_PERSONAL_DATA: (
        "Сожалеем о возникшей ситуации. "
        "Мы не можем подтвердить или опровергнуть факт утечки по одному сообщению: "
        "это требует проверки. "
        "Просим описать, что именно, где и когда вы увидели, не повторяя сами персональные данные, "
        "и не отправлять в чате пароли и коды подтверждения. "
        "Мы проверим обращение."
    ),
    CATEGORY_UNSUPPORTED_LANGUAGE: UNSUPPORTED_LANGUAGE_CUSTOMER_TEXT,
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

# Critical categories always get their deterministic text, whatever the model wrote: free-form
# model wording on health, hazard, mass-incident, threat, fraud and personal-data cases is exactly
# where an unsupported promise, diagnosis or admission would do the most damage. Personal-data
# exposure is listed for both its HIGH and CRITICAL forms (one category, one text).
HARD_TEMPLATE_CATEGORIES: frozenset[str] = frozenset({
    CATEGORY_HEALTH,
    CATEGORY_FOREIGN_OBJECT,
    CATEGORY_MASS_INCIDENT,
    CATEGORY_THREAT,
    CATEGORY_FRAUD,
    CATEGORY_PERSONAL_DATA,
})

# Generic drafts for a category without its own text. None of them is category- or
# symptom-specific; the CRITICAL one in particular must stay neutral (it used to be the health
# text, which reached fraud, threat and data-exposure cases).
GENERIC_RISK_DRAFTS: dict[RiskLevel, str] = {
    RiskLevel.CRITICAL: (
        "Благодарим за обращение. "
        "Ситуация требует приоритетной проверки. "
        "Просим указать номер заказа и кратко описать, что произошло. "
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


def safe_customer_draft(category: str, risk: DeterministicRiskResult) -> str:
    """Return the safest available deterministic draft for the given category/risk."""
    if category in CATEGORY_DRAFT_TEMPLATES:
        return CATEGORY_DRAFT_TEMPLATES[category]
    return GENERIC_RISK_DRAFTS[risk.risk_floor]


def sanitize_customer_draft(
    text: str,
    violations: list[str] | tuple[str, ...],
    risk: DeterministicRiskResult,
    category: str = "",
) -> tuple[str, bool]:
    """Return ``(text, is_safe)``; any violation yields the deterministic draft and ``False``.

    Fragments are never stripped out of a rejected draft: that could leave an incoherent reply, so
    ``is_safe=False`` always means a deterministic replacement was used.
    """
    if not violations:
        return text, True
    return safe_customer_draft(category, risk), False
