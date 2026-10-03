"""Customer-visible text policy.

One deterministic gate over model-authored customer text. ``detect_draft_violations`` returns the
codes of every rule the text breaks; an empty list means the text may be shown to a customer.

The rules fall into semantic groups of bounded patterns (not an exact-string blacklist):

* internal vocabulary / AI voice / operator instructions / circular support references;
* operational commitments the application cannot back: refund or compensation promises,
  completed refunds, guaranteed outcomes, fixed refund/response deadlines, and claims that a case
  was registered, transferred, escalated or that staff were notified;
* admission of fault or liability;
* medical diagnosis, causal attribution and treatment advice;
* requests for sensitive payment / authentication data;
* markup (HTML tags, script-like URL schemes, Markdown links) that has no place in plain support text.

Neutral wording must keep passing: "вопрос о возврате требует проверки", "возможность возврата
зависит от результатов проверки", empathy without admission, and a recommendation to seek
professional medical help.

The module is stdlib-only on purpose. ``tests/unit/test_customer_text_policy.py`` pins the
prompt/policy contract: wording the system prompt prescribes must pass, wording it forbids must fail.
"""

from __future__ import annotations

import re
import unicodedata

_CITATION_MARKER_RE = re.compile(r"\s*\[S\d+\]")
_INVISIBLE_RE = re.compile("[­​-‏‪-‮⁠-⁤﻿]")

_FLAGS = re.IGNORECASE | re.DOTALL
# A bounded run of up to three words, used between anchors so no pattern can span a paragraph.
_GAP = r"(?:\w+\s+){0,3}"
_SENT = r"[^.!?\n]"

_RULE_SPECS: list[tuple[str, str]] = []


def _rule(code: str, *patterns: str) -> None:
    _RULE_SPECS.extend((pattern, code) for pattern in patterns)


def strip_citation_markers(text: str) -> str:
    """Remove inline ``[Sx]`` citation markers from customer-facing text."""
    cleaned = _CITATION_MARKER_RE.sub("", text)
    return re.sub(r" {2,}", " ", cleaned).strip()


def normalize_customer_text(text: str) -> str:
    """Normalize for matching only: fold compatibility forms, drop invisible characters, ё -> е."""
    folded = unicodedata.normalize("NFKC", text)
    folded = _INVISIBLE_RE.sub("", folded)
    return folded.replace("ё", "е").replace("Ё", "Е")


# ── Internal vocabulary and system voice ────────────────────────────────────────────────────────
_rule("risk-label-high-case", r"\bhigh[\s\-–—]кейс\b")
_rule("risk-label-critical-case", r"\bcritical[\s\-–—]кейс\b")
_rule("backtick-risk-label", r"`\s*(?:high|critical|low|medium)\s*`")
_rule("raw-english-risk-label-high", r"\bhigh\b")
_rule("raw-english-risk-label-critical", r"\bcritical\b")
_rule("risk-level-label", r"\bуровень\s+риска\b")
_rule(
    "system-meta-phrase",
    r"\bассистент\b.{0,60}(?:устанавл|класс|передае|отправл|обрабат|определ|не\s+может)",
)
_rule(
    "ai-system-description",
    r"\b(?:нейросет\w*|алгоритм|модел\w+)\b.{0,40}(?:класс|определ|оцен)",
)
_rule(
    "ai-first-person",
    r"\bя\s+(?:не\s+могу|не\s+имею|проверю|уточню|направлю|передаю|передам|сообщу|не\s+вправе)\b",
)

# ── Circular support redirect and operator instructions ─────────────────────────────────────────
_rule("circular-support-reference", r"\bслуж\w+\s+поддержк\w*\b")
_rule("circular-support-reference", r"\bобратитесь\s+в\s+(?:службу?\s+)?поддержк\w*\b")
_rule(
    "operator-instruction",
    r"\bсотрудник\b.{0,50}(?:должен|обязан|проверит|проверяет|уточнит|свяжется|обработает|выполнит|передаст)",
)
# 'требует (приоритетной) проверки сотрудником' is an escalation directive, not a service promise.
_rule("operator-instruction", r"требует\w*\s+(?:\w+\s+){0,2}проверк\w+\s+сотрудник\w*")
_rule("operator-instruction", r"\bстарш\w+\s+специалист\w*\b")
_rule("process-instruction", r"необходимо\s+(?:зафиксировать|зарегистрировать|документировать)\b")
_rule("process-instruction", r"необходимо\s+(?:зарегистрировать|передать)\s+обращение\b")
_rule(
    "internal-technical-term",
    r"\b(?:response[\s_]mode|generation[\s_]outcome|insufficient[\s_]context|response_mode)\b",
)

# ── Fallback phrases that must not reach the customer ───────────────────────────────────────────
_rule("ic-text-leaked", r"(?:в\s+доступных\s+материалах|в\s+базе\s+знаний).{0,30}недостаточно")
_rule("generic-error-text", r"(?:сейчас\s+)?не\s+удалось\s+подготовить\s+(?:подтвержд\w+\s+)?ответ")
_rule("generic-error-text", r"повторите\s+запрос\s+или\s+обратитесь")
_rule("ic-text-leaked", r"вопрос\s+требует\s+дополнительной\s+проверки\s*\.$")
_rule("vague-review-promise", r"мы\s+проверим\s+информацию")
_rule("vague-result-promise", r"сообщим\s+о\s+результате")

# ── Claims that the application registered / transferred / escalated / notified ─────────────────
# The application does not register, transfer or notify anything, so customer text must only
# describe the need for review ("обращение требует проверки"), never a completed or scheduled
# operational step.
_CLAIM_NOUN = r"(?:обращени\w+|заявк\w+|запрос\w*|жалоб\w+|претензи\w+|сообщени\w+)"
_CLAIM_AUX = r"(?:[\w-]+\s+){0,3}(?:(?:уже|успешно|был[аоие]?|будет|будут|нами)\s+)*"
_rule(
    "false-registration",
    rf"\b{_CLAIM_NOUN}\s+{_CLAIM_AUX}(?:зарегистрирован\w*|зафиксирован\w*|принят\w*|создан\w*|оформлен\w*|взят\w*\s+в\s+работу)\b",
    rf"\b(?:зарегистрирован\w*|зафиксирован\w*|оформлен\w*|создан\w*)\s+(?:\w+\s+){{0,2}}{_CLAIM_NOUN}",
    rf"\bмы\s+(?:уже\s+|успешно\s+)?(?:зарегистрировал\w*|зафиксировал\w*|оформил\w*|создали|приняли)\s+{_GAP}(?:{_CLAIM_NOUN}|случай|инцидент)",
)
_rule(
    "false-completed-transfer",
    rf"\b(?:ваше\s+)?{_CLAIM_NOUN}\s+{_CLAIM_AUX}(?:передан\w*|направлен\w*|отправлен\w*)\b",
    rf"\b(?:передан\w*|направлен\w*)\s+(?:\w+\s+){{0,2}}{_CLAIM_NOUN}",
    rf"\bмы\s+(?:уже\s+)?(?:передал\w*|направил\w*|отправил\w*)\s+{_GAP}(?:{_CLAIM_NOUN}|информаци\w+|данные|случай)",
)
_rule(
    "false-escalation",
    rf"\b{_CLAIM_NOUN}\s+{_CLAIM_AUX}эскалирован\w*",
    r"\bмы\s+(?:уже\s+)?эскалировал\w*",
)
_rule(
    "false-specialist-transfer",
    r"запрос\s+(?:уже\s+)?(?:был\s+)?отправлен\s+(?:юрист|специалист|старш)",
)
_rule(
    "false-notification",
    r"\b(?:сотрудник\w*|специалист\w*|менеджер\w*|служб\w+|команд\w+|отдел\w*)\s+(?:\w+\s+){0,2}(?:уже\s+)?(?:уведомлен\w*|проинформирован\w*|извещен\w*)",
    r"\bмы\s+(?:уже\s+)?(?:уведомили|проинформировали|известили)\s+(?:\w+\s+){0,2}(?:сотрудник\w*|специалист\w*|менеджер\w*|служб\w+|команд\w+)",
)

# ── Refund / compensation commitments ───────────────────────────────────────────────────────────
_MONEY_SUBJECT = r"(?:деньги|денег|средства|средств|сумма|сумму|оплата|оплату|платеж\w*)"
_rule(
    "refund-promise",
    # 'мы вернем вам деньги', 'мы произведем возврат', 'вернем средства после проверки'
    r"\b(?:вернем|возместим)\b",
    rf"\b(?:зачислим|перечислим|переведем|начислим|выплатим)\s+{_GAP}(?:{_MONEY_SUBJECT}|возврат\w*|компенсаци\w+|бонус\w+|баллы|\d+)",
    r"\bмы\s+(?:обязательно\s+|непременно\s+|полностью\s+)?(?:произведем|осуществим|оформим|выполним|сделаем|предоставим|обеспечим)\s+(?:\w+\s+){0,2}(?:возврат|возмещени\w+)",
    r"\bвам\s+(?:обязательно\s+|непременно\s+)?(?:будет\s+)?(?:верн[уе]т\w*|возмест\w+|возвращен\w*|выплат\w+|компенсир\w+)",
    r"\bвы\s+(?:обязательно\s+|непременно\s+)?получите\s+(?:\w+\s+){0,2}(?:возврат|возмещени\w+|компенсаци\w+|" + _MONEY_SUBJECT + r")",
)
_rule(
    "compensation-promise",
    r"\bмы\s+(?:обязательно\s+|непременно\s+|полностью\s+)?(?:выплатим|компенсируем|предоставим\s+(?:\w+\s+){0,2}компенсаци\w+|выдадим\s+(?:\w+\s+){0,2}(?:компенсаци\w+|промокод\w*|бонус\w*|скидк\w+))",
    r"\bкомпенсаци\w+\s+(?:\w+\s+){0,2}(?:будет|будут)\s+(?:\w+\s+){0,2}(?:выплачен\w*|начислен\w*|предоставлен\w*|перечислен\w*|одобрен\w*)",
)
_rule(
    "completed-refund",
    # 'возврат уже оформлен', 'возврат будет оформлен', 'деньги были возвращены', 'компенсация выплачена'
    r"\bвозврат\w*\s+(?:\w+\s+){0,2}(?:будет|будут|уже|был[аои]?|успешно)\s+(?:\w+\s+){0,2}(?:оформлен\w*|выполнен\w*|произведен\w*|осуществлен\w*|одобрен\w*|подтвержден\w*|совершен\w*|зачислен\w*|переведен\w*)",
    r"\b(?:возврат|возмещени\w+)\s+(?:средств\s+|денег\s+)?(?:одобрен\w*|подтвержден\w*|оформлен\w*|произведен\w*)",
    rf"\b{_MONEY_SUBJECT}\s+(?:\w+\s+){{0,2}}(?:будут|были|уже)\s+(?:\w+\s+){{0,2}}(?:возвращен\w*|зачислен\w*|переведен\w*|перечислен\w*)",
    rf"\b{_MONEY_SUBJECT}\s+(?:\w+\s+){{0,2}}(?:поступят|вернутся|придут)\b",
    r"\bкомпенсаци\w+\s+(?:уже\s+)?(?:был[аи]?\s+)?(?:выплачен\w*|начислен\w*|предоставлен\w*|перечислен\w*|одобрен\w*)",
)
_rule(
    "guaranteed-outcome",
    rf"\bгарантир\w+\s+{_GAP}(?:возврат|возмещени\w+|компенсаци\w+|{_MONEY_SUBJECT}|решени\w+)",
    r"\b(?:возврат|возмещени\w+|компенсаци\w+)\s+(?:\w+\s+){0,2}(?:гарантирован\w*|обеспечен\w*|обязательн\w*|точно)",
    r"\b(?:гарантированно|обязательно|непременно|точно)\s+(?:\w+\s+){0,2}(?:верн\w+|возмест\w+|компенсир\w+|выплат\w+|получите\s+(?:возврат|компенсаци))",
    r"\bвам\s+точно\s+(?:верн|выплат|возмест|компенсир)\w+",
)

# ── Fixed refund / response deadlines ───────────────────────────────────────────────────────────
# The application has no system behind any deadline, so none may be stated by customer text.
_NUMBER_WORD = r"(?:\d+(?:\s*[-–]\s*\d+)?|одн\w+|дв[ау]\w*|тр[еи]\w*|четыр\w+|пят\w+|шест\w+|сем\w+|восем\w+|девят\w+|десят\w+)"
_TIME_UNIT = r"(?:минут\w*|час\w*|(?:рабоч\w+\s+|календарн\w+\s+)?дн\w+|сутк\w+|суток|недел\w+)"
_PERIOD = rf"(?:{_NUMBER_WORD}\s+(?:рабоч\w+\s+|календарн\w+\s+)?{_TIME_UNIT}|суток|(?:одного\s+)?рабочего\s+дня|ближайш\w+\s+(?:час\w*|дн\w+))"
_rule(
    "deadline-commitment",
    rf"\b(?:ответим|сообщим|свяжемся|рассмотрим|обработаем|проверим|верн[еу]м|зачислим|перечислим|выплатим|решим|вышлем|отправим|направим|предоставим|дадим\s+ответ)\b{_SENT}{{0,60}}?\b(?:в\s+течение|за|не\s+позднее(?:\s+чем\s+через)?|через|в\s+срок(?:\s+до)?)\s+{_PERIOD}",
    rf"\bсрок\w*\s+(?:рассмотрени\w+|ответа|возврата|компенсаци\w+|зачислени\w+|выплат\w*|решени\w+|реагировани\w+){_SENT}{{0,40}}?{_PERIOD}",
    rf"\b(?:ответ|решение|возврат|компенсаци\w+|{_MONEY_SUBJECT})\b{_SENT}{{0,30}}?\b(?:будет|будут|поступ\w+|придут?|вернутся)\b{_SENT}{{0,30}}?\b(?:в\s+течение|за|через|не\s+позднее)\s+{_PERIOD}",
    rf"\b{_MONEY_SUBJECT}\s+(?:\w+\s+)?(?:поступят|придут|вернутся|будут\s+зачислены)\s+(?:сегодня|завтра|послезавтра)",
)

# ── Admission of fault / liability ──────────────────────────────────────────────────────────────
_rule(
    "fault-admission",
    # 'это наша вина', 'по нашей вине', 'за нашу ошибку'
    r"(?<!\bне\s)\bнаш\w*\s+(?:вин\w*|ошибк\w*|недосмотр\w*|недоработк\w*|упущени\w+)",
    r"\bза\s+(?:допущенн\w+|совершенн\w+)\s+(?:\w+\s+)?(?:ошибк\w+|нарушени\w+)",
    # 'мы виноваты', 'мы ошиблись', 'мы признаем вину', 'мы несем ответственность'
    r"\bмы\s+(?:\w+\s+){0,2}(?:виноват\w*|виновн\w*|ошиблись|подвели)\b",
    r"\bмы\s+(?:\w+\s+){0,2}(?:допустили|совершили)\s+(?:\w+\s+)?(?:ошибк\w+|нарушени\w+|сбой)",
    r"\bмы\s+(?:\w+\s+){0,2}нарушили\s+(?:\w+\s+){0,2}(?:правил\w+|обязательств\w+|прав\w+|услови\w+)",
    r"\bмы\s+(?:\w+\s+){0,2}признаем\s+(?:\w+\s+){0,2}(?:вин\w+|ошибк\w+|нарушени\w+|ответственност\w+)",
    r"\bмы\s+(?:\w+\s+){0,2}(?:несем|берем|принимаем)\s+(?:\w+\s+){0,2}ответственност\w+",
    r"\b(?:вина|ответственность)\s+(?:\w+\s+){0,2}лежит\s+на\s+(?:нас|сервисе|foodflow|компании|курьер\w*)",
    # blaming a named party before any check: 'курьер виноват', 'сотрудник допустил ошибку'
    r"\b(?:курьер\w*|сотрудник\w*|ресторан\w*|партнер\w*|сервис|foodflow|компания)\s+(?:\w+\s+)?(?:виноват\w*|допустил\w*|ошибся|ошиблись|нарушил\w*)",
)

# ── Medical diagnosis, causal attribution and treatment advice ──────────────────────────────────
# Allowed: a neutral recommendation to seek professional medical help ("обратитесь к врачу",
# "обратитесь за профессиональной медицинской помощью", "вызовите скорую помощь").
_SYMPTOM_WORD = r"(?:симптом\w*|самочувстви\w+|недомоган\w+|тошнот\w*|рвот\w*|боли?|болью|состояни\w+\s+здоровья)"
_rule(
    "medical-diagnosis",
    # naming a specific condition at all (the prompt forbids naming diseases)
    r"\b(?:пищев\w+\s+(?:отравлени\w+|токсикоинфекци\w+)|отравлени\w+|отравил\w+|интоксикаци\w+|токсикоинфекци\w+|гастроэнтерит\w*|гастрит\w*|сальмонел\w+|ботулизм\w*|дизентери\w+|аппендицит\w*|анафилакс\w+|анафилактическ\w+|(?:кишечн|желудочн)\w+\s+(?:инфекци\w+|расстройств\w+)|аллергическ\w+\s+(?:реакци\w+|шок\w*))\b",
    # asserting a condition about the customer
    r"\bу\s+вас\s+(?:\w+\s+){0,2}(?:отравлени\w+|инфекци\w+|аллерги\w+|заболевани\w+|расстройств\w+|интоксикаци\w+)",
    r"\bвы\s+(?:\w+\s+)?(?:отравились|заболели|заразились)\b",
    r"\b(?:симптомы|состояние|самочувствие)\s+(?:\w+\s+){0,2}(?:указыва\w+|свидетельству\w+|говор\w+|объясня\w+)\s+(?:на|о)\b",
    r"\bэто\s+(?:похоже\s+на|признак\w*|симптом\w*)\s+(?:\w+\s+){0,2}(?:отравлени\w+|инфекци\w+|аллерги\w+|заболевани\w+)",
    # attributing the symptoms to the product
    rf"\b{_SYMPTOM_WORD}\s+{_GAP}(?:вызван\w*|из-за|вследствие)\b",
    rf"\b{_SYMPTOM_WORD}\s+{_GAP}связан\w*\s+(?:\w+\s+){{0,2}}(?:употреблени\w+|нашей\s+еды|наш\w+\s+продукт\w*)",
    r"\b(?:из-за|вследствие|в\s+результате)\s+(?:\w+\s+){0,3}(?:нашей\s+еды|нашего\s+продукт\w*|употреблени\w+\s+(?:\w+\s+){0,2}(?:продукт\w*|блюд\w*|еды))",
)
_MEDICATION = (
    r"(?:таблетк\w*|лекарств\w*|препарат\w*|сорбент\w*|(?:активированн\w+\s+)?уго(?:ль|ля|лем)|смект\w+|регидрон\w*|"
    r"парацетамол\w*|ибупрофен\w*|антибиотик\w*|антигистамин\w*|обезболивающ\w*|слабительн\w*|капельниц\w*|клизм\w*|"
    r"диет\w*|раствор\w*|больше\s+(?:воды|жидкости)|воду|жидкост\w+)"
)
_rule(
    "medical-treatment",
    rf"\b(?:примите|принимайте|принять|выпейте|выпить|пейте|пить|купите|используйте|нанесите|наносите|сделайте|делайте|соблюдайте)\s+{_GAP}{_MEDICATION}",
    r"\bпромо[йю]\w*\s+желуд\w+|\bвызов\w+\s+рвот\w+",
    r"\b\d+(?:[.,]\d+)?\s*(?:мг|мкг|мл|таблет\w*|капсул\w*|капел\w+|капл\w+)\b",
)

# ── Sensitive payment / authentication data ─────────────────────────────────────────────────────
# 'Не отправляйте CVV' is the safe wording and must keep passing; only the request is rejected.
_rule(
    "sensitive-data-request",
    rf"(?<!\bне\s)\b(?:пришлите|присылайте|отправьте|отправляйте|укажите|сообщите|назовите|введите|продиктуйте|предоставьте|передайте|напишите|скиньте)\s+{_GAP}(?:cvv2?|cvc2?|пин(?:-?код\w*)?|pin|полн\w+\s+номер\w*\s+(?:банковск\w+\s+)?карт\w+|номер\w*\s+банковск\w+\s+карт\w+|одноразов\w+\s+(?:код\w*|парол\w+)|смс[-\s]?код\w*|код\w*\s+(?:из\s+(?:смс|sms)|подтвержден\w+|безопасност\w+)|парол\w+|срок\w*\s+действия\s+карт\w+)",
)

# ── Markup that has no place in a plain-text customer reply ─────────────────────────────────────
_rule(
    "markup-in-draft",
    r"<\s*[/!?]?\s*[a-z]",
    r"\b(?:javascript|vbscript)\s*:",
    r"\bdata\s*:\s*(?:text|image|application)/",
    r"!?\[[^\]\n]{0,300}\]\([^)\n]{0,500}\)",
)

_COMPILED_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, _FLAGS), code) for pattern, code in _RULE_SPECS
)

VIOLATION_CODES: frozenset[str] = frozenset(code for _, code in _COMPILED_RULES)


def detect_draft_violations(text: str) -> list[str]:
    """Return the violation code of every rule the customer draft breaks (empty = acceptable)."""
    normalized = normalize_customer_text(text).strip()
    return [code for pattern, code in _COMPILED_RULES if pattern.search(normalized)]
