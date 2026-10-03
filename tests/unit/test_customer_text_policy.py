"""Customer-visible text policy (Stage 2C, H2): commitments, fault, medical advice, prompt contract."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from customer_claims_rag.application.customer_text_policy import (
    FUTURE_COMMITMENT_CODE,
    VIOLATION_CODES,
    detect_draft_violations,
    normalize_customer_text,
    strip_citation_markers,
)

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "system_prompt.md"


def _codes(text: str) -> set[str]:
    return set(detect_draft_violations(text))


# ---------------------------------------------------------------------------
# Unsupported commitments: refund / compensation / completed refund / guarantee / deadline
# ---------------------------------------------------------------------------

_REFUND_PROMISES = [
    "Мы вернем вам деньги после проверки.",
    "Мы вернём вам деньги.",
    "Мы обязательно вернём средства.",
    "Вам вернут деньги в ближайшее время.",
    "Мы произведём возврат средств.",
    "Мы оформим возврат.",
    "Вы получите полный возврат.",
]
_COMPENSATION_PROMISES = [
    "Мы выплатим компенсацию.",
    "Мы компенсируем неудобства.",
    "Компенсация будет выплачена на ваш счёт.",
    "Мы предоставим вам компенсацию.",
]
_COMPLETED_REFUNDS = [
    "Возврат уже оформлен.",
    "Возврат будет оформлен.",
    "Возврат средств будет произведён.",
    "Деньги были возвращены на карту.",
    "Деньги будут возвращены.",
    "Средства поступят на ваш счёт.",
    "Компенсация уже выплачена.",
    "Возврат одобрен.",
]
_GUARANTEES = [
    "Мы гарантируем возврат средств.",
    "Вам точно вернут деньги.",
    "Возврат гарантирован.",
]
_DEADLINES = [
    "Мы ответим в течение 24 часов.",
    "Мы рассмотрим обращение в течение 1 рабочего дня.",
    "Срок рассмотрения возврата составляет 5 рабочих дней.",
    "Мы свяжемся с вами в течение суток.",
    "Деньги поступят завтра.",
    "Ответ будет направлен в течение двух рабочих дней.",
]
_FALSE_ACTIONS = [
    "Ваше обращение зарегистрировано.",
    "Мы зарегистрировали ваше обращение.",
    "Запрос о возврате принят.",
    "Мы зафиксировали ваше обращение.",
    "Обращение принято в работу.",
    "Ваше обращение передано специалисту.",
    "Мы передали ваше обращение сотруднику.",
    "Обращение будет эскалировано.",
    "Сотрудник уже уведомлен.",
]


@pytest.mark.parametrize("text", _REFUND_PROMISES)
def test_refund_promise_is_rejected(text: str) -> None:
    assert "refund-promise" in _codes(text)


@pytest.mark.parametrize("text", _COMPENSATION_PROMISES)
def test_compensation_promise_is_rejected(text: str) -> None:
    assert "compensation-promise" in _codes(text)


@pytest.mark.parametrize("text", _COMPLETED_REFUNDS)
def test_completed_refund_is_rejected(text: str) -> None:
    assert "completed-refund" in _codes(text)


@pytest.mark.parametrize("text", _GUARANTEES)
def test_guaranteed_outcome_is_rejected(text: str) -> None:
    assert "guaranteed-outcome" in _codes(text)


@pytest.mark.parametrize("text", _DEADLINES)
def test_fixed_deadline_is_rejected(text: str) -> None:
    assert "deadline-commitment" in _codes(text)


@pytest.mark.parametrize("text", _FALSE_ACTIONS)
def test_registration_transfer_or_notification_claim_is_rejected(text: str) -> None:
    assert _codes(text) & {
        "false-registration",
        "false-completed-transfer",
        "false-escalation",
        "false-notification",
    }


# ---------------------------------------------------------------------------
# Future operational promises: the application reviews, answers and contacts nobody later
# ---------------------------------------------------------------------------

_FUTURE_PROMISES = [
    # first-person future review / check
    "Мы проверим обращение.",
    "Мы рассмотрим обращение.",
    "Мы приоритетно проверим обращение.",
    "Проверим обстоятельства заказа.",
    "Сверим описание обращения с данными по доставке.",
    "Мы будем проверять заказ.",
    # later notification / contact
    "После проверки сообщим результат.",
    "После проверки сообщим, возможен ли возврат средств.",
    "Мы сообщим после проверки.",
    "Мы свяжемся после проверки.",
    "О результатах сообщим отдельно.",
    "Если потребуется дополнительная информация, уточним её отдельно.",
    "Мы вернёмся с ответом.",
    "Будем на связи.",
    # passive-future form of the same promise
    "Обращение будет рассмотрено.",
    "Статус оплаты и доставки будет проверен.",
    "Проверка будет проводиться по правилам сервиса.",
    "Дальнейшее рассмотрение будет вестись по регламенту.",
    # registration / acceptance / transfer promised for later
    "Мы зарегистрируем ваше обращение.",
    "Мы передадим обращение сотруднику.",
    "Мы эскалируем обращение.",
    "Мы примем обращение в работу.",
]


@pytest.mark.parametrize("text", _FUTURE_PROMISES)
def test_future_operational_promise_is_rejected(text: str) -> None:
    assert FUTURE_COMMITMENT_CODE in _codes(text)


def test_future_operational_promise_cannot_hide_behind_invisible_characters() -> None:
    assert FUTURE_COMMITMENT_CODE in _codes("Мы про​верим обращение.")  # zero-width space
    assert FUTURE_COMMITMENT_CODE in _codes("Мы свя­жемся после проверки.")  # soft hyphen
    assert FUTURE_COMMITMENT_CODE in _codes("МЫ РАССМОТРИМ ОБРАЩЕНИЕ.")
    assert FUTURE_COMMITMENT_CODE in _codes("Мы разберём ситуацию.")  # ё folds to е


# ---------------------------------------------------------------------------
# Fault / liability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Это наша вина.",
        "Мы виноваты в произошедшем.",
        "Приносим извинения за нашу ошибку.",
        "Мы признаем свою вину.",
        "Мы допустили ошибку при сборке заказа.",
        "Курьер виноват в задержке.",
        "Произошло по нашей вине.",
        "Мы несём ответственность за случившееся.",
        "Компания нарушила ваши права.",
    ],
)
def test_admission_of_fault_is_rejected(text: str) -> None:
    assert "fault-admission" in _codes(text)


# ---------------------------------------------------------------------------
# Medical diagnosis / causal attribution / treatment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "У вас пищевое отравление.",
        "Похоже на отравление.",
        "Вероятно, это аллергическая реакция.",
        "Ваши симптомы вызваны употреблением продукта.",
        "Вы отравились.",
        "Симптомы указывают на инфекцию.",
        "Тошнота вызвана нашей едой.",
    ],
)
def test_diagnosis_is_rejected(text: str) -> None:
    assert "medical-diagnosis" in _codes(text)


@pytest.mark.parametrize(
    "text",
    [
        "Примите активированный уголь.",
        "Выпейте таблетку от тошноты.",
        "Пейте больше воды.",
        "Примите лекарство и отдохните.",
        "Принимайте 2 таблетки в день.",
        "Промойте желудок.",
        "Купите регидрон.",
    ],
)
def test_treatment_advice_is_rejected(text: str) -> None:
    assert "medical-treatment" in _codes(text)


# ---------------------------------------------------------------------------
# Sensitive-data requests and markup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Пришлите CVV.",
        "Укажите полный номер карты.",
        "Назовите код из смс.",
        "Отправьте пароль от аккаунта.",
    ],
)
def test_sensitive_data_request_is_rejected(text: str) -> None:
    assert "sensitive-data-request" in _codes(text)


@pytest.mark.parametrize(
    "text",
    [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        'Нажмите <a href="javascript:alert(1)">сюда</a>',
        "[сюда](javascript:alert(1))",
        "Смотрите [инструкцию](https://example.com/x)",
        "Нажмите javascript:alert(1)",
        "<SCRIPT>alert(1)</SCRIPT>",
        "＜script＞alert(1)＜/script＞",  # full-width brackets fold to ASCII under NFKC
    ],
)
def test_markup_in_draft_is_rejected(text: str) -> None:
    assert "markup-in-draft" in _codes(text)


# ---------------------------------------------------------------------------
# Neutral wording must keep passing
# ---------------------------------------------------------------------------

_NEUTRAL = [
    # facts the application knows and the need for review, with no promise of later action
    "Сообщение получено.",
    "Обращение требует проверки.",
    "Вопрос требует проверки.",
    "Обращение требует приоритетной проверки.",
    "Вопрос о возврате требует проверки.",
    "Возможность возврата зависит от результатов проверки.",
    "Возможность возврата будет определена после проверки каждого заказа.",
    "Статус и история доставки по вашему заказу требуют проверки.",
    # empathy and acknowledgement without admission
    "Сожалеем, что так вышло. Описанные обстоятельства требуют проверки.",
    "Приносим извинения за доставленные неудобства.",
    "Понимаем, что ситуация неприятна, и благодарим за обращение.",
    "Мы получили ваше сообщение.",
    "Сообщение получено. Чтобы проверить ситуацию, направьте номер заказа.",
    # cautious medical escalation
    "Если самочувствие ухудшилось, рекомендуем обратиться за профессиональной медицинской помощью.",
    "При ухудшении самочувствия обратитесь к врачу.",
    "При угрозе здоровью вызовите скорую помощь.",
    "Не употребляйте продукт и сохраните упаковку.",
    "По одному обращению невозможно установить причину симптомов.",
    # payment security: refusing to take sensitive data is the safe wording
    "Не отправляйте в чате CVV, PIN, полный номер карты и коды из SMS.",
    "Пожалуйста, не присылайте пароль.",
    "Укажите номер заказа, дату и сумму списания.",
    "Укажите маскированные последние цифры карты.",
    # customer-side windows and plain information
    "Просим сохранить упаковку в течение трёх дней.",
    "Сумма заказа 990 руб. указана верно.",
    "Больше информации поможет быстрее рассмотреть обращение.",
    "Сделайте фотографии упаковки в том виде, в котором получили заказ.",
]


@pytest.mark.parametrize("text", _NEUTRAL)
def test_neutral_wording_is_not_rejected(text: str) -> None:
    assert detect_draft_violations(text) == []


# ---------------------------------------------------------------------------
# Matching hygiene
# ---------------------------------------------------------------------------

def test_invisible_characters_do_not_evade_the_policy() -> None:
    assert "refund-promise" in _codes("Мы в​ернём вам деньги.")
    assert "fault-admission" in _codes("Это на­ша вина.")


def test_normalization_folds_yo_and_compatibility_forms() -> None:
    assert normalize_customer_text("Вернём") == "Вернем"
    assert normalize_customer_text("＜b＞") == "<b>"


def test_policy_exposes_every_rule_group() -> None:
    expected = {
        "refund-promise", "compensation-promise", "completed-refund", "guaranteed-outcome",
        "deadline-commitment", "false-registration", "false-completed-transfer",
        "false-escalation", "false-notification", "fault-admission", "medical-diagnosis",
        "medical-treatment", "sensitive-data-request", "markup-in-draft",
        FUTURE_COMMITMENT_CODE,
    }
    assert expected <= VIOLATION_CODES
    # The two narrow phrase rules the future-promise group replaced must not linger beside it.
    assert not {"vague-review-promise", "vague-result-promise"} & VIOLATION_CODES


def test_citation_markers_are_stripped_before_customer_text() -> None:
    assert strip_citation_markers("Проверим заказ [S1]. Затем ответим [S2].") == (
        "Проверим заказ. Затем ответим."
    )


@pytest.mark.parametrize(
    ("unit", "repeat"),
    [
        ("слово ", 4000),
        ("мы ", 7000),
        ("возврат ", 2500),
        ("наш ", 5000),
        ("укажите ", 2500),
        ("< ", 10000),
        ("[a", 10000),
        ("в течение 5 ", 1700),
        ("симптомы ", 2200),
        ("обращение ", 4000),
        ("будет ", 4000),
        ("проверка будет ", 2500),
        ("будем ", 4000),
    ],
    ids=[
        "words", "we", "refund", "our", "specify", "angle", "bracket", "period", "symptoms",
        "matter", "will-be", "check-will-be", "we-will",
    ],
)
def test_policy_is_bounded_on_adversarial_text(unit: str, repeat: int) -> None:
    text = unit * repeat
    started = time.perf_counter()
    detect_draft_violations(text)
    assert time.perf_counter() - started < 2.0


# ---------------------------------------------------------------------------
# Prompt / policy contract: the prompt must not prescribe wording the policy rejects
# ---------------------------------------------------------------------------

# Representative safe wording quoted from prompts/system_prompt.md. Each phrase must still occur in
# the prompt (so editing the prompt forces a revisit of this contract) and must pass the policy.
_PROMPT_PRESCRIBED = [
    "сожалеем, что так вышло",
    "просим указать номер заказа",
    "сообщение получено",
    "обращение требует проверки",
    "вопрос требует проверки",
    "возможность решения зависит от результатов проверки",
    "вопрос о возврате требует проверки",
    "вопрос о компенсации требует проверки",
    "возможность возврата зависит от результатов проверки",
    "если самочувствие ухудшилось, рекомендуем обратиться за профессиональной медицинской помощью",
]

# Wording the prompt lists as forbidden: the policy must enforce the same prohibition.
_PROMPT_FORBIDDEN = [
    ("обращение передано", {"false-completed-transfer"}),
    ("Ваше обращение было передано", {"false-completed-transfer"}),
    ("обращение будет эскалировано", {"false-escalation"}),
    ("мы проверим", {FUTURE_COMMITMENT_CODE}),
    ("мы рассмотрим", {FUTURE_COMMITMENT_CODE}),
    ("после проверки сообщим", {FUTURE_COMMITMENT_CODE}),
    ("мы сообщим после проверки", {FUTURE_COMMITMENT_CODE}),
    ("мы свяжемся после проверки", {FUTURE_COMMITMENT_CODE}),
    ("обращение будет рассмотрено", {FUTURE_COMMITMENT_CODE}),
    ("запрос отправлен", {"false-completed-transfer", "false-specialist-transfer"}),
    ("обращение зарегистрировано", {"false-registration"}),
    ("обращение принято в работу", {"false-registration"}),
    ("мы зафиксировали обращение", {"false-registration"}),
    ("сотрудник уже уведомлён", {"false-notification"}),
    ("мы вернём вам деньги", {"refund-promise"}),
    ("возврат будет оформлен", {"completed-refund"}),
    ("мы выплатим компенсацию", {"compensation-promise"}),
    ("вам точно вернут деньги", {"guaranteed-outcome", "refund-promise"}),
    ("ответим в течение 24 часов", {"deadline-commitment"}),
    ("деньги поступят завтра", {"deadline-commitment", "completed-refund"}),
    ("это наша вина", {"fault-admission"}),
    ("мы виноваты", {"fault-admission"}),
    ("курьер виноват", {"fault-admission"}),
    ("у вас пищевое отравление", {"medical-diagnosis"}),
    ("примите таблетку", {"medical-treatment"}),
]


@pytest.fixture(scope="module")
def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("phrase", _PROMPT_PRESCRIBED)
def test_prompt_prescribed_wording_passes_the_policy(phrase: str, prompt_text: str) -> None:
    assert phrase in prompt_text, "prompt no longer prescribes this wording: update the contract"
    sentence = phrase[0].upper() + phrase[1:] + "."
    assert detect_draft_violations(sentence) == []


@pytest.mark.parametrize(("phrase", "expected"), _PROMPT_FORBIDDEN)
def test_prompt_forbidden_wording_is_rejected_by_the_policy(
    phrase: str,
    expected: set[str],
    prompt_text: str,
) -> None:
    assert phrase.lower() in prompt_text, "prompt no longer forbids this wording: update the contract"
    sentence = phrase[0].upper() + phrase[1:] + "."
    assert _codes(sentence) & expected


def test_prompt_prescribes_no_future_promise_anywhere_outside_its_forbidden_list() -> None:
    """The rule applies to every sentence of the prompt, not just to the phrases pinned above.

    The only place the promise wording may appear is the section that lists what is forbidden.
    """
    text = PROMPT_PATH.read_text(encoding="utf-8")
    sections = re.split(r"^#{2,3} ", text, flags=re.MULTILINE)
    offenders = []
    for section in sections:
        title, _, body = section.partition("\n")
        if title.startswith("Что запрещено"):
            continue
        for line in body.splitlines():
            for sentence in re.split(r"(?<=[.!?;])\s+", line):
                if FUTURE_COMMITMENT_CODE in _codes(sentence):
                    offenders.append((title, sentence.strip()[:120]))
    assert offenders == []


def test_prompt_no_longer_prescribes_wording_the_policy_rejects(prompt_text: str) -> None:
    # Stage 2C reconciliation was done on the prompt side, not by weakening the policy: the
    # phrasings the prompt used to prescribe are still rejected (circular support reference /
    # 'требует проверки сотрудником' operator instruction) and are no longer in the prompt.
    stale = [
        "служба поддержки должна проверить возможность возврата",
        "сотруднику необходимо проверить",
        "вопрос требует проверки сотрудником",
    ]
    for phrase in stale:
        assert phrase not in prompt_text
    assert detect_draft_violations("Служба поддержки должна проверить возможность возврата.") != []
    assert detect_draft_violations("Вопрос требует проверки сотрудником.") != []
