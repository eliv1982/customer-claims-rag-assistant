"""The canonical corpus must not contradict what the application does.

The application classifies, retrieves, and drafts a customer text for staff to review. It does not
register, accept, transfer, escalate or notify anything, so the customer-output policy
(``application/customer_text_policy.py``, enforced on model text) forbids a draft from claiming
any of that, and the system prompt lists the same phrases as forbidden. The knowledge base is the
model's source material: wording in it that *is* such a claim, or that tells the assistant it may
say so, pulls drafts towards text the policy then has to replace.

Deterministic checks over the documents the canonical manifest selects (documents outside the
corpus are not part of the product and are not checked until they are included):

1. **Customer-facing wording** (the response templates, the safe alternatives and the modular
   phrases of ``09_response_style_and_templates``) must pass the same policy gate a model draft
   passes, and must not name a registered / accepted / recorded / transferred state at all.
2. **Permissions**: no sentence may allow the assistant to tell the customer that the case was
   registered, accepted, recorded or transferred, unless the same sentence forbids exactly that.
3. **Future operational promises**: no sentence of any document may promise that the application
   will review, answer, get in touch, register or transfer something later ("мы проверим",
   "после проверки сообщим", "мы свяжемся после проверки", "обращение будет рассмотрено"), and
   wording a permission sentence quotes as acceptable must pass the runtime gate. This check does
   not keep its own phrase list: it applies ``FUTURE_COMMITMENT_CODE`` of the runtime customer
   policy to every sentence, so the corpus and the model-output policy cannot drift apart. A
   company-policy fact such as "срок рассмотрения - до 5 рабочих дней" is not a promise the
   application makes and stays in the knowledge base.

Process descriptions in the staff voice ("the support team registers the case", "a high-risk case
is handed to an employee") are not claims the draft makes and are not touched; the knowledge base
states once, in ``01_service_overview`` and ``07_complaint_handling_procedure``, that the
application performs none of those steps.

The detectors are run against the frozen historical corpus too, which contains the old wording:
a detector that found nothing there would prove nothing about the current corpus.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from customer_claims_rag.application.customer_text_policy import (
    FUTURE_COMMITMENT_CODE,
    VIOLATION_CODES,
    detect_draft_violations,
)
from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    load_canonical_corpus_manifest,
    normalized_source_text,
)

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "experiments" / "corpus" / "historical_pre_2d2_15doc_v1"
TEMPLATES_DOCUMENT = "09_response_style_and_templates"

# A case or message said to be registered / accepted / recorded / transferred / escalated.
STATUS_CLAIM = re.compile(
    r"(?:зарегистрирова\w+|принят\w*|зафиксирова\w+|передан\w*|направлен\w*|эскалирован\w*"
    r"|взят\w*\s+в\s+работу)",
    re.IGNORECASE,
)
# A sentence that allows or recommends saying something to the customer.
PERMISSION = re.compile(
    r"(?:допустим\w*"
    r"|можно\s+(?:прямо\s+)?(?:сказать|сообщить|написать)"
    r"|может\s+(?:только\s+)?(?:сказать|сообщить|написать|подтвердить)"
    r"|корректн\w+\s+формулировк\w*"
    r"|верный\s+ответ"
    r"|сообщает\s+(?:только\s+)?о\b"
    r"|сообщают,?\s+что"
    r"|подтвердить\s+принятие)",
    re.IGNORECASE,
)
# The same sentence forbids it, so the status words are being quoted as the thing not to say.
NEGATION = re.compile(
    r"(?:недопустим|нельзя|запрещ|не\s+утвержда\w+"
    r"|\bне\s+(?:\w+\s+){0,2}(?:может|вправе|должен|следует|означает|утверждает|выполняет|сообщает|обещает|равн))",
    re.IGNORECASE,
)


def customer_wording_blocks(text: str) -> list[tuple[str, str]]:
    """The passages of the templates document that are meant to be copied to a customer."""
    blocks: list[tuple[str, str]] = []
    for match in re.finditer(r"\*\*Шаблон\*\*\n\n(.+?)(?:\n\n|\Z)", text, re.DOTALL):
        blocks.append(("template", match.group(1).strip()))
    for match in re.finditer(r"\*\*ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО:\*\*\s*«(.+?)»", text):
        blocks.append(("safe-alternative", match.group(1).strip()))
    for match in re.finditer(r'^- "(.+?)"\s*$', text, re.MULTILINE):
        blocks.append(("module", match.group(1).strip()))
    return blocks


def _as_customer_text(block: str) -> str:
    return re.sub(r"\{[a-z_]+\}", "X", block).replace("`", "")


def wording_findings(text: str) -> list[str]:
    findings = []
    for kind, block in customer_wording_blocks(text):
        plain = _as_customer_text(block)
        violations = detect_draft_violations(plain)
        if violations:
            findings.append(f"{kind} breaks the output policy {violations}: {block[:80]}")
        if STATUS_CLAIM.search(plain):
            findings.append(f"{kind} names a registered/accepted/transferred state: {block[:80]}")
    return findings


def permission_findings(text: str) -> list[str]:
    findings = []
    for paragraph in text.split("\n"):
        for sentence in re.split(r"(?<=[.!?;])\s+", paragraph):
            if STATUS_CLAIM.search(sentence) and PERMISSION.search(sentence) and not NEGATION.search(sentence):
                findings.append(sentence.strip()[:160])
    return findings


def _sentences(text: str):
    for paragraph in text.split("\n"):
        yield from re.split(r"(?<=[.!?;])\s+", paragraph)


QUOTED_WORDING = re.compile(r"«(.+?)»|\"(.+?)\"")


def future_promise_findings(text: str) -> list[str]:
    """Sentences that promise later review / reply / contact / registration / transfer.

    The test of what counts is the runtime customer-output policy itself. A sentence that forbids
    the wording ("нельзя обещать: «мы проверим»", the НЕДОПУСТИМО column) is quoting it, not saying it.
    """
    return [
        sentence.strip()[:160]
        for sentence in _sentences(text)
        if FUTURE_COMMITMENT_CODE in detect_draft_violations(sentence) and not NEGATION.search(sentence)
    ]


def permitted_wording_findings(text: str) -> list[str]:
    """Wording a permission sentence quotes as acceptable must pass the whole runtime gate."""
    findings = []
    for sentence in _sentences(text):
        if PERMISSION.search(sentence) and not NEGATION.search(sentence):
            for quoted in QUOTED_WORDING.findall(sentence):
                wording = next(part for part in quoted if part)
                violations = detect_draft_violations(wording)
                if violations:
                    findings.append(f"{violations}: {wording[:80]}")
    return findings


def _read(directory: Path, document_id: str) -> str:
    return normalized_source_text((directory / f"{document_id}.md").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus_documents() -> dict[str, str]:
    manifest = load_canonical_corpus_manifest(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    return {
        document_id: _read(ROOT / manifest.source_dir, document_id)
        for document_id in manifest.document_ids
    }


def test_the_check_covers_exactly_the_canonical_documents(corpus_documents: dict[str, str]) -> None:
    assert len(corpus_documents) == 10
    assert TEMPLATES_DOCUMENT in corpus_documents
    assert not any(document_id[:2] in {"11", "12", "13", "14", "15"} for document_id in corpus_documents)


def test_the_templates_document_still_yields_its_customer_wording(corpus_documents: dict[str, str]) -> None:
    """Guards the extraction itself: 13 templates, 12 safe alternatives, 14 modular phrases."""
    blocks = customer_wording_blocks(corpus_documents[TEMPLATES_DOCUMENT])
    counts = {kind: sum(1 for k, _ in blocks if k == kind) for kind in ("template", "safe-alternative", "module")}
    assert counts == {"template": 13, "safe-alternative": 12, "module": 14}


def test_customer_wording_passes_the_output_policy_and_claims_no_status(corpus_documents: dict[str, str]) -> None:
    assert wording_findings(corpus_documents[TEMPLATES_DOCUMENT]) == []


def test_no_document_allows_the_assistant_to_claim_a_status(corpus_documents: dict[str, str]) -> None:
    findings = {
        document_id: permission_findings(text)
        for document_id, text in corpus_documents.items()
        if permission_findings(text)
    }
    assert findings == {}


def test_the_knowledge_base_states_that_the_application_registers_and_transfers_nothing(
    corpus_documents: dict[str, str],
) -> None:
    assert (
        "регистрацию обращения, принятие в работу, передачу сотруднику и уведомления выполняет "
        "персонал FoodFlow вне приложения"
    ) in corpus_documents["01_service_overview"]
    assert (
        "Регистрацию обращения и его передачу сотруднику выполняет персонал FoodFlow; "
        "приложение этого не выполняет"
    ) in corpus_documents["07_complaint_handling_procedure"]


def test_no_document_promises_a_future_operational_action(corpus_documents: dict[str, str]) -> None:
    findings = {
        document_id: future_promise_findings(text)
        for document_id, text in corpus_documents.items()
        if future_promise_findings(text)
    }
    assert findings == {}


def test_wording_the_corpus_permits_passes_the_runtime_gate(corpus_documents: dict[str, str]) -> None:
    findings = {
        document_id: permitted_wording_findings(text)
        for document_id, text in corpus_documents.items()
        if permitted_wording_findings(text)
    }
    assert findings == {}


def test_corpus_scan_follows_the_runtime_rule_and_keeps_no_phrase_list_of_its_own(monkeypatch) -> None:
    """Remove the rule from the runtime policy and the corpus scan stops flagging: it has no copy."""
    from customer_claims_rag.application import customer_text_policy as policy

    assert FUTURE_COMMITMENT_CODE in VIOLATION_CODES
    promise = "Мы проверим обращение после получения данных."
    assert future_promise_findings(promise)
    assert wording_findings(f"**Шаблон**\n\n{promise}\n\n")

    monkeypatch.setattr(
        policy,
        "_COMPILED_RULES",
        tuple(rule for rule in policy._COMPILED_RULES if rule[1] != FUTURE_COMMITMENT_CODE),
    )
    assert future_promise_findings(promise) == []
    assert wording_findings(f"**Шаблон**\n\n{promise}\n\n") == []


def test_the_corpus_keeps_its_company_policy_facts_about_timing(corpus_documents: dict[str, str]) -> None:
    """Review periods are policy facts of the company; they are not the assistant's promises.

    The runtime policy still stops a draft from stating a deadline (``deadline-commitment``), but the
    knowledge base the model reads must keep describing the 1 / 5 / 10 working-day orientation.
    """
    facts = {
        "04_refund_policy": "Для стандартных случаев ориентир составляет до 5 рабочих дней",
        "07_complaint_handling_procedure": "Стандартное внутреннее рассмотрение занимает до пяти рабочих дней",
        "10_customer_faq": "стандартный кейс — до **5 рабочих дней**, сложный — до **10 рабочих дней**",
    }
    for document_id, fact in facts.items():
        assert fact in corpus_documents[document_id], f"{document_id} lost the policy fact: {fact!r}"
        assert future_promise_findings(fact) == []


def test_kb_prompt_and_runtime_policy_agree_on_the_future_promise_rule(corpus_documents: dict[str, str]) -> None:
    """The three surfaces say the same thing: the KB table and the prompt forbid it, the policy rejects it."""
    templates = corpus_documents[TEMPLATES_DOCUMENT]
    row = next(line for line in templates.splitlines() if "«Мы проверим обращение»" in line)
    forbidden, _reason, safe = (cell.strip() for cell in row.strip().strip("|").split("|"))
    assert forbidden.startswith("**НЕДОПУСТИМО:**")
    for quoted in re.findall(r"«(.+?)»", forbidden):
        assert FUTURE_COMMITMENT_CODE in detect_draft_violations(quoted), quoted
    safe_wording = re.search(r"«(.+?)»", safe)
    assert safe_wording and detect_draft_violations(safe_wording.group(1)) == []

    prompt = (ROOT / "prompts" / "system_prompt.md").read_text(encoding="utf-8").lower()
    for promise in ("мы проверим", "после проверки сообщим", "мы свяжемся после проверки"):
        assert promise in prompt
        assert FUTURE_COMMITMENT_CODE in detect_draft_violations(promise.capitalize() + ".")
    assert detect_draft_violations("Обращение требует проверки.") == []


# --- the detectors are not vacuous: the historical corpus has the old wording ------------------


def test_the_detectors_find_the_wording_the_historical_corpus_had() -> None:
    wording = wording_findings(_read(HISTORICAL, TEMPLATES_DOCUMENT))
    assert len(wording) >= 15, wording
    assert any("Мы зафиксировали ваше обращение и передали его на проверку" in finding for finding in wording)
    permissions = [
        finding
        for document_id in (
            "01_service_overview",
            "03_order_changes_and_cancellations",
            "06_food_quality_and_packaging",
            "08_escalation_and_risk_rules",
            "10_customer_faq",
        )
        for finding in permission_findings(_read(HISTORICAL, document_id))
    ]
    assert len(permissions) >= 10, permissions


def test_the_future_promise_scan_finds_the_wording_the_historical_corpus_had() -> None:
    promises = [
        sentence
        for document_id in (
            "01_service_overview",
            "04_refund_policy",
            "06_food_quality_and_packaging",
            "08_escalation_and_risk_rules",
            "09_response_style_and_templates",
        )
        for sentence in future_promise_findings(_read(HISTORICAL, document_id))
    ]
    assert len(promises) >= 15, promises
    assert any("Мы проверим данные по доставке" in sentence for sentence in promises)
    assert any("будет проверен" in sentence for sentence in promises)


@pytest.mark.parametrize(
    "sentence",
    [
        "Допустимо писать, что обращение зарегистрировано.",
        "Ассистент может сказать, что запрос принят и передан на проверку.",
        "Корректная формулировка: обращение принято к обработке.",
        "Верный ответ здесь: запрос принят к проверке.",
        "Клиенту сообщают, что обращение принято.",
        "Допустимо: «обращение передано сотруднику для проверки».",
    ],
)
def test_permission_detector_flags_each_claim_shape(sentence: str) -> None:
    assert permission_findings(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Недопустимо писать, что обращение зарегистрировано.",
        "Ассистент может сказать, что запрос требует проверки, не утверждая, что он принят.",
        "Нельзя сообщать, что обращение передано сотруднику.",
        "Регистрация обращения означает, что сообщение принято сотрудником.",
        "Допустимо писать, что сообщение получено и обращение требует проверки.",
    ],
)
def test_permission_detector_accepts_prohibitions_and_neutral_wording(sentence: str) -> None:
    assert permission_findings(sentence) == []


@pytest.mark.parametrize(
    "template",
    [
        "Запрос по возврату средств по заказу {order_number} принят.",
        "Сообщение по заказу {order_number} зафиксировали.",
        "Обращение будет передано по приоритетному маршруту.",
        "Мы зафиксировали ваше обращение и передали его на проверку.",
    ],
)
def test_wording_check_flags_a_template_that_claims_a_status(template: str) -> None:
    text = f"**Шаблон**\n\n{template}\n\n"
    assert wording_findings(text)


# --- representative variants of the future-promise concept ----------------------------------------
# Not an exhaustive phrase list: each variant is one shape of "the application will do something
# later". What decides is the runtime policy's verdict; these cases prove the corpus checks apply
# it on every surface the corpus has (template, safe alternative, modular phrase, prose, permission).

FUTURE_PROMISE_VARIANTS = [
    "Мы проверим обращение.",
    "Мы рассмотрим обращение.",
    "Мы приоритетно проверим обращение.",
    "Проверим обстоятельства заказа.",
    "После проверки сообщим результат.",
    "После проверки мы сообщим, возможен ли возврат.",
    "Мы сообщим после проверки.",
    "Мы свяжемся после проверки.",
    "Мы свяжемся с вами.",
    "О результатах сообщим отдельно.",
    "Обращение будет рассмотрено.",
    "Проверка будет проводиться по правилам.",
    "Мы зарегистрируем обращение.",
    "Мы передадим обращение сотруднику.",
]

ACCEPTED_NEUTRAL_WORDING = [
    "Сообщение получено.",
    "Обращение требует проверки.",
    "Вопрос требует проверки.",
    "Возможность возврата зависит от результатов проверки.",
]


def _every_surface(wording: str) -> dict[str, list[str]]:
    """Run one wording through each corpus check, placed where the corpus puts such wording."""
    unquoted = wording.rstrip(".")
    return {
        "template": wording_findings(f"**Шаблон**\n\n{wording}\n\n"),
        "safe-alternative": wording_findings(f"| **ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО:** «{unquoted}». |"),
        "module": wording_findings(f'- "{wording}"\n'),
        "prose": future_promise_findings(f"Ассистент отвечает клиенту так: {wording}"),
        "permission": permitted_wording_findings(f"Допустимо писать: «{wording}»"),
    }


@pytest.mark.parametrize("wording", FUTURE_PROMISE_VARIANTS)
def test_every_corpus_surface_rejects_a_future_promise(wording: str) -> None:
    for surface, findings in _every_surface(wording).items():
        assert findings, f"{surface} accepted {wording!r}"


def test_template_findings_name_the_future_promise_rule() -> None:
    findings = wording_findings("**Шаблон**\n\nПосле проверки сообщим результат.\n\n")
    assert any(FUTURE_COMMITMENT_CODE in finding for finding in findings), findings


@pytest.mark.parametrize("wording", ACCEPTED_NEUTRAL_WORDING)
def test_every_corpus_surface_accepts_the_wording_that_states_facts_and_needs(wording: str) -> None:
    for surface, findings in _every_surface(wording).items():
        assert findings == [], f"{surface} rejected {wording!r}: {findings}"
    assert permission_findings(wording) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "Сотрудник проверит факты и при необходимости примет решение по процедуре.",
        "Срок рассмотрения по политике компании - до 5 рабочих дней.",
        "Стандартное внутреннее рассмотрение занимает до пяти рабочих дней.",
        "Нельзя обещать: «мы проверим обращение», «после проверки сообщим».",
        "| **НЕДОПУСТИМО:** «Мы свяжемся с вами». | Обещание связи. | **ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО:** «Обращение требует проверки». |",
    ],
)
def test_staff_process_policy_facts_and_prohibitions_are_not_future_promises(sentence: str) -> None:
    assert future_promise_findings(sentence) == []
