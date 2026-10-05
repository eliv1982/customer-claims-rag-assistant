"""Stage 2I: lexical boundaries, body-versus-product temperature, and the foreign-object tiers.

Three deterministic-rule defects the frozen cases exposed:

* a health term matched *inside* an unrelated word ('протекает' contains 'отек') and the bare word
  'температура' escalated every food-temperature complaint to a critical health case (T026);
* hair in a salad (T039) matched no rule although the policy documents (06, FAQ-36, H-03) make it a
  HIGH quality complaint, and a piece of metal ('кусок металла') matched nothing either;
* 'безопасный' was read as 'опасный' and 'неделе' / 'супермаркет' as food, so unrelated messages
  raised the critical dangerous-object rule.

The cases below pin the intended distinctions in both directions: what must stay critical, what
becomes HIGH, and what must not escalate at all.
"""

from __future__ import annotations

import pytest

from customer_claims_rag.application.claim_guidance import (
    CATEGORY_FOREIGN_OBJECT,
    REASON_CODE_CATEGORIES,
    derive_claim_category,
)
from customer_claims_rag.application.customer_templates import HARD_TEMPLATE_CATEGORIES
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.models import DeterministicRiskResult, RiskAssessmentRequest, RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode

HEALTH = RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION
DANGEROUS = RiskReasonCode.DANGEROUS_FOREIGN_OBJECT
UNCONFIRMED = RiskReasonCode.UNCONFIRMED_FOREIGN_OBJECT

# Frozen cases, verbatim (tests/01_test_questions.md).
T026 = "FF-33221, коробка помята сбоку, но контейнер закрыт, не протекает. Все равно требую возврат за блюдо."
T039 = "FF-30303, в салате волос. Фото есть. Это отвратительно — что дальше?"
T053 = "Третья доставка подряд по моему плану FF-PLAN-99 с ошибками в составе — каждый раз не то блюдо."


def _assess(query: str) -> DeterministicRiskResult:
    return assess_deterministic_risk(RiskAssessmentRequest(customer_query=query))


def _is_health(query: str) -> bool:
    return HEALTH in _assess(query).reason_codes


# ---------------------------------------------------------------------------
# Health lexicon: terms start at a word start
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # 'протекает' / 'протек' / 'потекло' contain 'отек'; each query has a consumption link
        "Заказ пришел, пакет протекает.",
        "Заказ доставили, контейнер протек, суп разлился.",
        "Из заказа потекло масло.",
        # 'россыпью' contains 'сыпь', 'безудержно' contains 'зуд'
        "Заказ привезли, все лежит россыпью.",
        "Заказ привезли, курьер безудержно ругался.",
    ],
)
def test_health_terms_do_not_match_inside_unrelated_words(query: str) -> None:
    assert not _is_health(query)


def test_a_word_inside_another_word_neither_is_a_symptom_nor_makes_the_consumption_link() -> None:
    # the symptom lexicon alone: the link here comes from 'после', not from the leak
    assert not _is_health("Заказ пришел, после доставки пакет протекает.")
    # the link lexicon alone: 'протекает' must not turn an order mention into a consumption link
    # for an unrelated stated weakness
    assert not _is_health("Заказ пришел, пакет протекает. У меня слабость.")
    # a real symptom next to a leak is still a health case
    assert _is_health("После заказа у меня слабость, а еще пакет протекает.")


@pytest.mark.parametrize(
    "query",
    [
        "После еды отек лица.",
        "После заказа отекло лицо.",
        "Поел ваш суп — сыпь по всему телу.",
        "После еды захрипел.",
        "Съел блюдо, зуд и крапивница.",
        "После заказа у ребенка отёк губ.",
    ],
)
def test_genuine_symptoms_still_escalate_after_the_boundary_fix(query: str) -> None:
    result = _assess(query)
    assert HEALTH in result.reason_codes
    assert result.risk_floor is RiskLevel.CRITICAL and result.priority_handoff


def test_t026_a_closed_leak_free_container_is_a_refund_request_not_a_health_case() -> None:
    result = _assess(T026)
    assert result.risk_floor is RiskLevel.MEDIUM
    assert result.reason_codes == (RiskReasonCode.REFUND_REQUEST,)


# ---------------------------------------------------------------------------
# Temperature: a body or a product
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Доставка приехала, блюдо было при низкой температуре.",
        "Заказ доставили, суп был при комнатной температуре, а не горячий.",
        "Доставка пришла, температура блюда ниже ожидаемой.",
        "Доставили еду, температура в контейнере была низкая.",
        "Заказ привезли в мороз, при низкой температуре на улице блюдо остыло.",
        "Заказ получил, температурный режим доставки нарушен.",
        "Еда пришла холодной, жалоба на температуру.",
        "Заказ доставили, блюдо хранили при высокой температуре в сумке.",
        "Заказ получил, сомнительная температура блюда.",
        "Заказ получил, сомнительная температура.",
        "Заказ пришел, суп комнатной температуры.",
        "Курьер довез заказ, но термосумка не держит температуру.",
        "Заказ пришел, на улице была низкая температура и суп остыл.",
        "Заказ доставили, температура блюда 40 градусов.",
        "Доставили еду, температура хранения 4 градуса.",
    ],
)
def test_product_and_environment_temperature_is_not_a_health_symptom(query: str) -> None:
    # 'Съел блюдо.' guarantees a consumption link, so only the temperature wording decides
    result = _assess(f"{query} Съел блюдо.")
    assert HEALTH not in result.reason_codes
    assert result.assessment_status is RiskAssessmentStatus.NO_SIGNAL


def test_the_appended_consumption_link_alone_is_not_a_health_signal() -> None:
    assert _assess("Заказ доставили. Съел блюдо.").assessment_status is RiskAssessmentStatus.NO_SIGNAL
    # ... and with a bare temperature it is: the link is real, so the masks above are what decide
    assert _is_health("Заказ доставили, температура. Съел блюдо.")


@pytest.mark.parametrize(
    "query",
    [
        "После вашей еды у меня поднялась температура.",
        "После заказа у ребенка температура 39.",
        "Поел ваш суп, высокая температура и озноб.",
        "После обеда из вашего заказа температура 38,5.",
        "После еды у мужа повышенная температура.",
        "Съел блюдо, при температуре 38 меня тошнило.",
        "После еды температура поднялась до 39.",
        "Заказ доставили, после блюда температура под 40.",
        "Съел блюдо, температура была 37,5.",
        "Съел суп, у сына жар и температура.",
        "Поел, в машине поднялась температура.",
        # 'сохраняется' is a fever that persists, not a failure to keep a temperature
        "После еды сохраняется высокая температура.",
        # a bare mention is neither: the floor errs towards escalation when it cannot tell
        "После еды температура.",
        "Поел блюдо, лежу с температурой.",
    ],
)
def test_body_temperature_after_consumption_still_escalates(query: str) -> None:
    result = _assess(query)
    assert HEALTH in result.reason_codes
    assert result.risk_floor is RiskLevel.CRITICAL and result.priority_handoff


@pytest.mark.parametrize(
    "query",
    [
        "Блюдо хранилось при низкой температуре, а после еды у ребенка температура 39.",
        "Заказ пришел при комнатной температуре, после еды у меня поднялась температура.",
        # one mention that fits both: a person's temperature beats the 'in the frost' frame around it
        "Заказ пришел, после еды в мороз у меня температура.",
    ],
)
def test_a_product_temperature_never_cancels_a_body_temperature(query: str) -> None:
    assert _is_health(query)


def test_temperature_without_a_consumption_link_is_not_assessed_as_health() -> None:
    assert _assess("Какая сегодня температура на улице?").assessment_status is RiskAssessmentStatus.NO_SIGNAL


# ---------------------------------------------------------------------------
# Foreign objects: critical when dangerous, HIGH when merely unconfirmed, nothing otherwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        T039,
        "В супе нашла волос.",
        "В блюде длинный волос, заказ FF-1.",
        "Волосы в еде, ужас.",
        "В моём салате был волос.",
        "Нашёл волос в пасте.",
        "Из супа вытащил волосок.",
        "В салате посторонний предмет.",
        "В еде нашёл инородное тело.",
        "В заказе инородный предмет, фото есть.",
    ],
)
def test_hair_and_unconfirmed_foreign_objects_are_a_high_quality_complaint(query: str) -> None:
    result = _assess(query)
    assert result.risk_floor is RiskLevel.HIGH
    assert result.reason_codes == (UNCONFIRMED,)
    assert result.handoff_required and not result.priority_handoff
    assert DANGEROUS not in result.reason_codes


@pytest.mark.parametrize(
    "query",
    [
        "В салате стекло.",
        "В еде игла.",
        "В блюде острый металлический осколок.",
        "В супе кусок стекла.",
        "В еде обнаружен острый предмет.",
        "В еде кусок металла.",
        "В еде частицы металла.",
        "В салате металлический предмет.",
        "В супе металлическая стружка.",
        "В супе острый осколок.",
        "В еде фрагмент упаковки, острый.",
    ],
)
def test_dangerous_objects_stay_critical_and_do_not_also_raise_the_unconfirmed_tier(query: str) -> None:
    result = _assess(query)
    assert result.risk_floor is RiskLevel.CRITICAL and result.priority_handoff
    assert result.reason_codes == (DANGEROUS,)


def test_a_dangerous_object_and_hair_in_one_message_keep_both_signals_at_the_maximum() -> None:
    result = _assess("В салате стекло. А еще в супе был волос.")
    assert result.risk_floor is RiskLevel.CRITICAL
    assert result.reason_codes == (DANGEROUS, UNCONFIRMED)


@pytest.mark.parametrize(
    "query",
    [
        # hair that is not in the food
        "Курьер с длинными волосами вручил заказ.",
        "В супермаркете нашёл волос.",
        # a hypothetical is a question, not a complaint
        "Что будет, если я найду волос в салате?",
        # ordinary quality and appearance complaints are not foreign objects
        "Салат невкусный, выглядит несвежим.",
        "Блюдо пересолено, соус растёкся.",
        "В заказе нет салфеток.",
        # a metallic taste is not a piece of metal
        "В салате металлический привкус.",
        "Привкус металла в супе.",
        "Вкус металла в блюде.",
    ],
)
def test_look_alikes_and_ordinary_complaints_raise_no_foreign_object_signal(query: str) -> None:
    result = _assess(query)
    assert DANGEROUS not in result.reason_codes
    assert UNCONFIRMED not in result.reason_codes


@pytest.mark.parametrize(
    "query",
    [
        # 'неделе' contains 'еде' and is not food
        "На прошлой неделе у меня разбилось стекло телефона.",
        # 'безопасный' contains 'опасн' and means the opposite
        "В еде фрагмент упаковки, но он совершенно безопасный.",
    ],
)
def test_the_dangerous_object_rule_respects_word_boundaries(query: str) -> None:
    assert DANGEROUS not in _assess(query).reason_codes


def test_t039_hair_in_a_salad_is_high_with_a_staff_handoff_and_no_priority() -> None:
    result = _assess(T039)
    assert result.assessment_status is RiskAssessmentStatus.RULE_MATCH
    assert result.risk_floor is RiskLevel.HIGH
    assert result.handoff_required and not result.priority_handoff
    assert [signal.rule_id for signal in result.risk_signals] == ["unconfirmed_foreign_object"]


def test_unconfirmed_foreign_object_uses_the_existing_quality_category_not_the_hazard_template() -> None:
    result = _assess(T039)
    category = derive_claim_category(result, "grounded_answer", T039)
    assert category == "Испорченный или некачественный продукт"
    assert category != CATEGORY_FOREIGN_OBJECT
    assert category not in HARD_TEMPLATE_CATEGORIES
    glass = _assess("В салате стекло.")
    assert derive_claim_category(glass, "grounded_answer", "В салате стекло.") == CATEGORY_FOREIGN_OBJECT


def test_every_reason_code_maps_to_a_claim_category() -> None:
    assert set(REASON_CODE_CATEGORIES) == set(RiskReasonCode)


# ---------------------------------------------------------------------------
# T053: a documented gap, not a rule (decision recorded in README "Known limitations")
# ---------------------------------------------------------------------------


def test_t053_repeated_failure_is_a_documented_gap_that_is_reported_not_guessed() -> None:
    """Policy 08 makes a repeated significant failure HIGH, but it is a judgement over several
    deliveries ('существенный', 'повторный') and the text states no failure the rules know: the
    wrong-item rule needs 'привезли не то блюдо'. A rule tuned to this one frozen case would be
    fitted to it, so the floor reports ``no_signal`` and staff judge it. If a rule is added later,
    this test must change together with the README limitation and a held-out set that measures
    its false positives.
    """
    result = _assess(T053)
    assert result.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert result.reason_codes == ()


# ---------------------------------------------------------------------------
# Food and consumption context: explicit word forms, not bare stems (Stage 2I.1)
# ---------------------------------------------------------------------------
# A bare stem matches inside other words ('ед' in 'среда', 'блюд' in 'соблюдены', 'доставк' in
# 'недоставка') and, from the word start, unrelated ones ('едва', 'единственный', 'супермаркет',
# 'супруг', 'последний'). Each phrase below used to count as food / consumption evidence only
# through such a match.

MASS_INCIDENT = RiskReasonCode.MASS_INCIDENT


@pytest.mark.parametrize(
    "query",
    [
        "В супе стекло.",
        "В супчике стекло.",
        "В моём супе кусок стекла.",
        "В еде игла.",
        "В салате стекло.",
    ],
)
def test_genuine_food_wording_is_still_a_food_context_for_a_dangerous_object(query: str) -> None:
    assert DANGEROUS in _assess(query).reason_codes


@pytest.mark.parametrize(
    "query",
    [
        "В супермаркете разбилось стекло витрины.",
        "Мой супруг разбил стекло телефона.",
        "Курьер едет, у него разбилось стекло телефона.",
        "На прошлой неделе у меня разбилось стекло телефона.",
        # 'блюд' inside 'соблюдены'
        "Правила соблюдены, у меня разбилось стекло телефона.",
    ],
)
def test_words_that_merely_contain_a_food_fragment_are_not_a_food_context(query: str) -> None:
    assert DANGEROUS not in _assess(query).reason_codes


@pytest.mark.parametrize(
    "query",
    [
        "После супа тошнит.",
        "Съел суп, тошнит.",
        "Поел супа, тошнит.",
        "После еды у меня слабость.",
        "После вашей еды у меня слабость.",
        "После вашего супа у меня слабость.",
        # 'после вашей доставки' is explicit next to 'после доставки'; it used to match through the
        # 'ед' inside 'недомогание'
        "После вашей доставки у меня слабость.",
        "Слабость и недомогание после вашей доставки",
    ],
)
def test_genuine_consumption_wording_still_links_a_symptom_to_the_order(query: str) -> None:
    assert _is_health(query)


@pytest.mark.parametrize(
    "query",
    [
        # eating verbs that used to match only because 'ед' sits inside them are listed on purpose
        "Обедал у вас вчера, тошнит.",
        "Пообедал у вас, тошнит.",
        "Отведал ваш салат, тошнит.",
        "Доедал порцию, потом тошнит.",
        "Съедено все, тошнит.",
    ],
)
def test_eating_verbs_stay_consumption_evidence_through_explicit_forms(query: str) -> None:
    assert _is_health(query)


@pytest.mark.parametrize(
    "query",
    [
        # 'ед' inside 'среда' / at the start of 'единственный' / 'едва'
        "Среда была жаркая, меня тошнит.",
        "Единственный выходной, у меня сыпь.",
        "Едва доехал до дома, меня тошнит.",
        # 'блюд' inside 'соблюдены'
        "Правила соблюдены, меня тошнит.",
        # 'доставк' inside 'недоставка'
        "Недоставка вчера, меня тошнит.",
        "Недоставка вчера, после этого у меня слабость.",
        # 'блюд' inside 'соблюдены' as the food anchor, and as the second anchor after a delivery verb
        "Правила соблюдены, после этого у меня слабость.",
        "Курьер доставил, правила соблюдены, у меня слабость.",
        # 'суп' at the start of 'супруг' / 'супермаркет'
        "Супруг сказал, что после работы у меня слабость.",
        "После супермаркета у меня слабость.",
        # 'после' at the start of 'последний'
        "Заказ был давно. Последний раз у меня слабость.",
    ],
)
def test_unrelated_words_with_a_short_lexical_fragment_are_not_consumption_evidence(query: str) -> None:
    assert not _is_health(query)


def test_a_symptom_beside_a_real_order_mention_is_still_linked() -> None:
    # the same sentences with a genuine order word: the link is real again
    assert _is_health("Заказ был давно. Последний раз после заказа у меня слабость.")
    assert _is_health("Заказ получил. Среда была жаркая, меня тошнит.")


def test_mass_incident_context_words_are_word_forms_not_fragments() -> None:
    # 'еды' inside 'предыдущих', 'после' at the start of 'последних'
    assert MASS_INCIDENT not in _assess("Несколько человек говорили о предыдущих курьерах.").reason_codes
    assert MASS_INCIDENT not in _assess("Несколько человек писали в последних сообщениях.").reason_codes
    assert MASS_INCIDENT in _assess("Несколько человек после еды заболели.").reason_codes
    assert MASS_INCIDENT in _assess("Несколько клиентов получили заказ из одной партии и им плохо стало.").reason_codes


def test_hair_in_soup_is_found_and_hair_in_a_supermarket_is_not() -> None:
    assert UNCONFIRMED in _assess("В супе волос.").reason_codes
    assert UNCONFIRMED in _assess("В супчике нашёл волос.").reason_codes
    assert UNCONFIRMED not in _assess("В супермаркете нашёл волос.").reason_codes
