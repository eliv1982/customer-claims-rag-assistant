"""Deterministic risk matchers and assessment (3B.1 safety floor).

Matching conventions (see ``risk.matching``):

* A severe signal is a *candidate span*. Exclusion phrases are *local masks*: they cancel only
  the candidates they overlap and can never disable a valid signal elsewhere in the message.
* Proximity relations ("A ... B") are ordered anchor chains with a bounded gap, not ``.*``
  regexes, so matching stays near-linear on adversarial input.
* The raw message is capped at ``MAX_CUSTOMER_QUERY_CHARS`` by ``RiskAssessmentRequest``
  before any rule runs.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from customer_claims_rag.risk.language import is_supported_language
from customer_claims_rag.risk.matching import (
  ANCHOR_WINDOW,
  SENTENCE_BREAK,
  Masks,
  Span,
  chain_spans,
  has_unmasked,
  spans,
)
from customer_claims_rag.risk.models import (
  DeterministicRiskResult,
  RiskAssessmentRequest,
  RiskLevel,
  RiskSignal,
  risk_rank,
)
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.validator import (
  build_deterministic_risk_result,
  build_unsupported_language_result,
)

# Bounded same-line gap used inside a pattern that relates two nearby phrases. No pattern in
# this module uses an unbounded wildcard, so no regex can backtrack across the whole message.
_GAP = r"[^\n]{0,%d}" % ANCHOR_WINDOW

_DELAY_CONTEXT = re.compile(
  r"(?:"
  r"опозда\w*"
  r"|задерж\w*"
  r"|после\s+(?:конца\s+)?интервал"
  r"|прошл[оа]"
  r"|жду\s+уже"
  r"|уже"
  r"|курьер\w*\s+(?:все\s+еще\s+)?нет"
  r"|заказа\s+нет"
  r"|нет\s+заказа"
  r"|не\s+получил"
  r"|где\s+мой\s+заказ"
  r")",
)

_FUTURE_DURATION_PREFIX = re.compile(
  r"(?:"
  r"через\s*$"
  r"|запланирован\w*\s+через\s*$"
  r")",
)

# A duration followed by a promise ('2 часа привезут') is a future promise, not a delay.
_FUTURE_DURATION_TAIL = re.compile(
  r"^\s*(?:минут|мин\.?|час)[^\n]{0,40}(?:привезут|приедет|будет\s+достав)",
)

_NEGATIVE_DURATION = re.compile(
  r"(?:"
  r"[-−]\s*\d+\s*(?:минут|мин\.?|час)"
  r"|\bминус\s+\d+\s*(?:минут|мин\.?|час)"
  r")",
)

_COMPOSITE_NUMERIC_HOURS_MINUTES = re.compile(
  r"(\d+(?:[.,]\d+)?)\s*(?:час(?:а|ов|у)?|ч\.?)\s+(?:и\s+)?(\d+)\s*(?:минут(?:ы|а|у)?|мин\.?)",
)

_COMPOSITE_WORD_HOURS_MINUTES = re.compile(
  r"(один|одна|два|две|три|четыре|пять|шесть)\s+час(?:а|ов|у)?\s+(?:и\s+)?(\d+)\s*(?:минут(?:ы|а|у)?|мин\.?)",
)

_MINUTES_PATTERN = re.compile(
  r"(?<![\d№#])(\d+)\s*(?:минут(?:ы|а|у)?|мин\.?)(?!\s*(?:рубл|₽))",
)

_HOURS_NUMERIC = re.compile(
  r"(?<![\d№#])(\d+(?:[.,]\d+)?)\s*(?:час(?:а|ов|у)?|ч\.?)",
)

_GREATER_THAN_TWO_HOURS = re.compile(r"больше\s+двух\s+час")

_EXACT_TWO_HOURS = re.compile(r"(?:ровно\s+)?(?:два|2)\s*час(?:а|ов|у)?(?!\s+\d)")

_WORD_HOUR_PATTERN = re.compile(
  r"\b(один|одна|два|две|три|четыре|пять|шесть)\s+час(?:а|ов|у)?(?!\s+\d)",
)

_WORD_HOURS: dict[str, Decimal] = {
  "один": Decimal("60"),
  "одна": Decimal("60"),
  "два": Decimal("120"),
  "две": Decimal("120"),
  "три": Decimal("180"),
  "четыре": Decimal("240"),
  "пять": Decimal("300"),
  "шесть": Decimal("360"),
}

_HALF_HOUR_PATTERN = re.compile(
  r"(?:два\s+с\s+половиной|2[.,]5)\s*час",
)

_NON_DELIVERY = re.compile(
    r"(?:"
    r"вообще\s+не\s+(?:доставил\w*|привез\w*)"
    r"|заказ\s+так\s+и\s+не\s+приехал"
    r"|весь\s+заказ\s+не\s+приехал"
    r"|(?:мне\s+)?ничего\s+не\s+(?:доставил\w*|привез\w*)"
    r"|полн\w+\s+недоставк\w*"
    r"|статус\s+не\s+[«\"']?доставлен\w*"
    # passive-participle forms: 'не был доставлен', 'не была доставлена'
    r"|не\s+(?:был[аои]?\s+)?доставлен\w*(?!\s+(?:вовремя|в\s+срок|своевременно))"
    r")",
)

# 'вчера ... не доставили' / 'позавчера ... не приехал' (anchor chain, see _non_delivery_total_spans)
_NON_DELIVERY_DAY_ANCHOR = re.compile(r"(?:вчера|позавчера)")
_NON_DELIVERY_DAY_FAILURE = re.compile(r"(?:не\s+доставил\w*|не\s+приехал\w*)")

# Active past-tense delivery failure: 'не доставили', 'его не доставили', etc.
_NON_DELIVERY_ACTIVE = re.compile(
    r"(?:"
    r"(?:так\s+и\s+)?не\s+доставил\w*(?!\s+(?:вовремя|в\s+срок|своевременно))"
    r"|заказ\s+(?:так\s+и\s+)?не\s+доставил\w*"
    r"|(?:его|её|им|нам|мне|тебе|вам)\s+не\s+доставил\w*"
    r"|мне\s+не\s+доставил\w*\s+(?:оплаченн\w+\s+)?заказ"
    r")",
)

# Partial missing-item phrasing must stay on the missing-item rule, not total non-delivery.
_NON_DELIVERY_PARTIAL_EXCLUSION = re.compile(
    r"часть\s+заказ\w*\s+не\s+доставил\w*",
)

# In-progress delivery within an open interval is not final non-delivery
# (anchor chain: start phrase ... end phrase, see _match_non_delivery).
_NON_DELIVERY_IN_PROGRESS_START = re.compile(
    r"(?:"
    r"(?:еще|ещё)\s+не\s+доставил\w*"
    r"|не\s+доставил\w*\s*,?\s*но"
    r")",
)
_NON_DELIVERY_IN_PROGRESS_END = re.compile(
    r"(?:"
    r"интервал\w*"
    r"|заканчива(?:ет|ется|ться)"
    r"|через\s+"
    r"|не\s+закончил"
    r"|не\s+истек"
    r")",
)

_NON_DELIVERY_INFORMATIONAL = re.compile(
    r"(?:"
    r"доставля(?:ют|ете|ем)\s+ли"
    r")",
)

_FALSE_DELIVERY = re.compile(
  r"(?:"
  r"статус\w*\s+[«\"']?доставлен\w*[»\"']?"
  r"|[«\"']?доставлен\w*[»\"']?\s*,?\s*но"
  r"|доставлен\w*" + _GAP + r"(?:не\s+получ|ничего\s+не\s+получ)"
  r")",
)

_MISSING_ITEM = re.compile(
  r"(?:"
  r"не\s+хватил\w*"
  r"|не\s+хватало\w*"
  r"|не\s+хватает\s+(?:одн\w*|позиц\w*)"
  r"|отсутств\w+\s+позиц\w*"
  r"|отсутствовал[аои]?\s+(?:одна?\s+)?(?:позиц\w*|блюд\w*|товар\w*)"
  r"|недостач\w*"
  r"|не\s+привез\w*\s+(?:одн\w+\s+)?позиц\w*"
  r"|не\s+довез\w*\s+(?:одн\w+\s+)?позиц\w*"
  r"|(?:в\s+)?заказ\w*\s+не\s+привез\w+"
  r"|(?:мне|нам)\s+не\s+положил\w+"
  r"|(?:мне\s+)?не\s+положил\w+"
  r"|не\s+было\s+(?:одн\w+\s+)?(?:позиц\w*|блюд\w*|товар\w*|напитк\w*)"
  r"|отсутствует\s+один\s+товар"
  r"|одн\w+\s+позиц\w*\s+нет"
  r"|забыл\w+\s+положить"
  r"|забыли\s+положить"
  r"|привез\w*\s+не\s+весь\s+заказ"
  r"|часть\s+заказ\w*\s+не\s+доставил\w*"
  r"|часть\s+заказ\w*\s+не\s+привезли"
  r")",
)

_PARTIAL_MISSING_EXCLUSION = re.compile(
  r"(?:"
  r"вообще\s+не\s+привез\w*"
  r"|(?:мне\s+)?ничего\s+не\s+(?:доставил\w*|привез\w*)"
  r"|весь\s+заказ\s+не\s+приехал"
  r")",
)

_INFORMATIONAL_MISSING_ITEM = re.compile(
  r"(?:"
  r"что\s+считается\s+недоста"
  r"|где\s+посмотреть\s+список\s+позиц"
  r"|можно\s+ли\s+удалить\s+одн\w+\s+позиц"
  r"|(?:еще\s+)?не\s+знаю" + _GAP + r"(?:привез|позиц)"
  r"|сколько\s+позиций"
  r")",
)

_REFUND_REQUEST = re.compile(
    r"(?:"
    r"требую\s+(?:полный\s+)?(?:возврат|возвратить\s+стоимост\w*)"
    r"|верните\s+деньг\w*"
    r"|верните\s+уплаченн\w+\s+сумм\w*"
    r"|верните\s+сумм\w*" + _GAP + r"(?:заплатил\w*|оплатил\w*)"
    r"|вернуть\s+деньг\w*"
    r"|(?:прошу|требую)\s+вернуть\s+уплаченн\w+"
    r"|полный\s+возврат"
    r"|возврат\s+(?:денег|средств|полной\s+стоимости)"
    r"|хочу\s+оформить\s+возврат"
    r"|прошу\s+компенсир\w+\s+стоимост\w*"
    r"|компенсац\w*\s+за"
    r")",
)

# 'какие ... правила возврата', 'где прочитать ... возврат' etc. are anchor chains (see
# _informational_refund_present); the rest are plain local phrases.
_INFORMATIONAL_REFUND = re.compile(
  r"(?:"
  r"можно\s+ли\s+(?:вообще\s+)?вернуть"
  r"|можно\s+ли\s+вернуть\s+уплаченн\w+"
  r"|где\s+посмотреть\s+сумм\w*\s+оплат\w*"
  r"|что\s+означает\s+уплаченн\w+\s+сумм\w*"
  r")",
)
_INFORMATIONAL_REFUND_CHAINS: tuple[tuple[re.Pattern[str], ...], ...] = (
  (re.compile(r"какие\s+"), re.compile(r"правил\w*\s+возврат")),
  (re.compile(r"где\s+прочитать\s+"), re.compile(r"(?:возврат|компенсац)")),
  (
    re.compile(r"объясните\s+"),
    re.compile(r"(?:срок\w*|правил\w*)"),
    re.compile(r"(?:возврат|компенсац|зачислен)"),
  ),
  (re.compile(r"что\s+такое\s+"), re.compile(r"(?:возврат|компенсац|сложн\w+\s+случа)")),
)

_PACKAGE_TAMPERING = re.compile(
  r"(?:"
  r"вскрыт\w*"
  r"|сорван\w*\s+пломб\w*"
  r"|пломб\w*\s+сорван\w*"
  r"|нарушен\w*\s+герметичност\w*"
  r"|герметичност\w*\s+нарушен\w*"
  r"|вмешательств\w*"
  r"|контейнер\w*\s+(?:был[аои]?\s+)?открыт\w*"
  r"|открыт\w*\s+(?:был[аои]?\s+)?контейнер\w*"
  r"|упаковк\w*\s+(?:был[аои]?\s+)?(?:открыт\w*|поврежден\w*)"
  r"|крышк\w*\s+(?:был[аои]?\s+)?открыт\w*"
  r"|крышк\w*\s+не\s+сидит"
  r"|крышк\w*\s+треснул\w*"
  r"|(?:контейнер|упаковк\w*)\s+приехал[аои]?\s+(?:уже\s+)?(?:открыт\w*|вскрыт\w*)"
  r"|защитн\w+\s+плёнк\w*\s+(?:был[аои]?\s+)?поврежден\w*"
  r"|защитн\w+\s+пленк\w*\s+(?:был[аои]?\s+)?поврежден\w*"
  r")",
)

_PACKAGE_TAMPERING_EXCLUSION = re.compile(
  r"(?:"
  r"можно\s+ли\s+(?:открыть|вскрыть)"
  r"|как\s+(?:правильно\s+)?(?:открыть|хранить)"
  r"|открыть\s+(?:контейнер|упаковку)\s+(?:для\s+хранения|чтобы|и\s+хранить)"
  r")",
)

_WRONG_ITEM = re.compile(
  r"(?:"
  r"привезли\s+(?:совсем\s+|не\s+то\s*,?\s*|другое\s+)блюд\w*"
  r"|не\s+то\s*,?\s*что\s+заказ(?:ывал|ала|али|ывала)"
  r"|перепутали\s+(?:одну\s+)?позиц\w*"
  r"|положили\s+другой\s+товар"
  r"|заказ\s+не\s+соответствует\s+составу"
  r"|прислали\s+(?:не\s+то|другое\s+блюд\w*)"
  r"|оказалас?\s+(?:не\s+та|другая)\s+(?:позиц\w*|блюд\w*)"
  r"|не\s+тот\s+товар"
  r"|не\s+ту\s+позиц\w*"
  r")",
)

_WRONG_ITEM_EXCLUSION = re.compile(
  r"(?:"
  r"привезли\s+именно\s+то"
  r"|именно\s+(?:то\s+блюдо|то,?\s*что|то\s+что)"
  r"|привезли\s+(?:правильн\w*|верн\w*|точно\s+то)"
  r")",
)

_FOOD_SPOILAGE = re.compile(
  r"(?:"
  r"порч[аеи]\w*"
  r"|испорт\w+"
  r"|(?:резкий|странный|необычн\w*)\s+(?:кисл\w*\s+)?запах"
  r"|похож\w*\s+на\s+порч\w*"
  r")",
)

# Explicit intent to go to court / regulator / police / media. Plain phrases are matched directly;
# 'verb ... target' relations are anchor chains (see _legal_candidate_spans).
_LEGAL_DIRECT = re.compile(
  r"(?:"
  r"жалоб\w*\s+(?:в|регулятор)"
  r"|подам\s+иск"
  r"|(?:решил\w*|решили)\s+подать\s+жалоб"
  r"|(?:решил\w*|решили)\s+обращаться"
  r")",
)
_LEGAL_ACTION_VERB = re.compile(
  r"(?:подам|подал\w*|обращ\w+|напиш\w*|писал\w*|направ\w+|передам|буду\s+писать|пойду)",
)
_LEGAL_TARGET = re.compile(
  r"(?:в|к)\s+"
  r"(?:суд|полици\w*|прокуратур\w*|роспотребнадзор\w*|регулятор\w*|контролирующ\w+\s+орган|надзорн\w+\s+орган)",
)
_LEGAL_IF_NOT = re.compile(r"если\s+не\s+")
_LEGAL_IF_NOT_RESOLVE = re.compile(r"(?:верн\w*|реш\w*)")
_LEGAL_IF_NOT_APPEAL = re.compile(r"(?:обращ\w*|подам)")
_LEGAL_IF_NOT_TARGET = re.compile(r"(?:суд|регулятор|полици)")
_LEGAL_FILE_IN_COURT = re.compile(r"подам\s+в\s+суд")
_LEGAL_DEMAND_SOLVE = re.compile(r"требую\s+решить")
_LEGAL_OTHERWISE_APPEAL = re.compile(r"иначе\s+обращ\w*")
_LEGAL_REGULATOR = re.compile(r"регулятор")
_LEGAL_PUBLICITY_VERB = re.compile(r"(?:передам|вылож\w*|опублик\w*|распростран\w+)")
_LEGAL_PUBLICITY_TARGET = re.compile(r"(?:сми|журналист\w*|соцсет\w*)")
_LEGAL_WRITE_VERB = re.compile(r"(?:напиш\w*|обращ\w*)")
_LEGAL_MEDIA = re.compile(r"(?:журналист\w*|сми)")

_INFORMATIONAL_LEGAL = re.compile(
  r"(?:"
  r"что\s+(?:означает|такое)\s+(?:\bсми\b|контролирующ\w+\s+орган|регулятор|прокуратур\w*)"
  r"|где\s+(?:посмотреть|найти)" + _GAP + r"(?:раздел\s+)?\bсми\b"
  r"|кто\s+такой\s+регулятор"
  r"|какие\s+функции\s+выполняет"
  r"|где\s+найти\s+информаци\w+" + _GAP + r"суд"
  r"|расскаж\w+" + _GAP + r"что\s+такое"
  r")",
)

# 'читал ... статья / новости ... суд' is an anchor chain (see _historical_legal_present).
_HISTORICAL_LEGAL_CHAIN: tuple[re.Pattern[str], ...] = (
  re.compile(r"читал\w+"),
  re.compile(r"(?:стать\w+|новост\w+)"),
  re.compile(r"(?:роспотребнадзор|суд|прокуратур)"),
)
_HISTORICAL_LEGAL = re.compile(
  r"(?:"
  r"в\s+новост\w+\s+говорил"
  r"|сми\s+часто\s+пиш"
  r")",
)

# 'не буду ... обращаться ... в суд' is an anchor chain (see _legal_declined_action_present).
_LEGAL_DECLINED_CHAIN: tuple[re.Pattern[str], ...] = (
  re.compile(r"не\s+(?:буду|собира\w*|планиру\w*)"),
  re.compile(r"(?:обращ\w*|писать|пис\w*|подавать|подам)"),
  re.compile(
    r"(?:суд|прокуратур|полици|\bсми\b|регулятор|контролирующ\w+\s+орган|роспотребнадзор)",
  ),
)
_LEGAL_DECLINED_ACTION = re.compile(
  r"(?:"
  r"(?:в\s+)?\bсми\b\s+обращаться\s+не\s+планиру"
  r"|(?:решил\w*|решили)\s+не\s+(?:подавать|обращ\w*|писать|пис\w*|идти|пойти)"
  r"|(?:передумал\w*|отказал\w*)\s+(?:обращ\w*|писать|пис\w*)"
  r")",
)

_LEGAL_NEAR_MISS = re.compile(
  r"(?:"
  r"судя\s+по"
  r"|полицейск\w+\s+сериал"
  r"|смотр\w*\s+полицейск\w+\s+сериал"
  r"|работает\s+в\s+суде"
  r"|раздел\s+" + _GAP + r"отзыв"
  r")",
)

_LOCAL_LEGAL_NEGATION = re.compile(
  r"(?:"
  r"не\s+(?:хочу|собира\w*|буду|будем)\s*$"
  r"|пока\s+не\s*(?:буду|обращ\w*|пойду|подам|напиш\w*|пис\w*)?\s*$"
  r"|не\s+(?:говорил\w*|обращал\w*|писал\w*)\s*(?:,\s*)?(?:что\s+)?$"
  r")",
)

# Demand for an official written answer. 'demand ... answer ... complaint' is an anchor chain
# (see _official_written_candidate_spans).
_OFFICIAL_DIRECT = re.compile(
  r"(?:"
  r"(?:мне\s+)?нуж\w+\s+официальн\w+\s+ответ\w*\s+на\s+(?:мою\s+)?претенз\w+"
  r"|(?:требую|прошу)\s+официальн\w+\s+ответ\s+в\s+связи\s+с"
  r")",
)
_OFFICIAL_DEMAND_VERB = re.compile(r"(?:требую|прошу)")
_OFFICIAL_ANSWER = re.compile(r"ответ")
_OFFICIAL_COMPLAINT_WORD = re.compile(
  r"(?:претенз\w+|жалоб\w+|обращени\w+|нарушени\w+|возврат|компенсац|инцидент|недоставк|поврежден|по\s+существу)",
)
_OFFICIAL_ASK_WRITTEN = re.compile(r"прошу\s+письменно\s+ответить")
_OFFICIAL_ASK_WRITTEN_TOPIC = re.compile(r"(?:обращени|жалоб|претенз|возврат)")
_OFFICIAL_PROVIDE = re.compile(r"(?:сообщ\w+\s+срок\w*|предостав\w+)")
_OFFICIAL_CLAIM = re.compile(r"претенз")

_INFORMATIONAL_OFFICIAL_RESPONSE = re.compile(
  r"(?:"
  r"где\s+посмотреть[^.?!;\n]{0,60}(?:правил\w*|письменн\w+)"
  r"|правил\w*\s+подготовк\w+[^.?!;\n]{0,60}письменн\w+"
  r"|что\s+(?:обычно\s+)?означает[^.?!;\n]{0,60}официальн\w+[^.?!;\n]{0,40}ответ"
  r"|можно\s+ли\s+получить\s+информаци\w+[^.?!;\n]{0,60}(?:срок\w*|ответ\w*)"
  r"|(?:еще\s+)?не\s+решил[^.?!;\n]{0,60}письменн\w+"
  r"|нужна\s+ли\s+мне\s+письменн\w+\s+претенз\w+"
  r")",
)

_OFFICIAL_WRITTEN_SERVICE_INFO = re.compile(
  r"(?:"
  r"ответ\s+о\s+(?:правил\w+|график)"
  r"|ответ[^.?!;\n]{0,60}правил\w+\s+доставк"
  r"|(?:рассказать|объяснить)[^.?!;\n]{0,60}(?:правил\w*|подписк|график\w*\s+работ)"
  r"|о\s+вашем\s+график"
  r"|как\s+работает\s+подписк"
  r")",
)

_OFFICIAL_WRITTEN_DECLINED = re.compile(
  r"(?:"
  r"не\s+требую\s+письменн\w+\s+ответ"
  r"|мне\s+не\s+нуж\w+\s+официальн\w+\s+ответ"
  r"|курьер\w*\s+письменно\s+ответил"
  r")",
)

_OFFICIAL_WRITTEN_RECEIVED = re.compile(
  r"(?:"
  r"получил\w*\s+официальн\w+\s+письменн\w+\s+ответ"
  r")",
)

_PERSONAL_DATA_HIGH = re.compile(
    r"(?:"
    r"чужое\s+имя"
    r"|чужой\s+адрес"
    r"|чужие\s+данн\w*"
    r"|данн\w*\s+друг\w+\s+клиент\w*"
    r"|чуж\w+\s+имя\s+и"
    r")",
)
# 'имя и ... адрес ... чужой / не могу' is an anchor chain (see _match_personal_data_high).
_PERSONAL_DATA_HIGH_CHAIN: tuple[re.Pattern[str], ...] = (
  re.compile(r"имя\s+и\s+"),
  re.compile(r"адрес"),
  re.compile(r"(?:чуж\w+|не\s+мо\w*)"),
)

_PERSONAL_DATA_CRITICAL = re.compile(
    r"(?:"
    r"массов\w+\s+утечк\w*"
    r"|серьезн\w+\s+утечк\w*"
    r"|(?:сотен|сотни)\s+клиент\w*"
    r"|списк\w*" + _GAP + r"(?:сотен|сотни)\s+клиент\w*"
    r"|утечк\w*\s+данн\w*" + _GAP + r"(?:сотен|сотни|мног\w+\s+клиент\w*)"
    r"|массов\w+\s+(?:публикац\w*|выгрузк\w*)\s+" + _GAP + r"данн\w*"
    r")",
)

# --- consumption link -------------------------------------------------------------------------
# The message must connect the complaint to eating / receiving the order. Every relation here is
# either a plain local pattern or an independent co-occurrence check: no wildcard spans the message.

_CONSUMPTION_DIRECT = re.compile(
  r"(?:"
  r"после\s+(?:еды|ужин\w*|обед\w*|завтрак\w*|блюд\w*|суп\w*|заказ\w*|доставк\w*|употреблен\w*|ваш\w+\s+(?:ед\w*|блюд\w*|суп\w*|заказ\w*))"
  r"|поел\w*"
  r"|съел\w*"
  r"|употребил\w*"
  r")",
)

# food anchor ... later 'после' / 'отравлен'
_FOOD_BEFORE_AFTER_FIRST = re.compile(r"(?:ед\w*|блюд\w*|суп\w*|заказ\w*)")
_FOOD_BEFORE_AFTER_SECOND = re.compile(r"(?:после|отравлен\w*)")

# delivery verb ... later 'после' / food word
_DELIVERY_BEFORE_FOOD_FIRST = re.compile(r"(?:доставк\w*|привез\w*|доставил\w*)")
_DELIVERY_BEFORE_FOOD_SECOND = re.compile(r"(?:после|ед\w*|блюд\w*)")

# core symptom word and food / order word anywhere in the message, in either order
_CORE_SYMPTOM = re.compile(
  r"(?:тошнит|рвот\w*|заболел\w*|аллерг\w*|задыха\w*|одышк\w*|трудно\s+дыш|тяжело\s+дыш|хрип\w*"
  r"|отравил\w*|сыпь\w*|зуд\w*|крапивниц\w*|отёк\w*|отек\w*)",
)
_GENERAL_MALAISE = re.compile(
  r"(?:мне\s+плохо|стало\s+плохо|плохо\s+стало|ухудшилось\s+самочувств\w*|стало?\s+нехорошо)",
)
_ORDER_OR_FOOD_WORD = re.compile(r"(?:заказ\w*|ед\w*|блюд\w*|доставк\w*)")

_HEALTH_SYMPTOM = re.compile(
  r"(?:"
  r"тошнот\w*"
  r"|тошнит"
  r"|рвот\w*"
  r"|боль\s+в\s+живот\w*"
  r"|дискомфорт\w*\s+в\s+живот\w*"
  r"|стало\s+плохо"
  r"|плохо\s+стало"
  r"|мне\s+(?:стало\s+)?плохо"
  r"|ухудшен\w+\s+самочувств\w*"
  r"|ухудшилось\s+самочувств\w*"
  r"|аллергическ\w+\s+реакц\w*"
  r"|отравлен\w*"
  r"|отравил\w*"
  r"|заболел\w*"
  r"|температур\w*"
  r"|вызвал\w*\s+скор\w*"
  r"|слабост\w*"
  r"|трудно\s+дышать"
  r"|тяжело\s+дышать"
  r"|не\s+могу\s+дышать"
  r"|не\s+хватает\s+воздуха"
  r"|задыха\w*"
  r"|одышк\w*"
  r"|трудно\s+вдохнуть"
  r"|перехватило\s+дыхание"
  r"|хрип\w*"
  r"|сыпь\w*"
  r"|зуд\w*"
  r"|крапивниц\w*"
  r"|отёк\w*"
  r"|отек\w*"
  # --- severe-health lexicon (bounded, high confidence) -------------------------------------
  # hospitalization
  r"|госпитализ\w*"
  r"|больниц\w*"
  r"|реанимац\w*"
  # ambulance
  r"|скор(?:ая|ую|ой)\s+помощ\w*"
  r"|(?:вызыва\w*|вызов\w*|приезжал\w*|приехал\w*|забрал\w*|увез\w*|увоз\w*|отвез\w*)\s+(?:\w+\s+){0,2}скор(?:ая|ую|ой)(?!\s+(?:достав|скорост))"
  r"|на\s+скорой(?!\s+(?:достав|скорост))"
  # loss of consciousness
  r"|(?:потерял\w*|теря\w+|потеря)\s+сознани\w*"
  r"|без\s+сознани\w*"
  r"|обморок\w*"
  # anaphylaxis / anaphylactic shock
  r"|анафилак\w*"
  # choking and bleeding (hazard protocol H-02 triggers)
  r"|удушь\w*"
  r"|кровотечени\w*"
  r")",
)

# Informational questions about a symptom. They mask only the symptom mention they cover; a real
# symptom stated elsewhere in the same message still counts.
_HEALTH_INFORMATIONAL = re.compile(
  r"(?:"
  r"что\s+делать[^.?!;\n]{0,60}трудно\s+дышать"
  r"|расскаж\w+[^.?!;\n]{0,40}что\s+означает[^.?!;\n]{0,40}одышк\w*"
  r"|что\s+означает[^.?!;\n]{0,40}одышк\w*"
  r"|что\s+такое\s+(?:пищевое\s+)?отравлен\w*"
  r"|что\s+(?:такое|означает)\s+(?:аллерг\w*|анафилакс\w*|пищевое)"
  r")",
)

# Phrases that look like a symptom but are not one (a symptom before the order, an app that
# 'works badly', 'hard to breathe' about pictures, a symptom placed months in the past ...).
# Each is a mask over that phrase only. Scope redirections such as 'вопрос только о возврате'
# are deliberately NOT here: a stated symptom is never cancelled by what else the customer asks.
_HEALTH_LOCAL_EXCLUSION = re.compile(
  r"(?:"
  r"(?:было|была)\s+плохо\s+[^.?!;\n]{0,40}(?:до\s+заказ|до\s+доставк)"
  r"|будто\s+от\s+не[её]\s+станет\s+плохо"
  r"|неприятно\s+смотреть"
  r"|плохо\s+работает"
  r"|трудно\s+дышать\s+картинк"
  r"|задыха\w*\s+после\s+пробежк"
  # temporal distancing: symptom explicitly placed months/years in the past as separate context
  r"|(?:сыпь|зуд|отёк|отек|крапивниц\w*)\s+(?:\w+\s+){0,4}(?:месяц|год|недел[юи])\s+(?:назад|тому)"
  r"|(?:месяц|год|недел[юи])\s+назад\s+(?:\w+\s+){0,4}(?:сыпь|зуд|отёк|отек|крапивниц\w*)"
  r")",
)

_STAFF = r"(?:курьер|сотрудник|менеджер)\w*"
_STAFF_SEVERE_SYMPTOM = (
  r"(?:тяжело\s+дыш|трудно\s+дыш|задыха|одышк|хрип|не\s+хватает\s+воздух"
  r"|госпитализ|больниц|реанимац|скор\w*\s+помощ|сознани|обморок|анафилак)"
)

# Symptom attributed to staff rather than to the customer (anchor chains, see _staff_subject_spans).
_STAFF_SYMPTOM_DIRECT = re.compile(
  _STAFF + r"\s+(?:стало\s+плохо|плохо\s+стало|заболел\w*|тошнит|рвот\w*)",
)
_STAFF_SUBJECT = re.compile(_STAFF + r"\s+")
_STAFF_SAID = re.compile(r"(?:сказал\w*|говор\w*)")
_STAFF_FEELS_BAD = re.compile(r"(?:ем\w+\s+)?плохо")
_STAFF_ON_THE_WAY = re.compile(r"(?:по\s+дорог\w*|на\s+лестниц\w*)")
_STAFF_ILL = re.compile(r"(?:заболел\w*|плохо)")
_STAFF_SEVERE = re.compile(_STAFF_SEVERE_SYMPTOM)
_STAFF_STARTED = re.compile(r"(?:появил\w*|начал\w*)")
_STAFF_RESPIRATORY = re.compile(r"(?:одышк|трудно\s+дыш|задыха|хрип)")
_AFTER_FOOD = re.compile(r"(?:после\s+еды|после\s+блюд\w*)")
_STAFF_WORD = re.compile(_STAFF)

_HEALTH_LOCAL_WINDOW = 70
_HEALTH_STAFF_LOOKAHEAD = 20
# A staff-subject phrase must fit into the local window around the symptom.
_STAFF_CHAIN_WINDOW = _HEALTH_LOCAL_WINDOW + _HEALTH_STAFF_LOOKAHEAD

_FOOD_CONTEXT = re.compile(
  r"(?:"
  r"(?:в\s+)?(?:еде|блюд\w*|салат\w*|суп\w*|продукт\w*|заказ\w*|тарелк\w*|контейнер\w*)"
  r")",
)

_DANGEROUS_OBJECT = re.compile(
  r"(?:"
  r"стекл\w*"
  r"|игл\w*"
  r"|остр\w+\s+(?:металлическ\w+\s+)?(?:осколк\w*|предмет\w*|металл\w*)"
  r"|металлическ\w+\s+осколк\w*"
  r"|(?:обнаружен\w*|найден\w*)\s+остр\w+\s+предмет\w*"
  r"|острый\s+предмет\w*"
  r"|(?:остр\w+|тверд\w+|опасн\w+)\s+(?:пластик\w*|осколк\w*|фрагмент\w*)"
  r")",
)
# 'пластик / осколок / фрагмент ... острый / опасный / травма / риск' (anchor chain)
_DANGEROUS_FRAGMENT = re.compile(r"(?:пластик\w*|осколк\w*|фрагмент\w*)")
_DANGEROUS_HAZARD_WORD = re.compile(r"(?:остр\w+|опасн\w+|травм\w*|риск\w*)")

# Look-alike objects. Each masks only the object phrase it covers.
_FOREIGN_OBJECT_EXCLUSION = re.compile(
  r"(?:"
  r"волос\w*"
  r"|мягк\w+\s+пластик\w*"
  r"|пластик\w*(?!\s*(?:остр\w+|опасн\w+|тверд\w+|фрагмент\w*))"
  r"|металлическ\w+\s+привкус"
  r"|на\s+стол\w*[^.?!;\n]{0,60}нож"
  r"|курьер\w*[^.?!;\n]{0,60}металлическ\w+\s+сумк\w*"
  r"|крошк\w+\s+хлеб"
  r")",
)

_SYMPTOM_CLUSTER = re.compile(r"(?:плохо\s+стало|стало\s+плохо|заболел\w*|отравил\w*|симптом\w*)")
_MASS_COUNT_PEOPLE = re.compile(r"(?:двум|троим|трем|трое|четырем|пятерым|\d+\s+люд\w*)")
_MASS_GROUP_A = re.compile(r"(?:нескольк\w+\s+(?:люд\w*|человек|клиент\w*)|у\s+всей\s+семь\w*)")
_MASS_GROUP_B = re.compile(r"(?:нескольк\w+\s+клиент\w*|несколько\s+человек)")
_MASS_GROUP_B_CONTEXT = re.compile(r"(?:после|партии|еды|заказ\w*)")
_MASS_BATCH = re.compile(r"(?:партии|партия)")
_MASS_SIMILAR_SYMPTOMS = re.compile(r"у\s+всех\s+похож\w+\s+симптом\w*")
_MASS_COUNTED_PEOPLE = re.compile(
  r"(?:сразу\s+)?(?:двум|троим|трем|трое|четырем|пятерым|\d+)\s+(?:люд\w*|человек)",
)
_MASS_COUNTED_FAMILY = re.compile(
  r"(?:двум|троим|трем|трое|четырем|пятерым|\d+)\s+член\w+\s+семь\w*",
)

# Look-alike non-health phrases. Each masks only the mass-incident candidate it overlaps.
_MASS_INCIDENT_EXCLUSION = re.compile(
  r"(?:"
  r"нескольк\w+\s+заказ\w*\s+опоздал\w*"
  r"|плохо\s+работает\s+приложен\w*"
  r"|массов\w+\s+проблем\w*"
  r"|массов\w+\s+акци\w*"
  r")",
)

_MASS_HEARSAY = re.compile(
  r"(?:"
  r"^(?:я\s+)?(?:читал\w*|слышал\w*|говорят|пишут|сообщили)"
  r"|^(?:в\s+)?(?:новост\w*|интернет\w*)"
  r")",
)

_CLAUSE_SPLIT = re.compile(r"[.;]|,\s*но\s+|,\s*а\s+")

# Legal-intent clauses: sentence ends, line breaks, commas and 'но'.
_LEGAL_CLAUSE_SEPARATOR = re.compile(r"[.!?;\n]+|\s+но\s+|,\s*но\s+|,\s*")

_FRAUD_DIRECT = re.compile(
  r"(?:"
  r"несанкционированн\w+\s+списан\w*"
  r"|подменил\w*\s+реквизит\w*"
  r"|повторн\w+\s+неизвестн\w+\s+списан\w*"
  r"|странн\w+\s+списан\w*"
  r")",
)
_FRAUD_DEBITED_MONEY = re.compile(r"списал\w*\s+деньг\w*")
_FRAUD_NO_CONSENT = re.compile(r"без\s+(?:моего\s+)?соглас")
_FRAUD_DEBIT_VERB = re.compile(r"(?:списыва\w+|списал\w+|списан\w*)")
_FRAUD_UNKNOWN_MARKER = re.compile(r"(?:неизвестн\w+|повторн\w+|без\s+соглас|не\s+совершал\w*)")
_FRAUD_REPEAT_MARKER = re.compile(r"(?:повторн\w+|нескольк\w+\s+раз|продолжа\w+)")
_FRAUD_DEBIT_OR_OPERATION = re.compile(r"(?:списан\w*|операц\w*)")
_FRAUD_UNKNOWN_MARKER_STRICT = re.compile(r"(?:неизвестн\w+|без\s+соглас|не\s+совершал\w*)")
_FRAUD_UNKNOWN_OR_REPEATED = re.compile(r"(?:неизвестн\w+|повторн\w+)")
_FRAUD_DEBIT_OPERATION_OR_SUM = re.compile(r"(?:списан\w*|операц\w*|сумм\w*)")
_FRAUD_ACCOUNT_SUBJECT = re.compile(r"(?:карт\w*|счет\w*)")
_FRAUD_REPEAT_TWICE = re.compile(r"(?:повторн\w+|нескольк\w+\s+раз)")
_FRAUD_DEBIT_VERB_PLAIN = re.compile(r"(?:списан\w*|списыва\w*)")
_FRAUD_PAYMENT_WORD = re.compile(r"оплат\w*")
_FRAUD_UNKNOWN_DETAILS = re.compile(r"неизвестн\w+\s+реквизит\w*")
_FRAUD_ACCOUNT_USE = re.compile(r"(?:использовал\w*|оформля\w*)")
_FRAUD_MY_ACCOUNT = re.compile(r"(?:мой\s+аккаунт|мою\s+карт\w*)")
_FRAUD_ACCOUNT_WORD = re.compile(r"(?:аккаунт\w*|карт\w*)")
_FRAUD_FOREIGN = re.compile(r"(?:чуж\w+|мошенник\w*|без\s+моего\s+соглас)")
_FRAUD_SIGNS = re.compile(r"признак\w*\s+")
_FRAUD_SCAM_WORD = re.compile(r"мошен")
_FRAUD_FINANCIAL_WORD = re.compile(r"(?:списан|аккаунт|карт|реквизит)")

# Routine / authorised charge wording. Each masks only the charge phrase it covers.
_FRAUD_CHARGE_EXCLUSION = re.compile(
  r"(?:"
  r"списал\w*\s+деньг\w*\s+за\s+(?:мой\s+)?заказ"
  r"|сумма\s+списал\w*\s+один\s+раз\s+корректно"
  r"|обычн\w+\s+списан\w*"
  r"|когда\s+пройдет\s+[^.?!;\n]{0,60}списан\w*"
  r"|не\s+понима\w*,\s+когда\s+пройдет"
  r")",
)

# Rhetorical accusation of the company. It masks only the accusation itself: it never cancels a
# concrete unknown-charge statement elsewhere in the message. Prompt-injection strings such as
# 'ignore instructions' or 'risk_level = critical' are intentionally NOT exclusions: text can
# never lower the floor.
_FRAUD_RHETORICAL = re.compile(
  r"(?:"
  r"(?:вы|это)\s+(?:\w+\s+){0,3}мошенник\w*"
  r"|мошенничеств\w*\?"
  r"|похож\w*\s+на\s+развод"
  r")",
)

_STAFF_TARGET = (
  r"(?:курьер\w*|сотрудник\w*|менеджер\w*|оператор\w*|работник\w*|представител\w+\s+компани\w*)"
)

_STAFF_TARGET_PATTERN = re.compile(_STAFF_TARGET)
_THREAT_ACTION = re.compile(r"(?:я|мы)\s+(?:удар\w*|найду\w*|расправ\w*|причин\w+)")
_THREAT_HARM = re.compile(r"(?:не\s+поздоровится|удар\w*|вред)")
_THREAT_SUBJECT = re.compile(r"(?:я|мы)\s+")
_THREAT_PHYSICAL_SHOW = re.compile(r"(?:ему|ей)\s+физическ\w+\s+покаж\w+")

# Procedural / reversed-role wording. Each masks only the phrase it covers.
_THREAT_EXCLUSION = re.compile(
  r"(?:"
  r"угрожа\w*\s+(?:написать|обратиться|оставить|подать)[^.?!;\n]{0,80}(?:жалоб\w*|суд|отзыв|полици|менеджер\w*)"
  r"|(?:курьер\w*|менеджер\w*|сотрудник\w*|оператор\w*)[^.?!;\n]{0,60}(?:угрожа\w*|угроз\w+)[^.?!;\n]{0,40}(?:мне|клиент)"
  r"|если\s+не\s+верн\w*[^.?!;\n]{0,80}(?:полици|суд)"
  r"|(?:я|мы)\s+физическ\w+\s+(?:покаж\w+|передам\w+)\s+(?:документ|бумаг)"
  r"|(?:я|мы)\s+физическ\w+\s+передам\w+\s+бумаг"
  r"|(?:я|мы)\s+покаж\w+\s+" + _STAFF_TARGET + r"[^.?!;\n]{0,60}где\s+лежит\s+заказ"
  r")",
)

_DELAY_MEDIUM_THRESHOLD = Decimal("30")
_DELAY_HIGH_THRESHOLD = Decimal("120")


def normalize_for_matching(text: str) -> str:
  """Deterministic normalization for rule matching only."""
  normalized = unicodedata.normalize("NFC", text)
  normalized = normalized.replace("ё", "е").replace("Ё", "Е")
  return normalized.casefold()


def _has_delay_context(text: str) -> bool:
  return _DELAY_CONTEXT.search(text) is not None


def _parse_word_hours(word: str) -> Decimal | None:
  return _WORD_HOURS.get(word)


def _hours_to_minutes(value: str) -> Decimal:
  return Decimal(value.replace(",", ".")) * Decimal("60")


def _is_future_duration(text: str, start: int) -> bool:
  window = text[max(0, start - 40):start]
  if _FUTURE_DURATION_PREFIX.search(window):
    return True
  trailing = text[start: min(len(text), start + 40)]
  if _FUTURE_DURATION_TAIL.search(trailing):
    return True
  return False


def extract_delay_minutes(normalized: str) -> list[Decimal]:
  """Extract delay durations in minutes from normalized customer text."""
  if not _has_delay_context(normalized):
    return []

  # A negative duration ("-121 минут") is ignored itself; it must not cancel a genuine
  # duration stated elsewhere in the message.
  negative = Masks(spans(_NEGATIVE_DURATION, normalized))

  minutes: list[Decimal] = []
  consumed_spans: list[tuple[int, int]] = []

  if _GREATER_THAN_TWO_HOURS.search(normalized):
    minutes.append(Decimal("121"))

  for match in _COMPOSITE_NUMERIC_HOURS_MINUTES.finditer(normalized):
    if negative.overlaps(match.span()):
      continue
    total = _hours_to_minutes(match.group(1)) + Decimal(match.group(2))
    minutes.append(total)
    consumed_spans.append(match.span())

  for match in _COMPOSITE_WORD_HOURS_MINUTES.finditer(normalized):
    if negative.overlaps(match.span()):
      continue
    word_hours = _parse_word_hours(match.group(1))
    if word_hours is not None:
      total = word_hours + Decimal(match.group(2))
      minutes.append(total)
      consumed_spans.append(match.span())

  def _span_consumed(start: int, end: int) -> bool:
    return any(start >= span_start and end <= span_end for span_start, span_end in consumed_spans)

  for match in _MINUTES_PATTERN.finditer(normalized):
    if negative.overlaps(match.span()):
      continue
    if _span_consumed(*match.span()):
      continue
    if _is_future_duration(normalized, match.start(1)):
      continue
    value = int(match.group(1))
    if value <= 0:
      continue
    minutes.append(Decimal(value))

  for match in _HOURS_NUMERIC.finditer(normalized):
    if negative.overlaps(match.span()):
      continue
    if _span_consumed(*match.span()):
      continue
    if _is_future_duration(normalized, match.start(1)):
      continue
    minutes.append(_hours_to_minutes(match.group(1)))

  if _HALF_HOUR_PATTERN.search(normalized):
    minutes.append(Decimal("150"))

  if _EXACT_TWO_HOURS.search(normalized):
    minutes.append(Decimal("120"))

  for match in _WORD_HOUR_PATTERN.finditer(normalized):
    if _span_consumed(*match.span()):
      continue
    word_hours = _parse_word_hours(match.group(1))
    if word_hours is not None:
      minutes.append(word_hours)

  return minutes


def _match_delay_over_30(normalized: str) -> bool:
  values = extract_delay_minutes(normalized)
  return any(value > _DELAY_MEDIUM_THRESHOLD for value in values)


def _match_delay_over_120(normalized: str) -> bool:
  values = extract_delay_minutes(normalized)
  return any(value > _DELAY_HIGH_THRESHOLD for value in values)


def _occurs_before(normalized: str, first: re.Pattern[str], second: re.Pattern[str]) -> bool:
  """True when ``second`` occurs somewhere after the first occurrence of ``first``."""
  match = first.search(normalized)
  return match is not None and second.search(normalized, match.end()) is not None


def _has_consumption_link(normalized: str) -> bool:
  if _CONSUMPTION_DIRECT.search(normalized):
    return True
  if _occurs_before(normalized, _FOOD_BEFORE_AFTER_FIRST, _FOOD_BEFORE_AFTER_SECOND):
    return True
  if _occurs_before(normalized, _DELIVERY_BEFORE_FOOD_FIRST, _DELIVERY_BEFORE_FOOD_SECOND):
    return True
  if _ORDER_OR_FOOD_WORD.search(normalized) is None:
    return False
  return (
    _CORE_SYMPTOM.search(normalized) is not None
    or _GENERAL_MALAISE.search(normalized) is not None
  )


def _staff_subject_spans(normalized: str) -> list[Span]:
  """Spans where a symptom is attributed to courier / staff instead of the customer."""
  # A staff subject only attributes a symptom inside its own sentence: a courier mentioned in an
  # earlier sentence must not turn the customer's later symptom into a staff symptom.
  def _chain(*anchors: re.Pattern[str]) -> list[Span]:
    return chain_spans(normalized, anchors, window=_STAFF_CHAIN_WINDOW, stop=SENTENCE_BREAK)

  found = spans(_STAFF_SYMPTOM_DIRECT, normalized)
  found += _chain(_STAFF_SUBJECT, _STAFF_SAID, _STAFF_FEELS_BAD)
  found += _chain(_STAFF_SUBJECT, _STAFF_ON_THE_WAY, _STAFF_ILL)
  found += _chain(_STAFF_SUBJECT, _STAFF_SEVERE)
  found += _chain(_STAFF_SUBJECT, _STAFF_STARTED, _STAFF_RESPIRATORY)
  found += _chain(_AFTER_FOOD, _STAFF_WORD)
  return sorted(found)


def _is_staff_attributed_symptom(staff_spans: list[Span], symptom_start: int) -> bool:
  """A staff-subject phrase that fits the local window around the symptom attributes it."""
  window_start = max(0, symptom_start - _HEALTH_LOCAL_WINDOW)
  window_end = symptom_start + _HEALTH_STAFF_LOOKAHEAD
  index = bisect_left(staff_spans, (window_start, -1))
  while index < len(staff_spans):
    start, end = staff_spans[index]
    if start > window_end:
      return False
    if end <= window_end:
      return True
    index += 1
  return False


def _match_health_symptoms(normalized: str) -> bool:
  if not _has_consumption_link(normalized):
    return False
  # Informational questions and look-alike phrases cancel only the symptom mention they
  # cover. A severe symptom stated elsewhere in the message is never cancelled by them.
  masks = Masks(
    spans(_HEALTH_LOCAL_EXCLUSION, normalized) + spans(_HEALTH_INFORMATIONAL, normalized),
  )
  staff_spans: list[Span] | None = None
  for match in _HEALTH_SYMPTOM.finditer(normalized):
    if masks.overlaps(match.span()):
      continue
    if staff_spans is None:
      staff_spans = _staff_subject_spans(normalized)
    if not _is_staff_attributed_symptom(staff_spans, match.start()):
      return True
  return False


def _non_delivery_total_spans(normalized: str) -> list[Span]:
  """Spans of explicit total non-delivery wording."""
  return spans(_NON_DELIVERY, normalized) + chain_spans(
    normalized,
    (_NON_DELIVERY_DAY_ANCHOR, _NON_DELIVERY_DAY_FAILURE),
  )


def _match_missing_item(normalized: str) -> bool:
  if _INFORMATIONAL_MISSING_ITEM.search(normalized):
    return False
  if _non_delivery_total_spans(normalized) or _PARTIAL_MISSING_EXCLUSION.search(normalized):
    return False
  return _MISSING_ITEM.search(normalized) is not None


def _match_package_tampering(normalized: str) -> bool:
  # 'можно ли открыть ...' cancels only the candidate it overlaps, so a separate genuine
  # tampering statement elsewhere in the message still matches.
  masks = Masks(spans(_PACKAGE_TAMPERING_EXCLUSION, normalized))
  return has_unmasked(spans(_PACKAGE_TAMPERING, normalized), masks)


def _match_wrong_item(normalized: str) -> bool:
  if _WRONG_ITEM_EXCLUSION.search(normalized):
    return False
  return _WRONG_ITEM.search(normalized) is not None


def _split_clauses(normalized: str) -> list[str]:
  parts = _CLAUSE_SPLIT.split(normalized)
  return [part.strip() for part in parts if part.strip()]


def _is_hearsay_clause(clause: str) -> bool:
  return _MASS_HEARSAY.search(clause.strip()) is not None


def _match_mass_incident(normalized: str) -> bool:
  for clause in _split_clauses(normalized):
    if _is_hearsay_clause(clause):
      continue
    candidates = _mass_incident_candidate_spans(clause)
    if not candidates:
      continue
    # Look-alike non-health phrases cancel only the candidate they overlap.
    masks = Masks(spans(_MASS_INCIDENT_EXCLUSION, clause))
    if has_unmasked(candidates, masks):
      return True
  return False


def _mass_incident_candidate_spans(clause: str) -> list[Span]:
  found = spans(_MASS_SIMILAR_SYMPTOMS, clause) + spans(_MASS_COUNTED_PEOPLE, clause)
  found += chain_spans(clause, (_MASS_GROUP_A, _SYMPTOM_CLUSTER))
  found += chain_spans(clause, (_MASS_GROUP_B, _MASS_GROUP_B_CONTEXT))
  found += chain_spans(clause, (_MASS_BATCH, _SYMPTOM_CLUSTER))
  found += chain_spans(clause, (_MASS_COUNTED_FAMILY, _SYMPTOM_CLUSTER))
  found += chain_spans(clause, (_SYMPTOM_CLUSTER, _MASS_COUNT_PEOPLE))
  return found


def _is_local_legal_negated(normalized: str, match_start: int) -> bool:
  prefix = normalized[max(0, match_start - 50):match_start]
  return _LOCAL_LEGAL_NEGATION.search(prefix) is not None


def _legal_candidate_spans(clause: str) -> list[Span]:
  found = spans(_LEGAL_DIRECT, clause)
  found += chain_spans(clause, (_LEGAL_ACTION_VERB, _LEGAL_TARGET))
  found += chain_spans(
    clause,
    (_LEGAL_IF_NOT, _LEGAL_IF_NOT_RESOLVE, _LEGAL_IF_NOT_APPEAL, _LEGAL_IF_NOT_TARGET),
  )
  found += chain_spans(clause, (_LEGAL_IF_NOT, _LEGAL_FILE_IN_COURT))
  found += chain_spans(clause, (_LEGAL_DEMAND_SOLVE, _LEGAL_OTHERWISE_APPEAL, _LEGAL_REGULATOR))
  found += chain_spans(clause, (_LEGAL_PUBLICITY_VERB, _LEGAL_PUBLICITY_TARGET))
  found += chain_spans(clause, (_LEGAL_WRITE_VERB, _LEGAL_MEDIA))
  return sorted(found)


def _historical_legal_present(clause: str) -> bool:
  return _HISTORICAL_LEGAL.search(clause) is not None or bool(
    chain_spans(clause, _HISTORICAL_LEGAL_CHAIN),
  )


def _legal_declined_action_present(clause: str) -> bool:
  return _LEGAL_DECLINED_ACTION.search(clause) is not None or bool(
    chain_spans(clause, _LEGAL_DECLINED_CHAIN),
  )


def _clause_has_legal_escalation(
  clause: str,
  *,
  full_text: str,
  clause_offset: int,
) -> bool:
  if _INFORMATIONAL_LEGAL.search(clause) or _historical_legal_present(clause):
    return False
  if _legal_declined_action_present(clause):
    return False
  # Look-alike words ('судя по', 'работает в суде' ...) cancel only the candidate they overlap.
  near_miss = Masks(spans(_LEGAL_NEAR_MISS, clause))
  for start, end in _legal_candidate_spans(clause):
    if near_miss.overlaps((start, end)):
      continue
    if _is_local_legal_negated(full_text, clause_offset + start):
      continue
    return True
  return False


def _legal_clauses(normalized: str) -> list[tuple[str, int]]:
  """Split into sentence / comma / 'но' clauses, keeping each clause's offset in the text."""
  clauses: list[tuple[str, int]] = []

  def _append(start: int, end: int) -> None:
    raw = normalized[start:end]
    clause = raw.strip()
    if clause:
      clauses.append((clause, start + (len(raw) - len(raw.lstrip()))))

  cursor = 0
  for separator in _LEGAL_CLAUSE_SEPARATOR.finditer(normalized):
    _append(cursor, separator.start())
    cursor = separator.end()
  _append(cursor, len(normalized))
  return clauses


def _match_legal_escalation(normalized: str) -> bool:
  for clause, clause_offset in _legal_clauses(normalized):
    if _clause_has_legal_escalation(
      clause,
      full_text=normalized,
      clause_offset=clause_offset,
    ):
      return True
  return False


def _fraud_candidate_spans(normalized: str) -> list[Span]:
  found = spans(_FRAUD_DIRECT, normalized)
  found += chain_spans(normalized, (_FRAUD_DEBITED_MONEY, _FRAUD_NO_CONSENT))
  found += chain_spans(normalized, (_FRAUD_DEBIT_VERB, _FRAUD_UNKNOWN_MARKER))
  found += chain_spans(
    normalized,
    (_FRAUD_REPEAT_MARKER, _FRAUD_DEBIT_OR_OPERATION, _FRAUD_UNKNOWN_MARKER_STRICT),
  )
  found += chain_spans(normalized, (_FRAUD_UNKNOWN_OR_REPEATED, _FRAUD_DEBIT_OPERATION_OR_SUM))
  found += chain_spans(
    normalized,
    (_FRAUD_ACCOUNT_SUBJECT, _FRAUD_REPEAT_TWICE, _FRAUD_DEBIT_VERB_PLAIN),
  )
  found += chain_spans(normalized, (_FRAUD_PAYMENT_WORD, _FRAUD_UNKNOWN_DETAILS))
  found += chain_spans(normalized, (_FRAUD_ACCOUNT_USE, _FRAUD_MY_ACCOUNT))
  found += chain_spans(normalized, (_FRAUD_ACCOUNT_WORD, _FRAUD_FOREIGN))
  found += chain_spans(normalized, (_FRAUD_SIGNS, _FRAUD_SCAM_WORD, _FRAUD_FINANCIAL_WORD))
  return found


def _match_fraud_indicators(normalized: str) -> bool:
  candidates = _fraud_candidate_spans(normalized)
  if not candidates:
    return False
  # Routine-charge wording and rhetorical accusations cancel only the candidate they overlap.
  # A concrete unknown-charge statement elsewhere in the message is never cancelled by them.
  masks = Masks(
    spans(_FRAUD_CHARGE_EXCLUSION, normalized) + spans(_FRAUD_RHETORICAL, normalized),
  )
  return has_unmasked(candidates, masks)


def _match_dangerous_foreign_object(normalized: str) -> bool:
  if not _FOOD_CONTEXT.search(normalized):
    return False
  candidates = spans(_DANGEROUS_OBJECT, normalized) + chain_spans(
    normalized,
    (_DANGEROUS_FRAGMENT, _DANGEROUS_HAZARD_WORD),
  )
  if not candidates:
    return False
  # Look-alike objects (hair, soft plastic, metallic taste ...) cancel only the object they
  # cover; a dangerous object stated elsewhere in the message still matches.
  masks = Masks(spans(_FOREIGN_OBJECT_EXCLUSION, normalized))
  return has_unmasked(candidates, masks)


def _match_direct_threat(normalized: str) -> bool:
  candidates = chain_spans(normalized, (_THREAT_ACTION, _STAFF_TARGET_PATTERN))
  candidates += chain_spans(normalized, (_STAFF_TARGET_PATTERN, _THREAT_HARM))
  candidates += chain_spans(normalized, (_THREAT_SUBJECT, _THREAT_PHYSICAL_SHOW))
  if not candidates:
    return False
  # Procedural / reversed-role wording cancels only the phrase it overlaps.
  masks = Masks(spans(_THREAT_EXCLUSION, normalized))
  return has_unmasked(candidates, masks)


def _informational_refund_present(normalized: str) -> bool:
  if _INFORMATIONAL_REFUND.search(normalized):
    return True
  return any(chain_spans(normalized, chain) for chain in _INFORMATIONAL_REFUND_CHAINS)


def _match_refund_request(normalized: str) -> bool:
  if _informational_refund_present(normalized):
    return False
  return _REFUND_REQUEST.search(normalized) is not None


def _match_personal_data_high(normalized: str) -> bool:
  return _PERSONAL_DATA_HIGH.search(normalized) is not None or bool(
    chain_spans(normalized, _PERSONAL_DATA_HIGH_CHAIN),
  )


def _match_non_delivery(normalized: str) -> bool:
  candidates = _non_delivery_total_spans(normalized) + spans(_NON_DELIVERY_ACTIVE, normalized)
  if not candidates:
    return False
  # Informational / in-progress / partial wording cancels only the non-delivery claim it
  # overlaps, never a separate total non-delivery statement elsewhere in the message.
  masks = Masks(
    spans(_NON_DELIVERY_INFORMATIONAL, normalized)
    + chain_spans(
      normalized,
      (_NON_DELIVERY_IN_PROGRESS_START, _NON_DELIVERY_IN_PROGRESS_END),
    )
    + spans(_NON_DELIVERY_PARTIAL_EXCLUSION, normalized),
  )
  return has_unmasked(candidates, masks)


def _official_written_candidate_spans(normalized: str) -> list[Span]:
  found = spans(_OFFICIAL_DIRECT, normalized)
  found += chain_spans(
    normalized,
    (_OFFICIAL_DEMAND_VERB, _OFFICIAL_ANSWER, _OFFICIAL_COMPLAINT_WORD),
  )
  found += chain_spans(normalized, (_OFFICIAL_ASK_WRITTEN, _OFFICIAL_ASK_WRITTEN_TOPIC))
  found += chain_spans(normalized, (_OFFICIAL_PROVIDE, _OFFICIAL_ANSWER, _OFFICIAL_CLAIM))
  return found


def _match_official_written_response(normalized: str) -> bool:
  candidates = _official_written_candidate_spans(normalized)
  if not candidates:
    return False
  # Informational, declined, already-received and service-info wording cancels only the demand
  # it overlaps; a genuine demand elsewhere in the message still matches.
  masks = Masks(
    spans(_INFORMATIONAL_OFFICIAL_RESPONSE, normalized)
    + spans(_OFFICIAL_WRITTEN_DECLINED, normalized)
    + spans(_OFFICIAL_WRITTEN_RECEIVED, normalized)
    + spans(_OFFICIAL_WRITTEN_SERVICE_INFO, normalized),
  )
  return has_unmasked(candidates, masks)


@dataclass(frozen=True)
class _RiskRule:
  reason_code: RiskReasonCode
  level: RiskLevel
  rule_id: str
  matcher: Callable[[str], bool]


_RULES: tuple[_RiskRule, ...] = (
  _RiskRule(
    RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION,
    RiskLevel.CRITICAL,
    "health_symptoms_after_consumption",
    _match_health_symptoms,
  ),
  _RiskRule(
    RiskReasonCode.DANGEROUS_FOREIGN_OBJECT,
    RiskLevel.CRITICAL,
    "dangerous_foreign_object",
    _match_dangerous_foreign_object,
  ),
  _RiskRule(
    RiskReasonCode.MASS_INCIDENT,
    RiskLevel.CRITICAL,
    "mass_incident",
    _match_mass_incident,
  ),
  _RiskRule(
    RiskReasonCode.FRAUD_INDICATORS,
    RiskLevel.CRITICAL,
    "fraud_indicators",
    _match_fraud_indicators,
  ),
  _RiskRule(
    RiskReasonCode.DIRECT_THREAT,
    RiskLevel.CRITICAL,
    "direct_threat",
    _match_direct_threat,
  ),
  _RiskRule(
    RiskReasonCode.DELAY_OVER_120_MINUTES,
    RiskLevel.HIGH,
    "delay_over_120_minutes",
    _match_delay_over_120,
  ),
  _RiskRule(
    RiskReasonCode.NON_DELIVERY,
    RiskLevel.HIGH,
    "non_delivery",
    _match_non_delivery,
  ),
  _RiskRule(
    RiskReasonCode.FALSE_DELIVERY_STATUS,
    RiskLevel.HIGH,
    "false_delivery_status",
    lambda text: _FALSE_DELIVERY.search(text) is not None,
  ),
  _RiskRule(
    RiskReasonCode.PACKAGE_TAMPERING,
    RiskLevel.HIGH,
    "package_tampering_explicit",
    _match_package_tampering,
  ),
  _RiskRule(
    RiskReasonCode.FOOD_SPOILAGE,
    RiskLevel.HIGH,
    "food_spoilage",
    lambda text: _FOOD_SPOILAGE.search(text) is not None,
  ),
  _RiskRule(
    RiskReasonCode.LEGAL_OR_REGULATORY_ESCALATION,
    RiskLevel.HIGH,
    "legal_or_regulatory_escalation",
    _match_legal_escalation,
  ),
  _RiskRule(
    RiskReasonCode.OFFICIAL_WRITTEN_RESPONSE,
    RiskLevel.HIGH,
    "official_written_response_demand",
    _match_official_written_response,
  ),
  _RiskRule(
    RiskReasonCode.PERSONAL_DATA_EXPOSURE,
    RiskLevel.CRITICAL,
    "personal_data_exposure_critical",
    lambda text: _PERSONAL_DATA_CRITICAL.search(text) is not None,
  ),
  _RiskRule(
    RiskReasonCode.PERSONAL_DATA_EXPOSURE,
    RiskLevel.HIGH,
    "personal_data_exposure_high",
    _match_personal_data_high,
  ),
  _RiskRule(
    RiskReasonCode.DELAY_OVER_30_MINUTES,
    RiskLevel.MEDIUM,
    "delay_over_30_minutes",
    _match_delay_over_30,
  ),
  _RiskRule(
    RiskReasonCode.MISSING_ITEM,
    RiskLevel.MEDIUM,
    "missing_item",
    _match_missing_item,
  ),
  _RiskRule(
    RiskReasonCode.REFUND_REQUEST,
    RiskLevel.MEDIUM,
    "refund_request",
    _match_refund_request,
  ),
  _RiskRule(
    RiskReasonCode.WRONG_ITEM,
    RiskLevel.MEDIUM,
    "wrong_item",
    _match_wrong_item,
  ),
)


def assess_deterministic_risk(
  request: RiskAssessmentRequest,
) -> DeterministicRiskResult:
  """Assess deterministic risk floor from customer query text only.

  The request model already bounds the message length, so every rule below runs on a text of at
  most ``MAX_CUSTOMER_QUERY_CHARS`` characters.

  Result semantics (see ``DeterministicRiskResult.assessment_status``):

  * a high or critical rule matched      -> ``RULE_MATCH`` (always wins, in any language);
  * text the Russian rules cannot assess -> ``UNSUPPORTED_LANGUAGE``: not classified, manual
    review required. Weaker (low / medium) matches in such text are not trusted either;
  * a lower rule matched                 -> ``RULE_MATCH``;
  * nothing matched in supported text    -> ``NO_SIGNAL``: the absence of a deterministic
    signal, never an affirmative low-risk conclusion.
  """
  normalized = normalize_for_matching(request.customer_query)
  matched: list[RiskSignal] = []

  for rule in _RULES:
    if rule.matcher(normalized):
      matched.append(
        RiskSignal(
          reason_code=rule.reason_code,
          level=rule.level,
          rule_id=rule.rule_id,
        ),
      )

  has_severe_match = any(
    risk_rank(signal.level) >= risk_rank(RiskLevel.HIGH) for signal in matched
  )
  if not has_severe_match and not is_supported_language(normalized):
    return build_unsupported_language_result()

  return build_deterministic_risk_result(matched)
