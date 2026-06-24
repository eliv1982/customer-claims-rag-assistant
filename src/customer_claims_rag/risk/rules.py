"""Deterministic risk matchers and assessment (3B.1 safety floor)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from customer_claims_rag.risk.models import (
  DeterministicRiskResult,
  RiskAssessmentRequest,
  RiskLevel,
  RiskSignal,
)
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.validator import build_deterministic_risk_result

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
    r"|(?:вчера|позавчера).*(?:не\s+доставил\w*|не\s+приехал\w*)"
    r"|полн\w+\s+недоставк\w*"
    r"|статус\s+не\s+[«\"']?доставлен\w*"
    r")",
)

_FALSE_DELIVERY = re.compile(
  r"(?:"
  r"статус\w*\s+[«\"']?доставлен\w*[»\"']?"
  r"|[«\"']?доставлен\w*[»\"']?\s*,?\s*но"
  r"|доставлен\w*.*(?:не\s+получ|ничего\s+не\s+получ)"
  r")",
)

_MISSING_ITEM = re.compile(
  r"(?:"
  r"не\s+хватил\w*"
  r"|не\s+хватает\s+(?:одн\w*|позиц\w*)"
  r"|отсутств\w+\s+позиц\w*"
  r"|недостач\w*"
  r"|не\s+привез\w*\s+(?:одн\w+\s+)?позиц\w*"
  r"|не\s+довез\w*\s+(?:одн\w+\s+)?позиц\w*"
  r"|(?:в\s+)?заказ\w*\s+не\s+привез\w+"
  r"|(?:мне\s+)?не\s+положил\w+"
  r"|отсутствует\s+один\s+товар"
  r"|одн\w+\s+позиц\w*\s+нет"
  r"|забыл\w+\s+положить"
  r"|забыли\s+положить"
  r"|привез\w*\s+не\s+весь\s+заказ"
  r"|часть\s+заказ\w*\s+не\s+доставил\w*"
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
  r"|(?:еще\s+)?не\s+знаю.*(?:привез|позиц)"
  r"|сколько\s+позиций"
  r")",
)

_REFUND_REQUEST = re.compile(
    r"(?:"
    r"требую\s+(?:полный\s+)?(?:возврат|возвратить\s+стоимост\w*)"
    r"|верните\s+деньг\w*"
    r"|верните\s+уплаченн\w+\s+сумм\w*"
    r"|верните\s+сумм\w*.*(?:заплатил\w*|оплатил\w*)"
    r"|вернуть\s+деньг\w*"
    r"|(?:прошу|требую)\s+вернуть\s+уплаченн\w+"
    r"|полный\s+возврат"
    r"|возврат\s+(?:денег|средств|полной\s+стоимости)"
    r"|хочу\s+оформить\s+возврат"
    r"|прошу\s+компенсир\w+\s+стоимост\w*"
    r"|компенсац\w*\s+за"
    r")",
)

_INFORMATIONAL_REFUND = re.compile(
  r"(?:"
  r"можно\s+ли\s+(?:вообще\s+)?вернуть"
  r"|можно\s+ли\s+вернуть\s+уплаченн\w+"
  r"|какие\s+.*правил\w*\s+возврат"
  r"|где\s+прочитать\s+.*(?:возврат|компенсац)"
  r"|где\s+посмотреть\s+сумм\w*\s+оплат\w*"
  r"|что\s+означает\s+уплаченн\w+\s+сумм\w*"
  r"|объясните\s+.*(?:срок\w*|правил\w*).*(?:возврат|компенсац|зачислен)"
  r"|что\s+такое\s+.*(?:возврат|компенсац|сложн\w+\s+случа)"
  r")",
)

_PACKAGE_TAMPERING = re.compile(
  r"(?:"
  r"вскрыт\w*"
  r"|сорван\w*\s+пломб\w*"
  r"|пломб\w*\s+сорван\w*"
  r"|нарушен\w*\s+герметичност\w*"
  r"|герметичност\w*\s+нарушен\w*"
  r"|вмешательств\w*"
  r"|контейнер\w*\s+открыт\w*"
  r"|открыт\w*\s+контейнер\w*"
  r"|крышк\w*\s+не\s+сидит"
  r"|крышк\w*\s+треснул\w*"
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

_LEGAL_ESCALATION_POSITIVE = re.compile(
  r"(?:"
  r"(?:подам|подал\w*|обращ\w+|напиш\w*|писал\w*|направ\w+|передам|буду\s+писать|пойду)"
  r".*(?:жалоб\w+)?.*(?:в|к)\s+"
  r"(?:суд|полици\w*|прокуратур\w*|роспотребнадзор\w*|регулятор\w*|контролирующ\w+\s+орган|надзорн\w+\s+орган)"
  r"|жалоб\w*\s+(?:в|регулятор)"
  r"|подам\s+иск"
  r"|если\s+не\s+.*(?:верн\w*|реш\w*).*(?:обращ\w*|подам).*(?:суд|регулятор|полици)"
  r"|если\s+не\s+.*\s+подам\s+в\s+суд"
  r"|требую\s+решить.*иначе\s+обращ\w*.*регулятор"
  r"|(?:передам|вылож\w*|опублик\w*|распростран\w+)"
  r".*(?:истори\w+|информац\w+|материал\w+|все).*(?:сми|журналист\w*|соцсет\w*)"
  r"|(?:передам|вылож\w*|опублик\w*|распростран\w+).*(?:сми|журналист\w*|соцсет\w*)"
  r"|(?:напиш\w*|обращ\w*).*(?:журналист\w*|сми)"
  r"|(?:решил\w*|решили)\s+подать\s+жалоб"
  r"|(?:решил\w*|решили)\s+обращаться"
  r"|подам\s+иск"
  r")",
)

_INFORMATIONAL_LEGAL = re.compile(
  r"(?:"
  r"что\s+(?:означает|такое)\s+(?:\bсми\b|контролирующ\w+\s+орган|регулятор|прокуратур\w*)"
  r"|где\s+(?:посмотреть|найти).*(?:раздел\s+)?\bсми\b"
  r"|кто\s+такой\s+регулятор"
  r"|какие\s+функции\s+выполняет"
  r"|где\s+найти\s+информаци\w+.*суд"
  r"|расскаж\w+.*что\s+такое"
  r")",
)

_HISTORICAL_LEGAL = re.compile(
  r"(?:"
  r"читал\w+.*(?:стать\w+|новост\w+).*(?:роспотребнадзор|суд|прокуратур)"
  r"|в\s+новост\w+\s+говорил"
  r"|сми\s+часто\s+пиш"
  r")",
)

_LEGAL_DECLINED_ACTION = re.compile(
  r"(?:"
  r"не\s+(?:буду|собира\w*|планиру\w*).*(?:обращ\w*|писать|пис\w*|подавать|подам).*(?:суд|прокуратур|полици|\bсми\b|регулятор|контролирующ\w+\s+орган|роспотребнадзор)"
  r"|(?:в\s+)?\bсми\b\s+обращаться\s+не\s+планиру"
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
  r"|раздел\s+.*отзыв"
  r")",
)

_LOCAL_LEGAL_NEGATION = re.compile(
  r"(?:"
  r"не\s+(?:хочу|собира\w*|буду|будем)\s*$"
  r"|пока\s+не\s*(?:буду|обращ\w*|пойду|подам|напиш\w*|пис\w*)?\s*$"
  r"|не\s+(?:говорил\w*|обращал\w*|писал\w*)\s*(?:,\s*)?(?:что\s+)?$"
  r")",
)

_ORDINARY_REVIEW = re.compile(
  r"(?:"
  r"напиш\w*\s+отзыв"
  r"|остав\w*\s+отзыв"
  r"|угрожа\w*\s+.*отзыв"
  r")",
)

_OFFICIAL_WRITTEN_RESPONSE_DEMAND = re.compile(
  r"(?:"
  r"(?:требую|прошу).*(?:предостав\w+|направ\w+)?.*(?:официальн\w+\s+)?(?:письменн\w+\s+)?ответ"
  r".*(?:претенз\w+|жалоб\w+|обращени\w+|нарушени\w+|возврат|компенсац|инцидент|недоставк|поврежден|по\s+существу)"
  r"|(?:мне\s+)?нуж\w+\s+официальн\w+\s+ответ\w*\s+на\s+(?:мою\s+)?претенз\w+"
  r"|(?:требую|прошу)\s+(?:официальн\w+\s+)?(?:письменн\w+\s+)?ответ.*(?:претенз|жалоб|обращени|возврат|недоставк|поврежден)"
  r"|(?:требую|прошу)\s+официальн\w+\s+ответ\s+в\s+связи\s+с"
  r"|прошу\s+письменно\s+ответить.*(?:обращени|жалоб|претенз|возврат)"
  r"|(?:сообщ\w+\s+срок\w*|предостав\w+).*(?:официальн\w+\s+)?(?:письменн\w+\s+)?ответ.*претенз"
  r")",
)

_INFORMATIONAL_OFFICIAL_RESPONSE = re.compile(
  r"(?:"
  r"где\s+посмотреть.*(?:правил\w*|письменн\w+)"
  r"|правил\w*\s+подготовк\w+.*письменн\w+"
  r"|что\s+(?:обычно\s+)?означает.*официальн\w+.*ответ"
  r"|можно\s+ли\s+получить\s+информаци\w+.*(?:срок\w*|ответ\w*)"
  r"|(?:еще\s+)?не\s+решил.*письменн\w+"
  r"|нужна\s+ли\s+мне\s+письменн\w+\s+претенз\w+"
  r")",
)

_OFFICIAL_WRITTEN_SERVICE_INFO = re.compile(
  r"(?:"
  r"ответ\s+о\s+(?:правил\w+|график)"
  r"|ответ.*правил\w+\s+доставк"
  r"|(?:рассказать|объяснить).*(?:правил\w*|подписк|график\w*\s+работ)"
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
    r"|имя\s+и\s+.*адрес.*(?:чуж\w+|не\s+мо\w*)"
    r")",
)

_PERSONAL_DATA_CRITICAL = re.compile(
    r"(?:"
    r"массов\w+\s+утечк\w*"
    r"|серьезн\w+\s+утечк\w*"
    r"|(?:сотен|сотни)\s+клиент\w*"
    r"|списк\w*.*(?:сотен|сотни)\s+клиент\w*"
    r"|утечк\w*\s+данн\w*.*(?:сотен|сотни|мног\w+\s+клиент\w*)"
    r"|массов\w+\s+(?:публикац\w*|выгрузк\w*)\s+.*данн\w*"
    r")",
)

_CONSUMPTION_LINK = re.compile(
  r"(?:"
  r"после\s+(?:еды|ужин\w*|обед\w*|завтрак\w*|блюд\w*|суп\w*|заказ\w*|доставк\w*|употреблен\w*|ваш\w+\s+(?:ед\w*|блюд\w*|суп\w*|заказ\w*))"
  r"|(?:ед\w*|блюд\w*|суп\w*|заказ\w*).*(?:после|отравлен\w*)"
  r"|(?:тошнит|рвот\w*|заболел\w*|аллерг\w*|задыха\w*|одышк\w*|трудно\s+дыш|тяжело\s+дыш|хрип\w*).*(?:заказ\w*|ед\w*|блюд\w*)"
  r"|(?:заказ\w*|ед\w*|блюд\w*).*(?:тошнит|рвот\w*|заболел\w*|аллерг\w*|задыха\w*|одышк\w*|трудно\s+дыш|тяжело\s+дыш|хрип\w*)"
  r"|(?:поел\w*|съел\w*|употребил\w*)"
  r"|(?:доставк\w*|привез\w*|доставил\w*).*(?:после|ед\w*|блюд\w*)"
  r")",
)

_HEALTH_SYMPTOM = re.compile(
  r"(?:"
  r"тошнот\w*"
  r"|тошнит"
  r"|рвот\w*"
  r"|боль\s+в\s+живот\w*"
  r"|дискомфорт\w*\s+в\s+живот\w*"
  r"|стало\s+плохо"
  r"|плохо\s+стало"
  r"|ухудшен\w+\s+самочувств\w*"
  r"|аллергическ\w+\s+реакц\w*"
  r"|отравлен\w*"
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
  r")",
)

_HEALTH_INFORMATIONAL = re.compile(
  r"(?:"
  r"что\s+делать.*трудно\s+дышать"
  r"|расскаж\w+.*что\s+означает.*одышк"
  r"|что\s+означает.*одышк"
  r")",
)

_HEALTH_GLOBAL_EXCLUSION = re.compile(
  r"(?:"
  r"(?:было|была)\s+плохо\s+.*(?:до\s+заказ|до\s+доставк)"
  r"|будто\s+от\s+не[её]\s+станет\s+плохо"
  r"|неприятно\s+смотреть"
  r"|плохо\s+работает"
  r"|трудно\s+дышать\s+картинк"
  r"|задыха\w*\s+после\s+пробежк"
  r")",
)

_STAFF_SYMPTOM_SUBJECT = re.compile(
  r"(?:"
  r"(?:курьер\w*|сотрудник\w*|менеджер\w*)\w*\s+(?:стало\s+плохо|плохо\s+стало|заболел\w*|тошнит|рвот\w*)"
  r"|(?:курьер\w*|сотрудник\w*|менеджер\w*)\w*\s+.*(?:сказал\w*|говор\w*).*(?:ем\w+\s+)?плохо"
  r"|(?:курьер\w*|сотрудник\w*|менеджер\w*)\w*\s+.*(?:по\s+дорог\w*|на\s+лестниц\w*).*(?:заболел\w*|плохо)"
  r"|(?:курьер\w*|сотрудник\w*|менеджер\w*)\w*\s+.*(?:тяжело\s+дыш|трудно\s+дыш|задыха|одышк|хрип|не\s+хватает\s+воздух)"
  r"|(?:курьер\w*|сотрудник\w*|менеджер\w*)\w*\s+.*(?:появил\w*|начал\w*).*(?:одышк|трудно\s+дыш|задыха|хрип)"
  r"|(?:после\s+еды|после\s+блюд\w*).*(?:курьер\w*|сотрудник\w*|менеджер\w*)"
  r")",
)

_HEALTH_LOCAL_WINDOW = 70

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
  r"|(?:пластик\w*|осколк\w*|фрагмент\w*).*(?:остр\w+|опасн\w+|травм\w*|риск\w*)"
  r")",
)

_FOREIGN_OBJECT_EXCLUSION = re.compile(
  r"(?:"
  r"волос\w*"
  r"|мягк\w+\s+пластик\w*"
  r"|пластик\w*(?!\s*(?:остр\w+|опасн\w+|тверд\w+|фрагмент\w*))"
  r"|металлическ\w+\s+привкус"
  r"|на\s+стол\w*.*нож"
  r"|курьер\w*.*металлическ\w+\s+сумк\w*"
  r"|крошк\w+\s+хлеб"
  r")",
)

_SYMPTOM_CLUSTER = r"(?:плохо\s+стало|стало\s+плохо|заболел\w*|отравил\w*|симптом\w*)"

_MASS_INCIDENT_HEALTH = re.compile(
  r"(?:"
  r"(?:нескольк\w+\s+(?:люд\w*|человек|клиент\w*)|у\s+всей\s+семь\w*).*" + _SYMPTOM_CLUSTER +
  r"|(?:нескольк\w+\s+клиент\w*|несколько\s+человек).*(?:после|партии|еды|заказ\w*)"
  r"|(?:партии|партия).*" + _SYMPTOM_CLUSTER +
  r"|у\s+всех\s+похож\w+\s+симптом\w*"
  r"|(?:сразу\s+)?(?:двум|троим|трем|трое|четырем|пятерым|\d+)\s+(?:люд\w*|человек)"
  r"|(?:двум|троим|трем|трое|четырем|пятерым|\d+)\s+член\w+\s+семь\w*.*" + _SYMPTOM_CLUSTER +
  r"|(?:после|партии|еды|блюд\w*|заказ\w*).*" + _SYMPTOM_CLUSTER + r".*(?:двум|троим|трем|трое|четырем|пятерым|\d+\s+люд\w*)"
  r"|" + _SYMPTOM_CLUSTER + r".*(?:двум|троим|трем|трое|четырем|пятерым|\d+\s+люд\w*)"
  r")",
)

_MASS_INCIDENT_GLOBAL_EXCLUSION = re.compile(
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

_FRAUD_INDICATORS = re.compile(
    r"(?:"
    r"несанкционированн\w+\s+списан\w*"
    r"|списал\w*\s+деньг\w*.*без\s+(?:моего\s+)?соглас"
    r"|(?:списыва\w+|списал\w+|списан\w*).*(?:неизвестн\w+|повторн\w+|без\s+соглас|не\s+совершал\w*)"
    r"|(?:повторн\w+|нескольк\w+\s+раз|продолжа\w+).*(?:списан\w*|операц\w*).*(?:неизвестн\w+|без\s+соглас|не\s+совершал\w*)"
    r"|(?:неизвестн\w+|повторн\w+).*(?:списан\w*|операц\w*|сумм\w*)"
    r"|(?:с\s+)?(?:карт\w*|счет\w*|счета).*(?:повторн\w+|нескольк\w+\s+раз).*(?:списан\w*|списыва\w*)"
    r"|подменил\w*\s+реквизит\w*"
    r"|оплат\w*.*неизвестн\w+\s+реквизит\w*"
    r"|повторн\w+\s+неизвестн\w+\s+списан\w*"
    r"|(?:использовал\w*|оформля\w*).*(?:мой\s+аккаунт|мою\s+карт\w*)"
    r"|(?:аккаунт\w*|карт\w*).*(?:чуж\w+|мошенник\w*|без\s+моего\s+соглас)"
    r"|странн\w+\s+списан\w*"
    r"|признак\w*\s+.*мошен.*(?:списан|аккаунт|карт|реквизит)"
    r")",
)

_FRAUD_CHARGE_EXCLUSION = re.compile(
  r"(?:"
  r"списал\w*\s+деньг\w*\s+за\s+(?:мой\s+)?заказ"
  r"|сумма\s+списал\w*\s+один\s+раз\s+корректно"
  r"|обычн\w+\s+списан\w*"
  r"|когда\s+пройдет\s+.*списан\w*"
  r"|не\s+понима\w*,\s+когда\s+пройдет"
  r")",
)

_FRAUD_EXCLUSION = re.compile(
  r"(?:"
  r"(?:вы|это)\s+.*мошенник\w*"
  r"|мошенничеств\w*\?"
  r"|похож\w*\s+на\s+развод"
  r"|ignore\s+instructions"
  r"|risk_level\s*=\s*critical"
  r")",
)

_STAFF_TARGET = (
  r"(?:курьер\w*|сотрудник\w*|менеджер\w*|оператор\w*|работник\w*|представител\w+\s+компани\w*)"
)

_DIRECT_THREAT = re.compile(
    r"(?:"
    r"(?:я|мы)\s+(?:удар\w*|найду\w*|расправ\w*|причин\w+).*" + _STAFF_TARGET +
    r"|" + _STAFF_TARGET + r".*(?:не\s+поздоровится|удар\w*|вред)"
    r"|(?:я|мы)\s+(?:.*\s+)?(?:ему|ей)\s+физическ\w+\s+покаж\w+"
    r")",
)

_THREAT_EXCLUSION = re.compile(
  r"(?:"
  r"угрожа\w*\s+(?:написать|обратиться|оставить|подать).*(?:жалоб\w*|суд|отзыв|полици|менеджер\w*)"
  r"|(?:курьер\w*|менеджер\w*|сотрудник\w*|оператор\w*).*(?:угрожа\w*|угроз\w+).*(?:мне|клиент)"
  r"|если\s+не\s+верн\w*.*(?:полици|суд)"
  r"|(?:я|мы)\s+физическ\w+\s+(?:покаж\w+|передам\w+)\s+(?:документ|бумаг)"
  r"|(?:я|мы)\s+физическ\w+\s+передам\w+\s+бумаг"
  r"|(?:я|мы)\s+покаж\w+\s+" + _STAFF_TARGET + r".*где\s+лежит\s+заказ"
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
  if re.search(r"^\s*(?:минут|мин\.?|час).*(?:привезут|приедет|будет\s+достав)", trailing):
    return True
  return False


def extract_delay_minutes(normalized: str) -> list[Decimal]:
  """Extract delay durations in minutes from normalized customer text."""
  if not _has_delay_context(normalized):
    return []

  if _NEGATIVE_DURATION.search(normalized):
    return []

  minutes: list[Decimal] = []
  consumed_spans: list[tuple[int, int]] = []

  if _GREATER_THAN_TWO_HOURS.search(normalized):
    minutes.append(Decimal("121"))

  for match in _COMPOSITE_NUMERIC_HOURS_MINUTES.finditer(normalized):
    total = _hours_to_minutes(match.group(1)) + Decimal(match.group(2))
    minutes.append(total)
    consumed_spans.append(match.span())

  for match in _COMPOSITE_WORD_HOURS_MINUTES.finditer(normalized):
    word_hours = _parse_word_hours(match.group(1))
    if word_hours is not None:
      total = word_hours + Decimal(match.group(2))
      minutes.append(total)
      consumed_spans.append(match.span())

  def _span_consumed(start: int, end: int) -> bool:
    return any(start >= span_start and end <= span_end for span_start, span_end in consumed_spans)

  for match in _MINUTES_PATTERN.finditer(normalized):
    if _span_consumed(*match.span()):
      continue
    if _is_future_duration(normalized, match.start(1)):
      continue
    value = int(match.group(1))
    if value <= 0:
      continue
    minutes.append(Decimal(value))

  for match in _HOURS_NUMERIC.finditer(normalized):
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


def _is_staff_attributed_symptom(normalized: str, symptom_start: int) -> bool:
  local_start = max(0, symptom_start - _HEALTH_LOCAL_WINDOW)
  local_span = normalized[local_start:symptom_start + 20]
  return _STAFF_SYMPTOM_SUBJECT.search(local_span) is not None


def _match_health_symptoms(normalized: str) -> bool:
  if _HEALTH_GLOBAL_EXCLUSION.search(normalized):
    return False
  if _HEALTH_INFORMATIONAL.search(normalized):
    return False
  if not _CONSUMPTION_LINK.search(normalized):
    return False
  for match in _HEALTH_SYMPTOM.finditer(normalized):
    if not _is_staff_attributed_symptom(normalized, match.start()):
      return True
  return False


def _match_missing_item(normalized: str) -> bool:
  if _INFORMATIONAL_MISSING_ITEM.search(normalized):
    return False
  if _NON_DELIVERY.search(normalized) or _PARTIAL_MISSING_EXCLUSION.search(normalized):
    return False
  return _MISSING_ITEM.search(normalized) is not None


def _split_clauses(normalized: str) -> list[str]:
  parts = _CLAUSE_SPLIT.split(normalized)
  return [part.strip() for part in parts if part.strip()]


def _is_hearsay_clause(clause: str) -> bool:
  return _MASS_HEARSAY.search(clause.strip()) is not None


def _match_mass_incident(normalized: str) -> bool:
  if _MASS_INCIDENT_GLOBAL_EXCLUSION.search(normalized):
    return False
  for clause in _split_clauses(normalized):
    if _is_hearsay_clause(clause):
      continue
    if _MASS_INCIDENT_HEALTH.search(clause):
      return True
  return False


def _is_local_legal_negated(normalized: str, match_start: int) -> bool:
  prefix = normalized[max(0, match_start - 50):match_start]
  return _LOCAL_LEGAL_NEGATION.search(prefix) is not None


def _clause_has_legal_escalation(
  clause: str,
  *,
  full_text: str,
  clause_offset: int,
) -> bool:
  if _INFORMATIONAL_LEGAL.search(clause) or _HISTORICAL_LEGAL.search(clause):
    return False
  if _LEGAL_DECLINED_ACTION.search(clause):
    return False
  if _ORDINARY_REVIEW.search(clause):
    for match in _LEGAL_ESCALATION_POSITIVE.finditer(clause):
      if _is_local_legal_negated(full_text, clause_offset + match.start()):
        continue
      return True
    return False
  for match in _LEGAL_ESCALATION_POSITIVE.finditer(clause):
    if _is_local_legal_negated(full_text, clause_offset + match.start()):
      continue
    return True
  return False


def _match_legal_escalation(normalized: str) -> bool:
  if _LEGAL_NEAR_MISS.search(normalized):
    return False
  if _INFORMATIONAL_LEGAL.search(normalized) and not re.search(
    r",\s*но\s+|\s+но\s+|,\s*",
    normalized,
  ):
    return False
  if _HISTORICAL_LEGAL.search(normalized) and not re.search(
    r",\s*но\s+|\s+но\s+|,\s*",
    normalized,
  ):
    return False

  clauses_with_offsets: list[tuple[str, int]] = []
  cursor = 0
  for part in re.split(r"(\s+но\s+|,\s*но\s+|,\s*)", normalized):
    if re.fullmatch(r"(\s+но\s+|,\s*но\s+|,\s*)", part):
      continue
    part = part.strip()
    if part:
      start = normalized.find(part, cursor)
      if start == -1:
        start = cursor
      clauses_with_offsets.append((part, start))
      cursor = start + len(part)

  if not clauses_with_offsets:
    clauses_with_offsets = [(normalized, 0)]

  for clause, clause_offset in clauses_with_offsets:
    if _clause_has_legal_escalation(
      clause,
      full_text=normalized,
      clause_offset=clause_offset,
    ):
      return True
  return False


def _match_fraud_indicators(normalized: str) -> bool:
  if _FRAUD_EXCLUSION.search(normalized):
    return False
  if _FRAUD_CHARGE_EXCLUSION.search(normalized):
    return False
  return _FRAUD_INDICATORS.search(normalized) is not None


def _match_dangerous_foreign_object(normalized: str) -> bool:
  if _FOREIGN_OBJECT_EXCLUSION.search(normalized):
    return False
  if not _FOOD_CONTEXT.search(normalized):
    return False
  return _DANGEROUS_OBJECT.search(normalized) is not None


def _match_direct_threat(normalized: str) -> bool:
  if _THREAT_EXCLUSION.search(normalized):
    return False
  return _DIRECT_THREAT.search(normalized) is not None


def _match_refund_request(normalized: str) -> bool:
  if _INFORMATIONAL_REFUND.search(normalized):
    return False
  return _REFUND_REQUEST.search(normalized) is not None


def _match_official_written_response(normalized: str) -> bool:
  if _INFORMATIONAL_OFFICIAL_RESPONSE.search(normalized):
    return False
  if _OFFICIAL_WRITTEN_DECLINED.search(normalized):
    return False
  if _OFFICIAL_WRITTEN_RECEIVED.search(normalized):
    return False
  if _OFFICIAL_WRITTEN_SERVICE_INFO.search(normalized):
    return False
  return _OFFICIAL_WRITTEN_RESPONSE_DEMAND.search(normalized) is not None


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
    lambda text: _NON_DELIVERY.search(text) is not None,
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
    lambda text: _PACKAGE_TAMPERING.search(text) is not None,
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
    lambda text: _PERSONAL_DATA_HIGH.search(text) is not None,
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
)


def assess_deterministic_risk(
  request: RiskAssessmentRequest,
) -> DeterministicRiskResult:
  """Assess deterministic risk floor from customer query text only."""
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

  return build_deterministic_risk_result(matched)
