# Схема метаданных документов и чанков

**Дата документа:** 2026-06-20  
**Проект:** customer-claims-rag-assistant  
**Компания:** FoodFlow (вымышленная)

---

## Назначение

Документ описывает единую схему метаданных для 10 документов базы знаний и их чанков. Метаданные используются при индексации, retrieval, фильтрации и разрешении конфликтов между источниками.

Внутренний реестр `docs/05_business_rules_registry.md` **не индексируется** в RAG и не использует эту схему для пользовательской базы.

---

## Уровни метаданных

| Уровень | Область | Назначение |
|---------|---------|------------|
| Документ | Весь файл в `data/02_clean_markdown/` | Идентификация, версия, статус, аудитория |
| Чанк | Фрагмент документа для embedding | Точный retrieval, риск, связи между документами |

---

## Обязательные метаданные документа

| Поле | Назначение | Формат | Допустимые значения / пример | Обязательное | Уровень |
|------|------------|--------|------------------------------|--------------|---------|
| `document_id` | Уникальный идентификатор документа | string, snake_case | `04_refund_policy` | Да | Документ |
| `title` | Человекочитаемое название | string | `Политика возвратов` | Да | Документ |
| `category` | Тематическая категория | string | `refunds`, `delivery`, `faq` | Да | Документ |
| `document_type` | Тип содержимого | enum | `policy`, `procedure`, `reference`, `faq`, `templates`, `guideline` | Да | Документ |
| `version` | Версия документа | string semver | `1.0.0` | Да | Документ |
| `status` | Статус жизненного цикла | enum | см. ниже | Да | Документ |
| `effective_date` | Дата начала действия версии | date `YYYY-MM-DD` | `2026-06-01` | Да | Документ |
| `last_updated` | Дата последнего изменения | date `YYYY-MM-DD` | `2026-06-20` | Да | Документ |
| `audience` | Целевая аудитория | string или list | `support_agent`, `customer_facing` | Да | Документ |
| `confidentiality` | Уровень конфиденциальности | enum | см. ниже | Да | Документ |
| `source_type` | Происхождение материала | enum | см. ниже | Да | Документ |
| `language` | Язык содержимого | enum | `ru` | Да | Документ |
| `priority` | Приоритет документа при конфликтах | enum | см. ниже | Да | Документ |

### `source_type`

| Значение | Описание |
|----------|----------|
| `internal_policy` | Внутренняя политика (доставка, возвраты, компенсации и т.д.) |
| `internal_reference` | Внутренний справочник (обзор сервиса, глоссарий) |
| `internal_procedure` | Внутренняя процедура обработки обращений (SOP поддержки) |
| `internal_guideline` | Внутренние руководства по стилю и шаблонам ответов |
| `internal_faq` | Внутренний FAQ для поддержки и ассистента |
| `sop` | Стандартная операционная процедура |
| `faq` | Частые вопросы и ответы (устаревшее общее значение; для `10_customer_faq` используется `internal_faq`) |
| `brand_guide` | Стиль коммуникаций и шаблоны ответов |

### Канонический словарь `category`

Поле `category` **не является закрытым enum** — допускаются новые значения при расширении базы. Для текущих 10 документов используются:

| Значение | Документы |
|----------|-----------|
| `general` | `01_service_overview` |
| `delivery` | `02_delivery_rules` |
| `orders` | `03_order_changes_and_cancellations` |
| `refunds` | `04_refund_policy` |
| `compensation` | `05_compensation_policy` |
| `food_quality` | `06_food_quality_and_packaging` |
| `complaints` | `07_complaint_handling_procedure` |
| `escalation` | `08_escalation_and_risk_rules` |
| `communication` | `09_response_style_and_templates` |
| `faq` | `10_customer_faq` |

Release expansion (stage 4B.1):

| Значение | Документы |
|----------|-----------|
| `payments` | `11_payment_security_and_dispute_handling` |

Release expansion (stage 4B.2):

| Значение | Документы |
|----------|-----------|
| `physical_hazard` | `13_physical_hazard_and_foreign_body_protocol` |

Release expansion (stage 4B.3):

| Значение | Документы |
|----------|-----------|
| `staff_safety` | `12_staff_safety_and_threat_handling` |

Release expansion (stage 4B.4):

| Значение | Документы |
|----------|-----------|
| `evidence` | `14_evidence_standards_and_incomplete_information` |

Release expansion (stage 4B.5):

| Значение | Документы |
|----------|-----------|
| `remedy_priority` | `15_conflicting_rules_and_remedy_priority` |

---

## Дополнительные метаданные чанка

| Поле | Назначение | Формат | Допустимые значения / пример | Обязательное | Уровень |
|------|------------|--------|------------------------------|--------------|---------|
| `chunk_id` | Уникальный ID чанка | string | `04_refund_policy__review_timelines__001` | Да | Чанк |
| `section` | Заголовок раздела H2 | string | `Сроки рассмотрения` | Да | Чанк |
| `subsection` | Заголовок подраздела H3 | string | `Стандартный случай` | Нет | Чанк |
| `topic` | Краткая тема для retrieval | string | `refund_review_5_days`, `faq-01`, `template-03` | Нет (рекомендуется) | Чанк |
| `risk_level` | Уровень риска содержимого чанка | enum | см. ниже | Нет | Чанк |
| `related_documents` | Связанные документы | list[string] | `["07_complaint_handling_procedure"]` | Нет | Чанк |
| `keywords` | Ключевые слова | list[string] | `["возврат", "5 рабочих дней"]` | Нет | Чанк |
| `supersedes` | ID предыдущего чанка/версии | string | `04_refund_policy__v0_9__001` | Нет | Чанк |
| `source_file` | Путь к исходному файлу (относительный POSIX) | string | `data/02_clean_markdown/04_refund_policy.md` | Да | Чанк |

Поля `topic` и `risk_level` на этапе ingestion MVP:

- `topic` — optional, но рекомендуется; заполняется для FAQ (`faq-NN`), templates (`template-NN`), forbidden rows (`forbidden-NN`) и иных chunks с явным semantic ID; для generic `chunk-NNN` может оставаться пустым.
- `risk_level` — optional; заполняется только при наличии явной канонической chunk-level разметки в источнике; **не вычисляется эвристически** на этапе ingestion; **не является** request risk level, который определяет LLM по сообщению клиента.

Поля документа (`document_id`, `title`, `status`, `version`, `priority`, `language` и др.) **наследуются** каждым чанком и дублируются в его метаданных для фильтрации без join.

---

## Допустимые значения перечислений

### `status`

| Значение | Описание |
|----------|----------|
| `draft` | Черновик, не используется в production-retrieval |
| `active` | Действующая версия, участвует в retrieval |
| `superseded` | Заменена новой версией, не используется |
| `archived` | Архив, не используется |

### `confidentiality`

| Значение | Описание |
|----------|----------|
| `public` | Может цитироваться клиенту (FAQ, общее описание сервиса) |
| `internal` | Для операторов и ассистента (большинство политик и SOP) |
| `restricted` | Ограниченный доступ (правила эскалации с деталями внутренних процессов) |

### `priority`

| Значение | Описание |
|----------|----------|
| `critical` | Высший приоритет при конфликте (возвраты, компенсации, эскалация) |
| `high` | Важные операционные документы |
| `medium` | Справочные и процедурные материалы средней критичности |
| `low` | Общий контекст, второстепенные справки |

### `risk_level`

| Значение | Описание |
|----------|----------|
| `low` | Справочная информация без финансовых или safety-последствий |
| `medium` | Опоздание, частичный возврат, обычная жалоба |
| `high` | Недоставка, вскрытая упаковка, юридические упоминания |
| `critical` | Угроза здоровью, посторонний предмет, ухудшение самочувствия |

### `language`

| Значение | Описание |
|----------|----------|
| `ru` | Русский язык (единственный поддерживаемый в MVP) |

---

## Рекомендуемые значения по документам

| `document_id` | `category` | `document_type` | `priority` | `confidentiality` |
|---------------|------------|-----------------|------------|-------------------|
| `01_service_overview` | `general` | `reference` | `high` | `public` |
| `02_delivery_rules` | `delivery` | `policy` | `high` | `internal` |
| `03_order_changes_and_cancellations` | `orders` | `policy` | `high` | `internal` |
| `04_refund_policy` | `refunds` | `policy` | `critical` | `internal` |
| `05_compensation_policy` | `compensation` | `policy` | `critical` | `internal` |
| `06_food_quality_and_packaging` | `food_quality` | `policy` | `high` | `internal` |
| `07_complaint_handling_procedure` | `complaints` | `procedure` | `high` | `internal` |
| `08_escalation_and_risk_rules` | `escalation` | `policy` | `critical` | `restricted` |
| `09_response_style_and_templates` | `communication` | `guideline` | `critical` | `internal` |
| `10_customer_faq` | `faq` | `faq` | `medium` | `internal` |

Release expansion (stage 4B.1):

| `document_id` | `category` | `document_type` | `priority` | `confidentiality` |
|---------------|------------|-----------------|------------|-------------------|
| `11_payment_security_and_dispute_handling` | `payments` | `policy` | `critical` | `internal` |

Release expansion (stage 4B.2):

| `document_id` | `category` | `document_type` | `priority` | `confidentiality` |
|---------------|------------|-----------------|------------|-------------------|
| `13_physical_hazard_and_foreign_body_protocol` | `physical_hazard` | `policy` | `critical` | `internal` |

Release expansion (stage 4B.3):

| `document_id` | `category` | `document_type` | `priority` | `confidentiality` |
|---------------|------------|-----------------|------------|-------------------|
| `12_staff_safety_and_threat_handling` | `staff_safety` | `policy` | `critical` | `internal` |

Release expansion (stage 4B.4):

| `document_id` | `category` | `document_type` | `priority` | `confidentiality` |
|---------------|------------|-----------------|------------|-------------------|
| `14_evidence_standards_and_incomplete_information` | `evidence` | `procedure` | `high` | `internal` |

Release expansion (stage 4B.5):

| `document_id` | `category` | `document_type` | `priority` | `confidentiality` |
|---------------|------------|-----------------|------------|-------------------|
| `15_conflicting_rules_and_remedy_priority` | `remedy_priority` | `policy` | `high` | `internal` |

---

## Правила разрешения конфликтов источников

При противоречии между retrieved-фрагментами ассистент и retrieval-слой руководствуются следующими правилами:

1. **Только `active`** — в ответе используются только чанки документов со статусом `active`.
2. **Новая версия** — при конфликте двух `active` (ошибка данных) побеждает более новая `effective_date` / `version`.
3. **Специализация** — профильная политика (`04_refund_policy`, `05_compensation_policy` и т.д.) имеет приоритет над общим FAQ (`10_customer_faq`).
4. **Безопасность и эскалация** — `08_escalation_and_risk_rules` и чанки с `risk_level: critical` имеют приоритет над шаблонами из `09_response_style_and_templates`.
5. **FAQ не создает правила** — если FAQ утверждает правило, которого нет в профильной политике, FAQ **игнорируется**; расхождение фиксируется для исправления контента.
6. **Неустранимое противоречие** — ассистент **не выбирает** правило самостоятельно; сообщает о недостаточности надежных данных и **передает обращение сотруднику**.

Иерархия согласована с разделом 13 реестра `docs/05_business_rules_registry.md`.

---

## Соглашения об именовании

### `document_id`

**Обязательно** совпадает с именем файла без расширения. Примеры: `01_service_overview`, `04_refund_policy`.

### `version`

Оформляется в формате **SemVer**, например `1.0.0`. Строковые значения вроде `"1.0"` не используются в новых документах.

### `confidentiality` для внутренней базы

- Политики и процедуры для ассистента поддержки обычно имеют `confidentiality: internal`.
- Справочный клиентский документ без внутренних процедур может иметь `confidentiality: public` (например, `01_service_overview`).
- Документы с деталями эскалации могут иметь `confidentiality: restricted`.

### `chunk_id`

Формат ingestion MVP: `{document_id}::{semantic_key}`

Примеры:

- `10_customer_faq::faq-01`
- `09_response_style_and_templates::template-01`
- `09_response_style_and_templates::forbidden-03`
- `02_delivery_rules::chunk-001`

Устаревший пример (не используется ingestion layer): `04_refund_policy__credit_timeline__002`

### `related_documents`

Список `document_id` связанных документов. Для FAQ — обязательно указывать профильные политики. **Markdown-якоря не используются**; связь только через метаданные и текстовые названия разделов в поле `section`.

---

## Пример метаданных документа (YAML)

```yaml
document_id: 04_refund_policy
title: Политика возвратов
category: refunds
document_type: policy
version: 1.0.0
status: active
effective_date: 2026-06-01
last_updated: 2026-06-20
audience:
  - support_agent
confidentiality: internal
source_type: internal_policy
language: ru
priority: critical
```

---

## Пример метаданных чанка (YAML)

```yaml
chunk_id: 04_refund_policy::chunk-001
document_id: 04_refund_policy
title: Политика возвратов
category: refunds
document_type: policy
version: 1.0.0
status: active
effective_date: 2026-06-01
last_updated: 2026-06-20
audience:
  - support_agent
confidentiality: internal
source_type: internal_policy
language: ru
priority: critical
section: Сроки рассмотрения обращения
subsection: Стандартный случай
topic: refund_review_5_business_days
related_documents:
  - 07_complaint_handling_procedure
  - 10_customer_faq
keywords:
  - возврат
  - рассмотрение
  - 5 рабочих дней
source_file: data/02_clean_markdown/04_refund_policy.md
```

Поле `risk_level` в примере опущено: на этапе ingestion оно не вычисляется автоматически.

---

## Использование метаданных в retrieval

| Сценарий | Фильтр / действие |
|----------|-------------------|
| Обычный запрос | `status: active`, `language: ru` |
| Финансовый вопрос | boost `priority: critical`, категории `refunds`, `compensation` |
| Рискованное обращение | boost `risk_level: high|critical`, документ `08_escalation_and_risk_rules` |
| FAQ-only hit без политики | понизить confidence; запросить профильный документ |
| Superseded chunk | исключить из индекса |

### Vector store metadata (retrieval index)

При индексации в Chroma сохраняется scalar-safe подмножество полей чанка. Сложные значения сериализуются детерминированно:

| Поле в index | Источник | Примечание |
|--------------|----------|------------|
| `chunk_id` | chunk | document id в Chroma |
| `document_id` | chunk | фильтрация / diagnostics |
| `source_path` | chunk | относительный POSIX путь |
| `chunk_type` | `ChunkRecord.strategy` | policy/faq/templates/... |
| `topic`, `risk_level` | chunk metadata | optional |
| `heading`, `section`, `subsection` | chunk | контекст retrieval output |
| `heading_path` | chunk | JSON list string |
| `title`, `category`, `document_type`, `status`, `language`, `document_priority` | chunk metadata | scalar filters |

Абсолютные пути, секреты и non-scalar значения в vector metadata не сохраняются. Mapping выполняется отдельным adapter-слоем без изменения исходного `ChunkRecord`.

---

## Связанные документы

- `docs/03_chunking_strategy.md` — правила формирования чанков
- `docs/05_business_rules_registry.md` — бизнес-правила и иерархия источников
- `docs/01_data_inventory.md` — состав базы знаний

---

*Схема относится к учебному проекту. FoodFlow — вымышленная компания.*
