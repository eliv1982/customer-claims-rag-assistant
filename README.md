# customer-claims-rag-assistant

Учебный RAG-ассистент для первичной обработки клиентских обращений, жалоб и претензий вымышленного сервиса доставки еды **FoodFlow**.

Ассистент помогает классифицировать обращение, находить релевантные правила в базе знаний и формировать проект ответа. Окончательные решения принимает сотрудник поддержки.

> **Важно:** FoodFlow — вымышленная компания. Все документы и правила созданы исключительно в учебных целях и не отражают политику реальной организации.

## Структура каталогов

| Каталог | Назначение |
|---------|------------|
| `data/01_raw/` | Исходные тексты документов базы знаний (`.txt`) |
| `data/02_clean_markdown/` | Очищенные версии документов в Markdown (`.md`) |
| `data/03_chunks/` | Сгенерированные чанки (JSONL) и статистика; не источник истины |
| `docs/` | Проектная документация: область проекта, инвентаризация, отчеты, стратегии |
| `prompts/` | Системный промпт и шаблон RAG-запроса |
| `tests/` | Тестовые вопросы, ожидаемые ответы, результаты прогонов |
| `deliverables/` | Итоговые артефакты проекта |

## Текущий статус

**Ingestion layer (hybrid chunking)** — реализованы загрузка clean Markdown, валидация метаданных, гибридный чанкинг и экспорт в JSONL. Embeddings, vector database, semantic retrieval, reranking и LLM-генерация ответов **еще не реализованы**.

Источником истины для базы знаний остаются файлы в `data/02_clean_markdown/`. Каталог `data/03_chunks/` содержит только сгенерированные артефакты и не редактируется вручную.

### Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Требуется Python 3.12+.

### Сборка чанков

```powershell
python -m customer_claims_rag.cli.build_chunks --verbose
```

По умолчанию:

- вход: `data/02_clean_markdown/`;
- выход: `data/03_chunks/chunks.jsonl`;
- статистика: `data/03_chunks/chunk_stats.json`.

### Тесты

```powershell
python -m pytest
```

С coverage:

```powershell
python -m pytest --cov=customer_claims_rag --cov-report=term-missing
```

## Известные ограничения текущего ingestion MVP

- Generic chunk IDs вида `chunk-NNN` могут сдвинуться при добавлении более раннего раздела в документ.
- На текущем real corpus overlap не требуется, хотя synthetic tests покрывают механизм overlap.
- Policy overlap унифицирован диапазоном strategy; per-document fine-tuning еще не применен.
- Target token range носит рекомендательный характер; grouping ориентируется на soft/hard limits.
- Chunk-level `risk_level` не вычисляется эвристически на этапе ingestion.
- Поле `topic` заполняется только у chunks с явным semantic ID (FAQ, template, forbidden row).
- Embeddings, vector database и semantic retrieval еще не реализованы.
- Output и stats должны находиться внутри permitted project root; перезапись source Markdown запрещена.
- Двухфайловый export (JSONL + stats) записывает оба temp-файла до replace; при сбое второго replace первый файл может уже быть обновлен.
