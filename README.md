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
| `data/04_index/` | Сгенерированный persistent vector index (Chroma) и `manifest.json` |
| `data/05_evaluation/` | Сгенерированные JSON-результаты retrieval evaluation; не источник истины |
| `docs/` | Проектная документация: область проекта, инвентаризация, отчеты, стратегии |
| `prompts/` | Системный промпт и шаблон RAG-запроса |
| `tests/` | Тестовые вопросы, ожидаемые ответы, результаты прогонов |
| `deliverables/` | Итоговые артефакты проекта |

## Текущий статус

**Ingestion layer (hybrid chunking)** — реализованы загрузка clean Markdown, валидация метаданных, гибридный чанкинг и экспорт в JSONL.

**Retrieval layer (baseline dense search)** — реализованы framework-isolated embeddings, persistent Chroma index, deterministic fingerprint/manifest, baseline semantic retrieval и CLI для сборки/поиска.

**Retrieval evaluation (60-case baseline)** — реализованы parser evaluation corpus, baseline retrieval evaluator, retrieval-only metrics, threshold sweep analysis, CLI и committed Markdown reports.

**Reranking A/B (stage 2C.1, source-authority-v1)** — baseline retrieval evaluation завершён; A/B candidate принят по quality criteria; stage 2C.1 закрыт repair/audit cycle.

**Vector pool expansion (stage 2C.2)** — candidate `vector top-24 → source-authority-v1 → final top-12` принят; это **selected retrieval configuration** для MVP.

**Hybrid lexical + vector (stage 2C.3)** — experiment `hybrid-lexical-vector-v1` выполнен; formal guardrails не пройдены (**rejected** для MVP selection). Retrieval experimentation **frozen** после 2C.3. Hybrid v1 **не** production-ready.

**Application layer (functional MVP)** — реализованы grounded generation, deterministic risk/handoff, citations и fallback handling; production composition root (`build_customer_claims_pipeline`), frozen retrieval `vector top-24 → source-authority-v1 → final top-12`, single-shot CLI (`answer-claim`) и локальный Streamlit UI.

**Functional MVP complete:** production retrieval, grounded generation, deterministic risk/handoff, single-shot CLI и локальный Streamlit interface реализованы и покрыты тестами. Deployment и production operations **не** входят в текущий scope.

**Еще не реализованы:** deployment/operations, authentication, chat history, document upload из UI, feedback collection, query rewriting, production threshold auto-selection.

Источником истины для базы знаний остаются файлы в `data/02_clean_markdown/`. Каталоги `data/03_chunks/` и `data/04_index/` содержат только сгенерированные артефакты.

## Запуск с нуля (Windows / PowerShell)

Самостоятельная инструкция для чистого компьютера. Все зависимости проекта описаны **только** в `pyproject.toml`; отдельные `requirements.txt` / `requirements-dev.txt` намеренно не используются, чтобы не поддерживать второй дублирующий список пакетов.

### Требования

- **Git** — клонирование репозитория;
- **Python 3.12** — версия зафиксирована в `pyproject.toml` (`requires-python = ">=3.12"`);
- **OpenAI API** — нужен только для реальной сборки vector index и semantic search с OpenAI embeddings;
- **Автоматические тесты** работают offline и **не требуют** OpenAI API key.

### Клонирование

```powershell
git clone https://github.com/eliv1982/customer-claims-rag-assistant.git
cd customer-claims-rag-assistant
```

### Создание виртуального окружения

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Если активация блокируется политикой выполнения PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Установка проекта

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
```

Кратко:

- `pyproject.toml` — единственный источник runtime- и dev-зависимостей;
- `-e` устанавливает пакет в **editable mode** (изменения в `src/` сразу доступны);
- `[dev]` добавляет pytest и pytest-cov для тестов и coverage.

### Настройка environment

Скопируйте шаблон и заполните ключ локально:

```powershell
Copy-Item .env.example .env
```

Откройте `.env` и задайте:

```env
OPENAI_API_KEY=
```

Файл `.env` **не коммитится** (см. `.gitignore`). Шаблон `.env.example` содержит безопасные placeholder-значения без секретов.

Проект автоматически загружает `.env` из корня репозитория при запуске CLI, `answer-claim`, Streamlit UI и чтении `ApplicationSettings` / `RetrievalSettings`. Уже установленные переменные процесса имеют **приоритет** над значениями из `.env`. Отсутствие `.env` не является ошибкой.

### Тесты

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m pytest -q -p no:cacheprovider
```

С coverage:

```powershell
python -m pytest --cov=customer_claims_rag --cov-report=term-missing
```

### Сборка ingestion chunks

```powershell
python -m customer_claims_rag.cli.build_chunks --verbose
```

По умолчанию:

- вход: `data/02_clean_markdown/`;
- выход: `data/03_chunks/chunks.jsonl`;
- статистика: `data/03_chunks/chunk_stats.json`.

OpenAI API key для этой команды **не требуется**.

### Сборка vector index

Требуется заполненный `OPENAI_API_KEY` в `.env` или в environment процесса.

```powershell
python -m customer_claims_rag.cli.build_index --rebuild
```

Для диагностики ошибок:

```powershell
python -m customer_claims_rag.cli.build_index --rebuild --verbose
```

Флаг `--verbose` выводит полный traceback; без него CLI показывает только краткое сообщение об ошибке.

По умолчанию:

- вход: `data/02_clean_markdown/` (через `CorpusBuilder`);
- index: `data/04_index/`;
- collection: `customer_claims`;
- embedding model: `text-embedding-3-small`.

`--rebuild` обязателен в MVP: выполняется destructive full rebuild.

### Поиск

```powershell
python -m customer_claims_rag.cli.search_index "Где мой заказ?"
```

```powershell
python -m customer_claims_rag.cli.search_index "Сколько времени занимает возврат денег?"
```

```powershell
python -m customer_claims_rag.cli.search_index "После еды мне стало плохо"
```

```powershell
python -m customer_claims_rag.cli.search_index "Заказ отмечен доставленным, но я его не получил" --json
```

### Production claim answer (single-shot)

Требуется собранный vector index (`data/04_index/`), заполненный `OPENAI_API_KEY` и настроенные generation env vars (см. `.env.example`).

Команда принимает одно обращение, выполняет production pipeline один раз и печатает стабильный JSON в stdout:

```powershell
answer-claim --message "Курьер привез вскрытый контейнер"
```

Эквивалент через модуль:

```powershell
python -m customer_claims_rag.cli.answer_claim --message "Курьер привез вскрытый контейнер"
```

JSON содержит customer-safe поля: `answer`, `response_mode`, `generation_outcome`, `risk_level`, handoff flags/notice и citations (`key`, `heading`, `document_id`).

### Streamlit MVP (локальный UI)

Установка UI-зависимостей:

```powershell
python -m pip install -e ".[ui]"
```

Требования те же, что и для `answer-claim`: заполненный `.env`, собранный vector index в `data/04_index/`, generation env vars из `.env.example`. UI **не пересобирает** индекс и не загружает документы.

Запуск:

```powershell
python -m streamlit run src/customer_claims_rag/ui/streamlit_app.py
```

Интерфейс принимает одно обращение, вызывает production pipeline один раз и показывает:

- проект ответа для клиента;
- уровень риска;
- уведомление о передаче сотруднику поддержки (если требуется);
- безопасные метки источников (`[S1] заголовок — document_id`);
- статус при недостатке контекста или сбое генерации.

Дополнительные параметры:

```powershell
python -m customer_claims_rag.cli.search_index "возврат" --top-k 4 --fetch-k 12 --threshold 0.35
```

Явный `--threshold` по-прежнему включает filtering; без него используется default из env (см. ниже).

### Generated artifacts

- Chroma index и `manifest.json` создаются в `data/04_index/` **внутри project root**;
- рекомендуемый путь — `data/04_index`;
- нельзя направлять `--index-dir` в `data/01_raw/`, `data/02_clean_markdown/` или внутрь `--input-dir`;
- generated contents **не коммитятся** (см. `.gitignore`);
- `data/04_index/.gitkeep` сохраняет структуру каталога в git;
- index можно безопасно пересобрать: `python -m customer_claims_rag.cli.build_index --rebuild`.

### Остановка окружения

```powershell
deactivate
```

### Переменные окружения (справочник)

| Переменная | Назначение | Default |
|------------|------------|---------|
| `OPENAI_API_KEY` | Ключ OpenAI для embeddings и generation | — |
| `OPENAI_EMBEDDING_MODEL` | Модель embeddings | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | Модель chat completion для generation | `gpt-4o-mini` |
| `RAG_INDEX_DIR` | Каталог vector index | `data/04_index` |
| `RAG_COLLECTION_NAME` | Имя Chroma collection | `customer_claims` |
| `RAG_TOP_K` | Максимум результатов поиска (retrieval CLI) | `4` |
| `RAG_FETCH_K` | Размер candidate pool (retrieval CLI) | `12` |
| `RAG_SIMILARITY_THRESHOLD` | Минимальная cosine similarity | `0.0` (baseline: без filtering) |
| `RAG_EMBEDDING_BATCH_SIZE` | Batch size при индексации | `64` |
| `GENERATION_TEMPERATURE` | Temperature для grounded generation | `0.0` |
| `GENERATION_TIMEOUT_SECONDS` | Timeout chat completion (сек.) | `60` |
| `GENERATION_MAX_RETRIES` | Retries chat completion | `2` |
| `GENERATION_MAX_OUTPUT_TOKENS` | Max output tokens | `1024` |
| `GENERATION_PROMPT_PATH` | Путь к prompt-файлу | `prompts/grounded_answer_v1.md` |

CLI-параметры переопределяют env-значения.

### Архитектура retrieval layer

```text
clean Markdown
  -> CorpusBuilder (ingestion, framework-independent)
  -> validated ChunkRecord list
  -> deterministic ordering + corpus fingerprint
  -> EmbeddingProvider (OpenAI adapter via langchain-openai)
  -> VectorStore (Chroma adapter, cosine space)
  -> manifest.json
  -> BaselineRetriever (fetch_k -> threshold -> top_k)
```

Код:

```text
src/customer_claims_rag/retrieval/
  models.py            # IndexManifest, SearchResult, SearchResponse
  ports.py             # EmbeddingProvider, VectorStore protocols
  fingerprint.py       # deterministic SHA-256 corpus fingerprint
  metadata_mapper.py   # ChunkRecord -> scalar Chroma metadata
  manifest.py          # atomic manifest read/write/validation
  index_builder.py     # full rebuild pipeline
  retriever.py         # baseline dense retrieval
  adapters/
    openai_embeddings.py
    chroma_store.py
    fake_embeddings.py # offline tests only
```

LangChain используется только в `adapters/openai_embeddings.py`. Ingestion core не зависит от LangChain и не меняет свои chunk models.

### Cosine similarity threshold

Chroma collection настроена на cosine space. Adapter преобразует raw distance в similarity централизованно:

```text
similarity = 1.0 - distance
```

Baseline retriever:

1. получает до `fetch_k` кандидатов;
2. при `threshold > 0` отфильтровывает по `similarity >= threshold`;
3. возвращает не более `top_k` результатов;
4. сортирует по similarity desc, tie-break по `chunk_id`.

**Default `RAG_SIMILARITY_THRESHOLD=0.0`** означает отсутствие automatic threshold filtering в baseline retrieval. Это диагностический режим для измерения recall и анализа кандидатов, а не production threshold.

Реальный smoke-run показал, что threshold **`0.70` слишком высок** для текущего embedding/index: даже тематически очевидные запросы (например, срок возврата, similarity ≈ 0.66) отсекались. Production threshold **пока не установлен**. Окончательное значение будет выбрано по результатам **60-case evaluation** (`tests/01_test_questions.md`). До калибровки **retrieval quality не считается подтвержденным**.

### Manifest и fingerprint

`data/04_index/manifest.json` генерируется только после успешной индексации (temp + replace). Содержит:

- `index_format_version`, `metadata_schema_version`;
- `collection_name`, `embedding_model`;
- `corpus_fingerprint`, `chunk_count`, `document_count`;
- `vector_dimension`, informational `created_at`.

Fingerprint зависит от `chunk_id`, `content`, canonical metadata, embedding model и index format version. Timestamp и абсолютные пути в fingerprint не входят.

При mismatch manifest vs runtime retriever возвращает понятную ошибку с инструкцией выполнить rebuild. Автоматический rebuild не выполняется.

Human-readable search output показывает rank, similarity, chunk/document IDs, heading, source path и короткий excerpt. JSON mode возвращает structured `SearchResponse` с diagnostics.

## Baseline retrieval evaluation (60 cases)

Formal retrieval-only evaluation на corpus `tests/01_test_questions.md` + `tests/02_expected_answers.md`.

```powershell
python -m customer_claims_rag.cli.evaluate_retrieval
```

Явные paths при необходимости:

```powershell
python -m customer_claims_rag.cli.evaluate_retrieval `
  --questions tests/01_test_questions.md `
  --expected tests/02_expected_answers.md `
  --index-dir data/04_index `
  --collection customer_claims `
  --embedding-model text-embedding-3-small `
  --top-k 12 `
  --fetch-k 12 `
  --threshold 0.0 `
  --output-json data/05_evaluation/retrieval_results.json
```

Требования:

- существующий index в `data/04_index/` (без rebuild);
- `OPENAI_API_KEY` для query embeddings в **real run**;
- automated tests работают **offline** через fake retriever/fixtures.

Outputs:

- `data/05_evaluation/retrieval_results.json` — generated machine-readable JSON (**ignored by Git**);
- `tests/03_test_results.md` — committed human-readable run summary;
- `tests/04_improvement_log.md` — committed baseline improvement log.

**Output consistency:** каждый файл записывается атомарно (temp + replace), но набор из трёх файлов **не** является общей транзакцией. Поле `evaluation_result_id` (SHA-256 canonical result) должно совпадать во всех трёх артефактах; расхождение означает partial или mixed run.

**Preflight:** CLI выполняет parser validation до первого embedding call; полный manifest/index preflight (`retriever.validate_index()`) — до цикла по кейсам. Fatal mismatch (`IndexManifestError`, missing manifest/index, embedding model/collection/schema mismatch, missing `OPENAI_API_KEY`) прерывает run с **nonzero exit** без success reports.

**Методология @k:** все метрики `@k` ограничивают **первые k raw chunks**, затем при необходимости дедуплицируют document IDs внутри этого окна. `Supporting source hit@4` считается только по кейсам с непустым `expected_supporting_documents` (denominator явно показывается как `hits/denominator`).

**Run metadata:** `git_commit` + `git_dirty` фиксируются до run; при dirty working tree commit hash не полностью идентифицирует evaluation implementation.

Baseline evaluation использует `threshold=0.0` для измерения raw recall **до** выбора production threshold. Метрики **не** оценивают качество LLM-ответов, risk/handoff classification, answer factuality или Markdown output contract. Reranking и production threshold — только после анализа baseline.

### Troubleshooting

| Симптом | Действие |
|---------|----------|
| `OPENAI_API_KEY is required` | Заполнить `OPENAI_API_KEY` в `.env` или export в PowerShell |
| `index manifest not found` | Выполнить `python -m customer_claims_rag.cli.build_index --rebuild` |
| `embedding model mismatch` | Пересобрать index с тем же `--embedding-model`, что и search CLI |
| `chunk count mismatch` | Выполнить rebuild после изменения corpus |
| `No results above threshold` | Проверить явный `--threshold` или `RAG_SIMILARITY_THRESHOLD`; default `0.0` не фильтрует |

## Известные ограничения

### Ingestion MVP

- Generic chunk IDs вида `chunk-NNN` могут сдвинуться при добавлении более раннего раздела в документ.
- На текущем real corpus overlap не требуется, хотя synthetic tests покрывают механизм overlap.
- Policy overlap унифицирован диапазоном strategy; per-document fine-tuning еще не применен.
- Target token range носит рекомендательный характер; grouping ориентируется на soft/hard limits.
- Chunk-level `risk_level` не вычисляется эвристически на этапе ingestion.
- Поле `topic` заполняется только у chunks с явным semantic ID (FAQ, template, forbidden row).
- Output и stats должны находиться внутри permitted project root; перезапись source Markdown запрещена.

### Retrieval MVP (baseline)

- Dense cosine retrieval with frozen **2C.2** candidate: `vector top-24 → source-authority-v1 → final top-12`.
- Stage **2C.3** hybrid BM25+RRF experiment completed and **rejected** for MVP; retrieval stage frozen.
- Нет query rewriting.
- Default threshold `0.0` — diagnostic baseline без automatic filtering; production value TBD после 60-case evaluation.
- Threshold `0.70` отвергнут smoke-run как слишком высокий для текущего index.
- Качество retrieval **не считается подтвержденным** до отдельного evaluation stage.
- Incremental indexing не поддерживается; только full rebuild.

### Retrieval evaluation (baseline)

- Только retrieval metrics; answer/risk/handoff quality не измеряются.
- Fallback cases (T006, T060) анализируются отдельно от source-recall aggregates.
- Threshold sweep выполняется post-hoc над сохраненными candidates без повторных embedding calls.
- Production threshold не выбирается автоматически по одной метрике.
- Улучшения (reranking, hybrid search) не внедряются до A/B rerun на том же corpus.
