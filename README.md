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
| `data/04_index/` | Default output of `build-index` / `search-index` / evaluation CLIs; на maintainer-машине может содержать historical 15-document index; **не** production index |
| `data/04_index_production/` | **Production index** для release posture `foodflow-10doc-release-v2` (10 documents, 215 chunks); строится из репозитория командой из `docs/07_index_provisioning.md`; не коммитится |
| `data/05_evaluation/` | Сгенерированные JSON-результаты retrieval evaluation; не источник истины |
| `docs/` | Проектная документация: область проекта, инвентаризация, отчеты, стратегии |
| `prompts/` | Системный промпт и шаблон RAG-запроса |
| `tests/` | Тестовые вопросы, ожидаемые ответы, результаты прогонов |
| `deliverables/` | Итоговые артефакты проекта: manual acceptance report, evidence index, screenshots |

## Текущий статус

**Ingestion layer (hybrid chunking)** — реализованы загрузка clean Markdown, валидация метаданных, гибридный чанкинг и экспорт в JSONL.

**Retrieval layer (baseline dense search)** — реализованы framework-isolated embeddings, persistent Chroma index, deterministic fingerprint/manifest, baseline semantic retrieval и CLI для сборки/поиска.

**Retrieval evaluation (60-case baseline)** — реализованы parser evaluation corpus, baseline retrieval evaluator, retrieval-only metrics, threshold sweep analysis, CLI и committed Markdown reports.

**Reranking A/B (stage 2C.1, source-authority-v1)** — baseline retrieval evaluation завершён; A/B candidate принят по quality criteria; stage 2C.1 закрыт repair/audit cycle.

**Vector pool expansion (stage 2C.2)** — candidate `vector top-24 → source-authority-v1 → final top-12` принят; это **selected retrieval configuration** для MVP.

**Hybrid lexical + vector (stage 2C.3)** — experiment `hybrid-lexical-vector-v1` выполнен; formal guardrails не пройдены (**rejected** для MVP selection). Retrieval experimentation **frozen** после 2C.3. Hybrid v1 **не** production-ready.

**Application layer (functional MVP)** — реализованы grounded generation, deterministic risk/handoff, citations и fallback handling; production composition root (`build_customer_claims_pipeline`), frozen retrieval `vector top-24 → source-authority-v1 → final top-12`, single-shot CLI (`answer-claim`) и локальный Streamlit UI.

**10-document production release posture (stage 4C.4-B, пересобран в 2D2)** — release descriptor `foodflow-10doc-release-v2` (canonical corpus `configs/corpus/foodflow_production_v1.json`: documents 01–10, index `data/04_index_production`, frozen pool@24 contract). Index воспроизводится из репозитория (`build-index --corpus-manifest ...`); production startup fail-closed и пересчитывает identity из хранимых записей. Rollback — операция над репозиторием (checkout предыдущего релиза и rebuild), не архив. Подробности: `docs/06_release_posture.md`.

**Expanded corpus index (stage 4C.1)** — historical 15-document index (333 chunks) — исторический evidence (локальный архив, больше не часть release posture); **не** production.

**Expanded corpus frozen regression (stage 4C.2)** — frozen 60-question regression выполнен для historical 10-document и 15-document index arms; артефакт `expanded_corpus_frozen_regression_v1`.

**Functional MVP complete:** production retrieval, grounded generation, deterministic risk/handoff, single-shot CLI и локальный Streamlit interface реализованы и покрыты тестами.

**Docker-based local delivery (stage 5B)** — reproducible Streamlit deployment через Docker Compose с bind-mount активного production index; fail-closed startup preflight. См. `docs/07_index_provisioning.md` и `docs/08_docker_runbook.md`.

**Russian production flow (финальный repair-пакет)** — весь пользовательский production flow русифицирован: generation, deterministic risk/handoff, category-specific fallback, risk rules на русском языке; client draft отделен от staff information; технические статусы вынесены в свернутый блок.

**Manual acceptance (stage 5C + targeted recheck R1–R4)** — основной набор 5 типовых + 2 out-of-scope сценариев, targeted recheck R1–R4 (оплаченная недоставка, задержка, жалоба на здоровье, возврат). Verdict: **PASS**. Все targeted ответы менее 30 секунд (9.90 / 3.66 / 3.65 / 2.58 с). Отчет: `deliverables/manual_acceptance_report.md`; evidence index: `deliverables/evidence/README.md`; скриншоты: `deliverables/evidence/manual_acceptance/`.

**Еще не реализованы:** authentication, chat history, document upload из UI, feedback collection, query rewriting, remote/public deployment. Клиентские черновики являются предварительными и проверяются сотрудником перед отправкой. Более естественные category-specific шаблоны и SLA-aware ответы — post-MVP improvements (`итеративная калибровка шаблонов по результатам пилотной эксплуатации`). Система не обучается самостоятельно на обращениях.

Источником истины для базы знаний остаются файлы в `data/02_clean_markdown/`. Каталоги `data/03_chunks/` и `data/04_index*/` — только локальные generated артефакты. **Свежий clone не содержит production index** — его нужно собрать из репозитория (`docs/07_index_provisioning.md`; нужен `OPENAI_API_KEY`).

## Запуск с нуля (Windows / PowerShell)

Самостоятельная инструкция для чистого компьютера. Все зависимости проекта описаны **только** в `pyproject.toml`; отдельные `requirements.txt` / `requirements-dev.txt` намеренно не используются, чтобы не поддерживать второй дублирующий список пакетов.

### Требования

- **Git** — клонирование репозитория;
- **Python 3.12 или новее** — минимальная поддерживаемая версия определяется `requires-python = ">=3.12"` в `pyproject.toml`;
- **OpenAI API key** — требуется для реальных embeddings/search и grounded generation в `answer-claim` и Streamlit UI; **не требуется** для ingestion (`build-chunks`) и offline automated tests;
- **Автоматические тесты** работают полностью offline и **не требуют** OpenAI API key.

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

Production retrieval использует `configs/release/production_posture.json` (default target `active`). Индекс `data/04_index_production` — локальный build artifact; после сборки проверьте:

```powershell
validate-release-posture
```

См. `docs/06_release_posture.md` для release identity, limitations и rollback.

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

### Сборка vector index (dev/evaluation)

> **Важно:** production index создаётся только командой с `--corpus-manifest` (см. `docs/07_index_provisioning.md`). Generic `build-index --rebuild` без manifest индексирует **все** документы директории и release-валидацию не пройдёт.

Требуется заполненный `OPENAI_API_KEY` в `.env` или в environment процесса.

Используйте явный disposable path для dev/evaluation, чтобы не затрагивать локальные индексы:

```powershell
python -m customer_claims_rag.cli.build_index `
  --rebuild `
  --index-dir data/04_index_dev_tmp
```

Для диагностики ошибок:

```powershell
python -m customer_claims_rag.cli.build_index `
  --rebuild `
  --index-dir data/04_index_dev_tmp `
  --verbose
```

Флаг `--verbose` выводит полный traceback; без него CLI показывает только краткое сообщение об ошибке.

По умолчанию:

- вход: `data/02_clean_markdown/` (через `CorpusBuilder`);
- index (build CLI default при отсутствии `--index-dir`): `data/04_index/` — dev/evaluation default, **не** production index;
- collection: `customer_claims`;
- embedding model: `text-embedding-3-small`.

`--rebuild` обязателен: выполняется destructive full rebuild. `--index-dir` явно задаёт целевой путь.

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

Требуется собранный production index (`data/04_index_production` per release descriptor), заполненный `OPENAI_API_KEY` и generation env vars (см. `.env.example`). Production index path задаётся **только** через `configs/release/production_posture.json`; `RAG_INDEX_DIR` на answer/UI flow **не влияет**.

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

Требования те же, что и для `answer-claim`: заполненный `.env`, собранный production index, generation env vars из `.env.example`. UI **не пересобирает** индекс и не загружает документы.

Запуск:

```powershell
python -m streamlit run src/customer_claims_rag/ui/streamlit_app.py
```

Интерфейс принимает одно обращение, вызывает production pipeline один раз и показывает:

- черновик ответа клиенту (предварительный; проверяется сотрудником перед отправкой);
- отдельную служебную информацию для сотрудника;
- категорию обращения;
- уровень риска;
- необходимость эскалации;
- рекомендуемый маршрут обработки;
- действия сотрудника;
- основания и источники;
- найденные материалы для ручной проверки;
- техническую информацию в свернутом блоке.

### Docker (рекомендуемый способ демонстрации)

Требования: Docker Desktop / Docker Engine + Compose, собранный production index на хосте, `OPENAI_API_KEY` в environment процесса.

1. Собрать index на хосте: `docs/07_index_provisioning.md`
2. Runbook: `docs/08_docker_runbook.md`

Кратко (PowerShell):

```powershell
$env:OPENAI_API_KEY = "your-key-here"
docker compose build
docker compose run --rm streamlit validate-release-posture
docker compose up
```

Откройте http://localhost:8501. Индекс монтируется read-write в `/app/data/04_index_production`; в image он **не** копируется. `RAG_INDEX_DIR` на production UI flow **не влияет**; selector — `RAG_RELEASE_TARGET` + descriptor.

Дополнительные параметры search-index (локальный Python, dev/evaluation):

```powershell
python -m customer_claims_rag.cli.search_index "возврат" --top-k 4 --fetch-k 12 --threshold 0.35
```

Явный `--threshold` по-прежнему включает filtering; без него используется default из env (см. ниже).

### Generated artifacts

- **Production release index** (`data/04_index_production`) — build artifact, строится из репозитория; see `docs/07_index_provisioning.md` and `docs/06_release_posture.md`;
- **Build/evaluation index** (`data/04_index/`) — default output path для `build-index` / `search-index` / evaluation CLIs (на maintainer-машине может содержать historical 15-document index для воспроизведения экспериментов);

  > **Предупреждение:** generic build command (`build-index --rebuild`) выполняет destructive full rebuild в `data/04_index/` (все документы директории). Для одноразовой dev/evaluation сборки укажите явный путь: `build-index --rebuild --index-dir data/04_index_dev_tmp`.

- Chroma index и `manifest.json` создаются **внутри project root**;
- нельзя направлять `--index-dir` в `data/01_raw/`, `data/02_clean_markdown/` или внутрь `--input-dir`;
- generated contents **не коммитятся** (см. `.gitignore`);
- `data/04_index/.gitkeep` сохраняет структуру каталога в git.

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
| `CUSTOMER_CLAIMS_PROJECT_ROOT` | Explicit project root for container/non-editable installs | unset (auto-detect from package layout) |
| `RAG_RELEASE_TARGET` | Production release target (в descriptor только `active`) | `active` (descriptor default) |
| `RAG_ACTIVE_INDEX_HOST_PATH` | Docker Compose **host** bind-mount path for active index only | `./data/04_index_production` |
| `RAG_INDEX_DIR` | Index path for **build/search/evaluation CLIs only** | `data/04_index` |
| `RAG_COLLECTION_NAME` | Имя Chroma collection — **build/search/evaluation CLI only**; production pipeline использует frozen contract | `customer_claims` |
| `RAG_TOP_K` | Максимум результатов поиска — **retrieval/evaluation CLI only**; production pipeline: final top-12 (frozen) | `4` |
| `RAG_FETCH_K` | Размер candidate pool — **retrieval/evaluation CLI only**; production pipeline: fetch24 (frozen) | `12` |
| `RAG_SIMILARITY_THRESHOLD` | Минимальная cosine similarity — **retrieval/evaluation CLI only**; production pipeline: threshold=0.0 (frozen) | `0.0` |
| `RAG_EMBEDDING_BATCH_SIZE` | Batch size при индексации | `64` |
| `GENERATION_TEMPERATURE` | Temperature для grounded generation | `0.0` |
| `GENERATION_TIMEOUT_SECONDS` | Timeout chat completion (сек.) | `60` |
| `GENERATION_MAX_RETRIES` | Retries chat completion | `2` |
| `GENERATION_MAX_OUTPUT_TOKENS` | Max output tokens | `1024` |
| `GENERATION_PROMPT_PATH` | Путь к prompt-файлу | `prompts/system_prompt.md` |

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

**Production retrieval contract (frozen):** `fetch24 / pool24 / final12 / threshold0.0 / source-authority-v1`.

**`threshold=0.0` — frozen release configuration.** Означает отсутствие similarity filtering после candidate retrieval: все fetch_k кандидаты передаются reranker'у, затем возвращается final top-k. Значение зафиксировано в `configs/retrieval/vector_pool_expansion_v1.json` и не подлежит изменению без нового release. Это не «ожидающий выбора production threshold», а осознанное решение, принятое по итогам baseline evaluation и A/B анализа.

Исторически: smoke-run показал, что threshold `0.70` был слишком высок для текущего embedding/index (тематически очевидные запросы отсекались). После 60-case baseline evaluation был выбран и заморожен контракт `threshold=0.0` с reranker `source-authority-v1`.

### Manifest и fingerprint

`data/04_index/manifest.json` генерируется только после успешной индексации (temp + replace). Содержит:

- `index_format_version`, `metadata_schema_version`;
- `collection_name`, `embedding_model`;
- `corpus_fingerprint`, `chunk_count`, `document_count`;
- `vector_dimension`, informational `created_at`.

Fingerprint зависит от `chunk_id`, `content`, canonical metadata, embedding model и index format version. Timestamp и абсолютные пути в fingerprint не входят.

При mismatch manifest vs runtime retriever возвращает понятную ошибку с инструкцией выполнить rebuild. Автоматический rebuild не выполняется.

Human-readable search output показывает rank, similarity, chunk/document IDs, heading, source path и короткий excerpt. JSON mode возвращает structured `SearchResponse` с diagnostics.

## Baseline retrieval evaluation (60 cases) — исторический этап

> **Примечание:** этот раздел описывает завершённый исторический этап (stage 2C), результаты которого привели к выбору frozen production retrieval contract `fetch24 / pool24 / final12 / threshold0.0 / source-authority-v1`. Раздел сохранён для справки и воспроизводимости; он **не описывает** текущий production release.

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

Baseline evaluation использовал `threshold=0.0` для измерения raw recall. Метрики **не** оценивают качество LLM-ответов, risk/handoff classification, answer factuality или Markdown output contract. По итогам baseline analysis был выбран и заморожен production contract; retrieval experimentation закрыто.

### Troubleshooting

| Симптом | Действие |
|---------|----------|
| `OPENAI_API_KEY is required` | Заполнить `OPENAI_API_KEY` в `.env` или export в PowerShell |
| `index manifest not found` (production `answer-claim` / UI) | Собрать index согласно `docs/07_index_provisioning.md`, затем `validate-release-posture` |
| `index manifest not found` (dev/evaluation CLI) | Пересобрать отдельный индекс: `build-index --rebuild --index-dir data/04_index_dev_tmp` |
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

### Retrieval (production frozen)

- Dense cosine retrieval, frozen production contract: `fetch24 / pool24 / final12 / threshold0.0 / source-authority-v1`.
- Stage **2C.3** hybrid BM25+RRF experiment завершён и **отклонён** для MVP; retrieval stage frozen.
- Нет query rewriting.
- `threshold=0.0` — frozen release configuration; similarity filtering после candidate retrieval не применяется. Контракт зафиксирован; выбор production threshold завершён.
- Incremental indexing не поддерживается; только full rebuild.

### Retrieval evaluation (исторический этап, завершён)

- Только retrieval metrics; answer/risk/handoff quality не измерялись.
- Fallback cases (T006, T060) анализировались отдельно от source-recall aggregates.
- Threshold sweep выполнялся post-hoc над сохраненными candidates без повторных embedding calls.
- По итогам baseline analysis выбран и заморожен production contract; retrieval experimentation закрыто.

### Application MVP (текущие ограничения)

- Нет authentication.
- Нет chat history.
- Нет document upload из UI.
- Нет feedback collection.
- Нет query rewriting.
- Нет remote/public deployment.
- Клиентские черновики являются предварительными и проверяются сотрудником перед отправкой.
- Более естественные category-specific шаблоны и SLA-aware ответы — post-MVP improvements (`итеративная калибровка шаблонов по результатам пилотной эксплуатации`).
- Система не обучается самостоятельно на обращениях.
- Documents 11–15 не входят в production support.
