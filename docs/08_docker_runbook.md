# Docker runbook (stage 5B)

Reproducible local deployment for the FoodFlow customer claims Streamlit UI using Docker Compose.

**Prerequisites:**

- Docker Desktop (Windows/macOS) or Docker Engine + Compose plugin (Linux)
- Production index built on the host from the repository (see `docs/07_index_provisioning.md`)
- `OPENAI_API_KEY` available in your shell environment (not committed to Git or baked into the image)

The production index is **bind-mounted** at runtime. It is **never** copied into the image.

Browser URL after startup: **http://localhost:8501**

---

## Windows PowerShell

### 1. Set required environment variables

```powershell
$env:OPENAI_API_KEY = "your-key-here"
```

Optional overrides:

```powershell
$env:RAG_RELEASE_TARGET = "active"
$env:RAG_ACTIVE_INDEX_HOST_PATH = ".\data\04_index_production"
```

`RAG_ACTIVE_INDEX_HOST_PATH` controls only the **host** side of the bind mount. The container path remains `/app/data/04_index_production` per the release descriptor.

The container sets `CUSTOMER_CLAIMS_PROJECT_ROOT=/app` so non-editable package installs resolve descriptor and prompt paths correctly.

### 2. Inspect Compose configuration

```powershell
docker compose config
```

Confirm:

- Service `streamlit` exposes port `8501`
- Volume target is `/app/data/04_index_production`
- `RAG_RELEASE_TARGET` defaults to `active`

### 3. Build the image

```powershell
docker compose build
```

### 4. Validate release posture in the container

```powershell
docker compose run --rm streamlit validate-release-posture
```

Expected: non-zero exit if the index is missing or wrong; success lines including `release_posture_id=foodflow-10doc-release-v2`, `index_matches_canonical_corpus=yes` and the corpus fingerprint when the mount is valid; otherwise the output ends with the command that builds the index.

### 5. Start the stack

```powershell
docker compose up
```

Detached mode:

```powershell
docker compose up -d
```

### 6. Health inspection

```powershell
docker compose ps
```

Healthy status appears after Streamlit starts and `/_stcore/health` responds.

Manual check:

```powershell
Invoke-WebRequest -Uri http://localhost:8501/_stcore/health -UseBasicParsing
```

### 7. View logs

```powershell
docker compose logs -f streamlit
```

Startup logs should show release posture diagnostics from the application factory (release ID, target, fingerprint). They must **not** contain your API key.

### 8. Stop and remove the stack

```powershell
docker compose down
```

This removes containers and networks but **does not** delete the host index directory.

---

## Linux / macOS shell

```bash
export OPENAI_API_KEY="your-key-here"
export RAG_ACTIVE_INDEX_HOST_PATH="./data/04_index_production"  # optional

docker compose config
docker compose build
docker compose run --rm streamlit validate-release-posture
docker compose up -d
docker compose ps
curl -fsS http://localhost:8501/_stcore/health
docker compose logs -f streamlit
docker compose down
```

---

## Expected startup order

1. Container entrypoint runs filesystem checks on the mounted index path.
2. `validate-release-posture` validates descriptor, manifest, frozen config, and Chroma collection (no OpenAI call).
3. On success, `exec` starts Streamlit on `0.0.0.0:8501`.
4. Application factory logs release posture lines on first pipeline build.
5. Compose/Docker health check polls `/_stcore/health`.

## Fail-closed errors

| Symptom | Likely cause |
|---------|----------------|
| `production index mount missing` | Volume not mounted or wrong `RAG_ACTIVE_INDEX_HOST_PATH` |
| `production index mount is empty` | Host directory exists but has no files |
| `chroma.sqlite3 not found` | The build did not finish, or the wrong host directory is mounted |
| `release posture validation failed` | The index does not match the canonical corpus, or the frozen config differs; the output names the mismatch and the build command |
| `Set OPENAI_API_KEY in the environment` | Compose variable unset (`docker compose config` / `up` fails early) |

## Troubleshooting

### Missing index

Build the host directory per `docs/07_index_provisioning.md`, then re-run `docker compose run --rm streamlit validate-release-posture`.

### Incorrect mount

Verify the host path:

```powershell
Test-Path .\data\04_index_production\manifest.json
Test-Path .\data\04_index_production\chroma.sqlite3
```

On Windows, prefer relative paths like `.\data\04_index_production` for `RAG_ACTIVE_INDEX_HOST_PATH`.

### Missing API key

Set `OPENAI_API_KEY` in the shell before `docker compose up`. Do not add the key to the Dockerfile or commit it to `.env`.

### Port already in use

Stop the other process on port 8501 or temporarily change the host mapping in `compose.yaml` (e.g. `"8502:8501"`).

### Unhealthy container

- Check logs: `docker compose logs streamlit`
- Confirm preflight passed and Streamlit is listening
- Increase `start_period` if startup is slow on first Chroma open

### Host permissions on writable index mount

The index mount is **read-write** because Chroma may update internal storage on first open (benign drift; see `docs/06_release_posture.md`). Ensure the host directory is writable by the container user. On Linux, fix ownership or permissions on the host path if validation fails with permission errors.

---

## Maintenance commands

Run without starting Streamlit:

```powershell
docker compose run --rm streamlit validate-release-posture
docker compose run --rm streamlit answer-claim --message "Тестовый запрос"
docker compose run --rm streamlit sh
```

`validate-release-posture` and `answer-claim` skip the Streamlit-specific preflight wrapper logic where appropriate; only the default `streamlit` command runs full preflight before serving.

---

## What not to do

- Do **not** copy the production index into the Docker build context or image.
- Do **not** set `RAG_INDEX_DIR` expecting it to change production retrieval in the UI.
- Do **not** mount any other index (for example a historical local archive such as `data/04_index`): the release descriptor accepts only an index that matches the canonical corpus.
