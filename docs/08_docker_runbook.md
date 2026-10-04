# Docker runbook (stage 5B, hardened in stage 2E2)

Reproducible local deployment of the FoodFlow customer claims Streamlit UI with Docker Compose.

**Prerequisites**

- Docker Desktop (Windows/macOS) or Docker Engine with the Compose plugin (Linux), and Python 3 on the host (the launcher below uses only its standard library)
- The production index, built on the host from the repository (`docs/07_index_provisioning.md`). It is bind-mounted at run time and is **never** copied into the image, the build context or Git
- `OPENAI_API_KEY` exported in the shell that runs the launcher (never committed, never baked into the image)

Browser URL after startup: **http://127.0.0.1:8501**

---

## The supported way to run it: `scripts/release_compose.py`

Run every Compose command through the launcher. A bare `docker compose` stops at a guard in `compose.yaml` with an instruction, on purpose:

```text
python scripts/release_compose.py <compose command> [args...]
```

Compose reads a project `.env` for variable interpolation by default, so an old repository `.env` could silently hand its OpenAI key to a release container. The launcher closes that:

- `OPENAI_API_KEY` must be set in the **process environment**, non-empty and without whitespace. A repository `.env` is never used as the credential source, even when it defines the variable (the message says it was ignored);
- Compose is invoked with an empty `--env-file`, a pinned `-f compose.yaml` and `--project-directory`, so no project `.env`, `COMPOSE_ENV_FILES`, `COMPOSE_FILE` or override file is picked up; Compose's own global options are refused as the first argument;
- the key reaches Compose in the environment of the child process, never on a command line, and is not printed. `config` prints the resolved environment, so its output is redacted (`OPENAI_API_KEY: [REDACTED]`).

| Situation | Result (exit status) |
|-----------|----------------------|
| Key exported in the shell | Compose runs; the container receives that key (`credential source=process environment`) |
| Only a repository `.env` defines the key | refused (`2`): "not set in the process environment ... `.env` ... deliberately ignored" |
| Key absent | refused (`2`) |
| Key empty, blank, or containing whitespace/newline | refused (`2`) |
| Bare `docker compose ...` | stops at `x-release-guard` ("run this stack through scripts/release_compose.py") |

Docker's runtime environment is visible to anyone who can talk to the Docker daemon (`docker inspect`); the launcher prevents an accidental bake, log or repository leak, not an authorized operator reading the container's environment.

---

## Windows PowerShell

```powershell
$env:OPENAI_API_KEY = "your-key-here"
# Optional:
$env:RAG_ACTIVE_INDEX_HOST_PATH = ".\data\04_index_production"   # host side of the index mount only
$env:RAG_HOST_PORT = "8501"

python scripts/release_compose.py config            # resolved configuration, key redacted
python scripts/release_compose.py build
python scripts/release_compose.py run --rm streamlit validate-release-posture
python scripts/release_compose.py up -d --wait
python scripts/release_compose.py ps
python scripts/release_compose.py logs -f streamlit
python scripts/release_compose.py down
```

## Linux / macOS

```bash
export OPENAI_API_KEY="your-key-here"
export RAG_ACTIVE_INDEX_HOST_PATH="./data/04_index_production"   # optional

python3 scripts/release_compose.py config
python3 scripts/release_compose.py build
python3 scripts/release_compose.py run --rm streamlit validate-release-posture
python3 scripts/release_compose.py up -d --wait
curl -fsS http://127.0.0.1:8501/_stcore/health
python3 scripts/release_compose.py down
```

`up -d --wait` is the form for automation: it exits non-zero if the container exits or never becomes healthy (for example when release validation fails). A plain attached `up` exits `0` even when the container it started has stopped with an error: read the logs, not that status.

Every launcher command needs the key in the environment, including `down`, `ps` and `logs`: Compose parses the whole file each time.

---

## What the image and the stack contain

| Property | Setting |
|----------|---------|
| Base | `python:3.12-slim`; dependencies resolve from the ranges in `pyproject.toml` at build time (there is no lock file, same as CI) |
| Packages | `pip install ".[ui]"` only: the application and Streamlit. No `[dev]`, pytest, Playwright, coverage or CI tooling; no tokenizer vocabulary (needed to construct/verify canonical chunk topology, not for already-bounded embedding requests) |
| Process user | `10001:10001`, dedicated account, no sudo, final `USER` is numeric and non-root |
| Application tree | root-owned and not writable by the process (`/app`, source, configs, prompts, scripts) |
| Root filesystem | read-only (`read_only: true`); the only writable places are a 64 MB `tmpfs` at `/tmp` and the index mount |
| Privileges | `cap_drop: ALL`, `no-new-privileges`; no privileged mode, host network, device or Docker-socket mount |
| Telemetry | Chroma and Streamlit usage statistics off; the browser address is set so Streamlit does not ask a third party for the host's public IP at start |
| Health check | `scripts/healthcheck.py` (standard library, loopback only, ignores proxy variables), 30 s interval, 60 s start period |
| Restart | `on-failure:3`: a failing release validation is retried a few times and then stays stopped, not looped forever |
| Image content | the canonical `data/02_clean_markdown/*.md` sources are included so startup can recompute source identity; no `.env`, `.git`, `.venv`, tests, docs, chunks, indexes, evaluation output, tokenizer cache or credential; the build context is an allowlist (`.dockerignore`) |

### Network exposure

Compose publishes the UI on **`127.0.0.1:8501` only** (container port `8501`). The UI has no authentication, so anything wider is an explicit opt-in:

```powershell
$env:RAG_BIND_ADDRESS = "0.0.0.0"     # all interfaces; the launcher warns
$env:RAG_HOST_PORT = "8600"
```

If you opt in, put an authenticating reverse proxy or a firewall in front of it. Publishing the port is only half of it: measured with the image defaults, the UI's websocket accepts the origins `http://127.0.0.1:8501` and `http://localhost:8501` and answers any other origin (a LAN address, another hostname) with HTTP 403, so a wider deployment must also allow the origin its users type (Streamlit's `server.corsAllowedOrigins`). `compose.yaml` does not forward such a setting and this repository neither ships nor tests that configuration. The container needs outbound HTTPS to the OpenAI API to answer a claim.

### The index mount is read-write, deliberately

Chroma opens its SQLite database read-write even to answer queries. Measured with the pinned Chroma: a read-only mount fails inside the library with `attempt to write a readonly database`, and a normal start rewrites bytes of `chroma.sqlite3` and the segment's `length.bin` (same file count, same logical contents, which is the benign drift described in `docs/06_release_posture.md`). So the mount is writable and is the **only** writable path besides `/tmp`; the application tree and the root filesystem are not.

The host directory must be writable by uid `10001`. Docker Desktop (Windows/macOS) mounts are permissive. On Linux, fix the host path if the entrypoint reports it is not writable:

```bash
chown -R 10001:10001 data/04_index_production
```

---

## Startup order (the supported production path)

1. **Configuration.** `OPENAI_API_KEY` must be present in the container environment (never printed). Otherwise: exit `2`.
2. **Writability.** If an index exists, it must be writable by the process. Otherwise a clear error, exit `1`, instead of an engine traceback.
3. **Full release validation.** `validate-release-posture` opens the index and checks its content against the canonical corpus (`index_integrity=store_recomputed`). Exit `0` only when `release_can_proceed=yes`; its own exit status is passed on.
4. **Only then** `exec` replaces the shell with Streamlit.

If any step fails, no server exists and the container exits non-zero. Nothing in startup builds, repairs, copies or creates an index (there is no fallback to `data/04_index`, to a historical directory, or to an empty index), and the static-only check mode is never used. Provisioning and serving are separate operations.

Only `streamlit ...` (the default command) is gated; `validate-release-posture`, `answer-claim` and `sh` run unchanged for maintenance. Replacing the command is an explicit operator action, and the application itself re-validates the release posture before it builds a pipeline, so it never answers from an invalid index either way. A healthy `/_stcore/health` is liveness only; it does not replace this gate.

## Fail-closed errors

| Symptom | Where | Meaning |
|---------|-------|---------|
| `OPENAI_API_KEY is not set ...` | launcher (exit `2`) | export it in the shell; a `.env` is ignored |
| `the production index directory does not exist: ...` | launcher (exit `2`) | the host directory is missing; Compose will not create it. The message carries the build command |
| `index_present=no`, `next_step=python -m customer_claims_rag.cli.build_index ...`, `the server was not started` | container, exit `1` | the mounted directory is empty or has no `chroma.sqlite3`; build the index on the host |
| `index_matches_canonical_corpus=no` and a mismatch (for example `vector store chunk count mismatch: store has 215, expected 216`) | container, exit `1` | stale, partial or foreign index; rebuild it |
| `the production index at ... is not writable by uid 10001` | container, exit `1` | read-only mount or wrong ownership; see above |
| `OPENAI_API_KEY is not set, is empty or contains whitespace in the container environment` | container, exit `2` | the container was started without the launcher |

## Troubleshooting

- **Missing index:** build the host directory per `docs/07_index_provisioning.md`, then `python scripts/release_compose.py run --rm streamlit validate-release-posture`.
- **Wrong mount:** `Test-Path .\data\04_index_production\manifest.json` and `chroma.sqlite3`; on Windows prefer a relative `RAG_ACTIVE_INDEX_HOST_PATH` such as `.\data\04_index_production`.
- **Port in use:** set `RAG_HOST_PORT` (the launcher forwards it).
- **Unhealthy container:** `python scripts/release_compose.py logs streamlit`; confirm the gate passed and Streamlit is listening; raise `start_period` if the first Chroma open is slow.

## Maintenance commands

```powershell
python scripts/release_compose.py run --rm streamlit validate-release-posture
python scripts/release_compose.py run --rm streamlit answer-claim --message "Тестовый запрос"
python scripts/release_compose.py run --rm streamlit sh
```

`run` mounts the index, so the host directory must exist. `answer-claim` needs the network and spends API credit.

## What not to do

- Do **not** copy the production index into the Docker build context or image.
- Do **not** start the stack with a bare `docker compose`, and do not define `RELEASE_COMPOSE_LAUNCHER` in a `.env` to get past the guard.
- Do **not** pass the key with `--build-arg` or store it in a file that is mounted into the container.
- Do **not** set `RAG_INDEX_DIR` expecting it to change production retrieval in the UI.
- Do **not** mount any other index (for example a historical local archive such as `data/04_index`): the release descriptor accepts only an index that matches the canonical corpus.

---

## Verification record (stage 2E2.1)

Measured against Docker Desktop 29.8.1 (Linux containers). The corrected production index had not been built, so the "valid index" rows used a throwaway **synthetic** index (hash-based vectors, built from the canonical corpus by the application's own build path in a scratch directory, no provider call). It exercises the release gate and the server; it is not a production index and says nothing about retrieval quality.

| Check | Result |
|-------|--------|
| Build, `pip check`, runtime user | builds; no broken requirements; uid `10001`; 0 writable files under `/app`; no pytest/Playwright/coverage |
| Build context | allowlist: `src`, `configs`, `prompts`, canonical `data/02_clean_markdown/*.md`, 2 runtime scripts, `pyproject.toml`, `README.md`; host `__pycache__`, `egg-info` and all other `data/` remain excluded |
| Image content | exactly the 15 source files declared as included/excluded by the corpus manifest; no `.env`, `.git`, `.venv`, tests, docs, chunks, index contents, tokenizer cache or evaluation output |
| Embedding preprocessing | with an empty cache, tokenizer functions counted and container networking disabled, a bounded query and a canonical-sized document reached a fake provider (HTTP transport) with zero tokenizer calls (`check_embedding_ctx_length=false`); input size is bounded by the project, not by LangChain: a document over 4000 characters, a batch over 512 documents, or a bare string is rejected locally with no provider call |
| Source identity gate | a valid 216-chunk synthetic index proceeds only with verified image sources; deleting or mutating one included source exits `1` with `release_can_proceed=no` before index acceptance |
| Missing host index directory | launcher refuses (exit `2`) with the build command; nothing created on the host |
| Empty index directory | container validation fails (exit `1`), prints the build command, server not started, host directory untouched |
| Stale index (215 of 216 chunks) | `vector store chunk count mismatch`, exit `1`, no server |
| Read-only mount / non-owner mount | clear "not writable by uid 10001" error, exit `1`, no traceback |
| Valid synthetic index, full hardening | healthy; published `127.0.0.1:8501` only (refused on all 5 non-loopback host addresses; the previous `8501:8501` answered on the LAN address); 0 paths written outside the index mount; in-app form renders; `docker stop` 1.3 s |
| Packaging smoke (entrypoint bypassed for the test only), no network | health endpoint answers; no external-IP lookup; the app itself still refuses without an index |
| Secret sentinel | not in image config, history, any of 19 layer blobs, build, startup or failure output |
