# Production index provisioning (stage 5B)

This document describes how to obtain, install, and verify the **active production Chroma index** for release `foodflow-10doc-release-v1`.

The index is a **local deployment artifact**. It is intentionally **not** stored in Git and **not** baked into the Docker image.

## Why the index is not in Git

- The Chroma store contains embeddings and persistent database files that are large, binary, and environment-specific.
- Release identity is defined by **semantic invariants** (descriptor, manifest fingerprint, collection metadata, frozen retrieval contract), not by committing binary blobs.
- A fresh `git clone` gives you application code and configuration only — **not** a runnable production retrieval store.
- Rebuilding the index is **out of scope** for provisioning; use the approved external archive or secure transfer from the project maintainer.

## Exact release identity

| Field | Value |
|-------|-------|
| Release / posture ID | `foodflow-10doc-release-v1` |
| Target | `active` |
| Descriptor | `configs/release/production_posture.json` |
| Local install path | `data/04_index_backup_10docs_215chunks` |
| Docker container path | `/app/data/04_index_backup_10docs_215chunks` |
| Corpus fingerprint | `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3` |
| Chunks / documents | `215` / `10` |
| Collection | `customer_claims` |
| Embedding model / dimension | `text-embedding-3-small` / `1536` |
| Frozen config hash | `ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048` |
| Retrieval contract | vector fetch `24` → pool `24` → final `12`, threshold `0.0` |
| Reranker | `source-authority-v1` |

Production target selection uses `RAG_RELEASE_TARGET` against the committed descriptor. `RAG_INDEX_DIR` applies only to development/evaluation CLIs (`build-index`, `search-index`, evaluation tools) and **does not** select the production answer or Streamlit flow.

## Expected directory layout

After provisioning, the active index directory must contain at minimum:

```text
data/04_index_backup_10docs_215chunks/
  manifest.json
  chroma.sqlite3
  <uuid-segment-directory>/
    data_level0.bin
    header.bin
    length.bin
    link_lists.bin
```

- **`manifest.json`** — committed semantic identity: corpus fingerprint, chunk/document counts, embedding model, collection name, vector dimension.
- **`chroma.sqlite3`** — Chroma persistent metadata store.
- **Segment directory** — HNSW vector segment files created by Chroma for the `customer_claims` collection.

Do **not** merge files from the rollback archive (`data/04_index`, 15 documents / 333 chunks) into the active directory. They are separate emergency artifacts with a different fingerprint.

## External archive or secure transfer

Obtain the approved index archive through your course instructor, project maintainer, or another **documented secure channel**.

This repository does **not** publish a public download URL or a committed transport checksum. When you receive an archive:

1. Verify it came from the trusted source.
2. Optionally record a transport checksum provided **with** the archive (SHA-256 of the `.zip` / `.tar` file) for your own transfer integrity check.
3. Do **not** confuse a transport/archive checksum with the **corpus fingerprint** in `manifest.json` or with unstable internal Chroma file hashes.

## Safe extraction procedure

### Local (non-Docker)

1. Clone the repository and install dependencies (see `README.md`).
2. Create the target directory if it does not exist:

   ```powershell
   New-Item -ItemType Directory -Force -Path data\04_index_backup_10docs_215chunks
   ```

3. Extract the archive **into** `data/04_index_backup_10docs_215chunks/` so that `manifest.json` and `chroma.sqlite3` sit directly inside that folder (not nested one level deeper).
4. Validate:

   ```powershell
   validate-release-posture
   ```

5. Confirm output includes:

   - `release_posture_id=foodflow-10doc-release-v1`
   - `selected_target=active`
   - `corpus_fingerprint=bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3`
   - `chunk_count=215`
   - `document_count=10`

### Docker

1. Provision the index on the host at the default path `data/04_index_backup_10docs_215chunks`, **or** set `RAG_ACTIVE_INDEX_HOST_PATH` to your host directory before `docker compose up`.
2. The container always expects the mount at `/app/data/04_index_backup_10docs_215chunks` (descriptor path; do not change).
3. See `docs/08_docker_runbook.md` for compose commands.

## Validation command

```powershell
validate-release-posture
```

In Docker (after build):

```powershell
docker compose run --rm streamlit validate-release-posture
```

This uses the shared release-posture boundary: descriptor, manifest fingerprint, frozen config hash, collection name, embedding model, vector dimension, and Chroma record counts. It does **not** call OpenAI.

## Failure behavior

| Condition | Result |
|-----------|--------|
| Directory missing | `validate-release-posture` / container preflight fails; Streamlit does not start |
| Empty directory | Container preflight fails with clear error |
| Missing `chroma.sqlite3` | Container preflight fails with clear error |
| Wrong fingerprint or counts | `ReleasePostureError` / non-zero exit from validator |
| Wrong collection, model, or frozen config | Non-zero exit from validator |
| Missing `OPENAI_API_KEY` | Production pipeline fails at runtime (embeddings/generation); posture validation itself does not need the key |

## Benign Chroma internal byte drift

Read-only runtime access may update internal persistent bytes in `chroma.sqlite3` and HNSW segment files **without** changing semantic identity. See `docs/06_release_posture.md` (stage 4C.4-B-R1).

**Authoritative validation boundaries:**

1. Descriptor target identity (`configs/release/production_posture.json`)
2. `manifest.json` corpus fingerprint and chunk/document counts
3. `validate-release-posture` semantic checks
4. Frozen retrieval metrics on the unchanged 60-question benchmark

Do **not** treat internal Chroma/HNSW byte-hash drift alone as corruption when semantic invariants remain stable.

## Rollback index (not provisioned by default)

The 15-document emergency archive lives at `data/04_index` (fingerprint `b9526128dad23e71e812fbd8f26452b8d6efa29e4faac898fa11f279b310e827`). It is a **separate** artifact, selected only with `RAG_RELEASE_TARGET=rollback`. Docker Compose does **not** mount it by default.

## No rebuild during provisioning

Provisioning means **installing the approved artifact**, not running `build-index` or regenerating embeddings. The default `build-index` CLI targets `data/04_index` with the full 15-document corpus and does **not** produce the active 10-document production fingerprint.
