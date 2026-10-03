# Production index provisioning

This document describes how to build and verify the **production Chroma index** for release `foodflow-10doc-release-v2`.

The index is a **build artifact**: it is produced from the repository by one command, it is not stored in Git, it is not copied from anywhere and it is not baked into the Docker image. Nothing in this procedure needs an archive, a maintainer or a private download.

## What a clone contains and what it does not

A fresh `git clone` contains everything deterministic: the cleaned Markdown corpus, the canonical corpus manifest (`configs/corpus/foodflow_production_v1.json`), the release descriptor, and the tools that build and verify the index. It does **not** contain the vectors: they come from the OpenAI embeddings API, so only the last step of the build needs credentials and network.

| Step | Needs |
|------|-------|
| select the canonical documents, verify their hashes | nothing |
| chunk with the real `cl100k_base` tokenizer, verify chunk topology and fingerprint | the tokenizer vocabulary in the local tiktoken cache (downloaded once by `tiktoken`) |
| embed the chunks | `OPENAI_API_KEY`, network |
| write the Chroma index and its manifest | nothing |
| validate the release | nothing (no OpenAI call) |

## Release identity

See `docs/06_release_posture.md` for the exact values (chunk payload digest, corpus fingerprint, counts, model, dimension) and for what each one proves.

| Field | Value |
|-------|-------|
| Release / posture ID | `foodflow-10doc-release-v2` |
| Target | `active` |
| Descriptor | `configs/release/production_posture.json` |
| Canonical corpus | `configs/corpus/foodflow_production_v1.json` |
| Local index path | `data/04_index_production` |
| Docker container path | `/app/data/04_index_production` |
| Collection | `customer_claims` |
| Embedding model / dimension | `text-embedding-3-small` / `1536` |

Production target selection uses `RAG_RELEASE_TARGET` against the committed descriptor. `RAG_INDEX_DIR` applies only to development and evaluation CLIs and does not select the production answer or Streamlit flow.

## Build the index

1. Install the project (see `README.md`) and set `OPENAI_API_KEY` in the environment (or in your own `.env`; the build tool loads it if present).
2. Check the corpus and see the plan, without any embedding request:

   ```powershell
   python -m customer_claims_rag.cli.build_index --corpus-manifest configs/corpus/foodflow_production_v1.json --index-dir data/04_index_production --collection customer_claims --embedding-model text-embedding-3-small --dry-run
   ```

   This verifies that the source files match the manifest, builds the chunks and compares chunk count, chunk payload digest and corpus fingerprint with the manifest, then prints the plan. It stops with an error, before anything is embedded or written, if any of that differs.

3. Build:

   ```powershell
   python -m customer_claims_rag.cli.build_index --corpus-manifest configs/corpus/foodflow_production_v1.json --index-dir data/04_index_production --collection customer_claims --embedding-model text-embedding-3-small --rebuild
   ```

   `--rebuild` is destructive for `data/04_index_production` only. `validate-release-posture` prints this exact command whenever the index is missing or does not match.

4. Validate:

   ```powershell
   validate-release-posture
   ```

   Expected: exit status `0` with `index_matches_canonical_corpus=yes`, `release_can_proceed=yes` and `index_integrity=store_recomputed`. Exit status `0` is only ever returned when the index content was verified; `validate-release-posture --skip-vector-store` is a static check that exits `3` (`release_can_proceed=not_established`) and is not a release gate.

The deterministic part alone (no credentials, no network after the vocabulary is cached) is `build-chunks --corpus-manifest configs/corpus/foodflow_production_v1.json`: it writes `data/03_chunks/chunks.jsonl` and verifies the same identity.

## Expected directory layout

```text
data/04_index_production/
  manifest.json
  chroma.sqlite3
  <uuid-segment-directory>/
    data_level0.bin
    header.bin
    length.bin
    link_lists.bin
```

- **`manifest.json`** is the index's attestation: corpus fingerprint, chunk payload digest, embedding and collection digests, counts, model, dimension. The validator compares it with the committed expectations and re-derives what it can from the stored records; it is never trusted on its own.
- **`chroma.sqlite3`** and the segment directory are Chroma's persistent store.

## Validate in Docker

```powershell
docker compose run --rm streamlit validate-release-posture
```

The index is built on the host (above) and bind-mounted at `/app/data/04_index_production`; set `RAG_ACTIVE_INDEX_HOST_PATH` to use another host directory. See `docs/08_docker_runbook.md`.

## Failure behavior

| Condition | Result |
|-----------|--------|
| Index directory missing or empty | `validate-release-posture` exits `1` with `index_present=no` and the build command; the container preflight fails; Streamlit does not start |
| Missing `chroma.sqlite3` | container preflight fails with a clear error |
| Index built from another corpus, stale, or hand-edited | exit `1`, `index_matches_canonical_corpus=no`, the mismatching value and the build command |
| Manifest without content digests | refused: the index was not written by this build path |
| Source files differ from the corpus manifest | `build-chunks` / `build-index` refuse to proceed; `validate-release-posture` reports `canonical_corpus_sources=mismatch (...)` |
| Missing `OPENAI_API_KEY` | the embedding step of the build, and production retrieval at runtime, fail; validation itself does not need the key |

## Replacing an index

Rebuilding with `--rebuild` replaces `data/04_index_production`. Older local indexes (for example the historical 15-document index at `data/04_index` or the release v1 backup directory) are not production indexes, are not read by the release flow and are not touched by this procedure; they matter only to the historical experiment checks (`docs/06_release_posture.md`).

## Benign Chroma internal byte drift

Read-only runtime access may update internal persistent bytes in `chroma.sqlite3` and HNSW segment files **without** changing semantic identity. Do not treat such drift alone as corruption; the validator checks the identity values listed in `docs/06_release_posture.md`.
