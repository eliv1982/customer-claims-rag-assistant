# Production release posture

This document describes the **selected production release** of the FoodFlow customer claims RAG assistant: which corpus it serves, what is reproducible from the repository, how the release is validated and what is deliberately not claimed.

| Part | File |
|------|------|
| Release descriptor (schema 2.0.0) | `configs/release/production_posture.json` |
| Canonical corpus manifest | `configs/corpus/foodflow_production_v1.json` |
| Frozen retrieval contract | `configs/retrieval/vector_pool_expansion_v1.json` |
| How to build the index | `docs/07_index_provisioning.md` |

## Canonical production corpus

Release **foodflow-10doc-release-v2** serves the corpus **foodflow-10doc-corpus-v2**: documents **01–10** of `data/02_clean_markdown/`, selected by `configs/corpus/foodflow_production_v1.json`. The manifest is the only place the selection is made. Build tools, the release descriptor and the validator read it; no code or document keeps its own list.

| ID | Document | In corpus |
|----|----------|-----------|
| 01 | Service overview | yes |
| 02 | Delivery rules | yes |
| 03 | Order changes and cancellations | yes |
| 04 | Refund policy | yes |
| 05 | Compensation policy | yes |
| 06 | Food quality and packaging | yes |
| 07 | Complaint handling procedure | yes |
| 08 | Escalation and risk rules | yes |
| 09 | Response style and templates | yes |
| 10 | Customer FAQ | yes |
| 11 | Payment security and dispute handling | no, excluded |
| 12 | Staff safety and threat handling | no, excluded |
| 13 | Physical hazard and foreign body protocol | no, excluded |
| 14 | Evidence standards and incomplete information | no, excluded |
| 15 | Conflicting rules and remedy priority | no, excluded |

Documents 11–15 stay in the source directory as experimental material. They are declared in the manifest with their reason and evidence, and every `*.md` file in the source directory must be declared as included or excluded: a new document cannot enter or silently stay out of the corpus.

**Why 11–15 are excluded.** The 15-document index failed the frozen 60-question regression (`tests/09_expanded_corpus_frozen_regression_results.md`, verdict `REJECT / REPAIR REQUIRED`, harmful high/critical regressions on T011, T035, T039, T040, T044, T046, T047, T051); two repair experiments were rejected as well (`tests/11_doc08_atomic_risk_units_results.md`, `tests/12_doc12_threat_atomic_units_results.md`). The decision still holds: the application's deterministic safety layer already covers payment-data, personal-data, threat, hazard and health cases with fixed customer texts that never come from the knowledge base, and nothing in the runtime depends on documents 11–15.

**Corpus content.** Documents 01–10 were corrected in stage 2D2 so that no sentence tells the assistant it may say a case is "registered", "accepted", "recorded" or "transferred" (the application registers, transfers and notifies nothing), and the response templates no longer contain such statements. `tests/unit/test_corpus_kb_policy.py` enforces this. Because the text changed, the corpus identity changed (below); the previous identity is kept as history, not as the current posture.

## Identity: what each value proves

| Value | Where | Reproducible from the repository alone? | Proves |
|-------|-------|------------------------------------------|--------|
| `source_sha256` per document | corpus manifest | yes (no tokenizer) | the source files are the ones that were approved |
| `chunk_count`, `chunk_payload_digest` | corpus manifest, index manifest | yes, with the real `cl100k_base` vocabulary | the chunk topology: chunk ids, texts, metadata and source paths. Independent of any embedding model |
| `corpus_fingerprint` | corpus manifest, index manifest, descriptor target | yes, with the real `cl100k_base` vocabulary | the same chunks **plus the embedding model name** and the index/metadata format versions |
| `embedding_digest` | index manifest | **no** (needs the OpenAI vectors) | attests the vectors the provider returned at build time |
| `collection_content_digest` | index manifest | **no** | attests what the Chroma collection held when the index was written (texts, metadata subset, vectors) |

Nothing here claims that embeddings can be regenerated from repository data. The corpus and chunk layer is reproducible and checked; the vectors are an attested build artifact.

## Exact release identity

| Field | Value |
|-------|-------|
| Release / posture ID | `foodflow-10doc-release-v2` |
| Default target | `active` (the only target) |
| Canonical corpus | `foodflow-10doc-corpus-v2` (`configs/corpus/foodflow_production_v1.json`) |
| Index path | `data/04_index_production` |
| Chunks / documents | `216` / `10` |
| Chunk payload digest | `f46fa97741c3047628cb2d06cc48d4a11444c889ed894189b5f488f1e4c1b393` |
| Corpus fingerprint (`text-embedding-3-small`) | `aa1005e12a33bf6fb0b3c0e5d38262a88ba7c5528d930fbc44202f3eb76502be` |
| Collection | `customer_claims` |
| Embedding model / dimension | `text-embedding-3-small` / `1536` |
| Frozen config hash | `ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048` |
| Retrieval contract | vector fetch `24` → pool `24` → final `12`, threshold `0.0` |
| Reranker | `source-authority-v1` |

The descriptor does not repeat the corpus values: its target references the corpus manifest and the validator reads the expectations from there. A test (`tests/unit/test_release_docs_consistency.py`) keeps this table equal to the committed configuration.

The previous release `foodflow-10doc-release-v1` (index `data/04_index_backup_10docs_215chunks`, fingerprint `bf3df0d4…`) was a manually provisioned index that could not be rebuilt from the repository. It is superseded, not maintained, and survives only as historical evidence (below).

## Release states and validation

`validate-release-posture` reports these separate facts:

| Line | Meaning |
|------|---------|
| `canonical_corpus_sources` | whether the source files in this checkout match the manifest (`verified`, `mismatch (...)`, or `not_present` in a deployment without the corpus) |
| `index_present` | whether a Chroma index exists at the target path |
| `index_matches_canonical_corpus` | `yes` / `no` / `not_checked`: whether the index matches the corpus, **recomputed from the stored records** |
| `static_validation` | `passed` when the descriptor, the manifests and the frozen config validated. It says nothing about the index content and is **not** a release verdict |
| `release_can_proceed` | `yes` only when the index content was verified against the canonical corpus; `no` otherwise, with the command that builds it; `not_established` when the static checks passed but the content was not verified |

The command is a release gate, and its exit status says which of the three verdicts it reached:

| Exit status | `release_can_proceed` | Meaning |
|-------------|-----------------------|---------|
| `0` | `yes` | readiness is established: the index content matches the canonical corpus |
| `1` | `no` | the release cannot proceed: missing, invalid or stale index, source mismatch, or an invalid descriptor |
| `3` | `not_established` | the static checks passed but the index content was not verified (`--skip-vector-store`) |

Exit status `2` stays argparse's usage error. `--skip-vector-store` still performs the static validation (manifests and frozen config, no Chroma), but because it cannot establish readiness it never exits `0`: automation must not treat it as a passing release gate.

**Recomputed from what is stored** (reading a few hundred records; no embedding request):

- chunk count and document ids;
- chunk payload digest and corpus fingerprint, from the stored chunk texts, ids and metadata, compared with the committed expectations **and** with the index manifest;
- collection content digest, from texts, metadata and vectors, compared with the index manifest;
- vector dimension, from the length of every stored vector.

**Trusted, because it cannot be checked offline:** that the vectors were produced by the model the manifest names (only the model *name*, bound into the fingerprint, and the dimension are checkable) and `embedding_digest`.

What this catches: a wrong corpus, a stale index (older chunk text), a manifest edited to claim the approved identity over different content, an edited chunk, vector or metadata in the store, an added or removed chunk, vectors of another dimension, and an index written by an older metadata schema. A manifest alone never passes.

Startup of `answer-claim` and the Streamlit UI runs the same validation; if the index is missing or does not match, they stop before any retrieval.

## Historical evidence (kept, not current)

The retrieval experiments (stages 2C–4C) ran on the 10- and 15-document corpus as it was **before** the stage 2D2 content corrections. That corpus is frozen in `experiments/corpus/historical_pre_2d2_15doc_v1/` (documents plus `snapshot_manifest.json` with their hashes and identities: 10-document release v1 `bf3df0d4…`/215 chunks, 15-document expansion `b9526128…`/333 chunks). The tracked experiment artifacts in `data/05_evaluation/` and the experiment tests are reproduced from that snapshot; the real-tokenizer lane rebuilds the chunks and compares the fingerprints. The snapshot is never the production corpus and must not be edited.

Retrieval quality measured on the old text (frozen 60-question benchmark) has **not** been re-measured on the corrected corpus: that needs the new index, which needs the OpenAI embedding API.

## Known limitations

- The vector index is a build artifact. A clone has the corpus, the manifests and the build tool, not a runnable index: build it (`docs/07_index_provisioning.md`) with an `OPENAI_API_KEY`.
- Documents 11–15 are not production-supported; the specialized behaviors they describe are not available as retrieved knowledge.
- Embedding vectors are attested, not reproducible (see above).
- `RAG_INDEX_DIR` is **not** used by the production answer/UI flow; it applies only to build/search/evaluation CLIs.

## Changing the corpus

1. Edit the documents.
2. `build-chunks --corpus-manifest configs/corpus/foodflow_production_v1.json --refresh-manifest` (needs the real `cl100k_base` vocabulary in the local tiktoken cache) recomputes `source_sha256` and `expected`; review the diff.
3. Commit the documents, the manifest and `tests/fixtures/canonical_corpus_chunk_topology_v2.json` together, and rebuild the index.

Until the index is rebuilt, `validate-release-posture` reports the old index as not matching.

## Rollback

There is no archive to copy. A rollback is a repository operation: check out the previous release commit (its corpus manifest, descriptor and documents) and rebuild the index from it. The release descriptor accepts only the committed corpus, so an index built from anything else is refused.

## Chroma internal storage drift

Read-only runtime access to a built Chroma index may update internal persistent files (`chroma.sqlite3`, HNSW segment files) **without changing the semantic index identity**: opening the collection, counting and reading records and similarity queries leave collection name, ids, counts, metadata, embeddings and rankings unchanged. Do not treat byte-level drift of those files as corruption; the identity above is what is verified.
