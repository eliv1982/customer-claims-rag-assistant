# Production release posture (stage 4C.4-B)

This document describes the **selected production release** for the FoodFlow customer claims RAG assistant.

## Supported production corpus

Production release **foodflow-10doc-release-v1** supports documents **01–10** only:

| ID | Document |
|----|----------|
| 01 | Service overview |
| 02 | Delivery rules |
| 03 | Order changes and cancellations |
| 04 | Refund policy |
| 05 | Compensation policy |
| 06 | Food quality and packaging |
| 07 | Complaint handling procedure |
| 08 | Escalation and risk rules |
| 09 | Response style and templates |
| 10 | Customer FAQ |

## Experimental documents

Documents **11–15** are **not** production-supported and remain experimental:

- 11 — Payment security and dispute handling
- 12 — Staff safety and threat handling
- 13 — Physical hazard and foreign body protocol
- 14 — Evidence standards and incomplete information
- 15 — Conflicting rules and remedy priority

## Deferred specialized capabilities

The following **specialized corpus capabilities** are deferred (not covered by the 10-document production index):

- Payment-security handling (document 11)
- Staff-threat handling (document 12)
- Physical-hazard handling (document 13)
- Specialized evidence guidance (document 14)
- Specialized remedy-priority guidance (document 15)

The assistant retains **general** deterministic risk classification, escalation, and handoff behavior via documents 07–08 and the application risk layer. Deferred items refer only to the specialized policy content in documents 11–15.

## Exact release identity

| Field | Value |
|-------|-------|
| Release / posture ID | `foodflow-10doc-release-v1` |
| Default target | `active` |
| Active index path | `data/04_index_backup_10docs_215chunks` |
| Corpus fingerprint | `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3` |
| Chunks / documents | `215` / `10` |
| Collection | `customer_claims` |
| Embedding model / dimension | `text-embedding-3-small` / `1536` |
| Frozen retrieval config | `configs/retrieval/vector_pool_expansion_v1.json` |
| Frozen config hash | `ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048` |
| Retrieval contract | vector fetch `24` → pool `24` → final `12`, threshold `0.0` |
| Reranker | `source-authority-v1` |

Descriptor: `configs/release/production_posture.json`

## Local deployment artifact

The physical 10-document Chroma index is a **separately provisioned local artifact**. It is intentionally **not** committed to Git.

- A fresh clone does **not** contain a runnable production index.
- Provision the index at `data/04_index_backup_10docs_215chunks` with fingerprint and metadata matching the descriptor.
- Production startup **fails closed** when the artifact is absent or mismatched.

Validate after provisioning:

```powershell
validate-release-posture
```

## Known limitations

- Production index must be provisioned locally; Git alone is insufficient for production retrieval.
- Documents 11–15 are excluded from production support.
- Emergency rollback returns to the previously rejected 15-document posture (archive only).
- Specialized deferred capabilities listed above are not available in production.
- Startup fails closed on index, manifest, or frozen-config mismatch.
- `RAG_INDEX_DIR` is **not** used by production answer/UI flow; it applies only to build/search/evaluation CLIs.

## Rollback procedure (emergency only)

Rollback does **not** copy, rename, rebuild, or overwrite either index directory.

1. Select the rollback target explicitly:

   ```powershell
   $env:RAG_RELEASE_TARGET = "rollback"
   ```

2. Validate release posture:

   ```powershell
   validate-release-posture --target rollback
   ```

3. Verify rollback fingerprint: `b9526128dad23e71e812fbd8f26452b8d6efa29e4faac898fa11f279b310e827` (`333` chunks / `15` documents).

4. Start the application (`answer-claim` or Streamlit UI).

5. Confirm diagnostics show `target_status=emergency_rollback_archive_only` — **not** the selected production release.

6. Return to active production:

   ```powershell
   $env:RAG_RELEASE_TARGET = "active"
   validate-release-posture
   ```

No directory copy, rename, or rebuild is required for rollback or return-to-active.

## Chroma internal storage drift (stage 4C.4-B-R1)

Read-only runtime access to a provisioned local Chroma index may update internal persistent files (`chroma.sqlite3`, HNSW segment files) **without changing semantic index identity**.

Investigation on disposable copies outside the repository (active and rollback indices) found:

- Opening `PersistentClient` / `ChromaVectorStore(open_existing=True)` may change byte hashes of `chroma.sqlite3` and HNSW `data_level0.bin` / `length.bin`.
- `get_collection`, record counts, document IDs, metadata, embeddings, manifest fingerprint, and representative similarity-search rankings remained identical across open, count/metadata read, and query steps.
- **Classification:** `BENIGN INTERNAL STORAGE DRIFT`.

Release identity is therefore defined by:

1. Descriptor target identity (`configs/release/production_posture.json`);
2. Manifest corpus fingerprint and chunk/document counts;
3. Semantic record validation (collection name, IDs, counts, supported documents);
4. Frozen retrieval metrics on the unchanged 60-question benchmark.

Do **not** treat internal Chroma/HNSW byte-hash drift alone as index corruption when semantic invariants and frozen metrics remain stable.
