# Deliverables and evidence map

What in this repository is current evidence, what is history, and what is only produced locally. Nothing historical is rewritten to look current; where an artifact is outdated, this page and a banner in the artifact say so.

## Current

| Path | What it is |
|------|------------|
| [`evidence/`](evidence/README.md) | Final acceptance evidence for release `foodflow-10doc-release-v2`: scenarios, release-gate output, retrieval summary, screenshots, manifest with provenance |
| [`../docs/06_release_posture.md`](../docs/06_release_posture.md) | The exact release identity and what each digest proves |
| [`../tests/01_test_questions.md`](../tests/01_test_questions.md), [`../tests/02_expected_answers.md`](../tests/02_expected_answers.md) | The 60 frozen evaluation cases and their expected sources (dataset fingerprint `339ee42b744e3f17...`) |
| [`../tests/fixtures/`](../tests/fixtures/) | Fixtures the test suite pins the canonical corpus topology with (`canonical_corpus_chunk_topology_v2.json`) |

## Historical (kept, labelled)

| Path | Why it is history |
|------|-------------------|
| [`historical/stage5c_release_v1/`](historical/stage5c_release_v1/README.md) | The Stage 5C manual acceptance package for release v1: 215 chunks, a manually provisioned index, pre-stage-2C CLI schema, garbled CLI captures, UI screenshots of the earlier UI, SHAs from before the history cleanup |
| [`../tests/03_test_results.md`](../tests/03_test_results.md), [`04_improvement_log.md`](../tests/04_improvement_log.md) | Baseline retrieval evaluation of 2026-06-21 on the release v1 corpus (215 chunks, fingerprint `bf3df0d4...`) |
| [`../tests/05_answer_level_test_results.md`](../tests/05_answer_level_test_results.md) | Manual prompt-design runs of 2026-06-21, before any code existed |
| [`../tests/06_reranking_ab_results.md`](../tests/06_reranking_ab_results.md), [`07_vector_pool_expansion_results.md`](../tests/07_vector_pool_expansion_results.md), [`08_hybrid_lexical_vector_results.md`](../tests/08_hybrid_lexical_vector_results.md) | Retrieval experiments on the release v1 corpus: the first two selected the production retrieval configuration, hybrid lexical + vector was rejected. The current corpus was re-measured in the final retrieval summary |
| [`../tests/09_expanded_corpus_frozen_regression_results.md`](../tests/09_expanded_corpus_frozen_regression_results.md), [`10_vector_pool_36_cap4_results.md`](../tests/10_vector_pool_36_cap4_results.md), [`11_doc08_atomic_risk_units_results.md`](../tests/11_doc08_atomic_risk_units_results.md), [`12_doc12_threat_atomic_units_results.md`](../tests/12_doc12_threat_atomic_units_results.md) | Experiments on the 15-document expansion (documents 11-15): the expansion was rejected (09), a candidate-generation repair was only a partial fix (10) and two further repairs were rejected (11, 12). They record the maintainer's workstation paths and commit SHAs from before the history cleanup inside their results, which are covered by result identities, so they are left unedited |
| `../data/05_evaluation/*.json` (tracked ones) and `../data/05_evaluation/embedding_snapshots/` | Machine-readable results of those experiments and fixed embedding snapshots they are replayed from. Reproduced by tests from [`../experiments/corpus/historical_pre_2d2_15doc_v1/`](../experiments/corpus/historical_pre_2d2_15doc_v1/snapshot_manifest.json), never from the live corpus |
| [`../tests/fixtures/frozen_retrieval_baseline_v1.json`](../tests/fixtures/frozen_retrieval_baseline_v1.json), [`historical_15doc_chunk_topology_v1.json`](../tests/fixtures/historical_15doc_chunk_topology_v1.json) | Frozen fixtures of the historical corpus and run |
| [`../docs/00_project_scope.md`](../docs/00_project_scope.md) to [`05_business_rules_registry.md`](../docs/05_business_rules_registry.md) | Data-preparation records of June 2026 (scope, inventory, cleaning, chunking, metadata, rules registry). Their "not yet done" statements describe the time of writing |

## Local only (gitignored, never committed)

| Path | What it is |
|------|------------|
| `data/04_index_production/` | The production Chroma index, built with live embeddings (`docs/07_index_provisioning.md`) |
| `data/05_evaluation/stage2g_pool_expansion.json` (+ `.md`) | Raw per-case output of the final retrieval run (about 3 MB); the committed `evidence/retrieval_summary.*` is derived from it and records its SHA-256 |
| `data/04_index/`, `data/04_index_experiments/`, `data/04_index_backup_*` | Development indexes and historical archives |
