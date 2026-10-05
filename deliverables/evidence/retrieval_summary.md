# Retrieval acceptance summary

Derived from the final retrieval evaluation of the production index (`vector-pool-expansion-v1`, mode `production-like`, real embeddings). Retrieval only: no model answers are scored here.

## Provenance

| Item | Value |
|------|-------|
| Raw artifact | `data/05_evaluation/stage2g_pool_expansion.json` (not tracked: gitignored, ~2978 KiB) |
| Raw artifact SHA-256 | `ff291709dbcd9f32d5c786ac53216aea3ad5157afb8d440643228bcc13123cc7` |
| Evaluation run (UTC) | 2026-10-05T04:48:59.846216Z |
| Evaluation code commit | `54b0336b275aae29e833fe5602284f90ca9aa30e` (working tree clean: true) |
| Summary derived by | `scripts/summarize_retrieval_evidence.py` at `0d5b94dbc57425b97309e66877f7b19cab4e46e9` |

## Identity

| Item | Value |
|------|-------|
| Index / corpus fingerprint | `aa1005e12a33bf6fb0b3c0e5d38262a88ba7c5528d930fbc44202f3eb76502be` |
| Chunks / documents | 216 / 10 |
| Embedding model, collection | `text-embedding-3-small`, `customer_claims` |
| Evaluation dataset fingerprint (60 frozen cases) | `339ee42b744e3f17df5c963b01895147c29d64c55163582293332baf11a36861` |
| Retrieval config hash | `ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048` |
| Reranker | `source-authority-v1` (`c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957`) |
| Contract | fetch 24 -> pool 24 -> final 12, threshold 0.0 |

## Metrics (production configuration: pool@24 -> source-authority-v1 -> final top-12)

| Metric | Value | Cases |
|--------|------:|------:|
| Hit@1 | 0.517 | 30/58 |
| Hit@4 | 0.897 | 52/58 |
| Hit@12 | 0.983 | 57/58 |
| Primary hit@1 | 0.397 | 23/58 |
| Primary hit@4 | 0.759 | 44/58 |
| Supporting hit@4 | 0.605 | 23/38 |
| MRR | 0.715 | 60 |

## Candidate-pool reachability of the expected primary document

| Pool | Primary | Supporting | Fully unreachable |
|------|--------:|-----------:|-------------------|
| pool@12 | 51/57 | 29/38 | T004, T040, T047 |
| pool@24 (production) | 56/57 | 35/38 | T004 |

| Risk | pool@12 | pool@24 |
|------|--------:|--------:|
| critical | 6/8 | 8/8 |
| high | 13/15 | 15/15 |
| medium | 18/19 | 19/19 |
| low | 14/15 | 14/15 |

## Comparison with the previously accepted baseline

Baseline: `data/05_evaluation/vector_pool_expansion_v1.json` (index `bf3df0d4...`, 215 chunks, run 2026-06-21T17:58:20.730986Z), measured on the corpus **before** the content corrections that produced the current corpus. Same frozen dataset: true; same retrieval config: true; same reranker config: true.

| Metric | Baseline | Current | Delta |
|--------|---------:|--------:|------:|
| Hit@1 | 0.517 | 0.517 | +0.000 |
| Hit@4 | 0.897 | 0.897 | +0.000 |
| Hit@12 | 0.983 | 0.983 | +0.000 |
| Primary hit@1 | 0.379 | 0.397 | +0.017 |
| Primary hit@4 | 0.741 | 0.759 | +0.017 |
| Supporting hit@4 | 0.605 | 0.605 | +0.000 |
| MRR | 0.715 | 0.715 | +0.000 |

Per-case regressions versus the baseline: **0**.
Per-case improvements: **1**.
- T011 (high): `primary_hit_at_4` False -> True

## High and critical cases: retrieval next to the deterministic customer-output behaviour

`Primary@4` is whether the expected primary document is among the first four results. The deterministic columns are what the application does for the case text regardless of retrieval: the risk floor, and which text the customer-output policy selects when the model returns an ordinary, policy-clean grounded draft (`category_template` = fixed text that never comes from the knowledge base or the model). `no_signal` means no rule matched.

| Case | Dataset risk | Primary@4 | Best primary rank (final top-12) | Assessment | Floor | Priority | Customer text |
|------|--------------|:---------:|:--------------------------------:|------------|-------|:--------:|---------------|
| T011 | high | yes | 4 | `rule_match` | `high` | no | `llm_draft` |
| T012 | high | yes | 1 | `rule_match` | `high` | no | `llm_draft` |
| T013 | high | yes | 2 | `rule_match` | `high` | no | `llm_draft` |
| T024 | high | yes | 1 | `rule_match` | `high` | no | `llm_draft` |
| T027 | high | yes | 3 | `rule_match` | `high` | no | `llm_draft` |
| T035 | high | no | 5 | `rule_match` | `high` | no | `llm_draft` |
| T036 | high | yes | 1 | `rule_match` | `high` | no | `llm_draft` |
| T037 | high | yes | 1 | `rule_match` | `high` | no | `llm_draft` |
| T039 | high | yes | 4 | `no_signal` | `low` | no | `llm_draft` |
| T040 | critical | no | 8 | `rule_match` | `critical` | yes | `category_template` |
| T041 | critical | yes | 1 | `rule_match` | `critical` | yes | `category_template` |
| T042 | critical | yes | 2 | `rule_match` | `critical` | yes | `category_template` |
| T044 | high | no | 11 | `rule_match` | `high` | no | `category_template` |
| T045 | critical | yes | 1 | `rule_match` | `critical` | yes | `category_template` |
| T046 | critical | yes | 4 | `rule_match` | `critical` | yes | `category_template` |
| T047 | critical | no | 7 | `rule_match` | `critical` | yes | `category_template` |
| T048 | critical | yes | 1 | `rule_match` | `critical` | yes | `category_template` |
| T050 | high | yes | 1 | `rule_match` | `high` | no | `llm_draft` |
| T051 | high | yes | 2 | `rule_match` | `high` | no | `llm_draft` |
| T052 | high | yes | 1 | `rule_match` | `high` | no | `llm_draft` |
| T053 | high | no | - | `no_signal` | `low` | no | `llm_draft` |
| T055 | high | no | 7 | `rule_match` | `high` | no | `llm_draft` |
| T057 | critical | yes | 1 | `rule_match` | `critical` | yes | `category_template` |

## Known limitations visible in this run

- Expected primary document outside pool@24 for every candidate: T004.
- Primary reachable in the pool but not in the first four (ranking-limited after pool expansion): T016, T040, T044, T047, T053.
- Primary document not in the first four, critical cases: T040, T047; high cases: T035, T044, T053, T055.
- Frozen cases labelled high or critical whose deterministic floor is lower or undetermined: T039, T053 (reported as `no_signal`, never as an affirmative low risk).
- Deterministic floor two or more levels above the dataset label (over-escalation): T026.

Pool expansion improves candidate reachability, not top-4 ordering: the source-authority reranker's bonus is small by design, so a large similarity gap is not overcome. The experiment was frozen after it; see `docs/06_release_posture.md`.
