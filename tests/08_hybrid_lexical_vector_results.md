# Hybrid lexical + vector results: hybrid-lexical-vector-v1

**Timestamp:** 2026-06-22T13:19:12.013593+00:00
**Experiment ID:** `hybrid-lexical-vector-v1`
**Version:** `1.0.0`
**Experiment mode:** `production-like`
**Config hash:** `00476d68eef021da58180b8b97c5e5c4a04f74cb1d1b8c8229d2871c70385b3e`
**Reranker:** `source-authority-v1` (hash `c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957`)
**Score adapter:** `rrf-base-score-adapter-v1`

## 1. Experiment contract

- One shared vector retrieval per case (`vector_k=24`, `threshold=0.0`)
- Lexical BM25 top-24 per case (no embeddings, no LLM)
- Equal-weight RRF (`rrf_k=60`, weights 1.0/1.0) with normalized RRF base score
- Baseline arm: vector top-24 → source-authority-v1 (vector similarity) → final top-12
- Candidate arm: vector + lexical → fusion top-24 → rrf-base-score-adapter-v1 → source-authority-v1 → final top-12
- Index fingerprint: `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3`
- Lexical index fingerprint: `0367cc65b016eed5865d2d831d940d52e997fb0979bc9e37fee64c1ab97a1441`
- Embedding model: `text-embedding-3-small`
- Expected sources used only for post-retrieval evaluation diagnostics

Exact lexical top-24 pools were reconstructed once using the frozen tokenizer, BM25 parameters, queries, and verified corpus fingerprint. The reconstruction did not rerun vector retrieval, fusion, source-authority ranking, or final ranking.

**Score semantics:** `normalized_rrf_score` is the reranker base score in the candidate arm. It is **not** vector similarity. Vector similarity is preserved separately in audit fields.

### Baseline arm terminology

Baseline arm этапа 2C.3 — закрытый candidate arm этапа 2C.2: vector pool@24, source-authority-v1 reranking, final top-12.

## 2. Candidate-generation reachability

| Pool | Primary reachable | Supporting reachable | Fully unreachable |
|------|------------------:|-------------------:|-------------------|
| vector@24 (baseline) | 56/57 | 35/38 | ['T004'] |
| fusion@24 (candidate) | 55/57 | 36/38 | ['T047'] |

- High primary reachability: 15/15 → 14/15
- Critical primary reachability: 8/8 → 7/8
- Vector-only primary reachable cases: none
- Lexical-only primary reachable cases: ['T004']
- Both-channel primary reachable cases: ['T001', 'T002', 'T003', 'T005', 'T007', 'T008', 'T009', 'T010', 'T011', 'T012', 'T013', 'T014', 'T015', 'T016', 'T017', 'T018', 'T019', 'T020', 'T021', 'T022', 'T023', 'T024', 'T025', 'T026', 'T027', 'T028', 'T029', 'T030', 'T031', 'T032', 'T033', 'T034', 'T035', 'T036', 'T037', 'T038', 'T039', 'T040', 'T041', 'T042', 'T043', 'T044', 'T045', 'T046', 'T047', 'T048', 'T049', 'T050', 'T051', 'T052', 'T053', 'T054', 'T055', 'T057', 'T058', 'T059']
- Neither-channel primary reachable cases: none

## 3. Final ranking metrics

| Metric | Baseline | Candidate | Delta |
|--------|---------:|----------:|------:|
| Hit@1 | 0.517 | 0.586 | +0.069 |
| Hit@4 | 0.897 | 0.966 | +0.069 |
| Hit@12 | 0.983 | 0.983 | +0.000 |
| Document recall@4 | 0.667 | 0.710 | +0.043 |
| Primary hit@4 | 0.741 | 0.862 | +0.121 |
| Supporting hit@4 | 0.605 | 0.579 | -0.026 |
| MRR | 0.715 | 0.768 | +0.053 |

## 4. Promotions and regressions

- Primary hit@4 promotions: ['T009', 'T011', 'T016', 'T035', 'T040', 'T044', 'T054', 'T055', 'T059']
- Primary hit@4 regressions: ['T028', 'T057']

## 5. Focus case diagnostics (reporting only)

### T004

- **Risk / category:** low / general
- **Expected primary:** ['07_complaint_handling_procedure']
- **Baseline vector pool@24 primary reachable:** False (best rank None)
- **Candidate fusion pool@24 primary reachable:** True (best rank 12)
- **Lexical primary best rank (pool):** 1
- **Lexical primary best rank (candidate audit):** 1
- **Primary in final top-12 (candidate):** True
- **Candidate primary hit@4:** False
- **Vector reachable (primary):** False
- **Lexical reachable (primary):** True
- **Retrieval channel (primary):** lexical_only

### T040

- **Risk / category:** critical / quality
- **Expected primary:** ['06_food_quality_and_packaging']
- **Baseline vector pool@24 primary reachable:** True (best rank 13)
- **Candidate fusion pool@24 primary reachable:** True (best rank 2)
- **Lexical primary best rank (pool):** 1
- **Lexical primary best rank (candidate audit):** 1
- **Primary in final top-12 (candidate):** True
- **Candidate primary hit@4:** True
- **Vector reachable (primary):** True
- **Lexical reachable (primary):** True
- **Retrieval channel (primary):** both

### T047

- **Risk / category:** critical / quality
- **Expected primary:** ['08_escalation_and_risk_rules']
- **Baseline vector pool@24 primary reachable:** True (best rank 15)
- **Candidate fusion pool@24 primary reachable:** False (best rank None)
- **Lexical primary best rank (pool):** 17
- **Lexical primary best rank (candidate audit):** None
- **Primary in final top-12 (candidate):** False
- **Candidate primary hit@4:** False
- **Vector reachable (primary):** True
- **Lexical reachable (primary):** True
- **Retrieval channel (primary):** both

### T053

- **Risk / category:** high / procedure
- **Expected primary:** ['08_escalation_and_risk_rules']
- **Baseline vector pool@24 primary reachable:** True (best rank 19)
- **Candidate fusion pool@24 primary reachable:** False (best rank None)
- **Lexical primary best rank (pool):** 18
- **Lexical primary best rank (candidate audit):** None
- **Primary in final top-12 (candidate):** False
- **Candidate primary hit@4:** False
- **Vector reachable (primary):** True
- **Lexical reachable (primary):** True
- **Retrieval channel (primary):** both

### T057

- **Risk / category:** critical / injection
- **Expected primary:** ['06_food_quality_and_packaging']
- **Baseline vector pool@24 primary reachable:** True (best rank 1)
- **Candidate fusion pool@24 primary reachable:** True (best rank 6)
- **Lexical primary best rank (pool):** 9
- **Lexical primary best rank (candidate audit):** None
- **Primary in final top-12 (candidate):** True
- **Candidate primary hit@4:** False
- **Vector reachable (primary):** True
- **Lexical reachable (primary):** True
- **Retrieval channel (primary):** both


## 6. Acceptance verdicts

**Candidate-generation verdict:** **rejected**
**Final-ranking verdict:** **rejected**

- Hard invariants pass: **True**
- Single vector retrieval pass: **True**
- Reranker config hash match: **True**
- Lexical index valid: **True**
- High primary reachability non-regressed: **False**
- Critical primary reachability non-regressed: **False**
- Fully unreachable non-increased: **True**
- Low/medium guardrails pass: **True**
- Overall primary hit@4 delta: **+0.121**
- Overall MRR delta: **+0.053**
- Critical primary regressions: **1**

Failure reasons:
- high primary reachability regressed
- critical primary reachability regressed
- high/critical primary reachability regressions > 0
- critical primary_hit@4 regressions > 0
- primary_hit@4 regressions > 0

## 7. Retrieval configuration selection

**hybrid-lexical-vector-v1:** rejected for MVP selection

**Reason:**
- critical primary regression T057
- primary regressions T028/T057
- high/critical reachability regression
- supporting hit@4 decrease

**Selected retrieval configuration:** stage 2C.2 candidate

```text
vector top-24
→ source-authority-v1
→ final top-12
```

No weighted RRF or post-hoc fusion tuning was performed.
Retrieval experimentation is frozen after stage 2C.3.

## 8. Limitations

- Hybrid fusion improves candidate reachability but does not guarantee top-4 gains.
- Normalized RRF base scores are not comparable to vector similarity magnitudes.
- Frozen source-authority-v1 max bonus (+0.03) cannot overcome large base-score gaps.

---

*Generated by hybrid lexical + vector evaluation (stage 2C.3).*
