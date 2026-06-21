# Reranking A/B results: source-authority-v1

**Timestamp:** 2026-06-21T11:10:20.850135+00:00
**Reranker ID:** `source-authority-v1`
**Version:** `1.0.0`
**Experiment mode:** `production-like`
**Config hash:** `c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957`
**Git commit:** `70caa1a9ad17234b956b3e6ae6f51d997e2c5269`
**Working tree dirty:** True

## 1. Experiment contract

- Shared candidate pool from one baseline retrieval call per case
- `fetch_k=12`, `top_k=12`, `threshold=0.00`
- Embedding model: `text-embedding-3-small`
- Index fingerprint: `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3`
- Evaluation dataset fingerprint: `339ee42b744e3f17df5c963b01895147c29d64c55163582293332baf11a36861`
- Reranker inputs: query, baseline candidates, `chunk_type`, similarity metadata only
- No oracle risk/category labels and no expected-source features in reranker

## 2. Baseline identity

- Frozen baseline evaluation result ID: `2132c861d4d03b3b99fb5413ae69fd3f0b45f3326d7b71671b4ea3a2a8106e0a`
- Computed baseline evaluation result ID: `9755754ccbafdfe77dbdad567bfc957b0aee8960424233f1dd219cc4b5c92faf`
- Evaluation result ID match: **False**
- Frozen index fingerprint: `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3`
- Run index fingerprint: `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3`
- Case count: **60**
- Exact ordered match: **32**
- Same-order float drift: **24**
- Same-set different-order: **4**
- Substantive mismatches: **0**
- Max similarity drift: **0.003309** (tolerance 0.004)
- Max distance drift: **0.003309**
- Frozen metrics match: **True**
- Shared pool exact match: **True**
- **Status:** **equivalent_with_embedding_drift**

Frozen baseline metrics and candidate sets are equivalent.
Exact ordered identity was not reproduced in 4/60 cases due live embedding drift.
No substantive candidate-set mismatch was found.

## 3. Overall baseline vs candidate metrics

| Metric | Baseline | Candidate | Delta |
|--------|---------:|----------:|------:|
| Hit@1 | 0.414 | 0.517 | +0.103 |
| Hit@4 | 0.793 | 0.897 | +0.103 |
| Hit@12 | 0.948 | 0.948 | +0.000 |
| Document recall@1 | 0.230 | 0.279 | +0.049 |
| Document recall@4 | 0.555 | 0.667 | +0.112 |
| Document recall@12 | 0.802 | 0.802 | +0.000 |
| Primary hit@1 | 0.276 | 0.379 | +0.103 |
| Primary hit@4 | 0.621 | 0.741 | +0.121 |
| Supporting hit@4 | 0.526 (20/38) | 0.605 (23/38) | +0.079 |
| MRR | 0.637 | 0.706 | +0.069 |
| Technical errors | 0 | 0 | — |

## 4. Risk-level deltas

| Risk | Cases | Baseline Hit@4 | Candidate Hit@4 | Δ Hit@4 | Baseline Primary@4 | Candidate Primary@4 | Δ Primary@4 | Δ MRR |
|------|------:|---------------:|----------------:|--------:|-------------------:|--------------------:|------------:|------:|
| critical | 8 | 0.625 | 0.750 | +0.125 | 0.625 | 0.750 | +0.125 | +0.010 |
| high | 15 | 0.800 | 0.867 | +0.067 | 0.533 | 0.667 | +0.133 | +0.089 |
| low | 16 | 0.812 | 0.875 | +0.062 | 0.600 | 0.667 | +0.067 | +0.104 |
| medium | 19 | 0.842 | 1.000 | +0.158 | 0.737 | 0.895 | +0.158 | +0.048 |

## 5. Category-level deltas

| Category | Cases | Baseline Hit@4 | Candidate Hit@4 | Δ Hit@4 | Baseline Primary@4 | Candidate Primary@4 | Δ Primary@4 | Δ MRR |
|----------|------:|---------------:|----------------:|--------:|-------------------:|--------------------:|------------:|------:|
| delivery | 10 | 0.800 | 1.000 | +0.200 | 0.500 | 0.700 | +0.200 | +0.042 |
| general | 5 | 0.800 | 0.800 | +0.000 | 0.600 | 0.600 | +0.000 | +0.100 |
| injection | 4 | 0.750 | 1.000 | +0.250 | 0.667 | 0.667 | +0.000 | +0.167 |
| orders | 7 | 0.857 | 1.000 | +0.143 | 0.714 | 1.000 | +0.286 | +0.071 |
| procedure | 7 | 0.857 | 0.857 | +0.000 | 0.571 | 0.571 | +0.000 | +0.048 |
| quality | 15 | 0.667 | 0.733 | +0.067 | 0.600 | 0.733 | +0.133 | +0.052 |
| refunds | 10 | 0.900 | 1.000 | +0.100 | 0.800 | 0.900 | +0.100 | +0.080 |

## 6. Promotions (primary hit@4 gained)

- **T010** (medium): baseline primary@4=False → candidate=True
- **T015** (medium): baseline primary@4=False → candidate=True
- **T017** (low): baseline primary@4=False → candidate=True
- **T023** (medium): baseline primary@4=False → candidate=True
- **T027** (high): baseline primary@4=False → candidate=True
- **T039** (high): baseline primary@4=False → candidate=True
- **T046** (critical): baseline primary@4=False → candidate=True

## 7. Regressions (primary hit@4 lost)

_No primary hit@4 regressions._

## 8. Document promotion/demotion counts

- `01_service_overview`: promotions=2, demotions=7, net_rank_change=-6
- `02_delivery_rules`: promotions=9, demotions=0, net_rank_change=9
- `03_order_changes_and_cancellations`: promotions=11, demotions=0, net_rank_change=11
- `04_refund_policy`: promotions=12, demotions=0, net_rank_change=13
- `05_compensation_policy`: promotions=5, demotions=0, net_rank_change=7
- `06_food_quality_and_packaging`: promotions=8, demotions=0, net_rank_change=9
- `07_complaint_handling_procedure`: promotions=2, demotions=10, net_rank_change=-8
- `08_escalation_and_risk_rules`: promotions=9, demotions=0, net_rank_change=12
- `09_response_style_and_templates`: promotions=0, demotions=7, net_rank_change=-13
- `10_customer_faq`: promotions=0, demotions=23, net_rank_change=-34

## 9. Candidate-generation diagnostics

### Primary candidate-generation failures

- **T004**
- **T016**
- **T040**
- **T044**
- **T047**
- **T053**

### Supporting candidate-generation failures

- **T004**
- **T022**
- **T035**
- **T036**
- **T040**
- **T042**
- **T049**
- **T050**
- **T051**

### Reranker fully not applicable

- **T004**
- **T040**
- **T047**

## 10. FAQ vs policy movement

- FAQ chunks promoted: **0**
- FAQ chunks demoted: **178**
- Policy chunks promoted: **223**
- Policy chunks demoted: **0**
- Escalation chunks promoted: **19**
- Escalation chunks demoted: **0**

## 11. T040 analysis

- **Risk / category:** critical / quality
- **Expected primary:** ['06_food_quality_and_packaging']
- **Expected supporting:** ['08_escalation_and_risk_rules']
- **Shared candidate IDs:** 12 chunks
- **Baseline primary hit@4:** False
- **Candidate primary hit@4:** False
- **primary_candidate_generation_failure:** True
- **supporting_candidate_generation_failure:** True
- **reranker_not_applicable:** True
- Expected document(s) absent from shared top-12 pool; reranking cannot recover them.

## 12. T047 analysis

- **Risk / category:** critical / quality
- **Expected primary:** ['08_escalation_and_risk_rules']
- **Expected supporting:** []
- **Shared candidate IDs:** 12 chunks
- **Baseline primary hit@4:** False
- **Candidate primary hit@4:** False
- **primary_candidate_generation_failure:** True
- **supporting_candidate_generation_failure:** False
- **reranker_not_applicable:** True
- Expected document(s) absent from shared top-12 pool; reranking cannot recover them.

## 13. Acceptance criteria result

**Verdict:** **candidate accepted**

- Hard invariants pass: **True**
- Baseline identity pass: **True**
- Shared pool pass: **True**
- Technical errors zero: **True**
- High+critical net primary promotions: **3** (pass=True)
- Critical primary regressions: **0** (pass=True)
- Overall primary hit@4 delta: **+0.121** (pass=True)
- Overall MRR delta: **+0.069** (pass=True)
- Low Hit@4 regressions: **0** (pass=True)
- Low MRR delta: **+0.104** (pass=True)
- Medium Hit@4 regressions: **0** (pass=True)
- Medium MRR delta: **+0.048** (pass=True)
- Fallback status preserved: **True**

## 14. Limitations

- Reranking only reorders the existing top-12 candidate pool; it cannot recover documents absent from baseline retrieval.
- `source-authority-v1` uses chunk_type authority only; no lexical, risk, or category signals.
- Production-like mode does not use predicted request risk or category.
- Source bonus can change ordering only within the configured maximum bonus zone; candidates with similarity gaps greater than max_source_bonus (0.03) cannot be overtaken solely by this bonus.

## 15. Production threshold

No production similarity threshold was selected in this experiment.

---

*Generated by reranking A/B evaluation (stage 2C.1).*
