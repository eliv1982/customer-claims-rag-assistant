# Результаты baseline retrieval evaluation

**Evaluation result ID:** `2132c861d4d03b3b99fb5413ae69fd3f0b45f3326d7b71671b4ea3a2a8106e0a`
**Дата прогона:** 2026-06-21T09:16:27.956555+00:00
**Git commit:** `46eb01cf84ca4e32b5641dc3ceef0b3ebe9356fa`
**Working tree dirty:** False
**Git status summary:** clean
**Index fingerprint:** `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3`
**Embedding model:** `text-embedding-3-small`
**Collection:** `customer_claims`
**Vector dimension:** 1536
**Chunk count:** 215
**Document count:** 10
**Threshold:** 0.00 (filtering disabled for baseline recall)
**top_k / fetch_k:** 12 / 12

> **Ограничение:** это retrieval-only evaluation. Метрики не оценивают качество LLM-ответов, risk/handoff classification, answer factuality или Markdown output contract.

> **Output consistency:** каждый файл записывается атомарно, но набор из трёх файлов не является общей транзакцией. `evaluation_result_id` должен совпадать во всех артефактах.

## Aggregate metrics

| Metric | Value |
|--------|------:|
| Total cases | 60 |
| Successfully evaluated | 60 |
| Technical errors | 0 |
| Source-recall cases | 58 |
| Fallback cases | 2 |
| Hit@1 | 0.414 |
| Hit@4 | 0.793 |
| Hit@12 | 0.948 |
| Document recall@1 | 0.230 |
| Document recall@4 | 0.555 |
| Document recall@12 | 0.802 |
| MRR | 0.637 |
| Primary source hit@1 | 0.276 |
| Primary source hit@4 | 0.621 |
| Supporting source hit@4 | 0.526 (20/38) |
| No-result rate | 0.000 |
| Mean top-1 similarity | 0.5257 |
| Median top-1 similarity | 0.5415 |
| Min top-1 similarity | 0.3046 |
| Max top-1 similarity | 0.6928 |

## Metrics by risk

| Risk | Cases | Hit@1 | Hit@4 | Hit@12 | MRR | Mean top-1 sim |
|------|------:|------:|------:|-------:|----:|---------------:|
| low | 16 | 0.312 | 0.812 | 0.938 | 0.589 | 0.5612 |
| medium | 19 | 0.474 | 0.842 | 1.000 | 0.706 | 0.5637 |
| high | 15 | 0.400 | 0.800 | 1.000 | 0.624 | 0.5093 |
| critical | 8 | 0.500 | 0.625 | 0.750 | 0.594 | 0.4317 |

## Metrics by category

| Category | Cases | Hit@1 | Hit@4 | Hit@12 | MRR | Mean top-1 sim |
|----------|------:|------:|------:|-------:|----:|---------------:|
| delivery | 10 | 0.400 | 0.800 | 1.000 | 0.642 | 0.5642 |
| general | 5 | 0.200 | 0.800 | 0.800 | 0.467 | 0.5672 |
| injection | 4 | 0.250 | 0.750 | 1.000 | 0.583 | 0.5110 |
| orders | 7 | 0.714 | 0.857 | 1.000 | 0.857 | 0.5786 |
| procedure | 7 | 0.286 | 0.857 | 1.000 | 0.607 | 0.5022 |
| quality | 15 | 0.400 | 0.667 | 0.867 | 0.572 | 0.4800 |
| refunds | 10 | 0.500 | 0.900 | 1.000 | 0.703 | 0.5496 |

## Threshold sweep

| Threshold | Cases w/ result | No-result rate | Hit@1 | Hit@4 | Primary hit@4 | MRR | Critical hit@4 | High hit@4 |
|----------:|----------------:|---------------:|------:|------:|--------------:|----:|---------------:|-----------:|
| 0.00 | 58 | 0.000 | 0.414 | 0.793 | 0.621 | 0.637 | 0.625 | 0.800 |
| 0.20 | 58 | 0.000 | 0.414 | 0.793 | 0.621 | 0.637 | 0.625 | 0.800 |
| 0.25 | 58 | 0.000 | 0.414 | 0.793 | 0.621 | 0.637 | 0.625 | 0.800 |
| 0.30 | 58 | 0.000 | 0.414 | 0.793 | 0.621 | 0.637 | 0.625 | 0.800 |
| 0.35 | 57 | 0.017 | 0.414 | 0.793 | 0.621 | 0.630 | 0.625 | 0.800 |
| 0.40 | 55 | 0.052 | 0.414 | 0.759 | 0.603 | 0.606 | 0.500 | 0.733 |
| 0.45 | 45 | 0.224 | 0.362 | 0.672 | 0.517 | 0.526 | 0.125 | 0.667 |
| 0.50 | 37 | 0.362 | 0.293 | 0.569 | 0.431 | 0.434 | 0.000 | 0.533 |
| 0.55 | 25 | 0.569 | 0.190 | 0.276 | 0.207 | 0.241 | 0.000 | 0.133 |
| 0.60 | 13 | 0.776 | 0.086 | 0.086 | 0.017 | 0.086 | 0.000 | 0.000 |
| 0.65 | 4 | 0.931 | 0.017 | 0.017 | 0.000 | 0.017 | 0.000 | 0.000 |
| 0.70 | 0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Fallback / no-grounding separation

- Fallback cases (`fallback_expected=true`): **2**
- Mean top-1 similarity: 0.3807
- Min / max top-1 similarity: 0.3046 / 0.4568
- These cases are excluded from source-recall aggregates; similarity distribution is tracked separately and is **not** fallback accuracy.
- **T006**: top-1 sim=0.3046, top doc=['10_customer_faq']
- **T060**: top-1 sim=0.4568, top doc=['03_order_changes_and_cancellations']

## Failed cases (primary source miss@4)

- **T002** (low): expected primary=['01_service_overview']; top retrieved @4=['10_customer_faq', '02_delivery_rules']; top-1 sim=0.4942; primary source found in top-12 but outside top-4
- **T004** (low): expected primary=['07_complaint_handling_procedure']; top retrieved @4=['04_refund_policy', '10_customer_faq']; top-1 sim=0.4616
- **T009** (medium): expected primary=['02_delivery_rules']; top retrieved @4=['10_customer_faq', '03_order_changes_and_cancellations']; top-1 sim=0.6025; primary source found in top-12 but outside top-4
- **T010** (medium): expected primary=['02_delivery_rules']; top retrieved @4=['03_order_changes_and_cancellations', '10_customer_faq']; top-1 sim=0.5422; primary source found in top-12 but outside top-4
- **T011** (high): expected primary=['02_delivery_rules', '08_escalation_and_risk_rules']; top retrieved @4=['10_customer_faq']; top-1 sim=0.5595; primary source found in top-12 but outside top-4
- **T015** (medium): expected primary=['02_delivery_rules']; top retrieved @4=['07_complaint_handling_procedure', '10_customer_faq']; top-1 sim=0.5407; primary source found in top-12 but outside top-4
- **T016** (medium): expected primary=['02_delivery_rules']; top retrieved @4=['10_customer_faq', '06_food_quality_and_packaging']; top-1 sim=0.4916
- **T017** (low): expected primary=['03_order_changes_and_cancellations']; top retrieved @4=['10_customer_faq']; top-1 sim=0.4852; primary source found in top-12 but outside top-4
- **T023** (medium): expected primary=['03_order_changes_and_cancellations', '05_compensation_policy']; top retrieved @4=['10_customer_faq']; top-1 sim=0.6582; primary source found in top-12 but outside top-4
- **T027** (high): expected primary=['06_food_quality_and_packaging', '08_escalation_and_risk_rules']; top retrieved @4=['10_customer_faq', '04_refund_policy']; top-1 sim=0.5025; primary source found in top-12 but outside top-4
- **T029** (low): expected primary=['07_complaint_handling_procedure']; top retrieved @4=['10_customer_faq', '04_refund_policy']; top-1 sim=0.6658; primary source found in top-12 but outside top-4
- **T035** (high): expected primary=['06_food_quality_and_packaging']; top retrieved @4=['10_customer_faq']; top-1 sim=0.5499; primary source found in top-12 but outside top-4
- **T039** (high): expected primary=['06_food_quality_and_packaging']; top retrieved @4=['10_customer_faq', '07_complaint_handling_procedure', '08_escalation_and_risk_rules']; top-1 sim=0.4340; primary source found in top-12 but outside top-4
- **T040** (critical): expected primary=['06_food_quality_and_packaging']; top retrieved @4=['10_customer_faq']; top-1 sim=0.4282
- **T044** (high): expected primary=['08_escalation_and_risk_rules']; top retrieved @4=['10_customer_faq', '07_complaint_handling_procedure']; top-1 sim=0.3997
- **T046** (critical): expected primary=['08_escalation_and_risk_rules']; top retrieved @4=['07_complaint_handling_procedure', '10_customer_faq']; top-1 sim=0.4436; primary source found in top-12 but outside top-4
- **T047** (critical): expected primary=['08_escalation_and_risk_rules']; top retrieved @4=['10_customer_faq', '02_delivery_rules']; top-1 sim=0.3946
- **T053** (high): expected primary=['08_escalation_and_risk_rules']; top retrieved @4=['02_delivery_rules', '07_complaint_handling_procedure', '05_compensation_policy']; top-1 sim=0.5175
- **T054** (low): expected primary=['07_complaint_handling_procedure']; top retrieved @4=['05_compensation_policy', '06_food_quality_and_packaging', '09_response_style_and_templates']; top-1 sim=0.3275; primary source found in top-12 but outside top-4
- **T055** (high): expected primary=['06_food_quality_and_packaging']; top retrieved @4=['10_customer_faq', '07_complaint_handling_procedure', '02_delivery_rules', '04_refund_policy']; top-1 sim=0.5601; primary source found in top-12 but outside top-4
- ... and 1 more

## Short interpretation

- Baseline hit@4 across source-recall cases: **79.3%**; primary source hit@4: **62.1%**.
- Supporting source hit@4: **0.526 (20/38)**.
- All @k metrics are computed from the first k raw chunks only.
- Similarity scores are cosine-derived distances, not probabilities.
- Threshold filtering was disabled (`threshold=0.0`) to measure raw recall.
- Critical cases: hit@4 **62.5%**, MRR **0.594**.
- High cases: hit@4 **80.0%**, MRR **0.624**.
- This run does **not** establish production readiness; review critical/high primary misses before any threshold or reranking decision.

---

*Generated by baseline retrieval evaluation (stage 2B).*
