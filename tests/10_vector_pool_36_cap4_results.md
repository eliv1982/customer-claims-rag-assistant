# Vector pool cap A/B: vector-pool-36-cap4-v1

**Timestamp:** 2026-06-25T18:57:51.327371+00:00
**Experiment ID:** `vector-pool-36-cap4-v1`
**Version:** `1.0.0`
**Git commit:** `b69efee4973e71e1f037a2c3b666328f5d54fe89`
**Config hash:** `dddf35dd503598a3987e24545c83b0964604638b027922fe6c300529d34fc083`
**Verdict:** `ACCEPTED AS PARTIAL CANDIDATE-GENERATION REPAIR`

## 1. Executive summary

Controlled candidate-generation experiment comparing production retrieval (`fetch_k=24`, pool@24, no cap) against deep retrieval with per-document cap (`fetch_k=48`, cap@4, pool@36) before frozen `source-authority-v1` reranking.

- Baseline primary hit@4: **0.621**
- Candidate primary hit@4: **0.621**
- Baseline MRR: **0.643** | Candidate MRR: **0.646**
- Primary pool reach: **51/57** -> **55/57**

## 2. Repository and artifact identity

- Index fingerprint: `b9526128dad23e71e812fbd8f26452b8d6efa29e4faac898fa11f279b310e827`
- Corpus: **15** documents / **333** chunks
- Benchmark fingerprint: `339ee42b744e3f17df5c963b01895147c29d64c55163582293332baf11a36861`
- Reranker: `source-authority-v1` (hash `c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957`)

## 3. Exact baseline/candidate configurations

### Baseline arm

- `fetch_k = 24`
- `candidate_pool_k = 24`
- `per_document_cap = none`
- `reranker = source-authority-v1`
- `final_top_k = 12`
- `threshold = 0.0`

### Candidate arm

- `fetch_k = 48`
- `per_document_cap = 4`
- `candidate_pool_k = 36`
- `reranker = source-authority-v1`
- `final_top_k = 12`
- `threshold = 0.0`

## 4. Exact cap algorithm

1. Retrieve vector results at `fetch_k=48` in similarity order.
2. Iterate results in original vector order.
3. Keep at most four chunks per `document_id`.
4. Stop at 36 pooled results or after all 48 retrieved results.
5. No backfill beyond `fetch@48`.
6. Reassign pool ranks from 1 without mutating input `SearchResult` objects.
7. Preserve original vector rank in diagnostics.
8. Pass shaped pool to `SourceAuthorityV1Reranker`; final results are top-12.

## 5. Aggregate A/B table (final top-12)

| Metric | Baseline | Candidate | Delta |
|--------|--------:|----------:|------:|
| Hit@1 | 0.483 | 0.483 | +0.000 |
| Hit@4 | 0.759 | 0.759 | +0.000 |
| Hit@12 | 0.897 | 0.914 | +0.017 |
| Primary hit@4 | 0.621 | 0.621 | +0.000 |
| Primary hit@12 | 0.897 | 0.914 | +0.017 |
| MRR | 0.643 | 0.646 | +0.003 |
| Document recall@4 | 0.537 | 0.537 | +0.000 |

### Stage 4C.2 baseline reproduction (baseline arm)

- Target primary hit@4: 0.621 (actual 0.621)
- Target primary hit@12: 0.897 (actual 0.897)
- Target MRR: 0.643 (actual 0.643)
- Target primary reach: 51/57 (actual 51/57)
- Target FAQ top-4 count: 36 (actual 36)

## 6. Reachability table

- Primary reachable: 51/57 -> 55/57
- High-risk primary reachable: 14/15 -> 14/15
- Critical primary reachable: 5/8 -> 7/8
- Baseline fully unreachable: ['T004', 'T040', 'T044', 'T046', 'T047']
- Candidate fully unreachable: ['T047']

## 7. Saturation/diversity table

| Metric | Baseline | Candidate |
|--------|--------:|----------:|
| Avg unique documents in pool | 7.40 | 10.45 |
| Avg max chunks from one document | 8.32 | 4.00 |
| Questions with doc count >= cap | 0 | 60 |
| Questions where cap removed chunk | 0 | 60 |
| Total removed by cap | 0 | 1201 |
| Avg pool size | 24.00 | 27.95 |
| Pools shorter than target | 0 | 58 |

## 8. FAQ comparison

- Questions with FAQ in final top-4: baseline **36** -> candidate **36**
- FAQ in top-4 without expected primary: baseline **10** -> candidate **10**
- Case IDs with changed FAQ top-4 presence: []

## 9. Documents 11–15 footprint (candidate arm)

| Document | Pools | Final top-12 |
|----------|------:|-------------:|
| `11_payment_security_and_dispute_handling` | 28 | 18 |
| `12_staff_safety_and_threat_handling` | 27 | 12 |
| `13_physical_hazard_and_foreign_body_protocol` | 34 | 22 |
| `14_evidence_standards_and_incomplete_information` | 40 | 19 |
| `15_conflicting_rules_and_remedy_priority` | 50 | 31 |

## 10. Required per-case analysis

### T004

- Expected primary: ['07_complaint_handling_procedure']
- Baseline vector ranks: {'11_payment_security_and_dispute_handling': [1, 2, 4, 7, 13, 15, 23], '14_evidence_standards_and_incomplete_information': [3, 12], '04_refund_policy': [5, 6, 9, 10, 22], '10_customer_faq': [8, 11, 14, 16, 17, 19, 20], '03_order_changes_and_cancellations': [18, 21], '09_response_style_and_templates': [24]}
- Candidate vector ranks: {'11_payment_security_and_dispute_handling': [1, 2, 4, 7, 13, 15, 23, 26], '14_evidence_standards_and_incomplete_information': [3, 12, 25, 45], '04_refund_policy': [5, 6, 9, 10, 22, 30, 32, 34, 36, 38, 41, 43], '10_customer_faq': [8, 11, 14, 16, 17, 19, 20, 31, 35, 40, 42, 48], '03_order_changes_and_cancellations': [18, 21, 29, 37], '09_response_style_and_templates': [24, 47], '02_delivery_rules': [27], '15_conflicting_rules_and_remedy_priority': [28, 39, 46], '06_food_quality_and_packaging': [33], '07_complaint_handling_procedure': [44]}
- Baseline pool ranks: {'11_payment_security_and_dispute_handling': [1, 2, 4, 7, 13, 15, 23], '14_evidence_standards_and_incomplete_information': [3, 12], '04_refund_policy': [5, 6, 9, 10, 22], '10_customer_faq': [8, 11, 14, 16, 17, 19, 20], '03_order_changes_and_cancellations': [18, 21], '09_response_style_and_templates': [24]}
- Candidate pool ranks: {'11_payment_security_and_dispute_handling': [1, 2, 4, 7], '14_evidence_standards_and_incomplete_information': [3, 12, 18, 26], '04_refund_policy': [5, 6, 9, 10], '10_customer_faq': [8, 11, 13, 14], '03_order_changes_and_cancellations': [15, 16, 21, 23], '09_response_style_and_templates': [17, 28], '02_delivery_rules': [19], '15_conflicting_rules_and_remedy_priority': [20, 24, 27], '06_food_quality_and_packaging': [22], '07_complaint_handling_procedure': [25]}
- Candidate pool document counts: {'11_payment_security_and_dispute_handling': 4, '14_evidence_standards_and_incomplete_information': 4, '04_refund_policy': 4, '10_customer_faq': 4, '03_order_changes_and_cancellations': 4, '09_response_style_and_templates': 2, '02_delivery_rules': 1, '15_conflicting_rules_and_remedy_priority': 3, '06_food_quality_and_packaging': 1, '07_complaint_handling_procedure': 1}
- Baseline final ranks: {'11_payment_security_and_dispute_handling': [1, 2, 4, 7, 10, 11], '14_evidence_standards_and_incomplete_information': [3, 12], '04_refund_policy': [5, 6, 8, 9]}
- Candidate final ranks: {'11_payment_security_and_dispute_handling': [1, 2, 4, 7], '14_evidence_standards_and_incomplete_information': [3, 10], '04_refund_policy': [5, 6, 8, 9], '03_order_changes_and_cancellations': [11, 12]}
- Primary hit@4: False -> False
- Primary hit@12: False -> False
- Primary reachable: False -> True
- Change: primary reachability False -> True; cap removed 20 chunks

### T016

- Expected primary: ['02_delivery_rules']
- Baseline vector ranks: {'10_customer_faq': [1, 4, 6, 8, 10, 14, 17, 19, 22], '13_physical_hazard_and_foreign_body_protocol': [2, 3, 7, 15, 16, 23], '06_food_quality_and_packaging': [5, 11, 20], '07_complaint_handling_procedure': [9, 24], '04_refund_policy': [12], '14_evidence_standards_and_incomplete_information': [13, 21], '09_response_style_and_templates': [18]}
- Candidate vector ranks: {'10_customer_faq': [1, 4, 6, 8, 10, 14, 17, 19, 22, 28, 35, 42], '13_physical_hazard_and_foreign_body_protocol': [2, 3, 7, 15, 16, 23, 26, 34, 37, 41, 48], '06_food_quality_and_packaging': [5, 11, 20, 30, 44, 45, 47], '07_complaint_handling_procedure': [9, 24, 25], '04_refund_policy': [12, 29, 31, 33, 46], '14_evidence_standards_and_incomplete_information': [13, 21, 36, 39], '09_response_style_and_templates': [18], '02_delivery_rules': [27, 32], '15_conflicting_rules_and_remedy_priority': [38, 40], '08_escalation_and_risk_rules': [43]}
- Baseline pool ranks: {'10_customer_faq': [1, 4, 6, 8, 10, 14, 17, 19, 22], '13_physical_hazard_and_foreign_body_protocol': [2, 3, 7, 15, 16, 23], '06_food_quality_and_packaging': [5, 11, 20], '07_complaint_handling_procedure': [9, 24], '04_refund_policy': [12], '14_evidence_standards_and_incomplete_information': [13, 21], '09_response_style_and_templates': [18]}
- Candidate pool ranks: {'10_customer_faq': [1, 4, 6, 8], '13_physical_hazard_and_foreign_body_protocol': [2, 3, 7, 13], '06_food_quality_and_packaging': [5, 10, 15, 21], '07_complaint_handling_procedure': [9, 17, 18], '04_refund_policy': [11, 20, 22, 24], '14_evidence_standards_and_incomplete_information': [12, 16, 25, 27], '09_response_style_and_templates': [14], '02_delivery_rules': [19, 23], '15_conflicting_rules_and_remedy_priority': [26, 28], '08_escalation_and_risk_rules': [29]}
- Candidate pool document counts: {'10_customer_faq': 4, '13_physical_hazard_and_foreign_body_protocol': 4, '06_food_quality_and_packaging': 4, '07_complaint_handling_procedure': 3, '04_refund_policy': 4, '14_evidence_standards_and_incomplete_information': 4, '09_response_style_and_templates': 1, '02_delivery_rules': 2, '15_conflicting_rules_and_remedy_priority': 2, '08_escalation_and_risk_rules': 1}
- Baseline final ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 5, 8, 9], '10_customer_faq': [3], '06_food_quality_and_packaging': [4, 6, 11], '04_refund_policy': [7], '07_complaint_handling_procedure': [10], '14_evidence_standards_and_incomplete_information': [12]}
- Candidate final ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 5, 8], '10_customer_faq': [3, 12], '06_food_quality_and_packaging': [4, 6, 10], '04_refund_policy': [7], '07_complaint_handling_procedure': [9], '14_evidence_standards_and_incomplete_information': [11]}
- Primary hit@4: False -> False
- Primary hit@12: True -> True
- Primary reachable: False -> True
- Change: primary reachability False -> True; cap removed 19 chunks

### T039

- Expected primary: ['06_food_quality_and_packaging']
- Baseline vector ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 5, 6, 7, 8, 9, 11, 15, 16, 18, 20, 21, 24], '10_customer_faq': [3, 17], '14_evidence_standards_and_incomplete_information': [4, 12, 14, 19, 23], '07_complaint_handling_procedure': [10], '08_escalation_and_risk_rules': [13], '06_food_quality_and_packaging': [22]}
- Candidate vector ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 5, 6, 7, 8, 9, 11, 15, 16, 18, 20, 21, 24, 31, 32, 36, 42, 44], '10_customer_faq': [3, 17, 33, 37, 47, 48], '14_evidence_standards_and_incomplete_information': [4, 12, 14, 19, 23, 25, 26, 27, 35, 38, 39, 40, 43, 45, 46], '07_complaint_handling_procedure': [10], '08_escalation_and_risk_rules': [13], '06_food_quality_and_packaging': [22, 29, 30, 34, 41], '15_conflicting_rules_and_remedy_priority': [28]}
- Baseline pool ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 5, 6, 7, 8, 9, 11, 15, 16, 18, 20, 21, 24], '10_customer_faq': [3, 17], '14_evidence_standards_and_incomplete_information': [4, 12, 14, 19, 23], '07_complaint_handling_procedure': [10], '08_escalation_and_risk_rules': [13], '06_food_quality_and_packaging': [22]}
- Candidate pool ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 5, 6], '10_customer_faq': [3, 11, 17, 19], '14_evidence_standards_and_incomplete_information': [4, 8, 10, 12], '07_complaint_handling_procedure': [7], '08_escalation_and_risk_rules': [9], '06_food_quality_and_packaging': [13, 15, 16, 18], '15_conflicting_rules_and_remedy_priority': [14]}
- Candidate pool document counts: {'13_physical_hazard_and_foreign_body_protocol': 4, '10_customer_faq': 4, '14_evidence_standards_and_incomplete_information': 4, '07_complaint_handling_procedure': 1, '08_escalation_and_risk_rules': 1, '06_food_quality_and_packaging': 4, '15_conflicting_rules_and_remedy_priority': 1}
- Baseline final ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 3, 6, 7, 8, 9, 10], '14_evidence_standards_and_incomplete_information': [4], '10_customer_faq': [5], '08_escalation_and_risk_rules': [11], '07_complaint_handling_procedure': [12]}
- Candidate final ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 3, 6], '14_evidence_standards_and_incomplete_information': [4, 9, 10, 12], '10_customer_faq': [5], '08_escalation_and_risk_rules': [7], '07_complaint_handling_procedure': [8], '06_food_quality_and_packaging': [11]}
- Primary hit@4: False -> False
- Primary hit@12: True -> True
- Primary reachable: True -> True
- Change: cap removed 29 chunks

### T040

- Expected primary: ['06_food_quality_and_packaging']
- Baseline vector ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 4, 5, 6, 7, 9, 10, 13, 17, 19, 20, 21], '10_customer_faq': [3, 8, 11, 12, 16, 22, 24], '14_evidence_standards_and_incomplete_information': [14, 15, 23], '15_conflicting_rules_and_remedy_priority': [18]}
- Candidate vector ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 4, 5, 6, 7, 9, 10, 13, 17, 19, 20, 21, 29, 30, 35, 38, 39], '10_customer_faq': [3, 8, 11, 12, 16, 22, 24, 28, 31, 32, 37, 47], '14_evidence_standards_and_incomplete_information': [14, 15, 23, 25, 34, 36, 46], '15_conflicting_rules_and_remedy_priority': [18, 40, 45], '07_complaint_handling_procedure': [26], '04_refund_policy': [27, 43], '06_food_quality_and_packaging': [33, 41, 42, 44, 48]}
- Baseline pool ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 4, 5, 6, 7, 9, 10, 13, 17, 19, 20, 21], '10_customer_faq': [3, 8, 11, 12, 16, 22, 24], '14_evidence_standards_and_incomplete_information': [14, 15, 23], '15_conflicting_rules_and_remedy_priority': [18]}
- Candidate pool ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 4, 5], '10_customer_faq': [3, 6, 7, 8], '14_evidence_standards_and_incomplete_information': [9, 10, 12, 13], '15_conflicting_rules_and_remedy_priority': [11, 17, 22], '07_complaint_handling_procedure': [14], '04_refund_policy': [15, 20], '06_food_quality_and_packaging': [16, 18, 19, 21]}
- Candidate pool document counts: {'13_physical_hazard_and_foreign_body_protocol': 4, '10_customer_faq': 4, '14_evidence_standards_and_incomplete_information': 4, '15_conflicting_rules_and_remedy_priority': 3, '07_complaint_handling_procedure': 1, '04_refund_policy': 2, '06_food_quality_and_packaging': 4}
- Baseline final ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 3, 4, 6, 7, 8, 9, 10], '10_customer_faq': [5, 11], '14_evidence_standards_and_incomplete_information': [12]}
- Candidate final ranks: {'13_physical_hazard_and_foreign_body_protocol': [1, 2, 3, 4], '10_customer_faq': [5, 6, 8, 11], '14_evidence_standards_and_incomplete_information': [7, 9, 12], '15_conflicting_rules_and_remedy_priority': [10]}
- Primary hit@4: False -> False
- Primary hit@12: False -> False
- Primary reachable: False -> True
- Change: primary reachability False -> True; cap removed 26 chunks

### T044

- Expected primary: ['08_escalation_and_risk_rules']
- Baseline vector ranks: {'12_staff_safety_and_threat_handling': [1, 3, 12, 21, 24], '14_evidence_standards_and_incomplete_information': [2, 4, 7, 8, 9, 11, 16, 19, 20], '13_physical_hazard_and_foreign_body_protocol': [5, 13, 22], '10_customer_faq': [6, 14], '11_payment_security_and_dispute_handling': [10, 17, 18], '07_complaint_handling_procedure': [15], '15_conflicting_rules_and_remedy_priority': [23]}
- Candidate vector ranks: {'12_staff_safety_and_threat_handling': [1, 3, 12, 21, 24, 34], '14_evidence_standards_and_incomplete_information': [2, 4, 7, 8, 9, 11, 16, 19, 20, 28, 29, 31, 44], '13_physical_hazard_and_foreign_body_protocol': [5, 13, 22, 35, 37, 42, 43, 46, 47], '10_customer_faq': [6, 14, 25, 26, 32, 40, 41, 48], '11_payment_security_and_dispute_handling': [10, 17, 18, 33, 45], '07_complaint_handling_procedure': [15], '15_conflicting_rules_and_remedy_priority': [23], '02_delivery_rules': [27, 36], '03_order_changes_and_cancellations': [30], '09_response_style_and_templates': [38], '06_food_quality_and_packaging': [39]}
- Baseline pool ranks: {'12_staff_safety_and_threat_handling': [1, 3, 12, 21, 24], '14_evidence_standards_and_incomplete_information': [2, 4, 7, 8, 9, 11, 16, 19, 20], '13_physical_hazard_and_foreign_body_protocol': [5, 13, 22], '10_customer_faq': [6, 14], '11_payment_security_and_dispute_handling': [10, 17, 18], '07_complaint_handling_procedure': [15], '15_conflicting_rules_and_remedy_priority': [23]}
- Candidate pool ranks: {'12_staff_safety_and_threat_handling': [1, 3, 10, 16], '14_evidence_standards_and_incomplete_information': [2, 4, 7, 8], '13_physical_hazard_and_foreign_body_protocol': [5, 11, 17, 24], '10_customer_faq': [6, 12, 19, 20], '11_payment_security_and_dispute_handling': [9, 14, 15, 23], '07_complaint_handling_procedure': [13], '15_conflicting_rules_and_remedy_priority': [18], '02_delivery_rules': [21, 25], '03_order_changes_and_cancellations': [22], '09_response_style_and_templates': [26], '06_food_quality_and_packaging': [27]}
- Candidate pool document counts: {'12_staff_safety_and_threat_handling': 4, '14_evidence_standards_and_incomplete_information': 4, '13_physical_hazard_and_foreign_body_protocol': 4, '10_customer_faq': 4, '11_payment_security_and_dispute_handling': 4, '07_complaint_handling_procedure': 1, '15_conflicting_rules_and_remedy_priority': 1, '02_delivery_rules': 2, '03_order_changes_and_cancellations': 1, '09_response_style_and_templates': 1, '06_food_quality_and_packaging': 1}
- Baseline final ranks: {'12_staff_safety_and_threat_handling': [1, 3, 8], '14_evidence_standards_and_incomplete_information': [2, 4, 6, 9, 11, 12], '13_physical_hazard_and_foreign_body_protocol': [5, 10], '11_payment_security_and_dispute_handling': [7]}
- Candidate final ranks: {'12_staff_safety_and_threat_handling': [1, 3, 8], '14_evidence_standards_and_incomplete_information': [2, 4, 6, 9], '13_physical_hazard_and_foreign_body_protocol': [5, 10], '11_payment_security_and_dispute_handling': [7, 11, 12]}
- Primary hit@4: False -> False
- Primary hit@12: False -> False
- Primary reachable: False -> False
- Change: cap removed 21 chunks

### T046

- Expected primary: ['08_escalation_and_risk_rules']
- Baseline vector ranks: {'11_payment_security_and_dispute_handling': [1, 4, 5, 9, 10, 14, 19, 22], '14_evidence_standards_and_incomplete_information': [2, 3, 6, 7, 8, 11, 12, 13, 15, 16], '07_complaint_handling_procedure': [17, 21], '10_customer_faq': [18], '12_staff_safety_and_threat_handling': [20], '13_physical_hazard_and_foreign_body_protocol': [23, 24]}
- Candidate vector ranks: {'11_payment_security_and_dispute_handling': [1, 4, 5, 8, 10, 13, 18, 20, 22, 26, 34, 37], '14_evidence_standards_and_incomplete_information': [2, 3, 6, 7, 9, 11, 12, 14, 15, 16, 29, 30, 32, 38, 43, 48], '07_complaint_handling_procedure': [17, 23, 27, 45, 47], '10_customer_faq': [19, 39, 40], '12_staff_safety_and_threat_handling': [21, 31], '13_physical_hazard_and_foreign_body_protocol': [24, 25], '05_compensation_policy': [28], '08_escalation_and_risk_rules': [33, 46], '02_delivery_rules': [35, 42], '01_service_overview': [36, 41], '15_conflicting_rules_and_remedy_priority': [44]}
- Baseline pool ranks: {'11_payment_security_and_dispute_handling': [1, 4, 5, 9, 10, 14, 19, 22], '14_evidence_standards_and_incomplete_information': [2, 3, 6, 7, 8, 11, 12, 13, 15, 16], '07_complaint_handling_procedure': [17, 21], '10_customer_faq': [18], '12_staff_safety_and_threat_handling': [20], '13_physical_hazard_and_foreign_body_protocol': [23, 24]}
- Candidate pool ranks: {'11_payment_security_and_dispute_handling': [1, 4, 5, 8], '14_evidence_standards_and_incomplete_information': [2, 3, 6, 7], '07_complaint_handling_procedure': [9, 12, 15, 26], '10_customer_faq': [10, 21, 22], '12_staff_safety_and_threat_handling': [11, 17], '13_physical_hazard_and_foreign_body_protocol': [13, 14], '05_compensation_policy': [16], '08_escalation_and_risk_rules': [18, 27], '02_delivery_rules': [19, 24], '01_service_overview': [20, 23], '15_conflicting_rules_and_remedy_priority': [25]}
- Candidate pool document counts: {'11_payment_security_and_dispute_handling': 4, '14_evidence_standards_and_incomplete_information': 4, '07_complaint_handling_procedure': 4, '10_customer_faq': 3, '12_staff_safety_and_threat_handling': 2, '13_physical_hazard_and_foreign_body_protocol': 2, '05_compensation_policy': 1, '08_escalation_and_risk_rules': 2, '02_delivery_rules': 2, '01_service_overview': 2, '15_conflicting_rules_and_remedy_priority': 1}
- Baseline final ranks: {'11_payment_security_and_dispute_handling': [1, 3, 4, 8, 9], '14_evidence_standards_and_incomplete_information': [2, 5, 6, 7, 10, 11, 12]}
- Candidate final ranks: {'11_payment_security_and_dispute_handling': [1, 3, 4, 8], '14_evidence_standards_and_incomplete_information': [2, 5, 6, 7], '07_complaint_handling_procedure': [9], '12_staff_safety_and_threat_handling': [10], '13_physical_hazard_and_foreign_body_protocol': [11, 12]}
- Primary hit@4: False -> False
- Primary hit@12: False -> False
- Primary reachable: False -> True
- Change: primary reachability False -> True; cap removed 21 chunks

### T047

- Expected primary: ['08_escalation_and_risk_rules']
- Baseline vector ranks: {'12_staff_safety_and_threat_handling': [1, 2, 3, 7, 8, 11, 13, 17, 20, 22, 24], '13_physical_hazard_and_foreign_body_protocol': [4, 6, 9, 14, 15, 16, 18], '14_evidence_standards_and_incomplete_information': [5, 10, 23], '10_customer_faq': [12, 19], '02_delivery_rules': [21]}
- Candidate vector ranks: {'12_staff_safety_and_threat_handling': [1, 2, 3, 7, 8, 11, 13, 17, 20, 22, 24, 30, 36, 38, 41, 46, 48], '13_physical_hazard_and_foreign_body_protocol': [4, 6, 9, 14, 15, 16, 18, 25, 31, 32, 42], '14_evidence_standards_and_incomplete_information': [5, 10, 23, 26, 28, 29, 33, 35, 37, 39, 45, 47], '10_customer_faq': [12, 19, 27, 34, 40, 44], '02_delivery_rules': [21], '11_payment_security_and_dispute_handling': [43]}
- Baseline pool ranks: {'12_staff_safety_and_threat_handling': [1, 2, 3, 7, 8, 11, 13, 17, 20, 22, 24], '13_physical_hazard_and_foreign_body_protocol': [4, 6, 9, 14, 15, 16, 18], '14_evidence_standards_and_incomplete_information': [5, 10, 23], '10_customer_faq': [12, 19], '02_delivery_rules': [21]}
- Candidate pool ranks: {'12_staff_safety_and_threat_handling': [1, 2, 3, 7], '13_physical_hazard_and_foreign_body_protocol': [4, 6, 8, 11], '14_evidence_standards_and_incomplete_information': [5, 9, 14, 15], '10_customer_faq': [10, 12, 16, 17], '02_delivery_rules': [13], '11_payment_security_and_dispute_handling': [18]}
- Candidate pool document counts: {'12_staff_safety_and_threat_handling': 4, '13_physical_hazard_and_foreign_body_protocol': 4, '14_evidence_standards_and_incomplete_information': 4, '10_customer_faq': 4, '02_delivery_rules': 1, '11_payment_security_and_dispute_handling': 1}
- Baseline final ranks: {'12_staff_safety_and_threat_handling': [1, 2, 3, 6, 7, 10, 11], '13_physical_hazard_and_foreign_body_protocol': [4, 5, 9, 12], '14_evidence_standards_and_incomplete_information': [8]}
- Candidate final ranks: {'12_staff_safety_and_threat_handling': [1, 2, 3, 6], '13_physical_hazard_and_foreign_body_protocol': [4, 5, 8, 9], '14_evidence_standards_and_incomplete_information': [7, 10, 12], '02_delivery_rules': [11]}
- Primary hit@4: False -> False
- Primary hit@12: False -> False
- Primary reachable: False -> False
- Change: cap removed 30 chunks

### T051

- Expected primary: ['07_complaint_handling_procedure']
- Baseline vector ranks: {'15_conflicting_rules_and_remedy_priority': [1, 3, 8, 11, 12], '10_customer_faq': [2, 9, 10, 14, 21, 24], '07_complaint_handling_procedure': [4, 7], '11_payment_security_and_dispute_handling': [5, 13, 15, 17, 20, 22], '04_refund_policy': [6, 16, 18, 19, 23]}
- Candidate vector ranks: {'15_conflicting_rules_and_remedy_priority': [1, 3, 8, 11, 12, 33, 36, 39, 43, 45, 48], '10_customer_faq': [2, 9, 10, 14, 21, 24, 28, 29, 38, 42, 44, 46], '07_complaint_handling_procedure': [4, 7], '11_payment_security_and_dispute_handling': [5, 13, 15, 17, 20, 22, 27, 30], '04_refund_policy': [6, 16, 18, 19, 23, 25, 31, 32, 34, 37], '12_staff_safety_and_threat_handling': [26], '08_escalation_and_risk_rules': [35], '05_compensation_policy': [40], '14_evidence_standards_and_incomplete_information': [41], '09_response_style_and_templates': [47]}
- Baseline pool ranks: {'15_conflicting_rules_and_remedy_priority': [1, 3, 8, 11, 12], '10_customer_faq': [2, 9, 10, 14, 21, 24], '07_complaint_handling_procedure': [4, 7], '11_payment_security_and_dispute_handling': [5, 13, 15, 17, 20, 22], '04_refund_policy': [6, 16, 18, 19, 23]}
- Candidate pool ranks: {'15_conflicting_rules_and_remedy_priority': [1, 3, 8, 11], '10_customer_faq': [2, 9, 10, 13], '07_complaint_handling_procedure': [4, 7], '11_payment_security_and_dispute_handling': [5, 12, 14, 16], '04_refund_policy': [6, 15, 17, 18], '12_staff_safety_and_threat_handling': [19], '08_escalation_and_risk_rules': [20], '05_compensation_policy': [21], '14_evidence_standards_and_incomplete_information': [22], '09_response_style_and_templates': [23]}
- Candidate pool document counts: {'15_conflicting_rules_and_remedy_priority': 4, '10_customer_faq': 4, '07_complaint_handling_procedure': 2, '11_payment_security_and_dispute_handling': 4, '04_refund_policy': 4, '12_staff_safety_and_threat_handling': 1, '08_escalation_and_risk_rules': 1, '05_compensation_policy': 1, '14_evidence_standards_and_incomplete_information': 1, '09_response_style_and_templates': 1}
- Baseline final ranks: {'15_conflicting_rules_and_remedy_priority': [1, 2, 5, 8, 9], '11_payment_security_and_dispute_handling': [3, 10, 11], '04_refund_policy': [4, 12], '07_complaint_handling_procedure': [6, 7]}
- Candidate final ranks: {'15_conflicting_rules_and_remedy_priority': [1, 2, 5, 8], '11_payment_security_and_dispute_handling': [3, 9, 10, 12], '04_refund_policy': [4, 11], '07_complaint_handling_procedure': [6, 7]}
- Primary hit@4: False -> False
- Primary hit@12: True -> True
- Primary reachable: True -> True
- Change: cap removed 25 chunks

## 11. Acceptance checklist

- [PASS] `primary_hit_at_4`: Primary hit@4 >= baseline raw value (baseline=0.6206896551724138, candidate=0.6206896551724138)
- [PASS] `mrr`: MRR >= 0.645 (baseline=None, candidate=0.6456896551724138)
- [PASS] `primary_hit_at_12`: Primary hit@12 >= 0.910 (baseline=None, candidate=0.9137931034482759)
- [PASS] `primary_pool_reach`: Primary pool reach >= 55/57 (baseline=None, candidate=55/57)
- [PASS] `high_risk_pool_reach`: High-risk pool reach >= 14/15 (baseline=None, candidate=14/15)
- [PASS] `critical_pool_reach`: Critical pool reach >= 7/8 (baseline=None, candidate=7/8)
- [PASS] `faq_top4_count`: FAQ top-4 count <= 36 (baseline=36, candidate=36)
- [PASS] `unreachable_subset`: Unreachable primaries subset of {T044, T047} (baseline=None, candidate=['T047'])
- [PASS] `reachability_T004`: T004 must become reachable (baseline=None, candidate=True)
- [PASS] `reachability_T016`: T016 must become reachable (baseline=None, candidate=True)
- [PASS] `reachability_T040`: T040 must become reachable (baseline=None, candidate=True)
- [PASS] `reachability_T046`: T046 must become reachable (baseline=None, candidate=True)
- [PASS] `no_reachability_regression`: No previously reachable primary becomes unreachable
- [PASS] `new_doc_preserved_T040`: Cap must not fully remove dedicated new document from T040 (baseline=None, candidate=13_physical_hazard_and_foreign_body_protocol)
- [PASS] `new_doc_preserved_T047`: Cap must not fully remove dedicated new document from T047 (baseline=None, candidate=12_staff_safety_and_threat_handling)
- [PASS] `technical_errors_zero`: No technical retrieval errors
- [PASS] `reranker_hash`: Frozen reranker config hash matches (baseline=None, candidate=c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957)

## 12. Verdict

**ACCEPTED AS PARTIAL CANDIDATE-GENERATION REPAIR**

## 13. Production unchanged

Current production retrieval config (`vector-pool-expansion-v1`, fetch@24, pool@24, no cap) and `FrozenRetrievalService` semantics were not modified by this experiment.

## 14. Remaining T044/T047 backlog

Unresolved cases explicitly allowed to remain unreachable: ['T044', 'T047']

## 15. Recommended next stage

Stage 4C.3C ranking/metadata refinement or targeted corpus repair (not implemented in this stage).

### Interpretation boundary

- Candidate is only a partial candidate-generation repair.
- Candidate is not release-ready and receives no automatic production promotion.
- Primary hit@4 restoration to historical Arm A levels is not achieved.

_Informational total evaluation latency: 25.5s_
