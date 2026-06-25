# Expanded corpus frozen regression: expanded_corpus_frozen_regression_v1

**Timestamp:** 2026-06-25T18:20:47.034214+00:00
**Evaluation ID:** `expanded_corpus_frozen_regression_v1`
**Frozen question set:** `foodflow-60-case-frozen-v1`
**Retrieval chain:** `vector-top-24 -> source-authority-v1 -> final-top-12`
**Verdict:** `REJECT / REPAIR REQUIRED`

high/critical harmful regressions detected: T011, T035, T039, T040, T044, T046, T047, T051

## 1. Index arms

| Arm | Path | Fingerprint | Chunks | Documents |
|-----|------|-------------|-------:|----------:|
| A historical | `C:\Users\eliv\Cursor_Projects\customer-claims-rag-assistant\data\04_index_backup_10docs_215chunks` | `bf3df0d4631f29f3...` | 215 | 10 |
| B production | `C:\Users\eliv\Cursor_Projects\customer-claims-rag-assistant\data\04_index` | `b9526128dad23e71...` | 333 | 15 |

Both arms used identical frozen retrieval config `vector-pool-expansion-v1` (hash `ff53ff9721ad86b1...`), reranker `source-authority-v1`, pool@24, final top-12, threshold 0.0.

## 2. Aggregate metrics (final top-12 after rerank)

| Metric | Arm A | Arm B | Delta (B−A) |
|--------|------:|------:|------------:|
| Hit@1 | 0.517 | 0.483 | -0.034 |
| Hit@4 | 0.897 | 0.759 | -0.138 |
| Hit@12 | 0.983 | 0.897 | -0.086 |
| Primary hit@4 | 0.741 | 0.621 | -0.121 |
| MRR | 0.715 | 0.643 | -0.072 |
| Document recall@4 | 0.667 | 0.537 | -0.129 |
| Document recall@12 | 0.862 | 0.759 | -0.103 |

## 3. Pool@24 reachability

- Primary reachable: Arm A 56/57 -> Arm B 51/57 (delta -5)
- Fully unreachable primaries (A): ['T004']
- Fully unreachable primaries (B): ['T004', 'T040', 'T044', 'T046', 'T047']
- High-risk primary reachable@24: 15/15 -> 14/15
- Critical primary reachable@24: 8/8 -> 5/8

## 4. Risk slices (final top-12)

| Risk | Arm A Hit@4 | Arm B Hit@4 | Arm A MRR | Arm B MRR |
|------|------------:|------------:|--------:|--------:|
| low | 0.875 | 0.812 | 0.693 | 0.664 |
| medium | 1.000 | 0.842 | 0.754 | 0.715 |
| high | 0.867 | 0.667 | 0.713 | 0.572 |
| critical | 0.750 | 0.625 | 0.667 | 0.562 |

## 5. FAQ dominance

- FAQ top-1 slots: Arm A 22 -> Arm B 16
- Questions with FAQ in top-4: Arm A 45 -> Arm B 36
- FAQ outranks all primaries in top-4: Arm A 25 -> Arm B 21

## 6. New document footprint (Arm B)

### `11_payment_security_and_dispute_handling`

- top-1: 4 (T004, T024, T025, T046)
- top-4: 10
- top-12: 18
- pool@24: 19
- likely helpful: T004
- irrelevant noise top-4: T003, T005, T015, T024, T025, T028, T032, T046, T051
- displaces frozen primary: T004

### `12_staff_safety_and_threat_handling`

- top-1: 4 (T044, T047, T052, T054)
- top-4: 5
- top-12: 10
- pool@24: 15
- likely helpful: T047
- irrelevant noise top-4: T044, T045, T052, T054
- displaces frozen primary: T047

### `13_physical_hazard_and_foreign_body_protocol`

- top-1: 3 (T016, T039, T040)
- top-4: 8
- top-12: 20
- pool@24: 25
- likely helpful: T040
- irrelevant noise top-4: T016, T025, T036, T039, T043
- displaces frozen primary: T027, T040, T047

### `14_evidence_standards_and_incomplete_information`

- top-1: 1 (T055)
- top-4: 9
- top-12: 19
- pool@24: 27
- likely helpful: T055
- irrelevant noise top-4: T003, T012, T039, T043, T044, T046, T052
- displaces frozen primary: T004, T055

### `15_conflicting_rules_and_remedy_priority`

- top-1: 2 (T031, T051)
- top-4: 10
- top-12: 27
- pool@24: 37
- likely helpful: T023, T027, T053
- irrelevant noise top-4: T012, T022, T031, T033, T051, T052, T056, T057
- displaces frozen primary: T027, T053

## 7. Special cases

### T004

- Risk: low; expected primary: ['07_complaint_handling_procedure']
- Arm A primary rank (final): None; Arm B: None
- Arm A top-4: ['04_refund_policy']
- Arm B top-4: ['11_payment_security_and_dispute_handling', '14_evidence_standards_and_incomplete_information']
- Primary in pool@24: A=False, B=False
- Classification: `strict_frozen_regression_with_semantically_stronger_new_source`
- new dedicated policy 11_payment_security_and_dispute_handling visible in top-4 while strict score unchanged

### T040

- Risk: critical; expected primary: ['06_food_quality_and_packaging']
- Arm A primary rank (final): 8; Arm B: None
- Arm A top-4: ['10_customer_faq']
- Arm B top-4: ['13_physical_hazard_and_foreign_body_protocol']
- Primary in pool@24: A=True, B=False
- Classification: `strict_frozen_regression_with_semantically_stronger_new_source`
- frozen primary strict score declined while a more specific new policy (13_physical_hazard_and_foreign_body_protocol) surfaced in top-4

### T047

- Risk: critical; expected primary: ['08_escalation_and_risk_rules']
- Arm A primary rank (final): 7; Arm B: None
- Arm A top-4: ['02_delivery_rules', '10_customer_faq', '04_refund_policy']
- Arm B top-4: ['12_staff_safety_and_threat_handling', '13_physical_hazard_and_foreign_body_protocol']
- Primary in pool@24: A=True, B=False
- Classification: `strict_frozen_regression_with_semantically_stronger_new_source`
- frozen primary strict score declined while a more specific new policy (12_staff_safety_and_threat_handling) surfaced in top-4

### T055

- Risk: high; expected primary: ['06_food_quality_and_packaging']
- Arm A primary rank (final): 7; Arm B: 11
- Arm A top-4: ['10_customer_faq', '02_delivery_rules', '07_complaint_handling_procedure', '04_refund_policy']
- Arm B top-4: ['14_evidence_standards_and_incomplete_information', '10_customer_faq', '02_delivery_rules', '07_complaint_handling_procedure']
- Primary in pool@24: A=True, B=True
- Classification: `strict_frozen_regression_with_semantically_stronger_new_source`
- new dedicated policy 14_evidence_standards_and_incomplete_information outranks frozen primary in top-4

### T023

- Risk: medium; expected primary: ['03_order_changes_and_cancellations', '05_compensation_policy']
- Arm A primary rank (final): 2; Arm B: 2
- Arm A top-4: ['10_customer_faq', '03_order_changes_and_cancellations']
- Arm B top-4: ['10_customer_faq', '03_order_changes_and_cancellations']
- Primary in pool@24: A=True, B=True
- Classification: `stable`
- no material frozen-metric change between index arms

### T027

- Risk: high; expected primary: ['06_food_quality_and_packaging', '08_escalation_and_risk_rules']
- Arm A primary rank (final): 3; Arm B: None
- Arm A top-4: ['10_customer_faq', '04_refund_policy', '06_food_quality_and_packaging', '08_escalation_and_risk_rules']
- Arm B top-4: ['10_customer_faq', '13_physical_hazard_and_foreign_body_protocol', '15_conflicting_rules_and_remedy_priority']
- Primary in pool@24: A=True, B=True
- Classification: `strict_frozen_regression_with_semantically_stronger_new_source`
- new dedicated policy 15_conflicting_rules_and_remedy_priority outranks frozen primary in top-4

### T053

- Risk: high; expected primary: ['08_escalation_and_risk_rules']
- Arm A primary rank (final): None; Arm B: None
- Arm A top-4: ['02_delivery_rules', '07_complaint_handling_procedure', '05_compensation_policy']
- Arm B top-4: ['02_delivery_rules', '15_conflicting_rules_and_remedy_priority', '07_complaint_handling_procedure']
- Primary in pool@24: A=True, B=True
- Classification: `strict_frozen_regression_with_semantically_stronger_new_source`
- new dedicated policy 15_conflicting_rules_and_remedy_priority visible in top-4 while strict score unchanged

## 8. Semantic displacement overlay

### T004 -> `11_payment_security_and_dispute_handling`

- more specific than frozen: True
- frozen primary still in Arm B top-12: False
- recommend extension set: True
- recommend benchmark modification: False
- do not modify frozen benchmark; add extension coverage

### T040 -> `13_physical_hazard_and_foreign_body_protocol`

- more specific than frozen: True
- frozen primary still in Arm B top-12: False
- recommend extension set: True
- recommend benchmark modification: False
- do not modify frozen benchmark; add extension coverage

### T047 -> `12_staff_safety_and_threat_handling`

- more specific than frozen: True
- frozen primary still in Arm B top-12: False
- recommend extension set: True
- recommend benchmark modification: False
- do not modify frozen benchmark; add extension coverage

### T055 -> `14_evidence_standards_and_incomplete_information`

- more specific than frozen: True
- frozen primary still in Arm B top-12: True
- recommend extension set: True
- recommend benchmark modification: False
- do not modify frozen benchmark; add extension coverage

### T023 -> `15_conflicting_rules_and_remedy_priority`

- more specific than frozen: True
- frozen primary still in Arm B top-12: True
- recommend extension set: True
- recommend benchmark modification: False
- do not modify frozen benchmark; add extension coverage

### T027 -> `15_conflicting_rules_and_remedy_priority`

- more specific than frozen: True
- frozen primary still in Arm B top-12: False
- recommend extension set: True
- recommend benchmark modification: False
- do not modify frozen benchmark; add extension coverage

### T053 -> `15_conflicting_rules_and_remedy_priority`

- more specific than frozen: True
- frozen primary still in Arm B top-12: False
- recommend extension set: True
- recommend benchmark modification: False
- do not modify frozen benchmark; add extension coverage

## 9. Paired per-question summary

| Case | Class | Harmful | Arm A rank | Arm B rank | Arm B top-4 |
|------|-------|---------|----------:|----------:|-------------|
| T001 | stable | False | 3 | 3 | 10_customer_faq, 03_order_changes_and_cancellations, 01_service_overview, 04_refund_policy |
| T002 | regression | False | 7 | 9 | 02_delivery_rules, 10_customer_faq |
| T003 | regression | True | 3 | 5 | 10_customer_faq, 14_evidence_standards_and_incomplete_information, 11_payment_security_and_dispute_handling |
| T004 | strict_frozen_regression_with_semantically_stronger_new_source | False | None | None | 11_payment_security_and_dispute_handling, 14_evidence_standards_and_incomplete_information |
| T005 | stable | False | 1 | 1 | 04_refund_policy, 07_complaint_handling_procedure, 11_payment_security_and_dispute_handling, 10_customer_faq |
| T006 | stable | False | None | None | 06_food_quality_and_packaging, 03_order_changes_and_cancellations, 04_refund_policy |
| T007 | stable | False | 2 | 2 | 03_order_changes_and_cancellations, 02_delivery_rules |
| T008 | stable | False | 2 | 2 | 10_customer_faq, 02_delivery_rules, 03_order_changes_and_cancellations |
| T009 | stable | False | 6 | 6 | 10_customer_faq, 03_order_changes_and_cancellations |
| T010 | stable | False | 3 | 3 | 03_order_changes_and_cancellations, 02_delivery_rules, 10_customer_faq |
| T011 | regression | True | 6 | 6 | 10_customer_faq, 03_order_changes_and_cancellations |
| T012 | stable | False | 1 | 1 | 02_delivery_rules, 14_evidence_standards_and_incomplete_information, 15_conflicting_rules_and_remedy_priority |
| T013 | stable | False | 2 | 2 | 01_service_overview, 02_delivery_rules, 07_complaint_handling_procedure, 03_order_changes_and_cancellations |
| T014 | stable | False | 2 | 2 | 10_customer_faq, 02_delivery_rules |
| T015 | regression | True | 3 | 6 | 07_complaint_handling_procedure, 11_payment_security_and_dispute_handling |
| T016 | regression | True | 9 | None | 13_physical_hazard_and_foreign_body_protocol, 10_customer_faq, 06_food_quality_and_packaging |
| T017 | stable | False | 3 | 3 | 10_customer_faq, 03_order_changes_and_cancellations |
| T018 | stable | False | 3 | 3 | 10_customer_faq, 03_order_changes_and_cancellations |
| T019 | stable | False | 1 | 1 | 03_order_changes_and_cancellations, 10_customer_faq |
| T020 | stable | False | 1 | 1 | 03_order_changes_and_cancellations, 10_customer_faq |
| T021 | stable | False | 1 | 1 | 03_order_changes_and_cancellations, 10_customer_faq |
| T022 | stable | False | 1 | 1 | 03_order_changes_and_cancellations, 10_customer_faq, 15_conflicting_rules_and_remedy_priority |
| T023 | stable | False | 2 | 2 | 10_customer_faq, 03_order_changes_and_cancellations |
| T024 | regression | False | 1 | 3 | 11_payment_security_and_dispute_handling, 04_refund_policy |
| T025 | regression | True | 3 | 7 | 11_payment_security_and_dispute_handling, 13_physical_hazard_and_foreign_body_protocol, 05_compensation_policy |
| T026 | stable | False | 1 | 1 | 04_refund_policy, 10_customer_faq, 06_food_quality_and_packaging |
| T027 | strict_frozen_regression_with_semantically_stronger_new_source | False | 3 | None | 10_customer_faq, 13_physical_hazard_and_foreign_body_protocol, 15_conflicting_rules_and_remedy_priority |
| T028 | stable | False | 1 | 1 | 04_refund_policy, 03_order_changes_and_cancellations, 11_payment_security_and_dispute_handling, 10_customer_faq |
| T029 | stable | False | 5 | 5 | 04_refund_policy, 10_customer_faq |
| T030 | stable | False | 1 | 1 | 06_food_quality_and_packaging, 07_complaint_handling_procedure, 10_customer_faq |
| T031 | regression | False | 2 | 3 | 15_conflicting_rules_and_remedy_priority, 10_customer_faq, 05_compensation_policy |
| T032 | stable | False | 2 | 2 | 10_customer_faq, 05_compensation_policy, 06_food_quality_and_packaging, 11_payment_security_and_dispute_handling |
| T033 | stable | False | 1 | 1 | 05_compensation_policy, 15_conflicting_rules_and_remedy_priority |
| T034 | stable | False | 2 | 2 | 10_customer_faq, 06_food_quality_and_packaging |
| T035 | regression | True | 5 | 5 | 10_customer_faq, 07_complaint_handling_procedure |
| T036 | stable | False | 1 | 1 | 06_food_quality_and_packaging, 10_customer_faq, 13_physical_hazard_and_foreign_body_protocol |
| T037 | stable | False | 1 | 1 | 06_food_quality_and_packaging, 10_customer_faq, 07_complaint_handling_procedure |
| T038 | stable | False | 1 | 1 | 06_food_quality_and_packaging, 10_customer_faq, 07_complaint_handling_procedure |
| T039 | regression | True | 4 | None | 13_physical_hazard_and_foreign_body_protocol, 14_evidence_standards_and_incomplete_information |
| T040 | strict_frozen_regression_with_semantically_stronger_new_source | True | 8 | None | 13_physical_hazard_and_foreign_body_protocol |
| T041 | stable | False | 1 | 1 | 06_food_quality_and_packaging, 10_customer_faq, 07_complaint_handling_procedure |
| T042 | stable | False | 2 | 2 | 10_customer_faq, 06_food_quality_and_packaging |
| T043 | stable | False | 1 | 1 | 06_food_quality_and_packaging, 13_physical_hazard_and_foreign_body_protocol, 14_evidence_standards_and_incomplete_information |
| T044 | regression | True | 11 | None | 12_staff_safety_and_threat_handling, 14_evidence_standards_and_incomplete_information |
| T045 | stable | False | 1 | 1 | 08_escalation_and_risk_rules, 12_staff_safety_and_threat_handling |
| T046 | regression | True | 4 | None | 11_payment_security_and_dispute_handling, 14_evidence_standards_and_incomplete_information |
| T047 | strict_frozen_regression_with_semantically_stronger_new_source | True | 7 | None | 12_staff_safety_and_threat_handling, 13_physical_hazard_and_foreign_body_protocol |
| T048 | stable | False | 1 | 1 | 06_food_quality_and_packaging, 08_escalation_and_risk_rules |
| T049 | stable | False | 2 | 2 | 10_customer_faq, 07_complaint_handling_procedure |
| T050 | stable | False | 1 | 1 | 07_complaint_handling_procedure, 10_customer_faq, 09_response_style_and_templates |
| T051 | regression | True | 2 | 6 | 15_conflicting_rules_and_remedy_priority, 11_payment_security_and_dispute_handling, 04_refund_policy |
| T052 | regression | False | 1 | 2 | 12_staff_safety_and_threat_handling, 08_escalation_and_risk_rules, 15_conflicting_rules_and_remedy_priority, 14_evidence_standards_and_incomplete_information |
| T053 | strict_frozen_regression_with_semantically_stronger_new_source | False | None | None | 02_delivery_rules, 15_conflicting_rules_and_remedy_priority, 07_complaint_handling_procedure |
| T054 | regression | False | 5 | 7 | 12_staff_safety_and_threat_handling, 05_compensation_policy, 06_food_quality_and_packaging |
| T055 | strict_frozen_regression_with_semantically_stronger_new_source | False | 7 | 11 | 14_evidence_standards_and_incomplete_information, 10_customer_faq, 02_delivery_rules, 07_complaint_handling_procedure |
| T056 | regression | True | None | None | 09_response_style_and_templates, 15_conflicting_rules_and_remedy_priority, 08_escalation_and_risk_rules |
| T057 | stable | False | 1 | 1 | 06_food_quality_and_packaging, 15_conflicting_rules_and_remedy_priority, 08_escalation_and_risk_rules |
| T058 | stable | False | 1 | 1 | 02_delivery_rules, 10_customer_faq, 03_order_changes_and_cancellations |
| T059 | regression | False | 9 | 10 | 10_customer_faq, 03_order_changes_and_cancellations, 04_refund_policy |
| T060 | stable | False | None | None | 03_order_changes_and_cancellations, 06_food_quality_and_packaging, 04_refund_policy |

## 10. Notes

- Strict frozen metrics use the unchanged 60-question benchmark expectations.
- Semantic overlay candidates: T004->11_payment_security_and_dispute_handling, T040->13_physical_hazard_and_foreign_body_protocol, T047->12_staff_safety_and_threat_handling, T055->14_evidence_standards_and_incomplete_information, T023->15_conflicting_rules_and_remedy_priority, T027->15_conflicting_rules_and_remedy_priority, T053->15_conflicting_rules_and_remedy_priority.
- Release-validation extension set remains pending for Stage 4C.3.
