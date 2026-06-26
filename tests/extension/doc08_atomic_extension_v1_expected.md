# Extension expected answers: doc08-atomic-risk-units-v1

**Benchmark ID:** `doc08-atomic-extension-v1`

---

## E001 — Чужая наклейка с именем

- **Expected risk level:** high
- **Primary sources:** `08_escalation_and_risk_rules`
- **Acceptable additional sources:** отсутствуют
- **Fallback expected:** нет
- **Rationale:** possible customer-data leak on label; doc08 privacy atomic unit.

---

## E002 — Чужой адрес в кабинете

- **Expected risk level:** high
- **Primary sources:** `08_escalation_and_risk_rules`
- **Acceptable additional sources:** отсутствуют
- **Fallback expected:** нет
- **Rationale:** foreign address exposure; doc08 address unit.

---

## E003 — Чужой телефон в receipt

- **Expected risk level:** high
- **Primary sources:** `08_escalation_and_risk_rules`
- **Acceptable additional sources:** отсутствуют
- **Fallback expected:** нет
- **Rationale:** foreign phone on receipt; doc08 contact exposure unit.

---

## E004 — Несколько чужих полей на документе

- **Expected risk level:** high
- **Primary sources:** `08_escalation_and_risk_rules`
- **Acceptable additional sources:** отсутствуют
- **Fallback expected:** нет
- **Rationale:** multi-field customer-data mismatch; doc08 leak route.

---

## E005 — Прямая угроза курьеру

- **Expected risk level:** critical
- **Primary sources:** `12_staff_safety_and_threat_handling`
- **Acceptable additional sources:** `08_escalation_and_risk_rules`
- **Fallback expected:** нет
- **Rationale:** staff threat operational primary is doc12; doc08 escalation boundary only.

---

## E006 — Условная угроза курьеру

- **Expected risk level:** critical
- **Primary sources:** `12_staff_safety_and_threat_handling`
- **Acceptable additional sources:** `08_escalation_and_risk_rules`
- **Fallback expected:** нет
- **Rationale:** conditional courier threat; doc12 must outrank doc08.

---

## E007 — Повторная встреча с курьером

- **Expected risk level:** critical
- **Primary sources:** `12_staff_safety_and_threat_handling`
- **Acceptable additional sources:** `08_escalation_and_risk_rules`
- **Fallback expected:** нет
- **Rationale:** find courier scenario; doc12 primary.

---

## E008 — Угроза оператору поддержки

- **Expected risk level:** critical
- **Primary sources:** `12_staff_safety_and_threat_handling`
- **Acceptable additional sources:** `08_escalation_and_risk_rules`
- **Fallback expected:** нет
- **Rationale:** staff threat to support operator; doc12 primary.

---

## E009 — Платежный спор без privacy/threat

- **Expected risk level:** medium
- **Primary sources:** `11_payment_security_and_dispute_handling`
- **Acceptable additional sources:** `07_complaint_handling_procedure`
- **Fallback expected:** нет
- **Rationale:** payment dispute domain; doc08 must not dominate.

---

## E010 — Металл в супе

- **Expected risk level:** critical
- **Primary sources:** `13_physical_hazard_and_foreign_body_protocol`
- **Acceptable additional sources:** `06_food_quality_and_packaging`
- **Fallback expected:** нет
- **Rationale:** physical hazard protocol primary.

---

## E011 — Нет фото, только описание

- **Expected risk level:** high
- **Primary sources:** `06_food_quality_and_packaging`
- **Acceptable additional sources:** `14_evidence_standards_and_incomplete_information`
- **Fallback expected:** нет
- **Rationale:** quality/evidence domain, not broad escalation magnet.

---

## E012 — Конфликт refund и chargeback

- **Expected risk level:** medium
- **Primary sources:** `15_conflicting_rules_and_remedy_priority`
- **Acceptable additional sources:** `04_refund_policy`
- **Fallback expected:** нет
- **Rationale:** remedy-priority conflict; doc15 primary.
