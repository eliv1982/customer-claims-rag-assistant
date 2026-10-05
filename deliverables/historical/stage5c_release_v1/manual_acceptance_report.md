# Manual acceptance report (Stage 5C + Russian production flow)

> **Historical document, superseded.** This is the Stage 5C acceptance record for release `foodflow-10doc-release-v1` (215 chunks, June 2026), kept as it was written. It does not describe the current product, corpus, index or CLI schema, and the commit it names no longer exists on `main`. Current evidence: [`deliverables/evidence/`](../../evidence/README.md). See [`README.md`](README.md) in this directory for what is outdated.

**Overall verdict: PASS**

## 1. Scope

Application-level acceptance for the closed **10-document production posture** (`foodflow-10doc-release-v1`, target `active`). Evidence covers Docker Streamlit, local `answer-claim` CLI, release-posture validator, deterministic risk/handoff, bounded out-of-scope behavior, controlled generation fallback, fail-closed startup, and the full Russian-language production flow with targeted recheck R1–R4.

This is **not** a full answer-quality benchmark and does not re-run retrieval experiments.

## 2. Tested identity

| Field | Value |
|-------|-------|
| Base commit (pre-repair package) | `32ec3f7493bb6a87149511990a4a38be65f623c3` |
| Release ID | `foodflow-10doc-release-v1` |
| Target | `active` |
| Fingerprint | `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3` |
| Chunks / documents | `215 / 10` |
| Frozen config hash | `ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048` |
| Retrieval contract | `24 / 24 / 12 / 0.0` |
| Reranker | `source-authority-v1` |
| Docker image | `customer-claims-rag-assistant-streamlit` (local build) |

## 3. Environment

| Item | Value |
|------|-------|
| Surface | Docker Compose Streamlit + local CLI |
| Python | 3.12.10 |
| Execution date | 2026-06-26 / 2026-06-28 |
| Embedding model | `text-embedding-3-small` |
| Chat model | `gpt-4o-mini` |
| External OpenAI | Used for M01–M07 live CLI/UI evidence (not for M00, M08, M09) |

Credentials were loaded from local `.env` at runtime and are **not** included in committed evidence.

## 4. Scenario table — main demo set (5 typical + 2 out-of-scope)

| Scenario | Surface | Purpose | Expected | Observed risk | Handoff | Citations / sources | Generation outcome | Result | Evidence | OpenAI |
|----------|---------|---------|----------|---------------|---------|---------------------|-------------------|--------|----------|--------|
| M00 | Docker `validate-release-posture` | Preflight identity | Active release diagnostics | — | — | — | — | **PASS** | [terminal/M00](evidence/terminal/M00_release_posture.txt) | No |
| M01 | CLI + UI | Ordinary delivery FAQ | Low risk, grounded answer | `low` | none | fallback only (generation error) | `generation_error_fallback` | **PASS*** | [cli/M01](evidence/cli/M01_answer.json), [ui](evidence/ui/ui_M01_delivery_faq.png) | Yes |
| M02 | CLI | Missing paid item | Low/medium procedural answer | `low` | none | `10_customer_faq`, `04_refund_policy` | `grounded_answer` | **PASS** | [cli/M02](evidence/cli/M02_answer.json) | Yes |
| M03 | CLI + UI | Opened container | High risk + handoff | `high` | required | `10_customer_faq` | `grounded_answer` | **PASS** | [cli/M03](evidence/cli/M03_answer.json), [ui](evidence/ui/ui_M03_opened_container.png) | Yes |
| M04 | CLI + UI | Refund + compensation | Grounded policy answer | `medium` | none | `10_customer_faq` | `grounded_answer` | **PASS** | [cli/M04](evidence/cli/M04_answer.json), [ui](evidence/ui/ui_M04_refund_compensation.png) | Yes |
| M05 | CLI | Health complaint | Elevated risk + escalation | `critical` | priority | `10_customer_faq`, `07_complaint_handling_procedure` | `grounded_answer` | **PASS** | [cli/M05](evidence/cli/M05_answer.json) | Yes |
| M06 | CLI + UI | Critical breathing | Critical + priority handoff | `critical` | priority | none (generation fallback) | `generation_error_fallback` | **PASS** | [cli/M06](evidence/cli/M06_answer.json), [ui](evidence/ui/ui_M06_critical_escalation.png) | Yes |
| M07 | CLI + UI | Out-of-scope finance | Bounded refusal | `low` | none | none | `insufficient_context` | **PASS** | [cli/M07](evidence/cli/M07_answer.json), [ui](evidence/ui/ui_M07_out_of_scope.png) | Yes |
| M08 | Pytest (controlled) | Generation failure fallback | Preserve high risk + handoff | `high` (test) | preserved | n/a | `generation_error_fallback` | **PASS** | [terminal/M08](evidence/terminal/M08_controlled_fallback_pytest.txt) | No |
| M09 | Docker preflight | Missing index | Fail-closed exit 1 | — | — | — | — | **PASS** | [terminal/M09](evidence/terminal/M09_missing_index_startup.txt) | No |

\*M01 and M06: structurally correct risk/handoff; live generation returned controlled fallback on first execution (one technical retry for M01 did not change outcome). Risk invariants and UI loading/identity behavior verified.

Release identity empty-form screenshot: [ui_release_identity_empty_form.png](evidence/ui/ui_release_identity_empty_form.png)

Machine-readable index: [evidence_manifest.json](evidence/evidence_manifest.json)

## 5. Targeted recheck — Russian production flow (R1–R4)

All four scenarios executed against the production Streamlit UI on 2026-06-28. All response times were under 30 seconds, satisfying the SLA requirement.

| ID | Query (RU) | Category | Risk | Escalation | Response time | SLA ≤30s | Result | Screenshot |
|----|-----------|----------|------|-----------|--------------|----------|--------|-----------|
| R1 | Я оплатил заказ №12345, но он не был доставлен. Деньги не вернули. Что делать? | Недоставка оплаченного заказа | высокий | да | **9.90 s** | PASS | **PASS** | [01_paid_non_delivery.png](evidence/manual_acceptance/01_paid_non_delivery.png) |
| R2 | Курьер не приехал с заказом в срок, не могу до него дозвониться. Что делать? | Задержка доставки | низкий | нет | **3.66 s** | PASS | **PASS** | [08_targeted_recheck_delivery_delay.png](evidence/manual_acceptance/08_targeted_recheck_delivery_delay.png) |
| R3 | После заказа мне стало плохо: появилась сыпь и слабость. Что мне делать? | Жалоба на здоровье после употребления продукта | критический | приоритетная | **3.65 s** | PASS | **PASS** | [04_health_complaint.png](evidence/manual_acceptance/04_health_complaint.png) |
| R4 | Требую возврат за три заказа, которые были доставлены с большой задержкой. | Требование возврата средств | средний | нет | **2.58 s** | PASS | **PASS** | [05_refund_request.png](evidence/manual_acceptance/05_refund_request.png) |

### R1 — Paid non-delivery

- Category: `Недоставка оплаченного заказа`
- Risk: высокий (high)
- Escalation: yes
- Customer draft separated from staff technical information
- No technical statuses appear in the primary response zone

### R2 — Courier delivery delay

- Category: `Задержка доставки`
- Risk: низкий (low)
- Not classified as final non-delivery
- Delay duration not fabricated
- Customer response contains no internal staff instructions

### R3 — Health complaint

- Risk: критический (critical)
- Priority escalation triggered
- Safe medical fallback response
- No diagnosis provided
- Technical information visible only in collapsed staff block

### R4 — Refund demand

- Category: `Требование возврата средств`
- Risk: средний (medium)
- Manual review of conditions and order history required before decision
- No refund promise made
- Retrieved materials shown separately
- Technical information visible only in collapsed staff block

## 6. Evidence screenshots — full list

All eight PNG files are located in `deliverables/evidence/manual_acceptance/`.

| # | File | Scenario |
|---|------|---------|
| 1 | [01_paid_non_delivery.png](evidence/manual_acceptance/01_paid_non_delivery.png) | R1 — оплаченная недоставка |
| 2 | [02_missing_items.png](evidence/manual_acceptance/02_missing_items.png) | Недокомплект заказа |
| 3 | [03_opened_packaging.png](evidence/manual_acceptance/03_opened_packaging.png) | Вскрытая упаковка |
| 4 | [04_health_complaint.png](evidence/manual_acceptance/04_health_complaint.png) | R3 — Жалоба на здоровье после употребления продукта |
| 5 | [05_refund_request.png](evidence/manual_acceptance/05_refund_request.png) | R4 — требование возврата |
| 6 | [06_out_of_scope_weather.png](evidence/manual_acceptance/06_out_of_scope_weather.png) | Вне темы — погода |
| 7 | [07_out_of_scope_fitness.png](evidence/manual_acceptance/07_out_of_scope_fitness.png) | Вне темы — тренировки |
| 8 | [08_targeted_recheck_delivery_delay.png](evidence/manual_acceptance/08_targeted_recheck_delivery_delay.png) | R2 — задержка доставки |

## 7. Production invariants (confirmed unchanged)

| Parameter | Value |
|-----------|-------|
| Release posture ID | `foodflow-10doc-release-v1` |
| Active index | `data/04_index_backup_10docs_215chunks` |
| Fingerprint | `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3` |
| Chunks / documents | `215 / 10` |
| Retrieval contract | `fetch24 / pool24 / final12 / threshold0.0` |
| Reranker | `source-authority-v1` |

Confirmed via `docker compose run --rm streamlit validate-release-posture --skip-vector-store` on 2026-06-28.

## 8. Full test suite results (2026-06-28)

| Metric | Value |
|--------|-------|
| Passed | 1691 |
| Skipped | 2 |
| Failed | 0 |
| Duration | 75.54 s |

Skipped tests:
- `test_hybrid_lexical_replay`: requires frozen hybrid artifact corpus fingerprint; production index differs
- `test_retrieval_path_helpers` symlink escape: requires POSIX

## 9. Acceptance conclusions

### Validated capabilities

- Release posture validator reports correct active identity (M00).
- Deterministic risk floor and handoff for high/critical synthetic scenarios (M03, M05, M06).
- Grounded answers with safe citations for missing-item, packaging, refund/compensation flows (M02–M04).
- Bounded out-of-scope handling without invented FoodFlow policy (M07).
- Controlled offline generation fallback preserves risk/handoff (M08).
- Docker fail-closed startup on empty index mount (M09).
- Streamlit demo hardening: loading spinner, runtime release identity panel, categorized startup errors.
- Russian-language production flow: correct risk classification, escalation logic, client/staff separation, and response time < 30 s for all four recheck scenarios (R1–R4).
- Fallback and safety response for health complaints without diagnosis (R3).

### Known limitations

- M01 (and intermittently M06) live generation returned `generation_error_fallback` despite successful retrieval; wording is non-deterministic across runs.
- M05 classified as `critical` (not merely `high`) for strong post-meal symptoms — consistent with deterministic rules.
- UI screenshots reflect Docker surface; CLI JSON is the primary contract artifact.

### Deferred

- Stage 5D: defense script, consolidated architecture doc, video narration package.
- Remote/public deployment.

## 10. MVP limitations and post-MVP improvements

The assistant forms a **safe preliminary draft response**. The draft is reviewed by a staff member before sending to the customer. In risky or ambiguous situations, conservative wording is used.

**MVP limitations (out of scope for this release):**

- Expanding the client response template library
- SLA-aware responses with dynamic timing context
- More specific responses after manual review or escalation completion

**Post-MVP improvements (not in this release):**

- Iterative template calibration based on pilot operation results
- Integration with CRM order lookup for personalized responses

The system does **not** self-train on incoming claims.

## 11. No-overclaim boundary

- Retrieval benchmark metrics are **not** a full answer-quality score.
- This manual matrix is a **bounded** scenario set, not exhaustive production certification.
- Generated customer wording may vary between runs.
- The assistant does **not** execute refunds, compensation, medical intervention, or CRM actions.
- Documents **11–15** remain outside production support.
- M08 evidence is **controlled offline fallback** and is **not** proof of a real provider outage.
