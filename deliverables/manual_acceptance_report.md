# Manual acceptance report (Stage 5C)

## 1. Scope

Application-level acceptance for the closed **10-document production posture** (`foodflow-10doc-release-v1`, target `active`). Evidence covers Docker Streamlit, local `answer-claim` CLI, release-posture validator, deterministic risk/handoff, bounded out-of-scope behavior, controlled generation fallback, and fail-closed startup.

This is **not** a full answer-quality benchmark and does not re-run retrieval experiments.

## 2. Tested identity

| Field | Value |
|-------|-------|
| Base commit (pre-5C UI/evidence) | `b5739c90d5192e5ed990f4e797c07c157476f17b` |
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
| Execution date | 2026-06-26 |
| Embedding model | `text-embedding-3-small` |
| Chat model | `gpt-4o-mini` |
| External OpenAI | Used for M01–M07 live CLI/UI evidence (not for M00, M08, M09) |

Credentials were loaded from local `.env` at runtime and are **not** included in committed evidence.

## 4. Scenario table

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

## 5. Acceptance conclusions

### Validated capabilities

- Release posture validator reports correct active identity (M00).
- Deterministic risk floor and handoff for high/critical synthetic scenarios (M03, M05, M06).
- Grounded answers with safe citations for missing-item, packaging, refund/compensation flows (M02–M04).
- Bounded out-of-scope handling without invented FoodFlow policy (M07).
- Controlled offline generation fallback preserves risk/handoff (M08).
- Docker fail-closed startup on empty index mount (M09).
- Streamlit demo hardening: loading spinner, runtime release identity panel, categorized startup errors.

### Known limitations

- M01 (and intermittently M06) live generation returned `generation_error_fallback` despite successful retrieval; wording is non-deterministic across runs.
- M05 classified as `critical` (not merely `high`) for strong post-meal symptoms — consistent with deterministic rules.
- UI screenshots reflect Docker surface; CLI JSON is the primary contract artifact.

### Deferred

- Stage 5D: defense script, consolidated architecture doc, video narration package.
- Remote/public deployment.

## 6. No-overclaim boundary

- Retrieval benchmark metrics are **not** a full answer-quality score.
- This manual matrix is a **bounded** scenario set, not exhaustive production certification.
- Generated customer wording may vary between runs.
- The assistant does **not** execute refunds, compensation, medical intervention, or CRM actions.
- Documents **11–15** remain outside production support.
- M08 evidence is **controlled offline fallback** and is **not** proof of a real provider outage.
