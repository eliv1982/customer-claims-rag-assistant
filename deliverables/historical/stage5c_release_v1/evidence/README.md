# Manual acceptance evidence

> **Historical, superseded** (release `foodflow-10doc-release-v1`). The files below are the Stage 5C record, unchanged; see the parent directory's `README.md` for what is outdated.

Committed artifacts supporting Stage 5C + Russian production flow acceptance for release `foodflow-10doc-release-v1`.

## Layout

| Path | Contents |
|------|----------|
| `terminal/` | Sanitized validator and fail-closed startup captures |
| `cli/` | Sanitized `answer-claim` JSON outputs |
| `ui/` | PNG screenshots from Docker Streamlit (Playwright) |
| `manual_acceptance/` | Final acceptance screenshots (8 PNG, renamed for evidence index) |

## manual_acceptance/ — 8 required screenshots

| File | Scenario |
|------|---------|
| `01_paid_non_delivery.png` | R1 — оплаченная недоставка (high risk, escalation) |
| `02_missing_items.png` | Недокомплект заказа |
| `03_opened_packaging.png` | Вскрытая упаковка (high risk, handoff) |
| `04_health_complaint.png` | R3 — жалоба на здоровье (critical risk, priority escalation) |
| `05_refund_request.png` | R4 — требование возврата (medium risk, manual review) |
| `06_out_of_scope_weather.png` | Вне темы — погода (bounded refusal) |
| `07_out_of_scope_fitness.png` | Вне темы — тренировки (bounded refusal) |
| `08_targeted_recheck_delivery_delay.png` | R2 — задержка доставки (low risk, no fabricated delay) |

All files verified non-empty (107 KB – 191 KB each) on 2026-06-28.

## Regeneration

Not reproducible any more: the release v1 index, the Docker flow and the capture scripts of that package were replaced (the capture script was removed). The current evidence is produced by `scripts/capture_final_acceptance.py`; see `deliverables/evidence/README.md`.
