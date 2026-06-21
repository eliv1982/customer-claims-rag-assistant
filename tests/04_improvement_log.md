# Improvement log: baseline retrieval

**Evaluation result ID:** `2132c861d4d03b3b99fb5413ae69fd3f0b45f3326d7b71671b4ea3a2a8106e0a`
**Дата baseline run:** 2026-06-21
**Git commit:** `46eb01cf84ca4e32b5641dc3ceef0b3ebe9356fa`
**Working tree dirty:** False

## Baseline configuration

- Baseline dense retrieval without reranking, source priority or risk-aware boosting
- Embedding model: `text-embedding-3-small`
- Collection: `customer_claims`
- Index fingerprint: `bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3`
- Evaluation threshold: `0.00` (no filtering for baseline recall)
- Candidate pool: fetch_k=12, metrics reported at k=1/4/12
- All @k metrics use the first k raw chunks, then deduplicate documents inside that window
- Retrieval-only metrics; LLM answer pipeline not evaluated

## Main failure patterns

- Overall primary source hit@4: **0.621** on 58 source-recall cases
- Supporting source hit@4: **0.526 (20/38)**
- Critical primary miss@4: **3** cases
- High primary miss@4: **7** cases
- FAQ or generic docs may appear instead of profile documents in top-4 chunks
- Technical errors: **0**

## Possible causes

- Semantic similarity favors FAQ chunks with overlapping wording
- Multiple valid documents share vocabulary without source-priority reranking
- Chunk boundaries split policy sections away from query-specific terms
- Boundary/adversarial cases still retrieve plausible but non-primary documents

## Candidate improvements (not implemented in 2B)

- Source-priority reranking after baseline recall measurement
- Risk-aware reranking for critical/high cases
- Hybrid BM25 + dense retrieval
- Query rewriting for long mixed messages
- Production threshold selection after separate review

## Explicitly not implemented yet

- LLM answer generation and factuality scoring
- Risk/handoff classification model evaluation
- Critical safety pass rate based on generated answers
- Markdown output contract validation

## Improvement policy

Do **not** claim retrieval improvement until an A/B rerun on this same 60-case corpus shows measurable delta with unchanged ingestion and corpus fingerprint.

## Critical/high primary misses

- **T040** (critical): expected primary ['06_food_quality_and_packaging']; top-4 docs @4 ['10_customer_faq']
- **T046** (critical): expected primary ['08_escalation_and_risk_rules']; top-4 docs @4 ['07_complaint_handling_procedure', '10_customer_faq']
- **T047** (critical): expected primary ['08_escalation_and_risk_rules']; top-4 docs @4 ['10_customer_faq', '02_delivery_rules']
- **T011** (high): expected primary ['02_delivery_rules', '08_escalation_and_risk_rules']; top-4 docs @4 ['10_customer_faq']
- **T027** (high): expected primary ['06_food_quality_and_packaging', '08_escalation_and_risk_rules']; top-4 docs @4 ['10_customer_faq', '04_refund_policy']
- **T035** (high): expected primary ['06_food_quality_and_packaging']; top-4 docs @4 ['10_customer_faq']
- **T039** (high): expected primary ['06_food_quality_and_packaging']; top-4 docs @4 ['10_customer_faq', '07_complaint_handling_procedure', '08_escalation_and_risk_rules']
- **T044** (high): expected primary ['08_escalation_and_risk_rules']; top-4 docs @4 ['10_customer_faq', '07_complaint_handling_procedure']
- **T053** (high): expected primary ['08_escalation_and_risk_rules']; top-4 docs @4 ['02_delivery_rules', '07_complaint_handling_procedure', '05_compensation_policy']
- **T055** (high): expected primary ['06_food_quality_and_packaging']; top-4 docs @4 ['10_customer_faq', '07_complaint_handling_procedure', '02_delivery_rules', '04_refund_policy']

---

*Baseline captured at stage 2B; changes require A/B rerun.*
