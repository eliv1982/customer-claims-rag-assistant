# Doc08 atomic risk units: doc08-atomic-risk-units-v1

**Timestamp:** 2026-06-26T02:53:59.085542+00:00
**Source commit:** `6de96148781a2c2eb1bf8850a4e9a647192bf85c`
**Source dirty:** `False`
**Artifact commit:** `pending`
**Reference:** `vector-pool-36-cap4-v1` / `candidate`
**Verdict:** `REJECTED`

## Baseline reproduction

- Passed: **True**
- Reference artifact: `C:/Users/eliv/Cursor_Projects/customer-claims-rag-assistant/data/05_evaluation/vector_pool_36_cap4_v1.json`

## Retrieval arms

| Arm | Index fingerprint | fetch_k | pool_k | cap | chunks |
|-----|-------------------|--------:|-------:|----:|-------:|
| baseline | `b9526128dad23e71...` | 48 | 36 | 4 | 333 |
| candidate | `d3c27f4a72e5f785...` | 48 | 36 | 4 | 337 |

## Frozen metrics

| Metric | Baseline | Candidate |
|--------|----------|-----------|
| Primary hit@4 | 0.620690 | 0.637931 |
| Primary hit@12 | 0.913793 | 0.965517 |
| MRR | 0.645690 | 0.675862 |
| Primary reach | 55/57 | 57/57 |
| High-risk reach | 14/15 | 15/15 |
| Critical reach | 7/8 | 8/8 |
| FAQ top-4 | 36 | 36 |

- Primary unreachable baseline: ['T044', 'T047']
- Primary unreachable candidate: []
- Fully unreachable baseline: ['T047']
- Fully unreachable candidate: []

## Doc08 chunk diff

- Baseline doc08 chunks: 17
- Candidate doc08 chunks: 21
- Added: ['08_escalation_and_risk_rules::chunk-018', '08_escalation_and_risk_rules::chunk-019', '08_escalation_and_risk_rules::chunk-020', '08_escalation_and_risk_rules::chunk-021']
- Removed: []
- Changed: 16
- Non-doc08 byte-identical: True

## Extension metrics

| Metric | Baseline | Candidate |
|--------|----------|-----------|
| Privacy hit@4 | 0/4 | 4/4 |
| Privacy hit@12 | 0/4 | 4/4 |
| Threat doc12 hit@4 | 2/4 | 2/4 |
| Threat doc12 in top-4 | 2/4 | 2/4 |
| Negative domain hit@4 | 4/4 | 4/4 |

## Extension case diagnostics

| Case | Baseline hit@4 | Candidate hit@4 | Delta |
|------|----------------|-----------------|-------|
| E001 | False | True | candidate_improves |
| E002 | False | True | candidate_improves |
| E003 | False | True | candidate_improves |
| E004 | False | True | candidate_improves |
| E005 | True | True | unchanged_pass |
| E006 | True | True | unchanged_pass |
| E007 | False | False | both_fail_not_doc08_regression |
| E008 | False | False | both_fail_not_doc08_regression |
| E009 | True | True | unchanged_pass |
| E010 | True | True | unchanged_pass |
| E011 | True | True | unchanged_pass |
| E012 | True | True | unchanged_pass |

## T044 / T047 diagnostics

### T044

- Expected primary: ['08_escalation_and_risk_rules']
- Baseline vector rank doc08: []
- Candidate vector rank doc08: [2, 4, 7, 29]
- Candidate pool rank doc08: 2
- Candidate final rank doc08: 2
- Baseline/Candidate final rank doc12: 1 / 1
- Best doc08 heading: Customer personal-data exposure — чужой адрес или фрагмент адреса
- Best doc08 similarity: 0.48777127265930176
- Top competing docs: ['12_staff_safety_and_threat_handling', '14_evidence_standards_and_incomplete_information', '12_staff_safety_and_threat_handling']
- Grounding relevant: True

### T047

- Expected primary: ['08_escalation_and_risk_rules']
- Baseline vector rank doc08: []
- Candidate vector rank doc08: [18, 39]
- Candidate pool rank doc08: 12
- Candidate final rank doc08: 11
- Baseline/Candidate final rank doc12: 1 / 1
- Best doc08 heading: Customer personal-data exposure — обзор и disambiguation
- Best doc08 similarity: 0.3865059018135071
- Top competing docs: ['12_staff_safety_and_threat_handling', '12_staff_safety_and_threat_handling', '12_staff_safety_and_threat_handling', '13_physical_hazard_and_foreign_body_protocol', '14_evidence_standards_and_incomplete_information']
- Grounding relevant: False


## Acceptance checklist

- [PASS] `reference_identity`: Reference artifact experiment identity (baseline=vector-pool-36-cap4-v1, candidate=vector-pool-36-cap4-v1)
- [PASS] `baseline_primary_hit_at_4`: Baseline primary hit@4 reproduces reference (baseline=0.6206896551724138, candidate=0.6206896551724138)
- [PASS] `baseline_primary_hit_at_12`: Baseline primary hit@12 reproduces reference (baseline=0.9137931034482759, candidate=0.9137931034482759)
- [PASS] `baseline_mrr`: Baseline MRR reproduces reference (baseline=0.6456896551724138, candidate=0.6456896551724138)
- [PASS] `baseline_primary_reach`: Baseline primary pool reach reproduces reference (baseline=55/57, candidate=55/57)
- [PASS] `baseline_high_risk_reach`: Baseline high-risk reach reproduces reference (baseline=14/15, candidate=14/15)
- [PASS] `baseline_critical_reach`: Baseline critical reach reproduces reference (baseline=7/8, candidate=7/8)
- [PASS] `baseline_primary_unreachable`: Baseline primary-unreachable set reproduces reference (baseline=['T044', 'T047'], candidate=['T044', 'T047'])
- [PASS] `baseline_fully_unreachable`: Baseline fully-unreachable set reproduces reference (baseline=['T047'], candidate=['T047'])
- [PASS] `baseline_faq_top4`: Baseline FAQ top-4 reproduces reference (baseline=36, candidate=36)
- [PASS] `t044_reachable`: T044 primary reachable (baseline=None, candidate=True)
- [PASS] `t044_hit12`: T044 primary hit@12 (baseline=None, candidate=True)
- [PASS] `primary_hit_at_4`: Primary hit@4 >= baseline (baseline=0.6206896551724138, candidate=0.6379310344827587)
- [PASS] `primary_hit_at_12`: Primary hit@12 >= baseline (baseline=0.9137931034482759, candidate=0.9655172413793104)
- [PASS] `mrr`: MRR >= 0.645 (baseline=0.6456896551724138, candidate=0.6758620689655173)
- [PASS] `primary_pool_reach`: Primary pool reach >= 56/57 (baseline=55/57, candidate=57/57)
- [PASS] `high_risk_reach`: High-risk reach = 15/15 (baseline=14/15, candidate=15/15)
- [PASS] `critical_reach`: Critical reach >= 7/8 (baseline=7/8, candidate=8/8)
- [PASS] `faq_top4`: FAQ top-4 <= 36 (baseline=36, candidate=36)
- [PASS] `no_new_unreachable`: No new unreachable primary (baseline=None, candidate=[])
- [PASS] `t047_doc12_top4`: T047 doc12 remains in final top-4 (baseline=None, candidate=1)
- [PASS] `unreachable_subset`: Unreachable primaries subset of {T044, T047} (baseline=['T044', 'T047'], candidate=[])
- [PASS] `ext_privacy_hit4`: Privacy doc08 hit@4 = 4/4 (baseline=None, candidate=4/4)
- [PASS] `ext_privacy_hit12`: Privacy doc08 hit@12 = 4/4 (baseline=None, candidate=4/4)
- [FAIL] `ext_threat_doc12_hit4`: Threat doc12 hit@4 = 4/4 (baseline=None, candidate=2/4)
- [FAIL] `ext_threat_doc12_top4`: Threat doc12 in final top-4 = 4/4 (baseline=None, candidate=2/4)
- [PASS] `ext_threat_rank`: Threat doc12 rank not worse than doc08 = 4/4 (baseline=None, candidate=4/4)
- [PASS] `ext_negative_hit4`: Negative domain hit@4 >= 3/4 (baseline=None, candidate=4/4)
- [PASS] `ext_negative_doc08_top1`: Doc08 not top-1 in negative cases (baseline=None, candidate=0)
- [PASS] `ext_negative_doc08_top4`: Doc08 top-4 in <= 1/4 negative cases (baseline=None, candidate=0/4)

## Interpretation boundary

- T047: doc12 remains product-correct primary for direct courier threats; doc08 acceptance as parent escalation source does not require outranking doc12.
- Frozen expected primary for T047 remains 08 per frozen benchmark; extension set validates doc12 specificity separately.
- Extension threat failures where baseline already fails are not classified as doc08 regression.
