# Doc08 atomic risk units: doc08-atomic-risk-units-v1

**Timestamp:** 2026-06-26T02:08:26.317450+00:00
**Source commit:** `c7459039ecfa51e3d51ea6fffcc836ca01e44413`
**Artifact commit:** `pending`
**Verdict:** `REJECTED`

## Retrieval arms

| Arm | Index fingerprint | fetch_k | pool_k | cap | chunks |
|-----|-------------------|--------:|-------:|----:|-------:|
| baseline | `b9526128dad23e71...` | 48 | 36 | 4 | 333 |
| candidate | `0b346061a20fb676...` | 48 | 36 | 4 | 337 |

## Frozen metrics

| Metric | Baseline | Candidate |
|--------|----------|-----------|
| Primary hit@4 | 0.621 | 0.638 |
| Primary hit@12 | 0.879 | 0.897 |
| MRR | 0.639 | 0.659 |

- Primary unreachable baseline: ['T004', 'T006', 'T016', 'T027', 'T039', 'T040', 'T044', 'T046', 'T047', 'T053', 'T055', 'T056', 'T060']
- Primary unreachable candidate: ['T004', 'T006', 'T016', 'T027', 'T039', 'T040', 'T046', 'T047', 'T053', 'T055', 'T056', 'T060']
- Fully unreachable candidate: ['T004', 'T027', 'T039', 'T040', 'T046', 'T047']

## Doc08 chunk diff

- Baseline doc08 chunks: 17
- Candidate doc08 chunks: 21
- Added: ['08_escalation_and_risk_rules::chunk-018', '08_escalation_and_risk_rules::chunk-019', '08_escalation_and_risk_rules::chunk-020', '08_escalation_and_risk_rules::chunk-021']
- Removed: []
- Changed: 16
- Non-doc08 byte-identical: True

## Doc08 footprint (candidate)

- Pool: 7
- Top-12: 7
- Top-4: 5

## T044 / T047 diagnostics

### T044

- Expected primary: ['08_escalation_and_risk_rules']
- Baseline vector rank doc08: []
- Candidate vector rank doc08: [2, 4, 7]
- Candidate pool rank doc08: 2
- Candidate final rank doc08: 2
- Baseline/Candidate final rank doc12: 1 / 1
- Best doc08 heading: Customer personal-data exposure — чужой адрес или фрагмент адреса
- Best doc08 similarity: 0.48783791065216064
- Top competing docs: ['12_staff_safety_and_threat_handling', '14_evidence_standards_and_incomplete_information', '12_staff_safety_and_threat_handling']
- Grounding relevant: True

### T047

- Expected primary: ['08_escalation_and_risk_rules']
- Baseline vector rank doc08: []
- Candidate vector rank doc08: []
- Candidate pool rank doc08: None
- Candidate final rank doc08: None
- Baseline/Candidate final rank doc12: 1 / 1
- Best doc08 heading: None
- Best doc08 similarity: None
- Top competing docs: ['12_staff_safety_and_threat_handling', '12_staff_safety_and_threat_handling', '12_staff_safety_and_threat_handling', '13_physical_hazard_and_foreign_body_protocol', '14_evidence_standards_and_incomplete_information']
- Grounding relevant: None

## Extension metrics (candidate)

- Privacy hit@4: 4/4
- Threat doc12 in top-4: 2/4
- Negative doc08 top-1: 0

## Acceptance checklist

- [PASS] `t044_reachable`: T044 primary reachable (baseline=None, candidate=True)
- [PASS] `t044_hit12`: T044 primary hit@12 (baseline=None, candidate=True)
- [PASS] `primary_hit_at_4`: Primary hit@4 >= baseline (baseline=0.6206896551724138, candidate=0.6379310344827587)
- [FAIL] `primary_hit_at_12`: Primary hit@12 >= baseline (baseline=None, candidate=0.896551724137931)
- [PASS] `mrr`: MRR >= 0.645 (baseline=None, candidate=0.6594827586206896)
- [PASS] `faq_top4`: FAQ top-4 <= 36 (baseline=None, candidate=36)
- [PASS] `no_new_unreachable`: No new unreachable primary (baseline=None, candidate=[])
- [PASS] `t047_doc12_top4`: T047 doc12 remains in final top-4 (baseline=None, candidate=1)
- [PASS] `ext_privacy_hit4`: Privacy doc08 hit@4 = 4/4 (baseline=None, candidate=4/4)
- [PASS] `ext_privacy_hit12`: Privacy doc08 hit@12 = 4/4 (baseline=None, candidate=4/4)
- [FAIL] `ext_threat_doc12_hit4`: Threat doc12 hit@4 = 4/4 (baseline=None, candidate=2/4)
- [FAIL] `ext_threat_doc12_top4`: Threat doc12 in final top-4 = 4/4 (baseline=None, candidate=2/4)
- [FAIL] `ext_threat_rank`: Threat doc12 rank not worse than doc08 = 4/4 (baseline=None, candidate=3/4)
- [PASS] `ext_negative_hit4`: Negative domain hit@4 >= 3/4 (baseline=None, candidate=4/4)
- [PASS] `ext_negative_doc08_top1`: Doc08 not top-1 in negative cases (baseline=None, candidate=0)
- [PASS] `ext_negative_doc08_top4`: Doc08 top-4 in <= 1/4 negative cases (baseline=None, candidate=0/4)

## Interpretation boundary

- T047: doc12 remains product-correct primary for direct courier threats; doc08 acceptance as parent escalation source does not require outranking doc12.
- Frozen expected primary for T047 remains 08 per frozen benchmark; extension set validates doc12 specificity separately.
