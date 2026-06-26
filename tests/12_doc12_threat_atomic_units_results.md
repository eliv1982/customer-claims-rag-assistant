# Doc12 threat atomic units: doc12-threat-atomic-units-v1

**Timestamp:** 2026-06-26T05:15:26.334754+00:00
**Source commit:** `7135b13acd4dc21afc23067ecfd3055395f55411`
**Source dirty:** `False`
**Artifact commit:** `pending`
**Reference:** `doc08-atomic-risk-units-v1` / `candidate`
**Verdict:** `REJECTED`

## Baseline reproduction

- Passed: **True**
- Reference artifact: `data/05_evaluation/doc08_atomic_risk_units_v1.json`
- Doc08 fingerprint unchanged: **True**
- Expected doc08 fingerprint: `c208e6527facd195497d26e9efe6d8b4f487dfdea59629e603205ff1f4ea0c39`

## Retrieval arms

| Arm | Index fingerprint | fetch_k | pool_k | cap | chunks |
|-----|-------------------|--------:|-------:|----:|-------:|
| baseline | `d3c27f4a72e5f785...` | 48 | 36 | 4 | 337 |
| candidate | `9a537ca87c76692c...` | 48 | 36 | 4 | 340 |

## Frozen metrics

| Metric | Baseline (doc08 exp.) | Candidate (doc12 overlay) |
|--------|----------------------:|--------------------------:|
| Primary hit@4 | 0.637931 | 0.637931 |
| Primary hit@12 | 0.965517 | 0.965517 |
| MRR | 0.675862 | 0.692693 |
| Primary reach | 57/57 | 57/57 |
| FAQ top-4 | 36 | 35 |

## Replay stability

- Integrity verdict: `PASS — EXPERIMENT INTEGRITY REPAIRED`
- All identical: **True**
- Repeated query runs identical: True
- Repeated full runs identical: True
- Independent rebuilds identical: True
- Candidate collection digest: `0b74e08cb2ab6ecde5e0502f5d267875a8b5e5e36532b6889957260142a4fc65`
- Candidate embedding digest: `0a28c6ab83dca8096578f84377ceef613b177b01c149262736d48a9f758a31f5`

- Promoted hit@4: []
- Regressed hit@4: []
- Promoted hit@12: []
- Regressed hit@12: []

## Doc12 chunk diff

- Document: `12_staff_safety_and_threat_handling`
- Baseline doc12 chunks: 23
- Candidate doc12 chunks: 26
- Added: ['12_staff_safety_and_threat_handling::chunk-024', '12_staff_safety_and_threat_handling::chunk-025', '12_staff_safety_and_threat_handling::chunk-026']
- Removed: []
- Changed: 22
- Non-doc12 byte-identical: True

## Extension metrics

| Metric | Baseline | Candidate |
|--------|----------|-----------|
| Privacy hit@4 | 4/4 | 4/4 |
| Threat doc12 hit@4 | 2/4 | 4/4 |
| Threat doc12 in top-4 | 2/4 | 4/4 |
| Negative domain hit@4 | 4/4 | 4/4 |

## Holdout metrics

| Metric | Baseline | Candidate |
|--------|----------|-----------|
| Positive doc12 hit@4 | 4/6 | 5/6 |
| Positive doc12 hit@12 | 4/6 | 6/6 |
| Positive rank vs doc08 ok | 4/6 | 6/6 |
| Negative domain hit@4 | 3/6 | 2/6 |
| Negative doc12 top-1 | 3 | 4 |
| Negative doc12 top-4 | 5/6 | 5/6 |

## Holdout case diagnostics

| Case | Baseline hit@4 | Candidate hit@4 | Delta |
|------|----------------|-----------------|-------|
| H001 | False | True | candidate_improves |
| H002 | True | True | unchanged_pass |
| H003 | True | True | unchanged_pass |
| H004 | True | True | unchanged_pass |
| H005 | False | False | both_fail_not_doc12_regression |
| H006 | True | True | unchanged_pass |
| H007 | False | False | both_fail_not_doc12_regression |
| H008 | True | False | candidate_regression |
| H009 | True | True | unchanged_pass |
| H010 | False | False | both_fail_not_doc12_regression |
| H011 | False | False | both_fail_not_doc12_regression |
| H012 | True | True | unchanged_pass |

## Acceptance checklist

- [PASS] `reference_identity`: Reference artifact experiment identity (baseline=doc08-atomic-risk-units-v1, candidate=doc08-atomic-risk-units-v1)
- [PASS] `baseline_index_fingerprint`: Baseline index fingerprint matches doc08 experimental candidate (baseline=d3c27f4a72e5f78582c6cc29b8cb6c9f26234f6582970ead602f44e5b375bad6, candidate=d3c27f4a72e5f78582c6cc29b8cb6c9f26234f6582970ead602f44e5b375bad6)
- [PASS] `baseline_doc08_fingerprint`: Baseline doc08 fingerprint matches expected overlay (baseline=c208e6527facd195497d26e9efe6d8b4f487dfdea59629e603205ff1f4ea0c39, candidate=c208e6527facd195497d26e9efe6d8b4f487dfdea59629e603205ff1f4ea0c39)
- [PASS] `baseline_primary_hit_at_4`: Baseline primary hit@4 reproduces doc08 experimental candidate (baseline=0.6379310344827587, candidate=0.6379310344827587)
- [PASS] `baseline_primary_hit_at_12`: Baseline primary hit@12 reproduces doc08 experimental candidate (baseline=0.9655172413793104, candidate=0.9655172413793104)
- [PASS] `baseline_mrr`: Baseline MRR reproduces doc08 experimental candidate (baseline=0.6758620689655173, candidate=0.6758620689655173)
- [PASS] `baseline_primary_reach`: Baseline primary pool reach reproduces doc08 experimental candidate (baseline=57/57, candidate=57/57)
- [PASS] `baseline_high_risk_reach`: Baseline high-risk reach reproduces doc08 experimental candidate (baseline=15/15, candidate=15/15)
- [PASS] `baseline_critical_reach`: Baseline critical reach reproduces doc08 experimental candidate (baseline=8/8, candidate=8/8)
- [PASS] `baseline_primary_unreachable`: Baseline primary-unreachable set reproduces doc08 experimental candidate (baseline=[], candidate=[])
- [PASS] `baseline_fully_unreachable`: Baseline fully-unreachable set reproduces doc08 experimental candidate (baseline=[], candidate=[])
- [PASS] `baseline_faq_top4`: Baseline FAQ top-4 reproduces doc08 experimental candidate (baseline=36, candidate=36)
- [PASS] `frozen_no_hit4_regression`: No per-case primary hit@4 regression vs doc08 experimental baseline (baseline=None, candidate=[])
- [PASS] `frozen_no_hit12_regression`: No per-case primary hit@12 regression vs doc08 experimental baseline (baseline=None, candidate=[])
- [PASS] `frozen_aggregate_hit4`: Aggregate primary hit@4 >= doc08 experimental baseline (baseline=0.6379310344827587, candidate=0.6379310344827587)
- [PASS] `frozen_aggregate_hit12`: Aggregate primary hit@12 >= doc08 experimental baseline (baseline=0.9655172413793104, candidate=0.9655172413793104)
- [PASS] `ext_privacy_hit4`: Privacy doc08 hit@4 = 4/4 (baseline=None, candidate=4/4)
- [PASS] `ext_privacy_hit12`: Privacy doc08 hit@12 = 4/4 (baseline=None, candidate=4/4)
- [PASS] `ext_threat_doc12_hit4`: Threat doc12 hit@4 = 4/4 (baseline=None, candidate=4/4)
- [PASS] `ext_threat_doc12_top4`: Threat doc12 in final top-4 = 4/4 (baseline=None, candidate=4/4)
- [PASS] `ext_threat_rank`: Threat doc12 rank not worse than doc08 = 4/4 (baseline=None, candidate=4/4)
- [PASS] `ext_negative_hit4`: Negative domain hit@4 >= 3/4 (baseline=None, candidate=4/4)
- [PASS] `ext_negative_doc08_top1`: Doc08 not top-1 in negative cases (baseline=None, candidate=0)
- [PASS] `ext_negative_doc08_top4`: Doc08 top-4 in <= 1/4 negative cases (baseline=None, candidate=0/4)
- [PASS] `holdout_positive_hit4`: Holdout positive doc12 hit@4 >= 5/6 (baseline=None, candidate=5/6)
- [PASS] `holdout_positive_hit12`: Holdout positive doc12 hit@12 = 6/6 (baseline=None, candidate=6/6)
- [PASS] `holdout_positive_rank`: Holdout positive doc12 rank vs doc08 ok = 6/6 (baseline=None, candidate=6/6)
- [FAIL] `holdout_negative_hit4`: Holdout negative domain hit@4 >= 5/6 (baseline=None, candidate=2/6)
- [FAIL] `holdout_negative_doc12_top1`: Holdout negative doc12 not top-1 (baseline=None, candidate=4)
- [FAIL] `holdout_negative_doc12_top4`: Holdout negative doc12 top-4 <= 1/6 (baseline=None, candidate=5/6)
- [PASS] `doc08_overlay_unchanged`: Doc08 overlay unchanged between baseline and candidate corpora (baseline=None, candidate=True)

## Interpretation boundary

- Baseline arm is the accepted doc08 experimental candidate corpus; doc12 overlay must not mutate doc08 bytes.
- Frozen acceptance uses per-case hit@4/hit@12 non-regression against that baseline.
- Holdout threat-positive cases require doc12 to outrank or match doc08 when both appear.
- Extension threat failures where baseline already fails are not doc12 regressions.
