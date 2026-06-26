# doc08-atomic-risk-units-v1

Controlled corpus overlay for Stage 4C.3C-B.

Replaces only `08_escalation_and_risk_rules` in the candidate arm. Canonical
production clean corpus (`data/02_clean_markdown`) remains unchanged.

## Hypothesis

Atomic risk-trigger and escalation-route units in document 08 improve semantic
reachability for customer personal-data exposure cases without displacing
specialized document 12 for staff-threat scenarios.

## Build candidate index

```bash
build-experiment-index \
  --experiment-id doc08_atomic_risk_units_v1 \
  --overlay-document experiments/corpus/doc08_atomic_risk_units_v1/08_escalation_and_risk_rules.md \
  --index-dir data/04_index_experiments/doc08_atomic_risk_units_v1 \
  --rebuild
```

## Evaluate

```bash
evaluate-doc08-atomic-ab \
  --config configs/experiments/doc08_atomic_risk_units_v1.json \
  --baseline-index-dir data/04_index \
  --candidate-index-dir data/04_index_experiments/doc08_atomic_risk_units_v1
```
