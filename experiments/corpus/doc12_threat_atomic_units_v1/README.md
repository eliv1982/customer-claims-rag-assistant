# doc12-threat-atomic-units-v1

Controlled stacked corpus overlay for Stage 4C.3D-B.

**Baseline arm:** doc08 experimental corpus (`doc08_atomic_risk_units_v1` index).

**Candidate arm:** same doc08 overlay + doc12 threat atomic overlay only.

Production index `data/04_index` is immutable reference only.

## Build candidate index

```bash
build-doc12-threat-index --rebuild
```

## Evaluate

```bash
evaluate-doc12-threat-atomic-ab \
  --config configs/experiments/doc12_threat_atomic_units_v1.json
```
