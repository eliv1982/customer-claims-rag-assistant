"""Committed static inputs: the frozen retrieval baseline and the real-tokenizer topology.

``FROZEN_RETRIEVAL_BASELINE`` is a sanitized copy of the 60-case retrieval run that the
A/B experiments are measured against. The real file
(``data/05_evaluation/retrieval_results.json``) is gitignored, so tests that exercise
experiment logic against the baseline read this fixture instead. Provenance and the
rules that keep it honest are in ``frozen_retrieval_baseline_v1.provenance.json`` and
``tests/unit/test_frozen_baseline_fixture.py``.

The fixture must exist: a missing or malformed file is a failure, never a skip.
"""

from __future__ import annotations

import json
from pathlib import Path

from customer_claims_rag.evaluation.models import EvaluationRun

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FROZEN_RETRIEVAL_BASELINE = FIXTURES_DIR / "frozen_retrieval_baseline_v1.json"
FROZEN_RETRIEVAL_BASELINE_PROVENANCE = FIXTURES_DIR / "frozen_retrieval_baseline_v1.provenance.json"
# chunk_id -> token_count of the release corpus under real cl100k_base (real-tokenizer lane)
REAL_TOKENIZER_TOPOLOGY = FIXTURES_DIR / "real_tokenizer_release_chunk_topology_v1.json"


def load_frozen_retrieval_baseline() -> EvaluationRun:
    return EvaluationRun.model_validate(
        json.loads(FROZEN_RETRIEVAL_BASELINE.read_text(encoding="utf-8"))
    )
