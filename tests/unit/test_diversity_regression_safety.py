"""Regression safety tests for diversity experiment scope freeze."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.application.frozen_retrieval import FrozenRetrievalService
from customer_claims_rag.application.settings import load_frozen_retrieval_config
from customer_claims_rag.evaluation.diversity_reporting import PROTECTED_ARTIFACT_PATHS
from customer_claims_rag.evaluation.expanded_corpus_regression_reporting import (
    DEFAULT_REGRESSION_JSON,
    DEFAULT_REGRESSION_MARKDOWN,
)
from customer_claims_rag.evaluation.pool_expansion_metrics import compute_pool_expansion_config_hash
from customer_claims_rag.evaluation.pool_expansion_metrics import load_pool_expansion_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
PRODUCTION_CONFIG_HASH = "ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048"
INDEX_MANIFEST = PROJECT_ROOT / "data" / "04_index" / "manifest.json"
BACKUP_MANIFEST = PROJECT_ROOT / "data" / "04_index_backup_10docs_215chunks" / "manifest.json"
ARTIFACT = PROJECT_ROOT / "data" / "05_evaluation" / "vector_pool_36_cap4_v1.json"


def test_production_retrieval_config_unchanged() -> None:
    config = load_pool_expansion_config(PRODUCTION_CONFIG)
    assert compute_pool_expansion_config_hash(config) == PRODUCTION_CONFIG_HASH
    frozen = load_frozen_retrieval_config(PRODUCTION_CONFIG)
    assert frozen.vector_fetch_k == 24
    assert frozen.vector_top_k == 24
    assert frozen.final_top_k == 12


def test_frozen_retrieval_service_semantics_unchanged() -> None:
    source = Path(FrozenRetrievalService.__module__.replace(".", "/") + ".py")
    assert source.exists() or Path("src/customer_claims_rag/application/frozen_retrieval.py").exists()


def test_protected_historical_artifacts_listed() -> None:
    assert DEFAULT_REGRESSION_JSON.name in {path.name for path in PROTECTED_ARTIFACT_PATHS}
    assert DEFAULT_REGRESSION_MARKDOWN.name in {path.name for path in PROTECTED_ARTIFACT_PATHS}


def test_retrieval_metrics_unchanged_in_existing_artifact() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    baseline = payload["baseline_ranking"]["aggregate_metrics"]
    candidate = payload["candidate_ranking"]["aggregate_metrics"]
    assert baseline["primary_source_hit_rate_at_4"] == pytest.approx(0.6206896551724138)
    assert candidate["primary_source_hit_rate_at_4"] == pytest.approx(0.6206896551724138)
    assert baseline["hit_rate_at_12"] == pytest.approx(0.896551724137931)
    assert candidate["hit_rate_at_12"] == pytest.approx(0.9137931034482759)
    assert baseline["mrr"] == pytest.approx(0.642816091954023)
    assert candidate["mrr"] == pytest.approx(0.6456896551724138)


def test_index_manifest_fingerprints_stable() -> None:
    production = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8"))
    backup = json.loads(BACKUP_MANIFEST.read_text(encoding="utf-8"))
    assert production["chunk_count"] == 333
    assert production["document_count"] == 15
    assert backup["chunk_count"] == 215
    assert backup["document_count"] == 10
