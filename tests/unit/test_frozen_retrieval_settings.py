"""Unit tests for frozen retrieval configuration."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from customer_claims_rag.application.settings import (
    FrozenRetrievalConfig,
    load_frozen_retrieval_config,
)
from customer_claims_rag.exceptions import RetrievalError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"


def _valid_file_payload(**overrides) -> dict:
    payload = {
        "experiment_id": "vector-pool-expansion-v1",
        "version": "1.0.0",
        "experiment_mode": "production-like",
        "baseline_pool_k": 12,
        "candidate_pool_k": 24,
        "final_top_k": 12,
        "threshold": 0.0,
        "reranker_id": "source-authority-v1",
        "reranker_config_hash": "c39c4608b6ed4f25bae2cda076a305c665e65155680f6778d6152dde01290957",
        "tie_breaking": [
            "rerank_score_desc",
            "baseline_similarity_desc",
            "baseline_rank_asc",
            "chunk_id_asc",
        ],
    }
    payload.update(overrides)
    return payload


def test_repository_frozen_config_loads_expected_values() -> None:
    config = load_frozen_retrieval_config(FROZEN_CONFIG_PATH)
    assert config.config_id == "vector-pool-expansion-v1"
    assert config.version == "1.0.0"
    assert config.vector_top_k == 24
    assert config.vector_fetch_k == 24
    assert config.candidate_pool_k == 24
    assert config.final_top_k == 12
    assert config.similarity_threshold == 0.0
    assert config.reranker_id == "source-authority-v1"


def test_load_equivalent_temp_json(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(_valid_file_payload()), encoding="utf-8")
    config = load_frozen_retrieval_config(path)
    assert config.config_id == "vector-pool-expansion-v1"
    assert config.vector_top_k == config.candidate_pool_k == config.vector_fetch_k


def _valid_direct_config_payload(**overrides) -> dict:
    payload = {
        "config_id": "vector-pool-expansion-v1",
        "version": "1.0.0",
        "vector_top_k": 24,
        "vector_fetch_k": 24,
        "similarity_threshold": 0.0,
        "candidate_pool_k": 24,
        "final_top_k": 12,
        "reranker_id": "source-authority-v1",
    }
    payload.update(overrides)
    return payload


def test_frozen_retrieval_config_json_serialization() -> None:
    config = load_frozen_retrieval_config(FROZEN_CONFIG_PATH)
    dumped = config.model_dump(mode="json")
    assert dumped["vector_top_k"] == 24
    assert dumped["similarity_threshold"] == 0.0


def test_frozen_retrieval_config_json_round_trip() -> None:
    original = load_frozen_retrieval_config(FROZEN_CONFIG_PATH)
    restored = FrozenRetrievalConfig.model_validate_json(original.model_dump_json())
    assert restored == original


def test_direct_valid_frozen_retrieval_config() -> None:
    config = FrozenRetrievalConfig(**_valid_direct_config_payload())
    assert config.config_id == "vector-pool-expansion-v1"
    assert config.vector_top_k == 24


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("baseline_pool_k", "12"),
        ("candidate_pool_k", "24"),
        ("final_top_k", "12"),
        ("threshold", "0.0"),
    ],
)
def test_load_rejects_numeric_strings_in_raw_json(
    tmp_path: Path,
    field_name: str,
    invalid_value: str,
) -> None:
    path = tmp_path / "numeric-string.json"
    path.write_text(
        json.dumps(_valid_file_payload(**{field_name: invalid_value})),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_frozen_retrieval_config(path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("experiment_id", 123),
        ("version", 100),
        ("reranker_id", 123),
    ],
)
def test_load_rejects_non_string_identifiers_in_raw_json(
    tmp_path: Path,
    field_name: str,
    invalid_value: int,
) -> None:
    path = tmp_path / "identifier-type.json"
    path.write_text(
        json.dumps(_valid_file_payload(**{field_name: invalid_value})),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_frozen_retrieval_config(path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("baseline_pool_k", True),
        ("candidate_pool_k", True),
        ("final_top_k", False),
        ("threshold", False),
    ],
)
def test_load_rejects_boolean_numeric_values_in_raw_json(
    tmp_path: Path,
    field_name: str,
    invalid_value: bool,
) -> None:
    path = tmp_path / "boolean-numeric.json"
    path.write_text(
        json.dumps(_valid_file_payload(**{field_name: invalid_value})),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_frozen_retrieval_config(path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("vector_top_k", "24"),
        ("vector_fetch_k", "24"),
        ("similarity_threshold", "0.0"),
        ("candidate_pool_k", "24"),
        ("final_top_k", "12"),
    ],
)
def test_direct_config_rejects_numeric_strings(
    field_name: str,
    invalid_value: str,
) -> None:
    with pytest.raises(ValidationError):
        FrozenRetrievalConfig(**_valid_direct_config_payload(**{field_name: invalid_value}))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("config_id", 123),
        ("version", 100),
        ("reranker_id", 123),
    ],
)
def test_direct_config_rejects_non_string_identifiers(
    field_name: str,
    invalid_value: int,
) -> None:
    with pytest.raises(ValidationError):
        FrozenRetrievalConfig(**_valid_direct_config_payload(**{field_name: invalid_value}))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("vector_top_k", True),
        ("candidate_pool_k", True),
        ("similarity_threshold", False),
    ],
)
def test_direct_config_rejects_boolean_numeric_values(
    field_name: str,
    invalid_value: bool,
) -> None:
    with pytest.raises(ValidationError):
        FrozenRetrievalConfig(**_valid_direct_config_payload(**{field_name: invalid_value}))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "vector_top_k": 24,
                "vector_fetch_k": 24,
                "candidate_pool_k": 24,
                "final_top_k": 25,
            },
            "final_top_k must be <= candidate_pool_k",
        ),
        (
            {
                "vector_top_k": 20,
                "vector_fetch_k": 20,
                "candidate_pool_k": 24,
                "final_top_k": 12,
            },
            "candidate_pool_k must be <= vector_top_k",
        ),
        (
            {
                "vector_top_k": 24,
                "vector_fetch_k": 20,
                "candidate_pool_k": 24,
                "final_top_k": 12,
            },
            "fetch_k must be >= top_k",
        ),
        (
            {
                "vector_top_k": 24,
                "vector_fetch_k": 24,
                "candidate_pool_k": 20,
                "final_top_k": 12,
            },
            "vector_top_k must equal candidate_pool_k",
        ),
        (
            {
                "vector_top_k": 20,
                "vector_fetch_k": 24,
                "candidate_pool_k": 20,
                "final_top_k": 12,
            },
            "vector_fetch_k must equal candidate_pool_k",
        ),
        (
            {
                "vector_top_k": 0,
                "vector_fetch_k": 0,
                "candidate_pool_k": 0,
                "final_top_k": 12,
            },
            "top_k must be >= 1",
        ),
        (
            {
                "similarity_threshold": 1.5,
            },
            "similarity_threshold must be in",
        ),
        (
            {"config_id": "   "},
            "field must be a non-empty string",
        ),
        (
            {"version": ""},
            "field must be a non-empty string",
        ),
        (
            {"reranker_id": " "},
            "field must be a non-empty string",
        ),
    ],
)
def test_frozen_retrieval_config_rejects_invalid_invariants(
    kwargs: dict,
    match: str,
) -> None:
    base = {
        "config_id": "vector-pool-expansion-v1",
        "version": "1.0.0",
        "vector_top_k": 24,
        "vector_fetch_k": 24,
        "similarity_threshold": 0.0,
        "candidate_pool_k": 24,
        "final_top_k": 12,
        "reranker_id": "source-authority-v1",
    }
    base.update(kwargs)
    with pytest.raises((ValidationError, RetrievalError, ValueError), match=match):
        FrozenRetrievalConfig(**base)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_frozen_retrieval_config(tmp_path / "missing.json")


def test_load_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_frozen_retrieval_config(path)


def test_load_missing_required_key(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.json"
    payload = _valid_file_payload()
    del payload["candidate_pool_k"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_frozen_retrieval_config(path)


def test_load_wrong_type(tmp_path: Path) -> None:
    path = tmp_path / "wrong-type.json"
    payload = _valid_file_payload(candidate_pool_k={"not": "int"})
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_frozen_retrieval_config(path)


def test_load_unknown_top_level_key(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    payload = _valid_file_payload(unexpected_field=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level"):
        load_frozen_retrieval_config(path)


def test_frozen_config_is_frozen() -> None:
    config = load_frozen_retrieval_config(FROZEN_CONFIG_PATH)
    with pytest.raises(ValidationError):
        config.final_top_k = 8  # type: ignore[misc]


def test_settings_module_does_not_import_evaluation() -> None:
    module = importlib.import_module("customer_claims_rag.application.settings")
    source_path = Path(module.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "customer_claims_rag.evaluation" not in alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "customer_claims_rag.evaluation" not in node.module
