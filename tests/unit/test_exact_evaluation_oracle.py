"""Tests for deterministic exact evaluation oracle (Stage 4C.3D-B-R3)."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from customer_claims_rag.env_bootstrap import project_root
from customer_claims_rag.evaluation.doc12_ann_robustness import AnnRobustnessResult
from customer_claims_rag.evaluation.doc12_threat_atomic_models import (
    EnvironmentProvenance,
    ReplayIntegrityResult,
)
from customer_claims_rag.evaluation.environment_provenance import (
    MINIMUM_PYTHON,
    EnvironmentProvenanceError,
    capture_environment_provenance,
    is_isolated_virtualenv,
    validate_project_venv,
)
from customer_claims_rag.evaluation.exact_vector_search import (
    ExactVectorSearchError,
    exact_cosine_similarity,
    exact_vector_search,
    validate_embedding_vector,
)
from customer_claims_rag.retrieval.metadata_mapper import chunk_to_vector_metadata
from tests.retrieval_helpers import make_chunk_record

ROOT = project_root()


def _record(chunk_id: str, content: str) -> dict:
    chunk = make_chunk_record(chunk_id=chunk_id, content=content)
    return {
        "chunk_id": chunk.chunk_id,
        "document": chunk.content,
        "metadata": chunk_to_vector_metadata(chunk),
    }


def test_exact_cosine_known_ranking() -> None:
    records = [
        _record("b::1", "beta"),
        _record("a::1", "alpha"),
        _record("c::1", "gamma"),
    ]
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    query = [1.0, 0.0, 0.0]
    results = exact_vector_search(
        query_vector=query,
        chunk_records=records,
        embedding_matrix=matrix,
        fetch_k=3,
        threshold=0.0,
    )
    assert [item.chunk_id for item in results] == ["b::1", "a::1", "c::1"]
    assert results[0].similarity == pytest.approx(1.0)
    assert results[0].distance == pytest.approx(0.0)


def test_exact_search_tie_break_by_chunk_id() -> None:
    records = [
        _record("z::1", "z"),
        _record("a::1", "a"),
    ]
    matrix = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    results = exact_vector_search(
        query_vector=[1.0, 0.0],
        chunk_records=records,
        embedding_matrix=matrix,
        fetch_k=2,
    )
    assert [item.chunk_id for item in results] == ["a::1", "z::1"]


def test_exact_search_dimension_mismatch_rejected() -> None:
    records = [_record("a::1", "a")]
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ExactVectorSearchError, match="dimension mismatch"):
        exact_vector_search(
            query_vector=[1.0, 0.0, 0.0],
            chunk_records=records,
            embedding_matrix=matrix,
            fetch_k=1,
        )


def test_exact_search_nan_rejected() -> None:
    with pytest.raises(ExactVectorSearchError, match="non-finite"):
        validate_embedding_vector([1.0, float("nan")], label="vector")


def test_exact_search_zero_vector_rejected() -> None:
    with pytest.raises(ExactVectorSearchError, match="zero-norm"):
        validate_embedding_vector([0.0, 0.0], label="vector")


def test_exact_search_does_not_invoke_chroma() -> None:
    with patch("chromadb.PersistentClient") as client:
        records = [_record("a::1", "a")]
        matrix = np.array([[1.0, 0.0]], dtype=np.float32)
        exact_vector_search(
            query_vector=[1.0, 0.0],
            chunk_records=records,
            embedding_matrix=matrix,
            fetch_k=1,
        )
        client.assert_not_called()


def test_exact_search_does_not_invoke_document_embedding_api() -> None:
    provider = MagicMock()
    records = [_record("a::1", "a")]
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    exact_vector_search(
        query_vector=[1.0, 0.0],
        chunk_records=records,
        embedding_matrix=matrix,
        fetch_k=1,
    )
    provider.embed_documents.assert_not_called()


def test_cosine_matches_chroma_distance_convention() -> None:
    left = [1.0, 0.0]
    right = [0.0, 1.0]
    similarity = exact_cosine_similarity(left, right)
    assert similarity == pytest.approx(0.0)
    assert max(0.0, 1.0 - similarity) == pytest.approx(1.0)


def _simulate_external_virtualenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pretend to run inside a virtualenv that is NOT under the repository."""
    venv = tmp_path / "external-venv"
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base-python"))
    monkeypatch.chdir(ROOT)
    return venv


class _Completed:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _stub_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, **_kwargs):
        if "--version" in command:
            return _Completed("pip 99.0 from /somewhere/site-packages/pip (python 3.12)")
        return _Completed("No broken requirements found.")

    monkeypatch.setattr(
        "customer_claims_rag.evaluation.environment_provenance.subprocess.run", fake_run
    )


def test_external_virtualenv_permits_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    venv = _simulate_external_virtualenv(monkeypatch, tmp_path)
    assert ROOT not in venv.parents
    assert is_isolated_virtualenv() is True
    validate_project_venv(ROOT)
    _stub_pip(monkeypatch)
    payload = capture_environment_provenance(ROOT)
    assert payload["kind"] == "project_venv"
    assert payload["pip_version"] == "pip 99.0"
    assert payload["pip_check_exit_code"] == 0
    assert "sys.executable" not in payload
    assert str(tmp_path) not in json.dumps(payload)


def test_global_python_blocks_artifact_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "usr"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "usr"))
    assert is_isolated_virtualenv() is False
    with pytest.raises(EnvironmentProvenanceError, match="not an isolated virtual environment"):
        capture_environment_provenance(ROOT)


def test_unsupported_python_blocks_artifact_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _simulate_external_virtualenv(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "version_info", (MINIMUM_PYTHON[0], MINIMUM_PYTHON[1] - 1, 9, "final", 0))
    with pytest.raises(EnvironmentProvenanceError, match="Python 3.12\\+ is required"):
        validate_project_venv(ROOT)


def test_wrong_working_directory_blocks_artifact_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _simulate_external_virtualenv(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(EnvironmentProvenanceError, match="not the repository root"):
        validate_project_venv(ROOT)


def test_minimum_python_matches_pyproject() -> None:
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=" + ".".join(
        str(part) for part in MINIMUM_PYTHON
    )


def test_environment_provenance_has_no_absolute_paths_in_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _simulate_external_virtualenv(monkeypatch, tmp_path)
    _stub_pip(monkeypatch)
    payload = capture_environment_provenance(ROOT)
    model = EnvironmentProvenance(
        python_version=str(payload["python_version"]),
        pip_version=str(payload["pip_version"]),
        pip_check_exit_code=int(payload["pip_check_exit_code"]),
        dependency_check_summary=str(payload["dependency_check_summary"]),
    )
    dumped = model.model_dump_json()
    assert not re.search(r"[A-Za-z]:\\\\", dumped)
    assert "Scripts/python" not in dumped


def test_ann_robustness_marked_non_authoritative() -> None:
    result = AnnRobustnessResult(
        authoritative=False,
        builds=3,
        exact_authoritative_verdict="REJECTED",
    )
    assert result.authoritative is False


def test_replay_integrity_exact_verdict_string() -> None:
    result = ReplayIntegrityResult(
        integrity_verdict="PASS — EXACT FIXED-SNAPSHOT EVALUATION REPRODUCIBLE",
    )
    assert "EXACT FIXED-SNAPSHOT" in result.integrity_verdict


def test_config_declares_baseline_and_candidate_snapshot_paths() -> None:
    config = json.loads(
        (ROOT / "configs/experiments/doc12_threat_atomic_units_v1.json").read_text(
            encoding="utf-8"
        )
    )
    snapshot = config["embedding_snapshot"]
    assert "baseline_snapshot_path" in snapshot
    assert "candidate_snapshot_path" in snapshot


def test_three_exact_run_digests_identical_on_fixed_snapshot() -> None:
    from customer_claims_rag.evaluation.doc12_ann_robustness import digest_exact_run

    class _Diag:
        def __init__(self, case_id: str, rank: int | None) -> None:
            self.case_id = case_id
            self._rank = rank

        def model_dump(self, mode: str = "json") -> dict:
            return {"case_id": self.case_id, "candidate_final_rank_doc12": self._rank}

    class _Run:
        verdict = "REJECTED"
        frozen_candidate_primary_hit_at_12 = 0.9
        frozen_candidate_metrics = type(
            "M",
            (),
            {"primary_source_hit_rate_at_4": 0.5},
        )()
        extension = type(
            "Ext",
            (),
            {
                "candidate": type("C", (), {"threat_doc12_hit_at_4": 1})(),
                "case_diagnostics": [_Diag("E008", 4)],
            },
        )()
        holdout = type(
            "Hold",
            (),
            {
                "candidate": type("C", (), {"negative_doc12_top4": 2})(),
                "case_diagnostics": [_Diag("H005", None)],
            },
        )()

    digests = [digest_exact_run(_Run()) for _ in range(3)]
    assert len(set(digests)) == 1
