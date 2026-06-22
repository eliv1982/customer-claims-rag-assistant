"""Tests for offline hybrid artifact diagnostic rebuild."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from customer_claims_rag.evaluation.hybrid_metrics import (
    extract_immutability_snapshot,
    rebuild_hybrid_diagnostics,
)
from customer_claims_rag.evaluation.hybrid_models import HybridEvaluationRun
from customer_claims_rag.evaluation.hybrid_reporting import (
    DEFAULT_HYBRID_JSON,
    rebuild_hybrid_outputs,
    render_hybrid_markdown,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = PROJECT_ROOT / DEFAULT_HYBRID_JSON


def _artifact_has_exact(run: HybridEvaluationRun) -> bool:
    return all(
        case.lexical_pool is not None
        and case.lexical_pool.exact
        and case.lexical_pool.candidates
        for case in run.case_results
    )


@pytest.fixture
def frozen_run() -> HybridEvaluationRun:
    if not ARTIFACT.exists():
        pytest.skip("frozen hybrid artifact not present")
    run = HybridEvaluationRun.model_validate(
        json.loads(ARTIFACT.read_text(encoding="utf-8"))
    )
    if not _artifact_has_exact(run):
        pytest.skip("exact lexical pools not yet reconstructed")
    return run


def test_json_round_trip(frozen_run: HybridEvaluationRun) -> None:
    restored = HybridEvaluationRun.model_validate(
        json.loads(frozen_run.model_dump_json())
    )
    assert restored.experiment.config_hash == frozen_run.experiment.config_hash


def test_rebuild_preserves_immutable_experiment_snapshot(frozen_run: HybridEvaluationRun) -> None:
    before = extract_immutability_snapshot(frozen_run)
    rebuilt = rebuild_hybrid_diagnostics(frozen_run)
    after = extract_immutability_snapshot(rebuilt)
    assert before == after


def test_rebuild_updates_channel_diagnostics_only(frozen_run: HybridEvaluationRun) -> None:
    rebuilt = rebuild_hybrid_diagnostics(frozen_run)
    after_t004 = next(case for case in rebuilt.case_results if case.case_id == "T004")
    assert after_t004.candidate_fusion_pool.retrieval_channel == "lexical_only"
    assert after_t004.candidate_fusion_pool.vector_reachable is False
    assert after_t004.candidate_fusion_pool.lexical_reachable is True
    again = rebuild_hybrid_diagnostics(rebuilt)
    assert again.model_dump() == rebuilt.model_dump()


def test_t004_integration_channel_lexical_only(frozen_run: HybridEvaluationRun) -> None:
    rebuilt = rebuild_hybrid_diagnostics(frozen_run)
    case = next(item for item in rebuilt.case_results if item.case_id == "T004")
    assert case.candidate_fusion_pool.retrieval_channel == "lexical_only"


def test_rebuild_does_not_invoke_retrieval_stack(
    frozen_run: HybridEvaluationRun,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("retrieval invoked during rebuild")

    monkeypatch.setattr(
        "customer_claims_rag.retrieval.retriever.BaselineRetriever.search",
        _boom,
    )
    monkeypatch.setattr(
        "customer_claims_rag.retrieval.lexical.retriever.LexicalRetriever.search",
        _boom,
    )
    rebuild_hybrid_diagnostics(frozen_run)


def test_markdown_from_rebuilt_json(frozen_run: HybridEvaluationRun) -> None:
    rebuilt = rebuild_hybrid_diagnostics(frozen_run)
    markdown = render_hybrid_markdown(rebuilt)
    assert "## 7. Retrieval configuration selection" in markdown
    assert "rejected for MVP selection" in markdown
    assert "stage 2C.2 candidate" in markdown
    assert "normalized_rrf_score" in markdown
    assert "is **not** vector similarity" in markdown
    assert "C:\\\\" not in markdown


def test_rebuild_outputs_no_retrieval(frozen_run: HybridEvaluationRun) -> None:
    out_dir = PROJECT_ROOT / "data" / "05_evaluation" / "_pytest_hybrid_rebuild_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = out_dir / "hybrid.json"
    output_md = PROJECT_ROOT / "tests" / "_pytest_hybrid_rebuild_tmp.md"
    try:
        with patch(
            "customer_claims_rag.evaluation.hybrid_reporting.rebuild_hybrid_diagnostics",
            wraps=rebuild_hybrid_diagnostics,
        ) as rebuild_mock:
            rebuild_hybrid_outputs(
                artifact_path=ARTIFACT,
                output_json=output_json,
                output_markdown=output_md,
                project_root=PROJECT_ROOT,
            )
            rebuild_mock.assert_called_once()
        restored = HybridEvaluationRun.model_validate(
            json.loads(output_json.read_text(encoding="utf-8"))
        )
        assert restored.experiment.config_hash == frozen_run.experiment.config_hash
    finally:
        output_json.unlink(missing_ok=True)
        output_md.unlink(missing_ok=True)
