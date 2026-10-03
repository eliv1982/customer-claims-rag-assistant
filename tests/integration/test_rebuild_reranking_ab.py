"""Unit and integration tests for A/B artifact round-trip and rebuild."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from customer_claims_rag.cli.rebuild_reranking_ab_report import run_rebuild
from customer_claims_rag.evaluation.ab_models import AbEvaluationRun
from customer_claims_rag.evaluation.ab_rebuild import (
    artifact_content_hash,
    load_ab_artifact,
    rebuild_ab_artifact,
    snapshot_immutable_fields,
)
from customer_claims_rag.evaluation.ab_reporting import render_ab_markdown
from customer_claims_rag.evaluation.models import EvaluationRun
from tests.frozen_fixtures import FROZEN_RETRIEVAL_BASELINE as FROZEN_BASELINE
from tests.frozen_fixtures import load_frozen_retrieval_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Tracked artifact: a clone always has it, so a missing file is a failure, not a skip.
FROZEN_AB = PROJECT_ROOT / "data" / "05_evaluation" / "reranking_ab_source_authority_v1.json"


@pytest.fixture
def frozen_ab_run() -> AbEvaluationRun:
    return load_ab_artifact(FROZEN_AB)


@pytest.fixture
def frozen_baseline_run() -> EvaluationRun:
    return load_frozen_retrieval_baseline()


def test_json_round_trip(frozen_ab_run: AbEvaluationRun) -> None:
    payload = frozen_ab_run.model_dump(mode="json")
    restored = AbEvaluationRun.model_validate(payload)
    assert restored.model_dump(mode="json") == payload


def test_baseline_identity_present_after_rebuild(
    frozen_ab_run: AbEvaluationRun,
    frozen_baseline_run: EvaluationRun,
) -> None:
    rebuilt = rebuild_ab_artifact(frozen_ab_run, frozen_baseline_run)
    assert rebuilt.baseline_identity.case_count == 60
    assert rebuilt.baseline_identity.identity_status == "equivalent_with_embedding_drift"
    assert rebuilt.baseline_identity.substantive_mismatch_count == 0


def test_markdown_renders_from_json_only(frozen_ab_run: AbEvaluationRun) -> None:
    markdown = render_ab_markdown(frozen_ab_run)
    assert "Baseline identity" in markdown or "## 2. Baseline identity" in markdown
    assert "exact frozen baseline identity confirmed" not in markdown.lower()
    assert "maximum bonus zone" in markdown


def test_rebuild_does_not_call_retriever(
    frozen_ab_run: AbEvaluationRun,
    frozen_baseline_run: EvaluationRun,
    tmp_path: Path,
) -> None:
    blocked = MagicMock(side_effect=AssertionError("retriever must not be called"))
    with patch(
        "customer_claims_rag.retrieval.factory.create_embedding_provider",
        blocked,
    ), patch(
        "customer_claims_rag.retrieval.factory.create_vector_store",
        blocked,
    ):
        rebuilt = rebuild_ab_artifact(frozen_ab_run, frozen_baseline_run)
    assert rebuilt.comparison.acceptance.candidate_accepted is True


def test_rebuild_cli_without_retrieval(tmp_path: Path) -> None:
    output_json = tmp_path / "rebuilt.json"
    output_md = tmp_path / "report.md"
    blocked = MagicMock(side_effect=AssertionError("retrieval blocked"))
    with patch(
        "customer_claims_rag.retrieval.factory.create_embedding_provider",
        blocked,
    ), patch(
        "customer_claims_rag.retrieval.factory.create_vector_store",
        blocked,
    ):
        code, rebuilt = run_rebuild(
            artifact_path=FROZEN_AB,
            frozen_baseline_path=FROZEN_BASELINE,
            output_json=output_json,
            output_markdown=output_md,
            project_root_path=PROJECT_ROOT,
        )
    assert code == 0
    assert output_json.exists()
    assert output_md.exists()
    assert rebuilt is not None


def test_config_hash_unchanged_by_rebuild(
    frozen_ab_run: AbEvaluationRun,
    frozen_baseline_run: EvaluationRun,
) -> None:
    before = frozen_ab_run.experiment.config_hash
    rebuilt = rebuild_ab_artifact(frozen_ab_run, frozen_baseline_run)
    assert rebuilt.experiment.config_hash == before


def test_candidate_order_unchanged_by_rebuild(
    frozen_ab_run: AbEvaluationRun,
    frozen_baseline_run: EvaluationRun,
) -> None:
    before = snapshot_immutable_fields(frozen_ab_run)
    rebuilt = rebuild_ab_artifact(frozen_ab_run, frozen_baseline_run)
    after = snapshot_immutable_fields(rebuilt)
    assert before == after


def test_artifact_has_no_secrets_or_absolute_paths(
    frozen_ab_run: AbEvaluationRun,
    frozen_baseline_run: EvaluationRun,
) -> None:
    rebuilt = rebuild_ab_artifact(frozen_ab_run, frozen_baseline_run)
    text = json.dumps(rebuilt.model_dump(mode="json"))
    assert "OPENAI" not in text
    assert "api_key" not in text.lower()
    assert re.search(r"[A-Za-z]:\\\\Users\\\\", text) is None
    assert "/Users/" not in text


def test_report_lists_unreachable_cases(
    frozen_ab_run: AbEvaluationRun,
    frozen_baseline_run: EvaluationRun,
) -> None:
    rebuilt = rebuild_ab_artifact(frozen_ab_run, frozen_baseline_run)
    markdown = render_ab_markdown(rebuilt)
    assert "T040" in markdown
    assert "T047" in markdown
    for case_id in ("T004", "T040", "T047"):
        case = next(item for item in rebuilt.case_results if item.case_id == case_id)
        if case.comparison.reranker_not_applicable:
            assert case_id in markdown


def test_report_t040_t047_critical_section(
    frozen_ab_run: AbEvaluationRun,
    frozen_baseline_run: EvaluationRun,
) -> None:
    markdown = render_ab_markdown(rebuild_ab_artifact(frozen_ab_run, frozen_baseline_run))
    assert "## 11. T040 analysis" in markdown
    assert "## 12. T047 analysis" in markdown
