"""Evaluation reporting tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.evaluation.metrics import (
    compute_case_metrics,
    primary_found_in_top12_outside_top4,
)
from customer_claims_rag.evaluation.reporting import (
    render_improvement_log_markdown,
    render_results_markdown,
    write_evaluation_outputs,
)
from customer_claims_rag.evaluation.result_id import compute_evaluation_result_id
from customer_claims_rag.exceptions import EvaluationOutputError, RetrievalError
from tests.evaluation_helpers import make_fake_evaluation_run, make_retrieved_chunk
from tests.frozen_fixtures import load_frozen_retrieval_baseline


def test_json_serializable_and_no_absolute_user_paths(temp_project: Path) -> None:
    run = make_fake_evaluation_run()
    json_path, _, _ = write_evaluation_outputs(
        run,
        output_json=temp_project / "data" / "05_evaluation" / "retrieval_results.json",
        output_results=temp_project / "tests" / "03_test_results.md",
        output_improvement_log=temp_project / "tests" / "04_improvement_log.md",
        project_root=temp_project.resolve(),
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_metadata"]["evaluation_result_id"] == run.run_metadata.evaluation_result_id
    serialized = json.dumps(payload)
    assert "C:\\Users" not in serialized
    assert "OPENAI_API_KEY" not in serialized


def test_markdown_contains_required_sections() -> None:
    run = make_fake_evaluation_run()
    markdown = render_results_markdown(run)
    for section in (
        "**Evaluation result ID:**",
        "**Working tree dirty:**",
        "## Aggregate metrics",
        "Supporting source hit@4",
        "## Threshold sweep",
        "retrieval-only evaluation",
    ):
        assert section in markdown


def test_result_id_matches_across_artifacts(temp_project: Path) -> None:
    run = make_fake_evaluation_run()
    result_id = compute_evaluation_result_id(run)
    run = run.model_copy(
        update={
            "run_metadata": run.run_metadata.model_copy(
                update={"evaluation_result_id": result_id},
            )
        }
    )
    json_path, results_path, log_path = write_evaluation_outputs(
        run,
        output_json=temp_project / "data" / "05_evaluation" / "retrieval_results.json",
        output_results=temp_project / "tests" / "03_test_results.md",
        output_improvement_log=temp_project / "tests" / "04_improvement_log.md",
        project_root=temp_project.resolve(),
    )
    assert result_id in json_path.read_text(encoding="utf-8")
    assert result_id in results_path.read_text(encoding="utf-8")
    assert result_id in log_path.read_text(encoding="utf-8")


def test_rejects_output_inside_clean_markdown(temp_project: Path) -> None:
    run = make_fake_evaluation_run()
    with pytest.raises(RetrievalError, match="clean Markdown"):
        write_evaluation_outputs(
            run,
            output_json=temp_project / "data" / "02_clean_markdown" / "bad.json",
            output_results=temp_project / "tests" / "03_test_results.md",
            output_improvement_log=temp_project / "tests" / "04_improvement_log.md",
            project_root=temp_project.resolve(),
        )


def test_partial_write_raises_output_error(temp_project: Path, monkeypatch) -> None:
    run = make_fake_evaluation_run()
    original_atomic = __import__(
        "customer_claims_rag.evaluation.reporting",
        fromlist=["_atomic_write_text"],
    )._atomic_write_text
    calls = {"count": 0}

    def flaky_atomic(path, content, *, temp_paths):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("write failed")
        return original_atomic(path, content, temp_paths=temp_paths)

    monkeypatch.setattr(
        "customer_claims_rag.evaluation.reporting._atomic_write_text",
        flaky_atomic,
    )
    with pytest.raises(EvaluationOutputError, match="partially updated"):
        write_evaluation_outputs(
            run,
            output_json=temp_project / "data" / "05_evaluation" / "retrieval_results.json",
            output_results=temp_project / "tests" / "03_test_results.md",
            output_improvement_log=temp_project / "tests" / "04_improvement_log.md",
            project_root=temp_project.resolve(),
        )


def test_improvement_log_contains_policy() -> None:
    markdown = render_improvement_log_markdown(make_fake_evaluation_run())
    assert "A/B rerun" in markdown
    assert "evaluation_result_id" not in markdown.lower() or "Evaluation result ID" in markdown


def _chunks_for_primary_annotation_test(
    *,
    primary_in_top4: bool,
    primary_in_5_12: bool,
    supporting_in_5_12: bool,
) -> list:
    chunks = [
        make_retrieved_chunk(rank=1, document_id="10_customer_faq", similarity=0.9),
        make_retrieved_chunk(rank=2, document_id="10_customer_faq", similarity=0.85),
        make_retrieved_chunk(rank=3, document_id="10_customer_faq", similarity=0.8),
    ]
    if primary_in_top4:
        chunks.append(make_retrieved_chunk(rank=4, document_id="01_service_overview", similarity=0.75))
    else:
        chunks.append(make_retrieved_chunk(rank=4, document_id="10_customer_faq", similarity=0.74))
    for rank in range(5, 13):
        document_id = "10_customer_faq"
        if rank == 6 and primary_in_5_12 and not primary_in_top4:
            document_id = "01_service_overview"
        elif rank == 7 and supporting_in_5_12:
            document_id = "02_delivery_rules"
        chunks.append(make_retrieved_chunk(rank=rank, document_id=document_id, similarity=0.7))
    return chunks


def test_primary_annotation_absent_when_only_supporting_in_deep_window() -> None:
    chunks = _chunks_for_primary_annotation_test(
        primary_in_top4=False,
        primary_in_5_12=False,
        supporting_in_5_12=True,
    )
    case = compute_case_metrics(
        test_id="T999",
        query="q",
        expected_risk="medium",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=["02_delivery_rules"],
        fallback_expected=False,
        category="general",
        retrieved_chunks=chunks,
    )
    assert primary_found_in_top12_outside_top4(chunks, case.expected_primary_documents) is False
    markdown = render_results_markdown(
        make_fake_evaluation_run().model_copy(update={"case_results": [case]}),
    )
    assert "primary source found in top-12 but outside top-4" not in markdown


def test_primary_annotation_present_when_primary_in_deep_window() -> None:
    chunks = _chunks_for_primary_annotation_test(
        primary_in_top4=False,
        primary_in_5_12=True,
        supporting_in_5_12=False,
    )
    case = compute_case_metrics(
        test_id="T998",
        query="q",
        expected_risk="medium",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=[],
        fallback_expected=False,
        category="general",
        retrieved_chunks=chunks,
    )
    assert primary_found_in_top12_outside_top4(chunks, case.expected_primary_documents) is True
    markdown = render_results_markdown(
        make_fake_evaluation_run().model_copy(update={"case_results": [case]}),
    )
    assert "primary source found in top-12 but outside top-4" in markdown


def test_primary_hit_at_4_excluded_from_failed_cases_section() -> None:
    chunks = _chunks_for_primary_annotation_test(
        primary_in_top4=True,
        primary_in_5_12=False,
        supporting_in_5_12=False,
    )
    case = compute_case_metrics(
        test_id="T997",
        query="q",
        expected_risk="low",
        expected_primary_documents=["01_service_overview"],
        expected_supporting_documents=[],
        fallback_expected=False,
        category="general",
        retrieved_chunks=chunks,
    )
    assert case.primary_hit_at_4 is True
    markdown = render_results_markdown(
        make_fake_evaluation_run().model_copy(update={"case_results": [case]}),
    )
    assert "## Failed cases (primary source miss@4)" in markdown
    assert "_No primary source miss@4 among source-recall cases._" in markdown


def test_t016_and_t044_render_without_false_primary_annotation() -> None:
    # The frozen 60-case run is committed (tests/fixtures), so this runs on a fresh clone.
    run = load_frozen_retrieval_baseline()
    markdown = render_results_markdown(run)
    for test_id in ("T016", "T044"):
        line = next(line for line in markdown.splitlines() if line.startswith(f"- **{test_id}**"))
        assert "primary source found in top-12 but outside top-4" not in line


def test_third_output_replace_failure(temp_project: Path, monkeypatch) -> None:
    run = make_fake_evaluation_run()
    original_atomic = __import__(
        "customer_claims_rag.evaluation.reporting",
        fromlist=["_atomic_write_text"],
    )._atomic_write_text
    calls = {"count": 0}

    def flaky_atomic(path, content, *, temp_paths):
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("write failed")
        return original_atomic(path, content, temp_paths=temp_paths)

    monkeypatch.setattr(
        "customer_claims_rag.evaluation.reporting._atomic_write_text",
        flaky_atomic,
    )

    json_path = temp_project / "data" / "05_evaluation" / "retrieval_results.json"
    results_path = temp_project / "tests" / "03_test_results.md"
    improvement_path = temp_project / "tests" / "04_improvement_log.md"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    improvement_path.write_text("PRE-RUN\n", encoding="utf-8")

    with pytest.raises(EvaluationOutputError, match="partially updated"):
        write_evaluation_outputs(
            run,
            output_json=json_path,
            output_results=results_path,
            output_improvement_log=improvement_path,
            project_root=temp_project.resolve(),
        )

    assert json_path.exists()
    json.loads(json_path.read_text(encoding="utf-8"))
    assert "Evaluation result ID" in results_path.read_text(encoding="utf-8")
    assert improvement_path.read_text(encoding="utf-8") == "PRE-RUN\n"
    assert run.run_metadata.evaluation_result_id in json_path.read_text(encoding="utf-8")
    assert run.run_metadata.evaluation_result_id in results_path.read_text(encoding="utf-8")
    assert not list(results_path.parent.glob("*.tmp"))
    assert not list(json_path.parent.glob("*.tmp"))
