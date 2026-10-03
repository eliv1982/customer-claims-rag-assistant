"""Integrity tests for doc08 experiment R1 repairs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.env_bootstrap import project_root
from customer_claims_rag.evaluation.doc08_atomic_contract import (
    DEFAULT_REFERENCE_ARTIFACT,
    THREAT_EXTENSION_IDS,
    classify_threat_case_delta,
    ensure_clean_source_tree,
    load_baseline_reference_oracle,
)
from customer_claims_rag.evaluation.doc08_atomic_evaluator import DirtySourceTreeError
from customer_claims_rag.evaluation.git_state import GitState
from customer_claims_rag.evaluation.models import CaseResult, RetrievedChunkResult
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.ingestion.corpus_overlay import (
    DOC08_DOCUMENT_ID,
    build_baseline_and_overlay_chunks,
    cleanup_overlay_temp_dir,
    compute_overlay_corpus_fingerprint,
    logical_source_path,
    normalize_overlay_corpus_source_paths,
)
from customer_claims_rag.retrieval.manifest import load_manifest
from tests.local_artifacts import requires_local_artifacts

ROOT = project_root()
OVERLAY = ROOT / "experiments/corpus/doc08_atomic_risk_units_v1/08_escalation_and_risk_rules.md"
CANONICAL = ROOT / "data/02_clean_markdown"
PRODUCTION_INDEX = ROOT / "data/04_index"
REFERENCE = ROOT / DEFAULT_REFERENCE_ARTIFACT
FROZEN_Q = ROOT / "tests/01_test_questions.md"
FROZEN_E = ROOT / "tests/02_expected_answers.md"


def _chunk_result(document_id: str, rank: int) -> RetrievedChunkResult:
    return RetrievedChunkResult(
        rank=rank,
        chunk_id=f"{document_id}::chunk-001",
        document_id=document_id,
        source_path=f"data/02_clean_markdown/{document_id}.md",
        similarity=0.5,
        distance=0.5,
        heading="h",
    )


def _case_result(*, test_id: str, docs: list[str], primary_hit_at_4: bool) -> CaseResult:
    chunks = [_chunk_result(doc_id, index + 1) for index, doc_id in enumerate(docs)]
    return CaseResult(
        test_id=test_id,
        query="q",
        expected_risk="high",
        expected_primary_documents=["12_staff_safety_and_threat_handling"],
        expected_supporting_documents=["08_escalation_and_risk_rules"],
        fallback_expected=False,
        category="procedure",
        retrieved_chunks=chunks,
        retrieved_document_ids=list(dict.fromkeys(docs)),
        retrieved_document_ids_at_4=list(dict.fromkeys(docs))[:4],
        primary_hit_at_4=primary_hit_at_4,
        hit_at_12=True,
        status="success",
    )


def test_overlay_fingerprint_stable_across_temp_roots(corpus_sandbox: Path) -> None:
    canonical = corpus_sandbox / CANONICAL.relative_to(ROOT)
    overlay = corpus_sandbox / OVERLAY.relative_to(ROOT)
    temp_a = corpus_sandbox / ".tmp" / "corpus_overlay" / "_repro_test_a"
    temp_b = corpus_sandbox / ".tmp" / "corpus_overlay" / "_repro_test_b"
    temp_a.mkdir(parents=True, exist_ok=True)
    temp_b.mkdir(parents=True, exist_ok=True)
    build_a = build_baseline_and_overlay_chunks(
        canonical_dir=canonical,
        overlay_document_path=overlay,
        permitted_root=corpus_sandbox,
        staging_parent=temp_a,
    )
    build_b = build_baseline_and_overlay_chunks(
        canonical_dir=canonical,
        overlay_document_path=overlay,
        permitted_root=corpus_sandbox,
        staging_parent=temp_b,
    )
    try:
        fp_a = compute_overlay_corpus_fingerprint(build_a.candidate_chunks)
        fp_b = compute_overlay_corpus_fingerprint(build_b.candidate_chunks)
        ids_a = [chunk.chunk_id for chunk in build_a.candidate_doc08_chunks]
        ids_b = [chunk.chunk_id for chunk in build_b.candidate_doc08_chunks]
        paths_a = {chunk.chunk_id: chunk.source_path for chunk in build_a.candidate_chunks}
        paths_b = {chunk.chunk_id: chunk.source_path for chunk in build_b.candidate_chunks}
        assert fp_a == fp_b
        assert ids_a == ids_b
        assert paths_a == paths_b
        assert all(path.startswith("data/02_clean_markdown/") for path in paths_a.values())
    finally:
        cleanup_overlay_temp_dir(build_a.temp_input_dir)
        cleanup_overlay_temp_dir(build_b.temp_input_dir)


def test_overlay_content_change_changes_fingerprint(corpus_sandbox: Path) -> None:
    build = build_baseline_and_overlay_chunks(
        canonical_dir=corpus_sandbox / CANONICAL.relative_to(ROOT),
        overlay_document_path=corpus_sandbox / OVERLAY.relative_to(ROOT),
        permitted_root=corpus_sandbox,
    )
    try:
        base_fp = compute_overlay_corpus_fingerprint(build.candidate_chunks)
        mutated = normalize_overlay_corpus_source_paths(build.candidate_chunks)
        doc08 = next(chunk for chunk in mutated if chunk.document_id == DOC08_DOCUMENT_ID)
        changed = doc08.model_copy(update={"content": doc08.content + "\n"})
        changed_chunks = [
            changed if chunk.document_id == DOC08_DOCUMENT_ID else chunk
            for chunk in mutated
        ]
        changed_fp = compute_overlay_corpus_fingerprint(changed_chunks)
        assert base_fp != changed_fp
    finally:
        cleanup_overlay_temp_dir(build.temp_input_dir)


def test_logical_source_path_is_platform_independent() -> None:
    assert logical_source_path(DOC08_DOCUMENT_ID) == (
        "data/02_clean_markdown/08_escalation_and_risk_rules.md"
    )


@requires_local_artifacts(
    PRODUCTION_INDEX / "manifest.json",
    why="manifest of the provisioned production index (built with live OpenAI embeddings)",
)
def test_production_fingerprint_unchanged() -> None:
    manifest = load_manifest(PRODUCTION_INDEX)
    assert manifest.corpus_fingerprint == (
        "b9526128dad23e71e812fbd8f26452b8d6efa29e4faac898fa11f279b310e827"
    )


def test_reference_oracle_identity() -> None:
    oracle = load_baseline_reference_oracle(REFERENCE)
    assert oracle.experiment_id == "vector-pool-36-cap4-v1"
    assert oracle.reference_arm == "candidate"
    assert oracle.primary_hit_at_4 == pytest.approx(0.6206896551724138)
    assert oracle.primary_hit_at_12 == pytest.approx(0.9137931034482759)
    assert oracle.mrr == pytest.approx(0.6456896551724138)
    assert oracle.reachability.candidate_primary_reachable == 55
    assert oracle.reachability.candidate_primary_total == 57
    assert list(oracle.primary_unreachable_case_ids) == ["T044", "T047"]
    assert list(oracle.fully_unreachable_case_ids) == ["T047"]


def test_frozen_denominator_is_57_for_primary_reach() -> None:
    cases = load_evaluation_corpus(questions_path=FROZEN_Q, expected_path=FROZEN_E)
    eligible = [
        case
        for case in cases
        if not case.fallback_expected and case.expected_primary_documents
    ]
    assert len(eligible) == 57


def test_threat_delta_both_fail_not_doc08_regression() -> None:
    baseline = _case_result(
        test_id="E007",
        docs=["08_escalation_and_risk_rules", "12_staff_safety_and_threat_handling"],
        primary_hit_at_4=False,
    )
    candidate = _case_result(
        test_id="E007",
        docs=["08_escalation_and_risk_rules", "12_staff_safety_and_threat_handling"],
        primary_hit_at_4=False,
    )
    assert (
        classify_threat_case_delta(case_id="E007", baseline=baseline, candidate=candidate)
        == "both_fail_not_doc08_regression"
    )


def test_threat_delta_candidate_regression() -> None:
    baseline = _case_result(
        test_id="E005",
        docs=["12_staff_safety_and_threat_handling"],
        primary_hit_at_4=True,
    )
    candidate = _case_result(
        test_id="E005",
        docs=["08_escalation_and_risk_rules"],
        primary_hit_at_4=False,
    )
    assert (
        classify_threat_case_delta(case_id="E005", baseline=baseline, candidate=candidate)
        == "candidate_regression"
    )


def test_evaluator_rejects_dirty_source_tree() -> None:
    with pytest.raises(DirtySourceTreeError):
        ensure_clean_source_tree(
            GitState(commit="abc", dirty=True, status_summary="1 changed path(s)"),
            allow_dirty=False,
        )


def test_reference_artifact_has_candidate_ranking_arm() -> None:
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert payload["candidate_ranking"]["aggregate_metrics"]["hit_rate_at_12"] == pytest.approx(
        0.9137931034482759
    )


def test_extension_threat_ids_include_e007_e008() -> None:
    assert "E007" in THREAT_EXTENSION_IDS
    assert "E008" in THREAT_EXTENSION_IDS
