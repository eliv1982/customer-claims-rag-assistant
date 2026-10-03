"""Tests for doc08 atomic corpus experiment infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.env_bootstrap import project_root
from customer_claims_rag.evaluation.diversity_metrics import compute_vector_pool_cap_config_hash, load_vector_pool_cap_config
from customer_claims_rag.evaluation.extension_parser import (
    compute_extension_dataset_fingerprint,
    load_extension_corpus,
)
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.ingestion.corpus_overlay import (
    DOC08_DOCUMENT_ID,
    build_baseline_and_overlay_chunks,
    cleanup_overlay_temp_dir,
    compute_doc08_chunk_diff,
    compute_doc08_fingerprint,
    compute_overlay_corpus_fingerprint,
    logical_source_path,
)
from customer_claims_rag.retrieval.manifest import load_manifest
from tests.local_artifacts import requires_local_artifacts

ROOT = project_root()
OVERLAY = ROOT / "experiments/corpus/doc08_atomic_risk_units_v1/08_escalation_and_risk_rules.md"
CANONICAL = ROOT / "data/02_clean_markdown"
PRODUCTION_INDEX = ROOT / "data/04_index"
CANDIDATE_INDEX = ROOT / "data/04_index_experiments/doc08_atomic_risk_units_v1"
CONFIG = ROOT / "configs/experiments/doc08_atomic_risk_units_v1.json"
FROZEN_Q = ROOT / "tests/01_test_questions.md"
FROZEN_E = ROOT / "tests/02_expected_answers.md"
EXT_Q = ROOT / "tests/extension/doc08_atomic_extension_v1_questions.md"
EXT_E = ROOT / "tests/extension/doc08_atomic_extension_v1_expected.md"
ARTIFACT = ROOT / "data/05_evaluation/doc08_atomic_risk_units_v1.json"


@pytest.fixture(scope="module")
def overlay_build(corpus_sandbox: Path):
    result = build_baseline_and_overlay_chunks(
        canonical_dir=corpus_sandbox / CANONICAL.relative_to(ROOT),
        overlay_document_path=corpus_sandbox / OVERLAY.relative_to(ROOT),
        permitted_root=corpus_sandbox,
    )
    yield result
    cleanup_overlay_temp_dir(result.temp_input_dir)


def test_overlay_replaces_only_doc08(overlay_build) -> None:
    diff = compute_doc08_chunk_diff(overlay_build.baseline_chunks, overlay_build.candidate_chunks)
    assert diff.non_doc08_byte_identical
    assert len(diff.added_chunk_ids) > 0 or len(diff.changed_chunk_ids) > 0


def test_non_doc08_chunks_byte_identical(overlay_build) -> None:
    baseline_non = {
        c.chunk_id: c.content
        for c in overlay_build.baseline_chunks
        if c.document_id != DOC08_DOCUMENT_ID
    }
    candidate_non = {
        c.chunk_id: c.content
        for c in overlay_build.candidate_chunks
        if c.document_id != DOC08_DOCUMENT_ID
    }
    assert baseline_non == candidate_non


def test_atomic_privacy_unit_present(overlay_build) -> None:
    headings = [c.heading for c in overlay_build.candidate_doc08_chunks]
    assert any("Customer personal-data exposure" in h for h in headings)
    assert any(
        "наклейк" in c.content.lower() or "receipt" in c.content.lower()
        for c in overlay_build.candidate_doc08_chunks
    )


def test_atomic_threat_escalation_boundary(overlay_build) -> None:
    threat_chunks = [
        c for c in overlay_build.candidate_doc08_chunks
        if "Threat escalation" in c.heading
    ]
    assert threat_chunks
    combined = "\n".join(c.content for c in threat_chunks).lower()
    assert "12_staff_safety_and_threat_handling" in combined
    assert "физически покажу" not in combined


def test_no_copied_doc12_threat_example() -> None:
    doc12 = (ROOT / "data/02_clean_markdown/12_staff_safety_and_threat_handling.md").read_text(encoding="utf-8")
    overlay_doc08 = OVERLAY.read_text(encoding="utf-8")
    assert "Если этот курьер снова приедет, я ему физически покажу" not in overlay_doc08
    assert "физически покажу" in doc12


def test_deterministic_candidate_chunk_ids(overlay_build, corpus_sandbox: Path) -> None:
    second = build_baseline_and_overlay_chunks(
        canonical_dir=corpus_sandbox / CANONICAL.relative_to(ROOT),
        overlay_document_path=corpus_sandbox / OVERLAY.relative_to(ROOT),
        permitted_root=corpus_sandbox,
    )
    try:
        ids_a = [c.chunk_id for c in overlay_build.candidate_doc08_chunks]
        ids_b = [c.chunk_id for c in second.candidate_doc08_chunks]
        assert ids_a == ids_b
    finally:
        cleanup_overlay_temp_dir(second.temp_input_dir)


def test_frozen_benchmark_unchanged() -> None:
    cases = load_evaluation_corpus(questions_path=FROZEN_Q, expected_path=FROZEN_E)
    assert len(cases) == 60
    t044 = next(c for c in cases if c.test_id == "T044")
    assert t044.expected_primary_documents == ["08_escalation_and_risk_rules"]


def test_extension_benchmark_schema() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cases = load_extension_corpus(
        questions_path=EXT_Q,
        expected_path=EXT_E,
        expected_ids=config["extension_benchmark"]["case_ids"],
    )
    assert len(cases) == 12
    fp = compute_extension_dataset_fingerprint(
        questions_path=EXT_Q,
        expected_path=EXT_E,
        benchmark_id=config["extension_benchmark"]["benchmark_id"],
    )
    assert len(fp) == 64


@requires_local_artifacts(
    PRODUCTION_INDEX / "manifest.json",
    why="manifest of the provisioned production index (built with live OpenAI embeddings)",
)
def test_production_index_fingerprint_unchanged() -> None:
    manifest = load_manifest(PRODUCTION_INDEX)
    assert manifest.chunk_count == 333
    assert manifest.corpus_fingerprint == "b9526128dad23e71e812fbd8f26452b8d6efa29e4faac898fa11f279b310e827"


def test_overlay_doc08_uses_logical_source_path(overlay_build) -> None:
    doc08_paths = {chunk.source_path for chunk in overlay_build.candidate_doc08_chunks}
    assert doc08_paths == {logical_source_path(DOC08_DOCUMENT_ID)}


@pytest.mark.real_tiktoken
def test_overlay_topology_matches_the_accepted_artifact(overlay_build) -> None:
    """Chunk counts and fingerprints of both arms, as recorded when the experiment was run.

    They hold only under real cl100k_base packing, so the default lane cannot check them.
    """
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    baseline_arm, candidate_arm = artifact["baseline_arm"], artifact["candidate_arm"]
    assert len(overlay_build.baseline_chunks) == baseline_arm["chunk_count"] == 333
    assert len(overlay_build.candidate_chunks) == candidate_arm["chunk_count"] == 337
    assert (
        compute_overlay_corpus_fingerprint(overlay_build.candidate_chunks)
        == candidate_arm["corpus_fingerprint"]
    )
    assert compute_doc08_fingerprint(overlay_build.baseline_chunks) == baseline_arm["doc08_fingerprint"]
    assert compute_doc08_fingerprint(overlay_build.candidate_chunks) == candidate_arm["doc08_fingerprint"]
    assert len(overlay_build.baseline_doc08_chunks) == 17
    assert len(overlay_build.candidate_doc08_chunks) == 21


def test_overlay_corpus_fingerprint_deterministic(overlay_build) -> None:
    fp = compute_overlay_corpus_fingerprint(overlay_build.candidate_chunks)
    assert len(fp) == 64


@requires_local_artifacts(
    PRODUCTION_INDEX / "manifest.json",
    CANDIDATE_INDEX / "manifest.json",
    why="manifests of the production index and the locally built doc08 experiment index",
)
def test_candidate_index_separate_from_production() -> None:
    prod = load_manifest(PRODUCTION_INDEX)
    cand = load_manifest(CANDIDATE_INDEX)
    assert cand.corpus_fingerprint != prod.corpus_fingerprint
    assert cand.corpus_fingerprint == (
        "d3c27f4a72e5f78582c6cc29b8cb6c9f26234f6582970ead602f44e5b375bad6"
    )
    assert cand.chunk_count == 337


def test_production_retrieval_config_hash_unchanged() -> None:
    cfg = load_vector_pool_cap_config(ROOT / "configs/retrieval/vector_pool_36_cap4_v1.json")
    expected = "dddf35dd503598a3987e24545c83b0964604638b027922fe6c300529d34fc083"
    assert compute_vector_pool_cap_config_hash(cfg) == expected


def test_artifact_has_per_arm_metadata() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["baseline_arm"]["fetch_k"] == 48
    assert payload["candidate_arm"]["fetch_k"] == 48
    assert payload["source_commit"]
    assert payload["source_dirty"] is False
    assert payload["artifact_commit"] is None
    assert payload["reference_experiment_id"] == "vector-pool-36-cap4-v1"
    assert payload["baseline_reproduction"]["passed"] is True


def test_baseline_reproduction_in_artifact() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["frozen_baseline_primary_hit_at_12"] == pytest.approx(0.9137931034482759)
    assert payload["frozen_baseline_metrics"]["mrr"] == pytest.approx(0.6456896551724138)
    assert payload["frozen_reachability_baseline"]["primary_reachable"] == 55
    assert payload["frozen_reachability_baseline"]["primary_denominator"] == 57


def test_t044_acceptance_in_artifact() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    t044 = next(d for d in payload["required_diagnostics"] if d["case_id"] == "T044")
    assert t044["candidate_primary_reachable"] is True
    assert t044["candidate_primary_hit_at_12"] is True


def test_t047_doc12_guardrail() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    t047 = next(d for d in payload["required_diagnostics"] if d["case_id"] == "T047")
    assert t047["candidate_final_rank_doc12"] == 1


def test_verdict_is_raw_string() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["verdict"] in {
        "ACCEPTED AS TARGETED CORPUS REPAIR",
        "REJECTED",
    }
