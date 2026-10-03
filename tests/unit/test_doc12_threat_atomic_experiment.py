"""Tests for doc12 threat atomic corpus experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.env_bootstrap import project_root
from customer_claims_rag.evaluation.doc12_threat_atomic_contract import (
    EXPECTED_DOC08_FINGERPRINT,
    HOLDOUT_NEGATIVE_IDS,
    HOLDOUT_POSITIVE_IDS,
    compute_holdout_dataset_fingerprint,
    load_holdout_corpus,
)
from customer_claims_rag.evaluation.extension_parser import load_extension_corpus
from customer_claims_rag.ingestion.corpus_overlay import (
    DOC08_DOCUMENT_ID,
    DOC12_DOCUMENT_ID,
    build_doc12_experimental_corpora,
    cleanup_overlay_temp_dir,
    compute_doc12_chunk_diff,
    compute_overlay_corpus_fingerprint,
    logical_source_path,
    verify_doc08_overlay_unchanged,
)

ROOT = project_root()
DOC08_OVERLAY = ROOT / "experiments/corpus/doc08_atomic_risk_units_v1/08_escalation_and_risk_rules.md"
DOC12_OVERLAY = ROOT / "experiments/corpus/doc12_threat_atomic_units_v1/12_staff_safety_and_threat_handling.md"
CANONICAL = ROOT / "data/02_clean_markdown"
CONFIG = ROOT / "configs/experiments/doc12_threat_atomic_units_v1.json"
ARTIFACT = ROOT / "data/05_evaluation/doc12_threat_atomic_units_v1.json"
HOLDOUT_Q = ROOT / "tests/holdout/doc12_threat_holdout_v1_questions.md"
HOLDOUT_E = ROOT / "tests/holdout/doc12_threat_holdout_v1_expected.md"
BASELINE_INDEX = ROOT / "data/04_index_experiments/doc08_atomic_risk_units_v1"
CANDIDATE_INDEX = ROOT / "data/04_index_experiments/doc12_threat_atomic_units_v1"

FORBIDDEN_OVERLAY_STRINGS = [
    "FF-EXT07",
    "FF-EXT08",
    "лично с ним разберусь после доставки",
    "я вас лично найду и накажу",
]


@pytest.fixture(scope="module")
def multi_overlay_build(historical_corpus_sandbox: Path):
    result = build_doc12_experimental_corpora(
        canonical_dir=historical_corpus_sandbox / CANONICAL.relative_to(ROOT),
        doc08_overlay_path=historical_corpus_sandbox / DOC08_OVERLAY.relative_to(ROOT),
        doc12_overlay_path=historical_corpus_sandbox / DOC12_OVERLAY.relative_to(ROOT),
        permitted_root=historical_corpus_sandbox,
    )
    yield result
    cleanup_overlay_temp_dir(result.temp_input_dir)


def test_holdout_fingerprint_stable() -> None:
    fp = compute_holdout_dataset_fingerprint(
        questions_path=HOLDOUT_Q,
        expected_path=HOLDOUT_E,
        benchmark_id="doc12-threat-holdout-v1",
    )
    assert len(fp) == 64


def test_holdout_loads_twelve_cases() -> None:
    cases = load_holdout_corpus(
        questions_path=HOLDOUT_Q,
        expected_path=HOLDOUT_E,
        expected_ids=list(HOLDOUT_POSITIVE_IDS) + list(HOLDOUT_NEGATIVE_IDS),
    )
    assert len(cases) == 12


@pytest.mark.real_tiktoken
def test_baseline_is_doc08_experimental_corpus(multi_overlay_build) -> None:
    assert len(multi_overlay_build.baseline_chunks) == 337
    assert multi_overlay_build.baseline_doc08_fingerprint == EXPECTED_DOC08_FINGERPRINT


@pytest.mark.real_tiktoken
def test_candidate_adds_only_doc12_overlay(multi_overlay_build) -> None:
    verify_doc08_overlay_unchanged(
        multi_overlay_build.baseline_chunks,
        multi_overlay_build.candidate_chunks,
        expected_doc08_fingerprint=EXPECTED_DOC08_FINGERPRINT,
    )
    diff = compute_doc12_chunk_diff(
        multi_overlay_build.baseline_chunks,
        multi_overlay_build.candidate_chunks,
    )
    assert diff.non_target_byte_identical
    assert len(diff.added_chunk_ids) > 0 or len(diff.changed_chunk_ids) > 0
    assert len(multi_overlay_build.candidate_doc12_chunks) > len(multi_overlay_build.baseline_doc12_chunks)


@pytest.mark.real_tiktoken
def test_overlay_topology_matches_the_accepted_artifact(multi_overlay_build) -> None:
    """Baseline/candidate chunk counts and fingerprints as recorded when the experiment was run.

    The experiment is about how doc12 is re-packed into atomic threat units, so the counts
    (doc12: 23 -> 26 chunks) exist only under real cl100k_base token boundaries.
    """
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    baseline_arm, candidate_arm = artifact["baseline_arm"], artifact["candidate_arm"]
    assert len(multi_overlay_build.baseline_chunks) == baseline_arm["chunk_count"] == 337
    assert len(multi_overlay_build.candidate_chunks) == candidate_arm["chunk_count"] == 340
    baseline_fingerprint = compute_overlay_corpus_fingerprint(multi_overlay_build.baseline_chunks)
    assert baseline_fingerprint == baseline_arm["corpus_fingerprint"]
    assert baseline_fingerprint == config["baseline_corpus"]["expected_index_fingerprint"]
    assert (
        compute_overlay_corpus_fingerprint(multi_overlay_build.candidate_chunks)
        == candidate_arm["corpus_fingerprint"]
    )
    assert len(multi_overlay_build.baseline_doc12_chunks) == 23
    assert len(multi_overlay_build.candidate_doc12_chunks) == 26


@pytest.mark.real_tiktoken
def test_atomic_implicit_courier_threat_chunk(multi_overlay_build) -> None:
    headings = [c.heading for c in multi_overlay_build.candidate_doc12_chunks]
    assert any("Example find courier" in h for h in headings)
    assert any("Implicit find or confront" in h for h in headings)


@pytest.mark.real_tiktoken
def test_atomic_support_operator_threat_chunk(multi_overlay_build) -> None:
    headings = [c.heading for c in multi_overlay_build.candidate_doc12_chunks]
    assert any("Threat to support operator" in h for h in headings)


def test_no_extension_exact_strings_in_overlay() -> None:
    text = DOC12_OVERLAY.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_OVERLAY_STRINGS:
        assert forbidden not in text


def test_overlay_fingerprint_stable_across_temp_roots(historical_corpus_sandbox: Path) -> None:
    temp_a = historical_corpus_sandbox / ".tmp" / "corpus_overlay" / "_doc12_repro_test_a"
    temp_b = historical_corpus_sandbox / ".tmp" / "corpus_overlay" / "_doc12_repro_test_b"
    temp_a.mkdir(parents=True, exist_ok=True)
    temp_b.mkdir(parents=True, exist_ok=True)
    sources = dict(
        canonical_dir=historical_corpus_sandbox / CANONICAL.relative_to(ROOT),
        doc08_overlay_path=historical_corpus_sandbox / DOC08_OVERLAY.relative_to(ROOT),
        doc12_overlay_path=historical_corpus_sandbox / DOC12_OVERLAY.relative_to(ROOT),
        permitted_root=historical_corpus_sandbox,
    )
    first = build_doc12_experimental_corpora(**sources, staging_parent=temp_a)
    second = build_doc12_experimental_corpora(**sources, staging_parent=temp_b)
    try:
        fp_a = compute_overlay_corpus_fingerprint(first.candidate_chunks)
        fp_b = compute_overlay_corpus_fingerprint(second.candidate_chunks)
        ids_a = [c.chunk_id for c in first.candidate_doc12_chunks]
        ids_b = [c.chunk_id for c in second.candidate_doc12_chunks]
        assert fp_a == fp_b
        assert ids_a == ids_b
    finally:
        cleanup_overlay_temp_dir(first.temp_input_dir)
        cleanup_overlay_temp_dir(second.temp_input_dir)


def test_doc12_logical_source_paths(multi_overlay_build) -> None:
    paths = {c.source_path for c in multi_overlay_build.candidate_doc12_chunks}
    assert paths == {logical_source_path(DOC12_DOCUMENT_ID)}


def test_non_doc12_chunks_byte_identical_to_baseline(multi_overlay_build) -> None:
    baseline_non = {
        c.chunk_id: c.content
        for c in multi_overlay_build.baseline_chunks
        if c.document_id != DOC12_DOCUMENT_ID
    }
    candidate_non = {
        c.chunk_id: c.content
        for c in multi_overlay_build.candidate_chunks
        if c.document_id != DOC12_DOCUMENT_ID
    }
    assert baseline_non == candidate_non


def test_config_schema() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "doc12-threat-atomic-units-v1"
    assert config["baseline_index_dir"] == "data/04_index_experiments/doc08_atomic_risk_units_v1"
    assert config["candidate_index_dir"] == "data/04_index_experiments/doc12_threat_atomic_units_v1"
    assert DOC08_DOCUMENT_ID in config["baseline_corpus"]["doc08_overlay_path"]
    assert DOC12_DOCUMENT_ID in config["candidate_overlay"]["overlay_path"]
