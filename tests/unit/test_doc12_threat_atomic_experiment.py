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
HOLDOUT_Q = ROOT / "tests/holdout/doc12_threat_holdout_v1_questions.md"
HOLDOUT_E = ROOT / "tests/holdout/doc12_threat_holdout_v1_expected.md"
BASELINE_INDEX = ROOT / "data/04_index_experiments/doc08_atomic_risk_units_v1"
CANDIDATE_INDEX = ROOT / "data/04_index_experiments/doc12_threat_atomic_units_v1"
PRODUCTION_INDEX = ROOT / "data/04_index"

FORBIDDEN_OVERLAY_STRINGS = [
    "FF-EXT07",
    "FF-EXT08",
    "лично с ним разберусь после доставки",
    "я вас лично найду и накажу",
]


@pytest.fixture(scope="module")
def multi_overlay_build():
    result = build_doc12_experimental_corpora(
        canonical_dir=CANONICAL,
        doc08_overlay_path=DOC08_OVERLAY,
        doc12_overlay_path=DOC12_OVERLAY,
        permitted_root=ROOT,
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


def test_baseline_is_doc08_experimental_corpus(multi_overlay_build) -> None:
    assert len(multi_overlay_build.baseline_chunks) == 337
    assert multi_overlay_build.baseline_doc08_fingerprint == EXPECTED_DOC08_FINGERPRINT


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


def test_atomic_implicit_courier_threat_chunk(multi_overlay_build) -> None:
    headings = [c.heading for c in multi_overlay_build.candidate_doc12_chunks]
    assert any("Example find courier" in h for h in headings)
    assert any("Implicit find or confront" in h for h in headings)


def test_atomic_support_operator_threat_chunk(multi_overlay_build) -> None:
    headings = [c.heading for c in multi_overlay_build.candidate_doc12_chunks]
    assert any("Threat to support operator" in h for h in headings)


def test_no_extension_exact_strings_in_overlay() -> None:
    text = DOC12_OVERLAY.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_OVERLAY_STRINGS:
        assert forbidden not in text


def test_overlay_fingerprint_stable_across_temp_roots() -> None:
    temp_a = ROOT / ".tmp" / "corpus_overlay" / "_doc12_repro_test_a"
    temp_b = ROOT / ".tmp" / "corpus_overlay" / "_doc12_repro_test_b"
    temp_a.mkdir(parents=True, exist_ok=True)
    temp_b.mkdir(parents=True, exist_ok=True)
    first = build_doc12_experimental_corpora(
        canonical_dir=CANONICAL,
        doc08_overlay_path=DOC08_OVERLAY,
        doc12_overlay_path=DOC12_OVERLAY,
        permitted_root=ROOT,
        staging_parent=temp_a,
    )
    second = build_doc12_experimental_corpora(
        canonical_dir=CANONICAL,
        doc08_overlay_path=DOC08_OVERLAY,
        doc12_overlay_path=DOC12_OVERLAY,
        permitted_root=ROOT,
        staging_parent=temp_b,
    )
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


@pytest.mark.skipif(not PRODUCTION_INDEX.joinpath("manifest.json").exists(), reason="production index missing")
def test_production_index_unchanged() -> None:
    from customer_claims_rag.retrieval.manifest import load_manifest

    manifest = load_manifest(PRODUCTION_INDEX)
    assert manifest.corpus_fingerprint == "b9526128dad23e71e812fbd8f26452b8d6efa29e4faac898fa11f279b310e827"
    assert manifest.chunk_count == 333


def test_config_schema() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["experiment_id"] == "doc12-threat-atomic-units-v1"
    assert config["baseline_index_dir"] == "data/04_index_experiments/doc08_atomic_risk_units_v1"
    assert config["candidate_index_dir"] == "data/04_index_experiments/doc12_threat_atomic_units_v1"
    assert DOC08_DOCUMENT_ID in config["baseline_corpus"]["doc08_overlay_path"]
    assert DOC12_DOCUMENT_ID in config["candidate_overlay"]["overlay_path"]
