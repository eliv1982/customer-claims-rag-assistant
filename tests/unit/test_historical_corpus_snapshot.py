"""The frozen historical corpus: what it is, that it is intact, and that it is kept apart.

``experiments/corpus/historical_pre_2d2_15doc_v1`` holds the 15 documents as they were when the
retrieval experiments ran. The live corpus (``data/02_clean_markdown``) was corrected afterwards,
so evidence about the old corpus (the tracked experiment artifacts and their fingerprints) is
reproduced from the snapshot. These tests need no tokenizer: they pin the snapshot's bytes and
tie its recorded identities to the tracked artifacts; the real-tokenizer lane then rebuilds the
chunks and compares the fingerprints (``test_real_tokenizer_golden.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    load_canonical_corpus_manifest,
    normalized_source_text,
)

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = ROOT / "experiments" / "corpus" / "historical_pre_2d2_15doc_v1"
SNAPSHOT = json.loads((SNAPSHOT_DIR / "snapshot_manifest.json").read_text(encoding="utf-8"))
EVALUATION = ROOT / "data" / "05_evaluation"


def test_snapshot_contains_exactly_the_recorded_documents() -> None:
    recorded = [entry["file"] for entry in SNAPSHOT["documents"]]
    on_disk = sorted(path.name for path in SNAPSHOT_DIR.glob("*.md"))
    assert recorded == on_disk
    assert len(recorded) == 15
    # a README.md next to the documents would be loaded as a (malformed) document
    assert "README.md" not in on_disk


def test_snapshot_documents_are_byte_stable() -> None:
    for entry in SNAPSHOT["documents"]:
        text = normalized_source_text((SNAPSHOT_DIR / entry["file"]).read_text(encoding="utf-8"))
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == entry["sha256"], entry["file"]


def test_snapshot_is_labelled_as_historical_evidence() -> None:
    assert SNAPSHOT["snapshot_id"] == "historical-pre-2d2-15doc-v1"
    assert "HISTORICAL EVIDENCE ONLY" in SNAPSHOT["description"]
    assert SNAPSHOT["tokenizer_encoding"] == "cl100k_base"
    for evidence in SNAPSHOT["evidence"]:
        assert (ROOT / evidence).is_file(), evidence


def test_snapshot_identities_match_the_tracked_experiment_artifacts() -> None:
    expanded = SNAPSHOT["identities"]["expanded_15doc"]
    release_v1 = SNAPSHOT["identities"]["release_10doc_v1"]
    assert (expanded["chunk_count"], release_v1["chunk_count"]) == (333, 215)

    doc08 = json.loads((EVALUATION / "doc08_atomic_risk_units_v1.json").read_text(encoding="utf-8"))
    assert doc08["baseline_arm"]["chunk_count"] == expanded["chunk_count"]
    assert doc08["baseline_arm"]["corpus_fingerprint"] == expanded["corpus_fingerprint"]

    regression = json.loads(
        (EVALUATION / "expanded_corpus_frozen_regression_v1.json").read_text(encoding="utf-8")
    )
    fingerprints = {value for value in _strings(regression) if len(value) == 64}
    assert expanded["corpus_fingerprint"] in fingerprints
    assert release_v1["corpus_fingerprint"] in fingerprints


def _strings(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [item for value in node.values() for item in _strings(value)]
    if isinstance(node, list):
        return [item for value in node for item in _strings(value)]
    return []


def test_snapshot_is_never_the_production_corpus() -> None:
    manifest = load_canonical_corpus_manifest(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    assert not manifest.source_dir.startswith("experiments")
    assert manifest.source_dir == "data/02_clean_markdown"
    release_descriptor = (ROOT / "configs" / "release" / "production_posture.json").read_text(encoding="utf-8")
    assert "experiments" not in release_descriptor
    assert "historical" not in release_descriptor


def test_the_historical_release_identity_is_not_the_canonical_one() -> None:
    """The corpus was corrected, so the identity changed on purpose; the old one stays history."""
    manifest = load_canonical_corpus_manifest(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    historical = SNAPSHOT["identities"]["release_10doc_v1"]
    assert manifest.expected.corpus_fingerprint != historical["corpus_fingerprint"]
    assert manifest.expected.chunk_payload_digest != historical["chunk_payload_digest"]
    assert list(manifest.document_ids) == historical["document_ids"]
