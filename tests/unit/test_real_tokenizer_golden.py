"""Production tokenization invariants, checked under the real ``cl100k_base`` vocabulary.

The tests below carry ``real_tiktoken``: the default lane skips them (its offline stand-in
tokenizer cannot reproduce production chunk packing) and the real-tokenizer lane runs them::

    python -m pytest --real-tiktoken -m real_tiktoken

Two corpora are covered, and they must not be confused:

* the **canonical production corpus** (``configs/corpus/foodflow_production_v1.json``): its
  chunk count, chunk payload digest and corpus fingerprint are the manifest's ``expected``
  block, and ``tests/fixtures/canonical_corpus_chunk_topology_v2.json`` adds the chunk
  boundaries so packing drift is reported chunk by chunk and not only as a changed digest;
* the **historical 15-document corpus** the retrieval experiments ran on, frozen in
  ``experiments/corpus/historical_pre_2d2_15doc_v1``: the 10-document release v1 and the
  15-document expansion identities recorded in its ``snapshot_manifest.json`` (the values the
  tracked experiment artifacts carry) must keep reproducing from the snapshot.

Tokenizer-independent behaviour of the corpus builder stays in the default lane
(``tests/integration/test_corpus_builder.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    build_canonical_corpus,
    compute_corpus_identity,
    load_canonical_corpus_manifest,
)
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.corpus_overlay import compute_doc08_fingerprint
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint
from customer_claims_rag.token_counter import TiktokenCounter
from tests.frozen_fixtures import CANONICAL_TOPOLOGY, HISTORICAL_TOPOLOGY
from tests.offline_tiktoken import OfflineEncoding

ROOT = Path(__file__).resolve().parents[2]
CORPUS_MANIFEST_PATH = ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE
SNAPSHOT_MANIFEST = json.loads(
    (ROOT / "experiments" / "corpus" / "historical_pre_2d2_15doc_v1" / "snapshot_manifest.json").read_text(
        encoding="utf-8"
    )
)
DOC08_ARTIFACT = json.loads(
    (ROOT / "data" / "05_evaluation" / "doc08_atomic_risk_units_v1.json").read_text(encoding="utf-8")
)
EMBEDDING_MODEL = "text-embedding-3-small"
RELEASE_V1_DOCUMENT_PREFIXES = {f"{number:02d}" for number in range(1, 11)}


@pytest.fixture(scope="module")
def canonical_build(corpus_sandbox: Path):
    manifest = load_canonical_corpus_manifest(CORPUS_MANIFEST_PATH)
    # verifies sources, tokenizer and identity against the manifest before returning
    return build_canonical_corpus(manifest, project_root=corpus_sandbox, token_counter=TiktokenCounter())


@pytest.fixture(scope="module")
def historical_chunks(historical_corpus_sandbox: Path) -> list[ChunkRecord]:
    builder = CorpusBuilder(token_counter=TiktokenCounter(), permitted_root=historical_corpus_sandbox)
    _, chunks = builder.build_from_directory(historical_corpus_sandbox / "data" / "02_clean_markdown")
    return chunks


def _fingerprint(chunks: list[ChunkRecord]) -> str:
    return compute_corpus_fingerprint(chunks, embedding_model=EMBEDDING_MODEL)


def _differences(actual: dict[str, int], golden: dict[str, int], limit: int = 12) -> list[str]:
    lines = [f"changed {key}: {golden[key]} -> {actual[key]}" for key in golden if key in actual and actual[key] != golden[key]]
    lines += [f"missing {key}" for key in golden if key not in actual]
    lines += [f"unexpected {key}" for key in actual if key not in golden]
    return lines[:limit] + ([f"... {len(lines) - limit} more"] if len(lines) > limit else [])


@pytest.mark.real_tiktoken
def test_the_real_cl100k_vocabulary_is_in_use() -> None:
    encoding = TiktokenCounter()._encoding
    assert not isinstance(encoding, OfflineEncoding)
    assert encoding.name == "cl100k_base"
    assert encoding.n_vocab == 100277
    assert encoding.encode("hello world") == [15339, 1917]


# --- canonical production corpus --------------------------------------------------------------


@pytest.mark.real_tiktoken
def test_canonical_corpus_reproduces_its_committed_identity(canonical_build) -> None:
    """The manifest's ``expected`` block, rebuilt from the repository with the real tokenizer."""
    manifest = canonical_build.manifest
    identity = canonical_build.identity
    assert manifest.tokenizer_encoding == "cl100k_base"
    assert identity.document_ids == manifest.document_ids
    assert identity.chunk_count == manifest.expected.chunk_count
    assert identity.chunk_payload_digest == manifest.expected.chunk_payload_digest
    assert identity.corpus_fingerprint == manifest.expected.corpus_fingerprint
    assert identity.embedding_model == manifest.expected.embedding_model == EMBEDDING_MODEL


@pytest.mark.real_tiktoken
def test_canonical_corpus_is_the_ten_production_documents(canonical_build) -> None:
    ids = {chunk.document_id[:2] for chunk in canonical_build.chunks}
    assert ids == RELEASE_V1_DOCUMENT_PREFIXES
    assert len(canonical_build.documents) == 10


@pytest.mark.real_tiktoken
def test_canonical_chunk_boundaries_match_the_golden_topology(canonical_build) -> None:
    golden = json.loads(CANONICAL_TOPOLOGY.read_text(encoding="utf-8"))
    chunks = sorted(canonical_build.chunks, key=lambda c: (c.document_id, c.chunk_index))
    actual = {chunk.chunk_id: chunk.token_count for chunk in chunks}
    assert golden["encoding"] == "cl100k_base"
    assert golden["chunk_count"] == len(golden["token_counts"]) == canonical_build.manifest.expected.chunk_count
    differences = _differences(actual, golden["token_counts"])
    assert not differences, "chunk topology drifted from the golden table:\n" + "\n".join(differences)
    assert list(actual) == list(golden["token_counts"]), "chunk order drifted from the golden table"


@pytest.mark.real_tiktoken
def test_building_the_selection_equals_building_the_directory_slice(corpus_sandbox: Path, canonical_build) -> None:
    """Explicit selection and 'directory minus the excluded files' are the same chunks."""
    builder = CorpusBuilder(token_counter=TiktokenCounter(), permitted_root=corpus_sandbox)
    _, everything = builder.build_from_directory(corpus_sandbox / "data" / "02_clean_markdown")
    selected = [c for c in everything if c.document_id in set(canonical_build.manifest.document_ids)]
    identity = compute_corpus_identity(selected, embedding_model=EMBEDDING_MODEL)
    assert identity.chunk_payload_digest == canonical_build.identity.chunk_payload_digest
    assert identity.corpus_fingerprint == canonical_build.identity.corpus_fingerprint


# --- historical evidence ----------------------------------------------------------------------


@pytest.mark.real_tiktoken
def test_historical_release_v1_corpus_reproduces_the_accepted_fingerprint(
    historical_chunks: list[ChunkRecord],
) -> None:
    release = SNAPSHOT_MANIFEST["identities"]["release_10doc_v1"]
    chunks = [chunk for chunk in historical_chunks if chunk.document_id[:2] in RELEASE_V1_DOCUMENT_PREFIXES]
    assert sorted({chunk.document_id for chunk in chunks}) == sorted(release["document_ids"])
    assert len(chunks) == release["chunk_count"] == 215
    assert _fingerprint(chunks) == release["corpus_fingerprint"]
    assert release["corpus_fingerprint"].startswith("bf3df0d4")


@pytest.mark.real_tiktoken
def test_historical_full_corpus_reproduces_the_expansion_fingerprint(
    historical_chunks: list[ChunkRecord],
) -> None:
    expanded = SNAPSHOT_MANIFEST["identities"]["expanded_15doc"]
    assert sorted({chunk.document_id for chunk in historical_chunks}) == sorted(expanded["document_ids"])
    assert len(historical_chunks) == expanded["chunk_count"] == 333
    assert _fingerprint(historical_chunks) == expanded["corpus_fingerprint"]
    assert expanded["corpus_fingerprint"].startswith("b9526128")
    # the same chunks, seen through the experiment that was run on them
    baseline_arm = DOC08_ARTIFACT["baseline_arm"]
    assert _fingerprint(historical_chunks) == baseline_arm["corpus_fingerprint"]
    assert compute_doc08_fingerprint(historical_chunks) == baseline_arm["doc08_fingerprint"]


@pytest.mark.real_tiktoken
def test_historical_chunk_boundaries_match_the_golden_topology(historical_chunks: list[ChunkRecord]) -> None:
    golden = json.loads(HISTORICAL_TOPOLOGY.read_text(encoding="utf-8"))
    actual = {chunk.chunk_id: chunk.token_count for chunk in historical_chunks}
    assert golden["encoding"] == "cl100k_base"
    assert golden["chunk_count"] == len(golden["token_counts"]) == 333
    differences = _differences(actual, golden["token_counts"])
    assert not differences, "chunk topology drifted from the golden table:\n" + "\n".join(differences)
    assert list(actual) == list(golden["token_counts"]), "chunk order drifted from the golden table"


def test_topology_checker_reports_drift() -> None:
    # tokenizer-independent, so it stays in the default lane
    golden = {"d::chunk-001": 500, "d::chunk-002": 400, "d::chunk-003": 300}
    drifted = {"d::chunk-001": 500, "d::chunk-002": 401, "d::chunk-004": 300}
    assert _differences(drifted, golden) == [
        "changed d::chunk-002: 400 -> 401",
        "missing d::chunk-003",
        "unexpected d::chunk-004",
    ]
    assert _differences(golden, golden) == []
