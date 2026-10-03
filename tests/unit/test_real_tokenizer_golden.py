"""Production tokenization invariants, checked under the real ``cl100k_base`` vocabulary.

The tests below carry ``real_tiktoken``: the default lane skips them (its offline stand-in
tokenizer cannot reproduce production chunk packing) and the real-tokenizer lane runs them::

    python -m pytest --real-tiktoken -m real_tiktoken

The expected values are not new audit numbers. The corpus fingerprints and chunk counts are
read from the release posture (``configs/release/production_posture.json``) and from the
tracked evaluation artifacts that were produced from the same chunks; the token-count table
(``tests/fixtures/real_tokenizer_release_chunk_topology_v1.json``) adds the chunk boundaries,
so packing drift is reported chunk by chunk and not only as a changed fingerprint.
Tokenizer-independent behaviour of the corpus builder stays in the default lane
(``tests/integration/test_corpus_builder.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.ingestion.corpus_overlay import compute_doc08_fingerprint
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint
from customer_claims_rag.token_counter import TiktokenCounter
from tests.frozen_fixtures import REAL_TOKENIZER_TOPOLOGY
from tests.offline_tiktoken import OfflineEncoding

ROOT = Path(__file__).resolve().parents[2]
POSTURE = json.loads((ROOT / "configs" / "release" / "production_posture.json").read_text(encoding="utf-8"))
DOC08_ARTIFACT = json.loads(
    (ROOT / "data" / "05_evaluation" / "doc08_atomic_risk_units_v1.json").read_text(encoding="utf-8")
)
EMBEDDING_MODEL = "text-embedding-3-small"
ACTIVE_DOCUMENT_PREFIXES = {f"{number:02d}" for number in range(1, 11)}


@pytest.fixture(scope="module")
def release_chunks(corpus_sandbox: Path) -> list[ChunkRecord]:
    builder = CorpusBuilder(token_counter=TiktokenCounter(), permitted_root=corpus_sandbox)
    _, chunks = builder.build_from_directory(corpus_sandbox / "data" / "02_clean_markdown")
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


@pytest.mark.real_tiktoken
def test_active_release_corpus_reproduces_the_accepted_fingerprint(release_chunks: list[ChunkRecord]) -> None:
    active = POSTURE["targets"]["active"]
    chunks = [chunk for chunk in release_chunks if chunk.document_id[:2] in ACTIVE_DOCUMENT_PREFIXES]
    assert sorted({chunk.document_id for chunk in chunks}) == sorted(active["supported_document_ids"])
    assert len(chunks) == active["expected_chunk_count"] == 215
    assert _fingerprint(chunks) == active["expected_corpus_fingerprint"]


@pytest.mark.real_tiktoken
def test_full_release_corpus_reproduces_the_accepted_archive_fingerprint(
    release_chunks: list[ChunkRecord],
) -> None:
    archive = POSTURE["targets"]["rollback"]
    assert sorted({chunk.document_id for chunk in release_chunks}) == sorted(archive["supported_document_ids"])
    assert len(release_chunks) == archive["expected_chunk_count"] == 333
    assert _fingerprint(release_chunks) == archive["expected_corpus_fingerprint"]
    # the same chunks, seen through the experiment that was run on them
    baseline_arm = DOC08_ARTIFACT["baseline_arm"]
    assert _fingerprint(release_chunks) == baseline_arm["corpus_fingerprint"]
    assert compute_doc08_fingerprint(release_chunks) == baseline_arm["doc08_fingerprint"]


@pytest.mark.real_tiktoken
def test_chunk_boundaries_match_the_golden_topology(release_chunks: list[ChunkRecord]) -> None:
    golden = json.loads(REAL_TOKENIZER_TOPOLOGY.read_text(encoding="utf-8"))
    actual = {chunk.chunk_id: chunk.token_count for chunk in release_chunks}
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
