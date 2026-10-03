"""Identity recomputed from a stored record equals identity computed from the chunk it came from.

The release validator hashes the records it reads back from Chroma and compares the result with
digests computed at build time from chunk records. That is only meaningful if both sides hash the
same structure, field for field, after the round trip through the store; these tests pin it, on
the real documents (stand-in tokenizer: the claim is about parity, not about chunk counts).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.retrieval.adapters.chroma_store import ChromaVectorStore
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint
from customer_claims_rag.retrieval.index_identity import (
    compute_chunk_payload_digest,
    compute_chunk_payload_digest_from_records,
    compute_collection_content_digest,
    compute_corpus_fingerprint_from_records,
)
from customer_claims_rag.retrieval.metadata_mapper import (
    canonical_metadata_for_fingerprint,
    canonical_metadata_from_vector_metadata,
    chunk_to_vector_metadata,
)
from customer_claims_rag.token_counter import TiktokenCounter
from tests.release_posture_helpers import write_consistent_index
from tests.retrieval_helpers import make_chunk_record

MODEL = "fake-embedding-model"


@pytest.fixture(scope="module")
def corpus_chunks(corpus_sandbox: Path):
    builder = CorpusBuilder(token_counter=TiktokenCounter(), permitted_root=corpus_sandbox.resolve())
    _, chunks = builder.build_from_directory(corpus_sandbox / "data" / "02_clean_markdown")
    return chunks


@pytest.fixture(scope="module")
def stored_records(corpus_chunks, tmp_path_factory: pytest.TempPathFactory):
    index_dir = tmp_path_factory.mktemp("parity_index") / "index"
    write_consistent_index(index_dir, corpus_chunks, embedding_model=MODEL)
    store = ChromaVectorStore(index_dir=index_dir, collection_name="customer_claims", open_existing=True)
    try:
        return store.export_collection_records()
    finally:
        store.close()


def test_vector_metadata_carries_the_version_the_fingerprint_hashes() -> None:
    chunk = make_chunk_record()
    assert chunk_to_vector_metadata(chunk)["version"] == chunk.metadata.version == "1.0.0"


def test_every_real_chunk_round_trips_to_the_same_canonical_metadata(corpus_chunks) -> None:
    assert len(corpus_chunks) > 100
    for chunk in corpus_chunks:
        assert canonical_metadata_from_vector_metadata(chunk_to_vector_metadata(chunk)) == (
            canonical_metadata_for_fingerprint(chunk)
        ), chunk.chunk_id


def test_parity_holds_for_every_optional_metadata_shape() -> None:
    """The real corpus fills few optional fields; chunks that fill all of them, or none, round-trip too."""
    base = make_chunk_record(topic="topic", risk_level="high")
    full = base.model_copy(
        update={
            "metadata": base.metadata.model_copy(
                update={"keywords": ["b", "a"], "related_documents": ["z", "y"], "subsection": "sub"}
            )
        }
    )
    bare = make_chunk_record(section=None)
    for chunk in (full, bare):
        assert canonical_metadata_from_vector_metadata(chunk_to_vector_metadata(chunk)) == (
            canonical_metadata_for_fingerprint(chunk)
        )
    assert canonical_metadata_for_fingerprint(full)["keywords"] == ["a", "b"]
    assert "keywords" not in canonical_metadata_for_fingerprint(bare)
    assert canonical_metadata_for_fingerprint(bare)["section"] is None


def test_digests_recomputed_from_stored_records_equal_the_build_time_digests(corpus_chunks, stored_records) -> None:
    assert compute_chunk_payload_digest_from_records(stored_records) == compute_chunk_payload_digest(corpus_chunks)
    assert compute_corpus_fingerprint_from_records(stored_records, embedding_model=MODEL) == (
        compute_corpus_fingerprint(corpus_chunks, embedding_model=MODEL)
    )


def test_the_recomputed_digests_do_not_depend_on_record_order(stored_records) -> None:
    reversed_records = list(reversed(stored_records))
    assert compute_chunk_payload_digest_from_records(reversed_records) == compute_chunk_payload_digest_from_records(
        stored_records
    )
    assert compute_collection_content_digest(reversed_records) == compute_collection_content_digest(stored_records)


def test_the_recomputed_fingerprint_is_bound_to_the_model_name(stored_records) -> None:
    assert compute_corpus_fingerprint_from_records(stored_records, embedding_model=MODEL) != (
        compute_corpus_fingerprint_from_records(stored_records, embedding_model="another-model")
    )
    # the payload digest is model independent
    assert compute_chunk_payload_digest_from_records(stored_records)


def test_a_record_without_a_hashed_field_raises_naming_the_field(stored_records) -> None:
    broken = [dict(stored_records[0], metadata={k: v for k, v in stored_records[0]["metadata"].items() if k != "version"})]
    with pytest.raises(KeyError, match="version"):
        compute_chunk_payload_digest_from_records(broken)
    with pytest.raises(KeyError, match="version"):
        compute_corpus_fingerprint_from_records(broken, embedding_model=MODEL)


@pytest.mark.parametrize("field", ["title", "heading", "heading_path", "chunk_type", "source_path", "document_priority"])
def test_every_hashed_field_changes_the_recomputed_digest(stored_records, field: str) -> None:
    baseline = compute_chunk_payload_digest_from_records(stored_records)
    changed = [dict(record, metadata=dict(record["metadata"])) for record in stored_records]
    value = changed[0]["metadata"][field]
    changed[0]["metadata"][field] = "[]" if field == "heading_path" and value != "[]" else f"{value}-edited"
    assert compute_chunk_payload_digest_from_records(changed) != baseline


def test_changed_text_and_changed_chunk_id_change_the_recomputed_digest(stored_records) -> None:
    baseline = compute_chunk_payload_digest_from_records(stored_records)
    text = [dict(record) for record in stored_records]
    text[0]["document"] += "!"
    assert compute_chunk_payload_digest_from_records(text) != baseline
    renamed = [dict(record) for record in stored_records]
    renamed[0]["chunk_id"] += "-x"
    assert compute_chunk_payload_digest_from_records(renamed) != baseline
