"""Corpus fingerprint tests."""

from __future__ import annotations

from tests.retrieval_helpers import make_chunk_record

from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint


def test_fingerprint_stable_for_same_corpus(fake_embedding_provider) -> None:
    chunks = [
        make_chunk_record(chunk_id="a::chunk-001", content="one"),
        make_chunk_record(chunk_id="b::chunk-001", content="two"),
    ]
    first = compute_corpus_fingerprint(chunks, embedding_model=fake_embedding_provider.model_name)
    second = compute_corpus_fingerprint(chunks, embedding_model=fake_embedding_provider.model_name)
    assert first == second


def test_fingerprint_independent_of_input_order(fake_embedding_provider) -> None:
    chunk_a = make_chunk_record(chunk_id="a::chunk-001", content="one")
    chunk_b = make_chunk_record(chunk_id="b::chunk-001", content="two")
    ordered = compute_corpus_fingerprint(
        [chunk_a, chunk_b],
        embedding_model=fake_embedding_provider.model_name,
    )
    reversed_order = compute_corpus_fingerprint(
        [chunk_b, chunk_a],
        embedding_model=fake_embedding_provider.model_name,
    )
    assert ordered == reversed_order


def test_fingerprint_changes_on_content(fake_embedding_provider) -> None:
    base = make_chunk_record(chunk_id="a::chunk-001", content="one")
    changed = make_chunk_record(chunk_id="a::chunk-001", content="two")
    first = compute_corpus_fingerprint([base], embedding_model=fake_embedding_provider.model_name)
    second = compute_corpus_fingerprint([changed], embedding_model=fake_embedding_provider.model_name)
    assert first != second


def test_fingerprint_changes_on_metadata(fake_embedding_provider) -> None:
    base = make_chunk_record(chunk_id="a::chunk-001", topic=None)
    changed = make_chunk_record(chunk_id="a::chunk-001", topic="delivery_delay")
    first = compute_corpus_fingerprint([base], embedding_model=fake_embedding_provider.model_name)
    second = compute_corpus_fingerprint([changed], embedding_model=fake_embedding_provider.model_name)
    assert first != second


def test_fingerprint_changes_on_model_name() -> None:
    chunks = [make_chunk_record()]
    first = compute_corpus_fingerprint(chunks, embedding_model="model-a")
    second = compute_corpus_fingerprint(chunks, embedding_model="model-b")
    assert first != second


def test_fingerprint_uses_relative_source_path(fake_embedding_provider) -> None:
    relative = make_chunk_record(source_path="data/02_clean_markdown/99_test_doc.md")
    fingerprint = compute_corpus_fingerprint(
        [relative],
        embedding_model=fake_embedding_provider.model_name,
    )
    assert "Users" not in fingerprint
    assert "C:" not in fingerprint
    assert "\\\\" not in fingerprint
