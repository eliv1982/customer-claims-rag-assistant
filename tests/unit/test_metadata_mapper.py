"""Metadata mapping tests."""

from __future__ import annotations

from tests.retrieval_helpers import make_chunk_record

from customer_claims_rag.retrieval.metadata_mapper import (
    chunk_to_vector_metadata,
    vector_metadata_to_search_hit,
)


def test_scalar_safe_mapping() -> None:
    chunk = make_chunk_record(topic="faq-01", risk_level="high")
    metadata = chunk_to_vector_metadata(chunk)
    assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())


def test_relative_posix_source_path() -> None:
    chunk = make_chunk_record(source_path="data/02_clean_markdown/99_test_doc.md")
    metadata = chunk_to_vector_metadata(chunk)
    assert metadata["source_path"] == "data/02_clean_markdown/99_test_doc.md"
    assert "\\" not in str(metadata["source_path"])


def test_heading_path_serialization_round_trip() -> None:
    chunk = make_chunk_record(heading_path=["A", "B"])
    metadata = chunk_to_vector_metadata(chunk)
    hit = vector_metadata_to_search_hit(
        content=chunk.content,
        metadata=metadata,
        distance=0.1,
    )
    assert hit.heading_path == ["A", "B"]


def test_round_trip_retrieval_fields() -> None:
    chunk = make_chunk_record(
        chunk_id="doc::chunk-002",
        topic="delivery",
        risk_level="medium",
        section="Delivery",
    )
    metadata = chunk_to_vector_metadata(chunk)
    hit = vector_metadata_to_search_hit(
        content=chunk.content,
        metadata=metadata,
        distance=0.25,
    )
    assert hit.chunk_id == chunk.chunk_id
    assert hit.document_id == chunk.document_id
    assert hit.source_path == chunk.source_path
    assert hit.chunk_type == chunk.strategy
    assert hit.topic == "delivery"
    assert hit.risk_level == "medium"
    assert hit.section == "Delivery"
    assert hit.similarity == 0.75


def test_original_chunk_not_mutated() -> None:
    chunk = make_chunk_record(heading_path=["Only"])
    original_heading_path = list(chunk.heading_path)
    _ = chunk_to_vector_metadata(chunk)
    assert chunk.heading_path == original_heading_path
