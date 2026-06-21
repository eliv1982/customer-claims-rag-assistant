"""Shared helpers for retrieval tests."""

from __future__ import annotations

from customer_claims_rag.models import ChunkMetadata, ChunkRecord, DocumentMetadata


def make_document_metadata(**overrides) -> DocumentMetadata:
    base = {
        "document_id": "99_test_doc",
        "title": "Тестовый документ",
        "category": "general",
        "document_type": "reference",
        "version": "1.0.0",
        "status": "active",
        "effective_date": "2026-06-01",
        "last_updated": "2026-06-20",
        "audience": "support_agent",
        "confidentiality": "internal",
        "source_type": "internal_reference",
        "language": "ru",
        "priority": "medium",
    }
    base.update(overrides)
    return DocumentMetadata(**base)


def make_chunk_record(
    *,
    chunk_id: str = "99_test_doc::chunk-001",
    content: str = "Документ: test | Раздел: Section\n---\nBody text.",
    document_id: str = "99_test_doc",
    source_path: str = "data/02_clean_markdown/99_test_doc.md",
    strategy: str = "reference",
    heading: str = "Section",
    heading_path: list[str] | None = None,
    topic: str | None = None,
    risk_level: str | None = None,
    section: str | None = "Section",
    chunk_index: int = 1,
) -> ChunkRecord:
    metadata = make_document_metadata(document_id=document_id)
    chunk_meta = ChunkMetadata.from_document(
        metadata,
        source_path=source_path,
        section=section,
        topic=topic,
        risk_level=risk_level,
    )
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        source_path=source_path,
        title=metadata.title,
        category=metadata.category,
        document_type=metadata.document_type,
        source_type=metadata.source_type,
        document_priority=metadata.priority,
        chunk_index=chunk_index,
        heading=heading,
        heading_path=heading_path or [heading],
        content=content,
        token_count=10,
        word_count=10,
        char_count=len(content),
        strategy=strategy,
        overlap_tokens=0,
        metadata=chunk_meta,
    )
