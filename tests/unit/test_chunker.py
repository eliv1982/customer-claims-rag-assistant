"""Unit tests for hybrid chunker."""

from __future__ import annotations

import pytest

from customer_claims_rag.config import LIMITS_BY_STRATEGY, POLICY_HARD_MAX
from customer_claims_rag.exceptions import ChunkingError, DuplicateChunkIdError
from customer_claims_rag.ingestion.chunker import HybridChunker
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.models import ChunkMetadata, ChunkRecord, DocumentMetadata, DocumentRecord
from customer_claims_rag.token_counter import FakeTokenCounter, TiktokenCounter


def _doc(
    content: str,
    *,
    document_id: str = "99_chunk_test",
    document_type: str = "policy",
) -> DocumentRecord:
    metadata = DocumentMetadata(
        document_id=document_id,
        title="Test",
        category="general",
        document_type=document_type,  # type: ignore[arg-type]
        version="1.0.0",
        status="active",
        effective_date="2026-06-01",
        last_updated="2026-06-20",
        audience="support",
        confidentiality="internal",
        source_type="internal_policy",
        language="ru",
        priority="high",
    )
    return DocumentRecord(
        metadata=metadata,
        source_path=f"data/02_clean_markdown/{document_id}.md",
        content=content,
        word_count=len(content.split()),
        char_count=len(content),
    )


def _faq_doc(content: str) -> DocumentRecord:
    return _doc(content, document_id="10_customer_faq", document_type="faq")


def test_faq_intro_prepended_to_faq_01_only() -> None:
    intro = "Этот FAQ дает краткие рабочие ответы и не заменяет профильные политики."
    content = f"""# FAQ

{intro}

## FAQ-01. First question?

**Ответ:** Answer one.

## FAQ-02. Second question?

**Ответ:** Answer two.
"""
    chunks = HybridChunker(TiktokenCounter()).chunk_document(_faq_doc(content))
    faq_chunks = [c for c in chunks if "::faq-" in c.chunk_id]
    assert len(faq_chunks) == 2
    faq_01 = next(c for c in faq_chunks if c.chunk_id.endswith("::faq-01"))
    assert intro in faq_01.content
    assert "FAQ-01" in faq_01.content
    faq_02 = next(c for c in faq_chunks if c.chunk_id.endswith("::faq-02"))
    assert intro not in faq_02.content
    assert all(c.overlap_tokens == 0 for c in faq_chunks)


def test_faq_one_h2_one_chunk() -> None:
    content = """# FAQ

## FAQ-01. First question?

**Ответ:** Answer one.

## FAQ-02. Second question?

**Ответ:** Answer two.
"""
    faq_chunks = [
        c for c in HybridChunker(TiktokenCounter()).chunk_document(_faq_doc(content))
        if "::faq-" in c.chunk_id
    ]
    assert len(faq_chunks) == 2


def test_faq_cannot_be_silently_split() -> None:
    long_answer = " ".join(["слово"] * 1200)
    content = f"""# FAQ

## FAQ-01. Long question?

**Ответ:** {long_answer}
"""
    with pytest.raises(ChunkingError, match="hard max"):
        HybridChunker(TiktokenCounter()).chunk_document(_faq_doc(content))


def test_forbidden_row_contains_trailing_safety() -> None:
    trailing = "Опасные формулировки не являются рабочими шаблонами."
    content = f"""# Style

## Запретные формулировки и безопасные альтернативы

Intro text.

| Недопустимая | Почему нельзя | ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО |
|---|---|---|
| **НЕДОПУСТИМО:** bad | reason | **ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО:** good |

{trailing}
"""
    doc = _doc(content, document_id="09_response_style_and_templates", document_type="guideline")
    chunks = HybridChunker(TiktokenCounter()).chunk_document(doc)
    forbidden = [c for c in chunks if "::forbidden-" in c.chunk_id]
    assert len(forbidden) == 1
    assert trailing in forbidden[0].content
    assert "НЕДОПУСТИМО" in forbidden[0].content
    assert "используйте вместо" in forbidden[0].content.lower()


def test_forbidden_row_missing_reason_raises() -> None:
    trailing = "Опасные формулировки не являются рабочими шаблонами."
    content = f"""# Style

## Запретные формулировки и безопасные альтернативы

| A | Почему нельзя | ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО |
|---|---|---|
| **НЕДОПУСТИМО:** bad |  | **ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО:** good |

{trailing}
"""
    doc = _doc(content, document_id="09_response_style_and_templates", document_type="guideline")
    with pytest.raises(ChunkingError, match="reason"):
        HybridChunker(TiktokenCounter()).chunk_document(doc)


def test_template_dangerous_without_safe_alternative_raises() -> None:
    content = """# Style

## Шаблоны проектов ответов

### 1. First

**НЕДОПУСТИМО:** dangerous phrase without neighbor safe text.
"""
    doc = _doc(content, document_id="09_response_style_and_templates", document_type="guideline")
    with pytest.raises(ChunkingError, match="safe alternative"):
        HybridChunker(TiktokenCounter()).chunk_document(doc)


def test_forbidden_row_missing_safe_alternative_raises() -> None:
    content = """# Style

## Запретные формулировки и безопасные альтернативы

| A | Почему нельзя | B |
|---|---|---|
| **НЕДОПУСТИМО:** bad | reason | no safe here |
"""
    doc = _doc(content, document_id="09_response_style_and_templates", document_type="guideline")
    with pytest.raises(ChunkingError, match="safe alternative"):
        HybridChunker(TiktokenCounter()).chunk_document(doc)


def test_forbidden_table_without_rows_raises() -> None:
    content = """# Style

## Запретные формулировки и безопасные альтернативы

Intro with **НЕДОПУСТИМО:** marker and **ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО:** and Почему нельзя.

| A | Почему нельзя | ИСПОЛЬЗУЙТЕ ВМЕСТО ЭТОГО |
|---|---|---|
"""
    doc = _doc(content, document_id="09_response_style_and_templates", document_type="guideline")
    with pytest.raises(ChunkingError, match="data row"):
        HybridChunker(TiktokenCounter()).chunk_document(doc)


def test_template_chunks_have_zero_overlap() -> None:
    content = """# Style

## Шаблоны проектов ответов

### 1. First

**Шаблон**

Text one.
"""
    doc = _doc(content, document_id="09_response_style_and_templates", document_type="guideline")
    chunks = HybridChunker(TiktokenCounter()).chunk_document(doc)
    templates = [c for c in chunks if "::template-" in c.chunk_id]
    assert len(templates) == 1
    assert templates[0].overlap_tokens == 0


def _policy_huge_content(words: int = 1500) -> str:
    huge = " ".join(["token"] * words)
    return f"""# Doc

## Huge block

{huge}
"""


def test_overlap_split_mandatory() -> None:
    chunks = HybridChunker(FakeTokenCounter()).chunk_document(_doc(_policy_huge_content()))
    assert len(chunks) > 1
    assert chunks[0].overlap_tokens == 0
    for chunk in chunks[1:]:
        assert chunk.overlap_tokens > 0


def test_overlap_within_limits() -> None:
    limits = LIMITS_BY_STRATEGY["policy"]
    chunks = HybridChunker(FakeTokenCounter()).chunk_document(_doc(_policy_huge_content()))
    assert len(chunks) > 1
    max_ratio = int(limits.soft_max * 0.20)
    for chunk in chunks[1:]:
        assert 0 < chunk.overlap_tokens <= limits.overlap_max
        assert chunk.overlap_tokens <= max_ratio or chunk.overlap_tokens <= limits.overlap_max


def test_overlap_content_matches_suffix() -> None:
    chunks = HybridChunker(FakeTokenCounter()).chunk_document(_doc(_policy_huge_content()))
    assert len(chunks) > 1
    bodies = [c.content.split("---\n", 1)[-1] for c in chunks]
    for idx in range(1, len(bodies)):
        prev, curr = bodies[idx - 1], bodies[idx]
        max_len = min(len(prev), len(curr))
        shared = 0
        for length in range(max_len, 0, -1):
            if curr.startswith(prev[-length:]):
                shared = length
                break
        assert shared > 0


def test_hard_max_boundary_policy() -> None:
    limits = LIMITS_BY_STRATEGY["policy"]
    under = " ".join(["w"] * 200)
    chunks = HybridChunker(FakeTokenCounter()).chunk_document(_doc(f"## S\n\n{under}"))
    assert len(chunks) == 1
    assert chunks[0].token_count < limits.hard_max

    splittable = " ".join(["w"] * (limits.hard_max + 50))
    split_chunks = HybridChunker(FakeTokenCounter()).chunk_document(_doc(f"## S\n\n{splittable}"))
    assert len(split_chunks) > 1
    assert all(c.token_count < limits.hard_max for c in split_chunks)

    huge_answer = " ".join(["word"] * limits.hard_max)
    faq_content = f"""# FAQ

## FAQ-01. Question?

**Ответ:** {huge_answer}
"""
    with pytest.raises(ChunkingError, match="hard max"):
        HybridChunker(FakeTokenCounter()).chunk_document(_faq_doc(faq_content))


def test_duplicate_chunk_id_production_path(tmp_path) -> None:
    builder = CorpusBuilder(token_counter=FakeTokenCounter(), permitted_root=tmp_path.resolve())
    meta = DocumentMetadata(
        document_id="d",
        title="T",
        category="g",
        document_type="reference",
        version="1.0.0",
        status="active",
        effective_date="2026-06-01",
        last_updated="2026-06-20",
        audience="support",
        confidentiality="internal",
        source_type="internal_reference",
        language="ru",
        priority="medium",
    )
    chunk_meta = ChunkMetadata.from_document(meta, source_path="data/02_clean_markdown/d.md")
    duplicate = ChunkRecord(
        chunk_id="d::chunk-001",
        document_id="d",
        source_path="data/02_clean_markdown/d.md",
        title="T",
        category="g",
        document_type="reference",
        source_type="internal_reference",
        document_priority="medium",
        chunk_index=1,
        heading="H",
        heading_path=["H"],
        content="Документ: d | Раздел: H\n---\nbody",
        token_count=5,
        word_count=5,
        char_count=5,
        strategy="reference",
        overlap_tokens=0,
        metadata=chunk_meta,
    )
    with pytest.raises(DuplicateChunkIdError, match="duplicate chunk_id"):
        builder.ensure_unique_chunk_ids([duplicate, duplicate])
