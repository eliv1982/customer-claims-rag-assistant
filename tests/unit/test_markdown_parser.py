"""Unit tests for Markdown heading parser."""

from __future__ import annotations

import pytest

from customer_claims_rag.exceptions import MarkdownStructureError
from customer_claims_rag.ingestion.markdown_parser import parse_heading_blocks
from customer_claims_rag.models import DocumentMetadata, DocumentRecord


def _record(content: str, doc_id: str = "99_parser_test") -> DocumentRecord:
    metadata = DocumentMetadata(
        document_id=doc_id,
        title="Test",
        category="general",
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
    return DocumentRecord(
        metadata=metadata,
        source_path=f"data/02_clean_markdown/{doc_id}.md",
        content=content,
        word_count=len(content.split()),
        char_count=len(content),
    )


def test_h1_h2_h3_structure() -> None:
    content = """# Doc

## Section A

Text A

### Sub A1

Sub text
"""
    blocks = parse_heading_blocks(_record(content))
    assert len(blocks) >= 1
    assert any("Sub text" in b.content or "Text A" in b.content for b in blocks)


def test_repeated_headings() -> None:
    content = """# Doc

## Section

First body.

## Section

Second body.
"""
    blocks = parse_heading_blocks(_record(content))
    assert len(blocks) == 2
    assert "First body" in blocks[0].content
    assert "Second body" in blocks[1].content


def test_tilde_fenced_code_block() -> None:
    content = """# Doc

## Section

~~~python
## not heading
~~~
"""
    blocks = parse_heading_blocks(_record(content))
    assert len(blocks) == 1
    assert "## not heading" in blocks[0].content


def test_h3_before_first_h2() -> None:
    content = """# Doc

### Early H3

Early content.

## Later H2

Later content.
"""
    blocks = parse_heading_blocks(_record(content))
    assert any("Early content" in b.content for b in blocks)


def test_crlf_line_endings() -> None:
    content = "# Doc\r\n\r\n## Section\r\n\r\nBody.\r\n"
    blocks = parse_heading_blocks(_record(content))
    assert blocks[0].content.endswith("Body.")


def test_blockquote_preserved() -> None:
    content = """# Doc

## Section

> quoted line
"""
    blocks = parse_heading_blocks(_record(content))
    assert "> quoted line" in blocks[0].content


def test_nested_list_preserved() -> None:
    content = """# Doc

## Section

- outer
  - inner
"""
    blocks = parse_heading_blocks(_record(content))
    assert "- outer" in blocks[0].content
    assert "inner" in blocks[0].content


def test_document_without_headings_raises() -> None:
    with pytest.raises(MarkdownStructureError):
        parse_heading_blocks(_record("Plain text only without headings."))


def test_fenced_code_block_with_heading_inside() -> None:
    content = """# Doc

## Real section

```python
## not a heading
```

After code.
"""
    blocks = parse_heading_blocks(_record(content))
    assert len(blocks) == 1
    assert "## not a heading" in blocks[0].content


def test_markdown_table_preserved() -> None:
    content = """# Doc

## Table section

| Col | Val |
|-----|-----|
| a   | 1   |
"""
    blocks = parse_heading_blocks(_record(content))
    assert "| Col | Val |" in blocks[0].content
