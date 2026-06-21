"""Content coverage helpers and integration tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.models import ChunkRecord

_FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_HEADING_ONLY = re.compile(r"^#{1,6}\s+")
_TABLE_SEPARATOR = re.compile(r"^\|[\s\-:|]+\|$")
_FAQ_SECTION = re.compile(r"(?=^## FAQ-\d+\.)", re.MULTILINE)
_MIN_FRAGMENT_LEN = 40


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _split_faq_sections(body: str) -> list[str]:
    parts = _FAQ_SECTION.split(body)
    if len(parts) <= 1:
        return [body]
    return [part.strip() for part in parts if part.strip()]


def extract_meaningful_fragments(markdown_body: str) -> list[str]:
    """Extract paragraphs, list items and table rows worth preserving."""
    fragments: list[str] = []
    seen: set[str] = set()

    def add_fragment(text: str) -> None:
        normalized = _normalize_whitespace(text)
        if len(normalized) >= _MIN_FRAGMENT_LEN and normalized not in seen:
            seen.add(normalized)
            fragments.append(normalized)

    sections = _split_faq_sections(markdown_body) if "## FAQ-" in markdown_body else [markdown_body]

    for section in sections:
        for line in section.split("\n"):
            line = line.strip()
            if not line:
                continue
            if _HEADING_ONLY.match(line):
                continue
            if _TABLE_SEPARATOR.match(line):
                continue
            add_fragment(line)

        for block in re.split(r"\n\s*\n", section.strip()):
            block = block.strip()
            if not block:
                continue
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            if not lines or all(_HEADING_ONLY.match(line) for line in lines):
                continue
            if all(line.startswith("|") for line in lines):
                for row in lines:
                    if not _TABLE_SEPARATOR.match(row):
                        add_fragment(row)
                continue
            add_fragment(block)

    return fragments


def load_markdown_body(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return _FRONT_MATTER.sub("", raw, count=1).strip()


def _fragment_present(fragment: str, combined: str) -> bool:
    if fragment in combined:
        return True
    if len(fragment) > _MIN_FRAGMENT_LEN and fragment[:_MIN_FRAGMENT_LEN] in combined:
        return True
    numbered = re.match(r"^\d+\.\s+", fragment)
    if numbered:
        body = fragment[numbered.end() :].strip()
        if len(body) >= _MIN_FRAGMENT_LEN:
            if body in combined or body[:_MIN_FRAGMENT_LEN] in combined:
                return True
    return False


def test_content_coverage_real_corpus(
    builder: CorpusBuilder,
    temp_project: Path,
) -> None:
    clean_input = temp_project / "data" / "02_clean_markdown"
    documents, chunks = builder.build_from_directory(clean_input)
    chunks_by_doc: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    missing: list[str] = []
    for doc in documents:
        body = load_markdown_body(clean_input / f"{doc.metadata.document_id}.md")
        combined = " ".join(
            _normalize_whitespace(c.content)
            for c in chunks_by_doc[doc.metadata.document_id]
        )
        for fragment in extract_meaningful_fragments(body):
            if not _fragment_present(fragment, combined):
                missing.append(f"{doc.metadata.document_id}: {fragment[:80]}...")

    assert not missing, "Missing fragments:\n" + "\n".join(missing[:10])
