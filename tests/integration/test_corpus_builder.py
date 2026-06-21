"""Integration tests on real clean Markdown corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_claims_rag.config import EXPECTED_FAQ_COUNT
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder

ARCHIVE_MARKERS = ("УСТАРЕЛО", "АРХИВ", "УДАЛИТЬ")
FAQ_PREAMBLE_PHRASE = "не заменяет профильные политики FoodFlow"
FORBIDDEN_TRAILING_PHRASE = "не должны попадать в retrieval как самостоятельные инструкции"
EXPECTED_DOC_IDS = [
    f"{i:02d}_{name}"
    for i, name in enumerate(
        [
            "service_overview",
            "delivery_rules",
            "order_changes_and_cancellations",
            "refund_policy",
            "compensation_policy",
            "food_quality_and_packaging",
            "complaint_handling_procedure",
            "escalation_and_risk_rules",
            "response_style_and_templates",
            "customer_faq",
        ],
        start=1,
    )
]


@pytest.fixture
def clean_input(temp_project: Path) -> Path:
    return temp_project / "data" / "02_clean_markdown"


def test_loads_ten_active_documents(builder: CorpusBuilder, clean_input: Path) -> None:
    documents, _ = builder.build_from_directory(clean_input)
    assert len(documents) == 10


def test_document_ids_present(builder: CorpusBuilder, clean_input: Path) -> None:
    documents, _ = builder.build_from_directory(clean_input)
    loaded_ids = {doc.metadata.document_id for doc in documents}
    for doc_id in EXPECTED_DOC_IDS:
        assert doc_id in loaded_ids


def test_faq_produces_45_chunks(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    faq_chunks = [c for c in chunks if "::faq-" in c.chunk_id]
    assert len(faq_chunks) == EXPECTED_FAQ_COUNT


def test_faq_preamble_in_faq_01_only(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    faq_chunks = [c for c in chunks if c.document_id == "10_customer_faq" and "::faq-" in c.chunk_id]
    faq_01 = next(c for c in faq_chunks if c.chunk_id.endswith("::faq-01"))
    assert FAQ_PREAMBLE_PHRASE in faq_01.content
    others = [c for c in faq_chunks if c.chunk_id != faq_01.chunk_id]
    assert all(FAQ_PREAMBLE_PHRASE not in c.content for c in others)


def test_forbidden_trailing_in_all_row_chunks(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    forbidden = [
        c for c in chunks
        if c.document_id == "09_response_style_and_templates" and "::forbidden-" in c.chunk_id
    ]
    assert len(forbidden) == 11
    assert all(FORBIDDEN_TRAILING_PHRASE in c.content for c in forbidden)


def test_relative_source_paths(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    for chunk in chunks:
        assert not Path(chunk.source_path).is_absolute()
        assert "\\" not in chunk.source_path
        assert chunk.source_path.startswith("data/02_clean_markdown/")
        assert ":" not in chunk.source_path[:2]


def test_all_chunk_ids_unique(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_jsonl_roundtrip(builder: CorpusBuilder, clean_input: Path, temp_project: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    out = temp_project / "data" / "03_chunks" / "chunks.jsonl"
    builder.export_jsonl(chunks, out)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(chunks)
    for line in lines:
        obj = json.loads(line)
        assert "chunk_id" in obj
        assert "content" in obj


def test_stats_json_created(builder: CorpusBuilder, clean_input: Path, temp_project: Path) -> None:
    documents, chunks = builder.build_from_directory(clean_input)
    stats = builder.compute_stats(documents, chunks)
    stats_path = temp_project / "data" / "03_chunks" / "chunk_stats.json"
    builder.export_stats(stats, stats_path)
    loaded = json.loads(stats_path.read_text(encoding="utf-8"))
    assert loaded["documents_total"] == 10
    assert loaded["faq_chunk_count"] == EXPECTED_FAQ_COUNT
    assert "average_tokens" in loaded


def test_does_not_write_to_project_data_dir(
    builder: CorpusBuilder,
    clean_input: Path,
    temp_project: Path,
    project_root: Path,
) -> None:
    out = temp_project / "data" / "03_chunks" / "chunks.jsonl"
    stats = temp_project / "data" / "03_chunks" / "chunk_stats.json"
    builder.build_and_export(clean_input, out, stats)
    assert not (project_root / "data" / "03_chunks" / "chunks.jsonl").exists()


def test_source_files_unchanged(builder: CorpusBuilder, clean_input: Path) -> None:
    before = {path.name: path.read_bytes() for path in sorted(clean_input.glob("*.md"))}
    builder.build_from_directory(clean_input)
    after = {path.name: path.read_bytes() for path in sorted(clean_input.glob("*.md"))}
    assert before == after


def test_no_archive_markers_in_chunks(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    for chunk in chunks:
        for marker in ARCHIVE_MARKERS:
            assert marker not in chunk.content


def test_no_yo_letter_in_generated_chunks(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    for chunk in chunks:
        assert "ё" not in chunk.content
        assert "Ё" not in chunk.content
        assert "\ufffd" not in chunk.content


def test_required_chunk_fields(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    for chunk in chunks:
        assert chunk.document_id
        assert chunk.source_path
        assert chunk.content.strip()
        assert chunk.strategy


def test_document_priority_not_request_risk(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    for chunk in chunks:
        assert hasattr(chunk, "document_priority")
        dumped = chunk.model_dump()
        assert "document_priority" in dumped
        assert dumped["document_priority"] in {"critical", "high", "medium", "low"}
