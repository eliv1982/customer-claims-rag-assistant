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
BASELINE_DOC_COUNT = 10
RELEASE_EXPANSION_DOC_COUNT = 4
EXPECTED_DOC_COUNT = BASELINE_DOC_COUNT + RELEASE_EXPANSION_DOC_COUNT

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
] + [
    "11_payment_security_and_dispute_handling",
    "12_staff_safety_and_threat_handling",
    "13_physical_hazard_and_foreign_body_protocol",
    "14_evidence_standards_and_incomplete_information",
]


@pytest.fixture
def clean_input(temp_project: Path) -> Path:
    return temp_project / "data" / "02_clean_markdown"


def test_loads_expected_active_documents(builder: CorpusBuilder, clean_input: Path) -> None:
    documents, _ = builder.build_from_directory(clean_input)
    assert len(documents) == EXPECTED_DOC_COUNT


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
    assert loaded["documents_total"] == EXPECTED_DOC_COUNT
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


PAYMENT_SECURITY_TERMS = (
    "CVV",
    "CVC",
    "chargeback",
    "чарджбэк",
    "несанкционированное списание",
    "полный номер карты",
)


def test_payment_security_document_terms(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    payment_chunks = [
        c for c in chunks if c.document_id == "11_payment_security_and_dispute_handling"
    ]
    assert payment_chunks
    combined = "\n".join(c.content for c in payment_chunks)
    for term in PAYMENT_SECURITY_TERMS:
        assert term in combined


PHYSICAL_HAZARD_TERMS = (
    "металлический осколок",
    "осколок стекла",
    "острый пластик",
    "инородный предмет в еде",
    "поврежденный зуб",
    "опасный предмет в блюде",
)

PHYSICAL_HAZARD_RULE_MARKERS = tuple(f"H-{index:02d}" for index in range(1, 9))

T040_VOCABULARY = ("металлический осколок", "кусок металла", "острый")


def test_physical_hazard_document_terms(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    hazard_chunks = [
        c for c in chunks if c.document_id == "13_physical_hazard_and_foreign_body_protocol"
    ]
    assert hazard_chunks
    combined = "\n".join(c.content for c in hazard_chunks)
    combined_lower = combined.lower()
    for term in PHYSICAL_HAZARD_TERMS:
        assert term in combined_lower
    for marker in PHYSICAL_HAZARD_RULE_MARKERS:
        assert marker in combined


def test_physical_hazard_t040_vocabulary_outside_examples(
    builder: CorpusBuilder,
    clean_input: Path,
) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    policy_chunks = [
        c
        for c in chunks
        if c.document_id == "13_physical_hazard_and_foreign_body_protocol"
        and "Пример" not in (c.heading or "")
    ]
    assert policy_chunks
    combined = "\n".join(c.content for c in policy_chunks).lower()
    for term in T040_VOCABULARY:
        assert term in combined


STAFF_SAFETY_TERMS = (
    "угроза курьеру",
    "ударю курьера",
    "найду курьера",
    "подам в суд",
    "вы еще пожалеете",
    "телефон курьера",
)

STAFF_SAFETY_RULE_MARKERS = tuple(f"S-{index:02d}" for index in range(1, 10))

T047_VOCABULARY = ("угроза курьеру", "физически покажу", "курьеру")


def test_staff_safety_document_terms(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    staff_chunks = [
        c for c in chunks if c.document_id == "12_staff_safety_and_threat_handling"
    ]
    assert staff_chunks
    combined = "\n".join(c.content for c in staff_chunks)
    combined_lower = combined.lower()
    for term in STAFF_SAFETY_TERMS:
        assert term in combined_lower
    for marker in STAFF_SAFETY_RULE_MARKERS:
        assert marker in combined


def test_staff_safety_t047_vocabulary_outside_examples(
    builder: CorpusBuilder,
    clean_input: Path,
) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    policy_chunks = [
        c
        for c in chunks
        if c.document_id == "12_staff_safety_and_threat_handling"
        and "Пример" not in (c.heading or "")
    ]
    assert policy_chunks
    combined_lower = "\n".join(c.content for c in policy_chunks).lower()
    for term in T047_VOCABULARY:
        assert term in combined_lower


def test_staff_safety_legal_vs_physical_threat_sections(
    builder: CorpusBuilder,
    clean_input: Path,
) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    staff_chunks = [
        c for c in chunks if c.document_id == "12_staff_safety_and_threat_handling"
    ]
    s01_chunks = [c for c in staff_chunks if "S-01" in c.content]
    s05_chunks = [c for c in staff_chunks if "S-05" in c.content]
    assert s01_chunks
    assert s05_chunks
    s01_text = "\n".join(c.content for c in s01_chunks).lower()
    s05_text = "\n".join(c.content for c in s05_chunks).lower()
    assert "ударю курьера" in s01_text or "угроза курьеру" in s01_text
    assert "подам в суд" in s05_text
    assert s01_chunks[0].chunk_id != s05_chunks[0].chunk_id


EVIDENCE_TERMS = (
    "нет номера заказа",
    "нет фото",
    "спорный статус доставки",
    "курьер говорит одно, клиент другое",
    "жалобу нельзя отклонять автоматически",
    "нельзя снижать риск из-за отсутствия фото",
    "неполные сведения",
    "противоречивые сведения",
)

EVIDENCE_RULE_MARKERS = tuple(f"E-{index:02d}" for index in range(1, 11))

T055_VOCABULARY = ("нет номера заказа", "нет фото", "неполные сведения", "вскрыт")


def test_evidence_standards_document_terms(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    evidence_chunks = [
        c for c in chunks if c.document_id == "14_evidence_standards_and_incomplete_information"
    ]
    assert len(evidence_chunks) == 25
    combined = "\n".join(c.content for c in evidence_chunks)
    combined_lower = combined.lower()
    for term in EVIDENCE_TERMS:
        assert term in combined_lower
    for marker in EVIDENCE_RULE_MARKERS:
        assert marker in combined


def test_evidence_t055_vocabulary_outside_examples(
    builder: CorpusBuilder,
    clean_input: Path,
) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    policy_chunks = [
        c
        for c in chunks
        if c.document_id == "14_evidence_standards_and_incomplete_information"
        and "Пример" not in (c.heading or "")
    ]
    assert policy_chunks
    combined_lower = "\n".join(c.content for c in policy_chunks).lower()
    for term in T055_VOCABULARY:
        assert term in combined_lower


def test_evidence_disputed_delivery_vs_payment_sections(
    builder: CorpusBuilder,
    clean_input: Path,
) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    evidence_chunks = [
        c for c in chunks if c.document_id == "14_evidence_standards_and_incomplete_information"
    ]
    delivery_chunks = [c for c in evidence_chunks if "E-05" in c.content]
    payment_chunks = [
        c
        for c in evidence_chunks
        if "CVV" in c.content or "несанкционирован" in c.content.lower()
    ]
    assert delivery_chunks
    assert payment_chunks
    delivery_text = "\n".join(c.content for c in delivery_chunks).lower()
    assert "заказ отмечен доставленным" in delivery_text
    assert delivery_chunks[0].chunk_id != payment_chunks[0].chunk_id


def test_evidence_no_auto_reject_and_no_risk_lowering(
    builder: CorpusBuilder,
    clean_input: Path,
) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    evidence_chunks = [
        c for c in chunks if c.document_id == "14_evidence_standards_and_incomplete_information"
    ]
    combined_lower = "\n".join(c.content for c in evidence_chunks).lower()
    assert "жалобу нельзя отклонять автоматически" in combined_lower
    assert "нельзя снижать риск из-за отсутствия фото" in combined_lower


def test_evidence_safe_vs_prohibited_evidence(
    builder: CorpusBuilder,
    clean_input: Path,
) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    e08_chunks = [
        c
        for c in chunks
        if c.document_id == "14_evidence_standards_and_incomplete_information"
        and "E-08" in c.content
    ]
    assert e08_chunks
    e08_text = "\n".join(c.content for c in e08_chunks).lower()
    assert "cvv" in e08_text
    assert "полный номер карты" in e08_text
    assert "замаскирован" in e08_text or "маскирован" in e08_text


def test_document_priority_not_request_risk(builder: CorpusBuilder, clean_input: Path) -> None:
    _, chunks = builder.build_from_directory(clean_input)
    for chunk in chunks:
        assert hasattr(chunk, "document_priority")
        dumped = chunk.model_dump()
        assert "document_priority" in dumped
        assert dumped["document_priority"] in {"critical", "high", "medium", "low"}
