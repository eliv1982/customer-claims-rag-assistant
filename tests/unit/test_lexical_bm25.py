"""Unit tests for local BM25 index."""

from __future__ import annotations

import pytest

from customer_claims_rag.exceptions import DuplicateChunkIdError
from customer_claims_rag.retrieval.lexical.bm25 import BM25Index
from customer_claims_rag.retrieval.lexical.corpus_loader import LexicalChunk


def _chunk(
    chunk_id: str,
    *,
    document_id: str,
    heading: str,
    content: str,
) -> LexicalChunk:
    return LexicalChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        heading=heading,
        chunk_type="policy",
        source_path=f"data/02_clean_markdown/{document_id}.md",
        metadata={"chunk_id": chunk_id, "document_id": document_id},
    )


def test_rare_exact_term_ranks_expected_document_higher() -> None:
    index = BM25Index(
        [
            _chunk("faq::faq-01", document_id="10_customer_faq", heading="FAQ", content="общий возврат"),
            _chunk(
                "07_complaint_handling_procedure::chunk-019",
                document_id="07_complaint_handling_procedure",
                heading="Безопасность",
                content="не запрашивать CVV или CVC в чате",
            ),
        ]
    )
    hits = index.search("пришлите CVV в чат", k=2)
    assert hits[0].document_id == "07_complaint_handling_procedure"


def test_deterministic_ties() -> None:
    chunks = [
        _chunk("b::chunk-002", document_id="b_doc", heading="H", content="одинаковый текст cvv"),
        _chunk("a::chunk-001", document_id="a_doc", heading="H", content="одинаковый текст cvv"),
    ]
    first = BM25Index(chunks).search("cvv", k=2)
    second = BM25Index(chunks).search("cvv", k=2)
    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
    assert first[0].chunk_id == "a::chunk-001"


def test_empty_query_returns_empty() -> None:
    index = BM25Index([_chunk("a::1", document_id="a", heading="H", content="text")])
    assert index.search("!!!", k=5) == []


def test_unknown_tokens_score_zero() -> None:
    index = BM25Index([_chunk("a::1", document_id="a", heading="H", content="известный термин")])
    hits = index.search("zzzzqqqq", k=1)
    assert hits == []


def test_duplicate_chunk_ids_rejected() -> None:
    chunk = _chunk("dup::1", document_id="a", heading="H", content="text")
    with pytest.raises(DuplicateChunkIdError):
        BM25Index([chunk, chunk])


def test_index_fingerprint_stable() -> None:
    chunks = [
        _chunk("a::1", document_id="a", heading="H", content="alpha"),
        _chunk("b::2", document_id="b", heading="H", content="beta"),
    ]
    fp1 = BM25Index(chunks).compute_fingerprint(corpus_fingerprint="abc")
    fp2 = BM25Index(chunks).compute_fingerprint(corpus_fingerprint="abc")
    assert fp1 == fp2
    assert fp1 != BM25Index(chunks).compute_fingerprint(corpus_fingerprint="def")
