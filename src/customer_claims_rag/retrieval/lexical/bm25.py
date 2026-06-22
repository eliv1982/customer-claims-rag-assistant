"""Local BM25 index for hybrid lexical retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from customer_claims_rag.exceptions import DuplicateChunkIdError
from customer_claims_rag.retrieval.lexical.corpus_loader import LexicalChunk
from customer_claims_rag.retrieval.lexical.preprocessor import TOKENIZER_VERSION, tokenize


@dataclass(frozen=True)
class LexicalHit:
    """Single lexical search hit with rank assigned by the retriever."""

    chunk_id: str
    document_id: str
    lexical_rank: int
    bm25_score: float
    matched_tokens: list[str]


class BM25Index:
    """Deterministic BM25 index built once per experiment run."""

    def __init__(
        self,
        chunks: list[LexicalChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("BM25 index requires at least one chunk")
        self.k1 = k1
        self.b = b
        self._chunks = list(chunks)
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(self._chunk_ids)) != len(self._chunk_ids):
            raise DuplicateChunkIdError("duplicate chunk_id in lexical corpus")

        self._documents = [self._document_text(chunk) for chunk in chunks]
        self._tokens = [tokenize(text) for text in self._documents]
        self.n_docs = len(self._tokens)
        self.avgdl = sum(len(tokens) for tokens in self._tokens) / self.n_docs
        self.df: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    @staticmethod
    def _document_text(chunk: LexicalChunk) -> str:
        return f"{chunk.heading}\n{chunk.content}"

    @property
    def chunk_count(self) -> int:
        return self.n_docs

    def compute_fingerprint(self, *, corpus_fingerprint: str) -> str:
        """Return stable fingerprint for the lexical index configuration."""
        payload = {
            "tokenizer_version": TOKENIZER_VERSION,
            "bm25_k1": self.k1,
            "bm25_b": self.b,
            "corpus_fingerprint": corpus_fingerprint,
            "chunk_count": self.n_docs,
            "chunk_ids": sorted(self._chunk_ids),
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def search(self, query: str, *, k: int) -> list[LexicalHit]:
        """Return top-k lexical hits sorted by BM25 score then chunk_id."""
        if k < 1:
            raise ValueError("k must be >= 1")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        scored: list[tuple[float, int, list[str]]] = []
        for doc_index, doc_tokens in enumerate(self._tokens):
            matched = sorted(query_set & set(doc_tokens))
            score = self._score(query_tokens, doc_tokens, len(doc_tokens))
            scored.append((score, doc_index, matched))

        scored.sort(key=lambda item: (-item[0], self._chunk_ids[item[1]]))
        hits: list[LexicalHit] = []
        for rank, (score, doc_index, matched) in enumerate(scored[:k], start=1):
            if score <= 0.0:
                break
            chunk = self._chunks[doc_index]
            hits.append(
                LexicalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    lexical_rank=rank,
                    bm25_score=score,
                    matched_tokens=matched,
                )
            )
        return hits

    def _score(self, query_tokens: list[str], doc_tokens: list[str], doc_len: int) -> float:
        total = 0.0
        for token in query_tokens:
            if token not in self.df:
                continue
            term_freq = doc_tokens.count(token)
            idf = math.log(1 + (self.n_docs - self.df[token] + 0.5) / (self.df[token] + 0.5))
            denom = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            total += idf * (term_freq * (self.k1 + 1)) / denom
        return total
