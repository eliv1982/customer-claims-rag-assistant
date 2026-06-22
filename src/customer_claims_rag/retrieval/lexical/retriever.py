"""Lexical retriever backed by a pre-built BM25 index."""

from __future__ import annotations

from customer_claims_rag.retrieval.lexical.bm25 import BM25Index, LexicalHit


class LexicalRetriever:
    """Search a frozen BM25 index without calling embeddings or LLMs."""

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, *, k: int) -> list[LexicalHit]:
        """Return top-k lexical hits with 1-based lexical ranks assigned."""
        return self.index.search(query, k=k)
