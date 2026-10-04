"""OpenAI embeddings adapter via langchain-openai."""

from __future__ import annotations

from typing import Sequence

from langchain_openai import OpenAIEmbeddings

from customer_claims_rag.exceptions import EmbeddingError
from customer_claims_rag.retrieval.embedding_validation import (
    validate_document_texts,
    validate_embedding_batch,
    validate_embedding_vector,
    validate_query_text,
)


class OpenAIEmbeddingProvider:
    """LangChain-backed OpenAI embedding provider.

    Text is bounded here, by the project, and not by LangChain: queries and documents are checked
    by ``validate_query_text`` and ``validate_document_texts`` before any request is made, so
    LangChain must not tokenize these already-bounded requests a second time.  Besides being
    redundant, its optional length check requires a tiktoken vocabulary/cache (and can download
    one) in an otherwise serving-only image.
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise EmbeddingError("embedding model name must not be empty")
        if not api_key:
            raise EmbeddingError(
                "OPENAI_API_KEY is required for OpenAI embeddings; "
                "set the environment variable before building or searching"
            )
        self._model_name = model_name
        self._vector_dimension: int | None = None
        self._client = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            # Query and document sizes are project-owned invariants.  Keep ordinary embedding
            # calls independent of cl100k_base; the real tokenizer remains mandatory when the
            # canonical chunks themselves are constructed and their topology is verified.
            check_embedding_ctx_length=False,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vector_dimension(self) -> int | None:
        return self._vector_dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        documents = validate_document_texts(texts)
        if not documents:
            return []
        try:
            vectors = self._client.embed_documents(documents)
        except Exception:
            raise EmbeddingError("OpenAI embedding request failed") from None
        validated = validate_embedding_batch(vectors, expected_count=len(documents))
        if self._vector_dimension is None and validated:
            self._vector_dimension = len(validated[0])
        if self._vector_dimension is not None:
            return validate_embedding_batch(
                validated,
                expected_count=len(documents),
                expected_dimension=self._vector_dimension,
            )
        return validated

    def embed_query(self, text: str) -> list[float]:
        cleaned = validate_query_text(text)
        try:
            vector = self._client.embed_query(cleaned)
        except Exception:
            raise EmbeddingError("OpenAI embedding request failed") from None
        validated = validate_embedding_vector(vector)
        if self._vector_dimension is None:
            self._vector_dimension = len(validated)
        return validate_embedding_vector(
            validated,
            expected_dimension=self._vector_dimension,
        )
