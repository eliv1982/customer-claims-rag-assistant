"""Embedding vector validation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from customer_claims_rag.exceptions import EmbeddingError
from customer_claims_rag.input_limits import MAX_CUSTOMER_QUERY_CHARS

# Size bounds for document embedding requests, counted in Unicode code points of the text exactly
# as it is sent.  They are project-owned so that no request depends on a tokenizer: the largest
# canonical chunk is 2,838 characters (891 cl100k tokens, 3,731 UTF-8 bytes) and the whole
# canonical corpus is 216 chunks, so both leave headroom for any supported build, including a
# batch size larger than the corpus (the default batch is 64).
MAX_EMBEDDING_DOCUMENT_CHARS = 4000
MAX_EMBEDDING_BATCH_DOCUMENTS = 512


def validate_embedding_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int | None = None,
) -> list[float]:
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise EmbeddingError(
            f"embedding dimension mismatch: expected {expected_dimension}, "
            f"got {len(vector)}"
        )
    result: list[float] = []
    for value in vector:
        if math.isnan(value) or math.isinf(value):
            raise EmbeddingError("embedding contains NaN or infinity")
        result.append(float(value))
    return result


def validate_embedding_batch(
    vectors: Sequence[Sequence[float]],
    *,
    expected_count: int | None = None,
    expected_dimension: int | None = None,
) -> list[list[float]]:
    if expected_count is not None and len(vectors) != expected_count:
        raise EmbeddingError(
            f"embedding count mismatch: expected {expected_count}, got {len(vectors)}"
        )
    return [
        validate_embedding_vector(vector, expected_dimension=expected_dimension)
        for vector in vectors
    ]


def validate_document_texts(texts: Sequence[str]) -> list[str]:
    """Return the document texts as a list, unchanged, if the whole batch is within bounds.

    Documents are embedded verbatim (unlike queries they are not stripped), so the bound applies
    to the raw text.  Error messages carry indexes and limits, never document content.
    """
    if isinstance(texts, (str, bytes, bytearray)) or not isinstance(texts, Sequence):
        raise EmbeddingError("document texts must be a sequence of strings")
    if len(texts) > MAX_EMBEDDING_BATCH_DOCUMENTS:
        raise EmbeddingError(
            f"embedding batch must not exceed {MAX_EMBEDDING_BATCH_DOCUMENTS} documents"
        )
    documents = list(texts)
    for index, text in enumerate(documents):
        if not isinstance(text, str):
            raise EmbeddingError(f"document text at index {index} must be a string")
        if not text.strip():
            raise EmbeddingError(f"document text at index {index} must not be empty")
        if len(text) > MAX_EMBEDDING_DOCUMENT_CHARS:
            raise EmbeddingError(
                f"document text at index {index} must not exceed "
                f"{MAX_EMBEDDING_DOCUMENT_CHARS} characters"
            )
    return documents


def validate_query_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise EmbeddingError("query text must not be empty")
    if len(cleaned) > MAX_CUSTOMER_QUERY_CHARS:
        raise EmbeddingError(
            f"query text must not exceed {MAX_CUSTOMER_QUERY_CHARS} characters"
        )
    return cleaned
