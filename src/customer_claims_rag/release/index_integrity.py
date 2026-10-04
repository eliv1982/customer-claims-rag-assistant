"""Recompute what a vector index actually contains, instead of trusting what it says it contains.

``manifest.json`` next to a Chroma index is self-attestation: whoever (or whatever) wrote it can
claim any fingerprint, count or model. The release validator therefore recomputes, from the records
that are really stored, every value that can be recomputed cheaply and offline, and compares the
result with both the manifest and the identity committed in the canonical corpus manifest:

========================  ===================================  ================================
value                     recomputed from the store            compared with
========================  ===================================  ================================
chunk count, document ids  record ids / metadata                canonical corpus manifest
chunk payload digest       chunk ids, texts, metadata           canonical corpus manifest + index
                                                                manifest
corpus fingerprint         the same, plus the model *name*      canonical corpus manifest + index
                                                                manifest
collection content digest  texts, metadata subset, vectors      index manifest
vector dimension           length of every stored vector        release descriptor + index manifest
========================  ===================================  ================================

What stays trusted, and why it cannot be otherwise offline: that the stored vectors were produced
by the named embedding model (only the name, via the fingerprint, and the dimension are checkable),
and ``embedding_digest`` (it hashes the vectors the provider returned, each component serialized
as an IEEE-754 half-precision value, 2 bytes big-endian; Chroma stores float32 copies, and rounding
float64 -> float32 -> half is not always the same as float64 -> half, so it cannot be re-derived
exactly from the store; ``collection_content_digest`` plays that role for the store itself).
Reading the records is a few hundred rows; no embedding request is ever made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from customer_claims_rag.exceptions import ReleasePostureError
from customer_claims_rag.retrieval.index_identity import (
    compute_chunk_payload_digest_from_records,
    compute_collection_content_digest,
    compute_corpus_fingerprint_from_records,
)
from customer_claims_rag.retrieval.ports import VectorStore


@dataclass(frozen=True)
class StoreIdentity:
    """Identity of an index as recomputed from its stored records."""

    chunk_count: int
    document_ids: frozenset[str]
    vector_dimensions: frozenset[int]
    chunk_payload_digest: str
    corpus_fingerprint: str
    collection_content_digest: str


def recompute_store_identity(
    vector_store: VectorStore,
    *,
    embedding_model: str,
    rebuild_hint: str,
) -> StoreIdentity:
    """Export the stored records and recompute the identity values from them."""
    export = getattr(vector_store, "export_collection_records", None)
    if not callable(export):
        raise ReleasePostureError(
            "vector store cannot export its records, so the index content cannot be verified "
            "against the canonical corpus"
        )
    try:
        records: list[dict[str, Any]] = list(export())
    except Exception as exc:
        raise ReleasePostureError(f"failed to read the stored index records: {exc}") from exc
    if not records:
        raise ReleasePostureError("the index holds no readable records; " + rebuild_hint)
    try:
        chunk_payload_digest = compute_chunk_payload_digest_from_records(records)
        corpus_fingerprint = compute_corpus_fingerprint_from_records(
            records,
            embedding_model=embedding_model,
        )
    except (KeyError, ValueError) as exc:
        raise ReleasePostureError(
            f"stored index records are incomplete or malformed ({exc!r}); the index was not "
            f"built by the current build path; {rebuild_hint}"
        ) from exc
    return StoreIdentity(
        chunk_count=len(records),
        document_ids=frozenset(str(record["metadata"]["document_id"]) for record in records),
        vector_dimensions=frozenset(len(record["embedding"]) for record in records),
        chunk_payload_digest=chunk_payload_digest,
        corpus_fingerprint=corpus_fingerprint,
        collection_content_digest=compute_collection_content_digest(records),
    )
