"""Map ingestion chunks to scalar-safe vector store metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.models import VectorSearchHit
from customer_claims_rag.retrieval.similarity import distance_to_similarity


def _normalize_source_path(source_path: str) -> str:
    return Path(source_path).as_posix()


def _serialize_heading_path(heading_path: list[str]) -> str:
    return json.dumps(heading_path, ensure_ascii=False, separators=(",", ":"))


def _deserialize_heading_path(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("heading_path metadata must be a JSON list")
    return [str(item) for item in parsed]


def canonical_metadata_for_fingerprint(chunk: ChunkRecord) -> dict[str, Any]:
    """Subset of chunk metadata included in corpus fingerprint."""
    metadata = chunk.metadata
    payload: dict[str, Any] = {
        "document_id": chunk.document_id,
        "source_path": _normalize_source_path(chunk.source_path),
        "chunk_type": chunk.strategy,
        "heading": chunk.heading,
        "heading_path": chunk.heading_path,
        "section": metadata.section,
        "subsection": metadata.subsection,
        "topic": metadata.topic,
        "risk_level": metadata.risk_level,
        "title": chunk.title,
        "category": chunk.category,
        "document_type": chunk.document_type,
        "status": metadata.status,
        "version": metadata.version,
        "priority": metadata.priority,
        "language": metadata.language,
    }
    if metadata.related_documents:
        payload["related_documents"] = sorted(metadata.related_documents)
    if metadata.keywords:
        payload["keywords"] = sorted(metadata.keywords)
    return payload


def canonical_metadata_from_vector_metadata(stored: Mapping[str, Any]) -> dict[str, Any]:
    """The fingerprint metadata subset, rebuilt from a stored record.

    The inverse view of ``chunk_to_vector_metadata`` for exactly the keys that
    ``canonical_metadata_for_fingerprint`` hashes, so that for every chunk
    ``canonical_metadata_from_vector_metadata(chunk_to_vector_metadata(chunk))`` equals
    ``canonical_metadata_for_fingerprint(chunk)``. Raises ``KeyError`` naming the first field a
    record lacks (an index written before the field existed).
    """
    payload: dict[str, Any] = {
        "document_id": str(stored["document_id"]),
        "source_path": _normalize_source_path(str(stored["source_path"])),
        "chunk_type": str(stored["chunk_type"]),
        "heading": str(stored["heading"]),
        "heading_path": _deserialize_heading_path(str(stored["heading_path"])),
        "section": _optional_str(stored.get("section")),
        "subsection": _optional_str(stored.get("subsection")),
        "topic": _optional_str(stored.get("topic")),
        "risk_level": _optional_str(stored.get("risk_level")),
        "title": str(stored["title"]),
        "category": str(stored["category"]),
        "document_type": str(stored["document_type"]),
        "status": str(stored["status"]),
        "version": str(stored["version"]),
        "priority": str(stored["document_priority"]),
        "language": str(stored["language"]),
    }
    related = _decode_json_list(stored.get("related_documents"))
    if related:
        payload["related_documents"] = sorted(related)
    keywords = _decode_json_list(stored.get("keywords"))
    if keywords:
        payload["keywords"] = sorted(keywords)
    return payload


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _decode_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("list metadata must be a JSON list")
    return [str(item) for item in parsed]


def chunk_to_vector_metadata(chunk: ChunkRecord) -> dict[str, str | int | float | bool]:
    """Convert a chunk to Chroma-compatible scalar metadata."""
    metadata = chunk.metadata
    store_metadata: dict[str, str | int | float | bool] = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_path": _normalize_source_path(chunk.source_path),
        "chunk_type": chunk.strategy,
        "heading": chunk.heading,
        "heading_path": _serialize_heading_path(chunk.heading_path),
        "title": chunk.title,
        "category": chunk.category,
        "document_type": chunk.document_type,
        "document_priority": chunk.document_priority,
        "status": metadata.status,
        "language": metadata.language,
        # Part of the corpus fingerprint payload, so the stored record must carry it for the
        # fingerprint to be recomputable from the store (release/index_integrity.py).
        "version": metadata.version,
    }
    if metadata.section is not None:
        store_metadata["section"] = metadata.section
    if metadata.subsection is not None:
        store_metadata["subsection"] = metadata.subsection
    if metadata.topic is not None:
        store_metadata["topic"] = metadata.topic
    if metadata.risk_level is not None:
        store_metadata["risk_level"] = metadata.risk_level
    if metadata.related_documents:
        store_metadata["related_documents"] = json.dumps(
            sorted(metadata.related_documents),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if metadata.keywords:
        store_metadata["keywords"] = json.dumps(
            sorted(metadata.keywords),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return store_metadata


def vector_metadata_to_search_hit(
    *,
    content: str,
    metadata: dict[str, Any],
    distance: float,
    backend_rank: int = 0,
) -> VectorSearchHit:
    heading_path_raw = metadata.get("heading_path", "[]")
    if isinstance(heading_path_raw, list):
        heading_path = [str(item) for item in heading_path_raw]
    else:
        heading_path = _deserialize_heading_path(str(heading_path_raw))

    topic = metadata.get("topic")
    risk_level = metadata.get("risk_level")
    section = metadata.get("section")
    subsection = metadata.get("subsection")

    return VectorSearchHit(
        chunk_id=str(metadata["chunk_id"]),
        document_id=str(metadata["document_id"]),
        content=content,
        source_path=_normalize_source_path(str(metadata["source_path"])),
        chunk_type=str(metadata["chunk_type"]),
        topic=str(topic) if topic is not None else None,
        risk_level=str(risk_level) if risk_level is not None else None,
        heading=str(metadata["heading"]),
        heading_path=heading_path,
        section=str(section) if section is not None else None,
        subsection=str(subsection) if subsection is not None else None,
        distance=float(distance),
        similarity=distance_to_similarity(float(distance)),
        backend_rank=backend_rank,
    )
