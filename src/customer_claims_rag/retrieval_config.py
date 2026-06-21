"""Retrieval and index configuration with env and CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from customer_claims_rag.config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FETCH_K,
    DEFAULT_INDEX_DIR,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
)
from customer_claims_rag.env_bootstrap import load_project_env
from customer_claims_rag.exceptions import RetrievalError


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value.strip())


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value.strip())


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return Path(value.strip())


@dataclass(frozen=True)
class RetrievalSettings:
    """Validated settings for index build and search."""

    index_dir: Path
    collection_name: str
    embedding_model: str
    top_k: int
    fetch_k: int
    similarity_threshold: float
    embedding_batch_size: int
    openai_api_key: str | None = None

    @classmethod
    def from_env(cls) -> RetrievalSettings:
        load_project_env()
        return cls(
            index_dir=_env_path("RAG_INDEX_DIR", DEFAULT_INDEX_DIR),
            collection_name=_env_str("RAG_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
            embedding_model=_env_str("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            top_k=_env_int("RAG_TOP_K", DEFAULT_TOP_K),
            fetch_k=_env_int("RAG_FETCH_K", DEFAULT_FETCH_K),
            similarity_threshold=_env_float(
                "RAG_SIMILARITY_THRESHOLD",
                DEFAULT_SIMILARITY_THRESHOLD,
            ),
            embedding_batch_size=_env_int(
                "RAG_EMBEDDING_BATCH_SIZE",
                DEFAULT_EMBEDDING_BATCH_SIZE,
            ),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def validate_search_params(
        self,
        *,
        top_k: int | None = None,
        fetch_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> tuple[int, int, float]:
        resolved_top_k = self.top_k if top_k is None else top_k
        resolved_fetch_k = self.fetch_k if fetch_k is None else fetch_k
        resolved_threshold = (
            self.similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )
        validate_top_k(resolved_top_k)
        validate_fetch_k(resolved_fetch_k, resolved_top_k)
        validate_similarity_threshold(resolved_threshold)
        return resolved_top_k, resolved_fetch_k, resolved_threshold

    def validate_batch_size(self, batch_size: int | None = None) -> int:
        resolved = self.embedding_batch_size if batch_size is None else batch_size
        if resolved < 1:
            raise RetrievalError("embedding batch size must be >= 1")
        return resolved


def validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise RetrievalError("top_k must be >= 1")


def validate_fetch_k(fetch_k: int, top_k: int) -> None:
    if fetch_k < top_k:
        raise RetrievalError("fetch_k must be >= top_k")


def validate_similarity_threshold(threshold: float) -> None:
    if not MIN_SIMILARITY_THRESHOLD <= threshold <= MAX_SIMILARITY_THRESHOLD:
        raise RetrievalError(
            f"similarity_threshold must be in [{MIN_SIMILARITY_THRESHOLD}, "
            f"{MAX_SIMILARITY_THRESHOLD}]"
        )
