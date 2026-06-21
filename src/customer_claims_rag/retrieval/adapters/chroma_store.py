"""Persistent Chroma vector store adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb

from customer_claims_rag.exceptions import VectorStoreError
from customer_claims_rag.models import ChunkRecord
from customer_claims_rag.retrieval.metadata_mapper import (
    chunk_to_vector_metadata,
    vector_metadata_to_search_hit,
)
from customer_claims_rag.retrieval.models import VectorSearchHit


class ChromaVectorStore:
    """Local persistent Chroma collection using cosine space."""

    def __init__(
        self,
        *,
        index_dir: Path,
        collection_name: str,
    ) -> None:
        self._index_dir = index_dir
        self._collection_name = collection_name
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._index_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def recreate_collection(self, *, embedding_dimension: int) -> None:
        if embedding_dimension < 1:
            raise VectorStoreError("embedding dimension must be >= 1")
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"chunk/embedding count mismatch: {len(chunks)} chunks, "
                f"{len(embeddings)} embeddings"
            )
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk_to_vector_metadata(chunk) for chunk in chunks]
        vectors = [list(vector) for vector in embeddings]

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            try:
                self._collection.add(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=vectors[start:end],
                )
            except Exception as exc:
                raise VectorStoreError(f"failed to add vectors to Chroma: {exc}") from exc

    def count(self) -> int:
        return int(self._collection.count())

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
    ) -> list[VectorSearchHit]:
        if k < 1:
            raise VectorStoreError("similarity search k must be >= 1")
        try:
            result = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma similarity search failed: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[VectorSearchHit] = []
        for content, metadata, distance in zip(documents, metadatas, distances, strict=True):
            if content is None or metadata is None or distance is None:
                continue
            hits.append(
                vector_metadata_to_search_hit(
                    content=content,
                    metadata=metadata,
                    distance=float(distance),
                )
            )
        return hits

    def close(self) -> None:
        self._client = None
        self._collection = None
