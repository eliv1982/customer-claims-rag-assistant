"""Deterministic vector index builder."""

from __future__ import annotations

import time
from pathlib import Path

from customer_claims_rag.config import METADATA_SCHEMA_VERSION
from customer_claims_rag.exceptions import DuplicateChunkIdError, IndexBuildError
from customer_claims_rag.ingestion.corpus_builder import CorpusBuilder
from customer_claims_rag.models import ChunkRecord, DocumentRecord
from customer_claims_rag.retrieval.fingerprint import compute_corpus_fingerprint, sort_chunks_deterministic
from customer_claims_rag.retrieval.manifest import (
    build_manifest,
    invalidate_manifest,
    write_manifest_atomic,
)
from customer_claims_rag.retrieval.models import IndexBuildReport
from customer_claims_rag.retrieval.ports import EmbeddingProvider, VectorStore


class IndexBuilder:
    """Build a persistent vector index from ingestion corpus."""

    def __init__(
        self,
        *,
        corpus_builder: CorpusBuilder,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        index_dir: Path,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise IndexBuildError("embedding batch size must be >= 1")
        self.corpus_builder = corpus_builder
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.index_dir = index_dir
        self.batch_size = batch_size

    def build_from_directory(
        self,
        input_dir: Path,
        *,
        rebuild: bool = True,
    ) -> IndexBuildReport:
        if not rebuild:
            raise IndexBuildError(
                "incremental indexing is not supported in MVP; use --rebuild"
            )

        started = time.perf_counter()
        documents, chunks = self.corpus_builder.build_from_directory(input_dir)
        self._validate_corpus(chunks)

        ordered_chunks = sort_chunks_deterministic(chunks)
        fingerprint = compute_corpus_fingerprint(
            ordered_chunks,
            embedding_model=self.embedding_provider.model_name,
        )

        embeddings = self._embed_all(ordered_chunks)
        if not embeddings:
            raise IndexBuildError("embedding provider returned no vectors")

        dimension = len(embeddings[0])
        invalidate_manifest(self.index_dir)
        try:
            self.vector_store.recreate_collection(embedding_dimension=dimension)
            self.vector_store.add_chunks(ordered_chunks, embeddings)
            indexed_count = self.vector_store.count()
            if indexed_count != len(ordered_chunks):
                raise IndexBuildError(
                    f"indexed chunk count mismatch: expected {len(ordered_chunks)}, "
                    f"got {indexed_count}"
                )

            manifest = build_manifest(
                collection_name=self.vector_store.collection_name,
                embedding_model=self.embedding_provider.model_name,
                corpus_fingerprint=fingerprint,
                chunk_count=len(ordered_chunks),
                document_count=len(documents),
                metadata_schema_version=METADATA_SCHEMA_VERSION,
                vector_dimension=dimension,
            )
            write_manifest_atomic(self.index_dir, manifest)
        except IndexBuildError:
            raise
        except Exception as exc:
            raise IndexBuildError(f"index rebuild failed: {exc}") from None

        elapsed = time.perf_counter() - started
        return IndexBuildReport(
            documents=len(documents),
            chunks=len(ordered_chunks),
            embedding_model=self.embedding_provider.model_name,
            collection_name=self.vector_store.collection_name,
            index_dir=str(self.index_dir),
            vector_dimension=dimension,
            fingerprint=fingerprint,
            elapsed_seconds=elapsed,
            status="success",
        )

    def _validate_corpus(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            raise IndexBuildError("corpus contains no chunks to index")
        try:
            self.corpus_builder.ensure_unique_chunk_ids(chunks)
        except DuplicateChunkIdError as exc:
            raise IndexBuildError(str(exc)) from exc

    def _embed_all(self, chunks: list[ChunkRecord]) -> list[list[float]]:
        texts = [chunk.content for chunk in chunks]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                batch_vectors = self.embedding_provider.embed_documents(batch)
            except Exception as exc:
                raise IndexBuildError("embedding batch failed") from None
            if len(batch_vectors) != len(batch):
                raise IndexBuildError(
                    f"embedding batch size mismatch: expected {len(batch)}, "
                    f"got {len(batch_vectors)}"
                )
            vectors.extend(batch_vectors)
        return vectors
