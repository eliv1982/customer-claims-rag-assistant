"""Baseline dense semantic retriever."""

from __future__ import annotations

from pathlib import Path

from customer_claims_rag.config import DEFAULT_SIMILARITY_THRESHOLD, INDEX_FORMAT_VERSION, METADATA_SCHEMA_VERSION
from customer_claims_rag.exceptions import SearchError
from customer_claims_rag.retrieval.manifest import load_manifest, validate_manifest_against_runtime
from customer_claims_rag.retrieval.models import SearchResponse, SearchResult, VectorSearchHit
from customer_claims_rag.retrieval.ports import EmbeddingProvider, VectorStore
from customer_claims_rag.retrieval_config import validate_fetch_k, validate_similarity_threshold, validate_top_k
from customer_claims_rag.retrieval.embedding_validation import validate_query_text


class BaselineRetriever:
    """Cosine similarity retrieval without reranking or query classification."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        index_dir: Path,
        top_k: int = 4,
        fetch_k: int = 12,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        validate_top_k(top_k)
        validate_fetch_k(fetch_k, top_k)
        validate_similarity_threshold(similarity_threshold)
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.index_dir = index_dir
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.similarity_threshold = similarity_threshold

    def search(self, query: str) -> SearchResponse:
        cleaned_query = validate_query_text(query)
        manifest = load_manifest(self.index_dir)
        store_count = self.vector_store.count()
        validate_manifest_against_runtime(
            manifest,
            collection_name=self.vector_store.collection_name,
            embedding_model=self.embedding_provider.model_name,
            chunk_count=store_count,
            expected_index_format_version=INDEX_FORMAT_VERSION,
            expected_metadata_schema_version=METADATA_SCHEMA_VERSION,
        )

        try:
            query_vector = self.embedding_provider.embed_query(cleaned_query)
        except Exception as exc:
            raise SearchError(f"search failed: {exc}") from None

        validate_manifest_against_runtime(
            manifest,
            collection_name=self.vector_store.collection_name,
            embedding_model=self.embedding_provider.model_name,
            chunk_count=store_count,
            expected_index_format_version=INDEX_FORMAT_VERSION,
            expected_metadata_schema_version=METADATA_SCHEMA_VERSION,
            query_vector_dimension=len(query_vector),
        )

        try:
            candidates = self.vector_store.similarity_search(query_vector, k=self.fetch_k)
        except Exception:
            raise SearchError("search failed") from None

        filtered = [
            hit for hit in candidates if hit.similarity >= self.similarity_threshold
        ]
        ranked = self._rank_hits(filtered)[: self.top_k]

        return SearchResponse(
            query=cleaned_query,
            top_k=self.top_k,
            fetch_k=self.fetch_k,
            similarity_threshold=self.similarity_threshold,
            results=[self._to_search_result(index + 1, hit) for index, hit in enumerate(ranked)],
            candidates_fetched=len(candidates),
            candidates_above_threshold=len(filtered),
            embedding_model=self.embedding_provider.model_name,
            collection_name=self.vector_store.collection_name,
        )

    def _rank_hits(self, hits: list[VectorSearchHit]) -> list[VectorSearchHit]:
        return sorted(
            hits,
            key=lambda hit: (-hit.similarity, hit.chunk_id),
        )

    @staticmethod
    def _to_search_result(rank: int, hit: VectorSearchHit) -> SearchResult:
        return SearchResult(
            rank=rank,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            content=hit.content,
            source_path=hit.source_path,
            chunk_type=hit.chunk_type,
            topic=hit.topic,
            risk_level=hit.risk_level,
            heading=hit.heading,
            heading_path=hit.heading_path,
            section=hit.section,
            subsection=hit.subsection,
            similarity=hit.similarity,
            distance=hit.distance,
        )
