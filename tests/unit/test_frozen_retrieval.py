"""Unit tests for FrozenRetrievalService."""

from __future__ import annotations

import ast
import importlib
from copy import deepcopy
from pathlib import Path

import pytest

from customer_claims_rag.application.frozen_retrieval import FrozenRetrievalService
from customer_claims_rag.application.ports import RetrievalPort
from customer_claims_rag.application.settings import (
    FrozenRetrievalConfig,
    load_frozen_retrieval_config,
)
from customer_claims_rag.exceptions import RetrievalError, SearchError
from customer_claims_rag.retrieval.models import SearchResponse, SearchResult
from customer_claims_rag.retrieval.reranker import (
    RerankedCandidate,
    RerankerConfig,
    SourceAuthorityV1Reranker,
    load_reranker_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "retrieval" / "vector_pool_expansion_v1.json"
RERANKER_CONFIG_PATH = PROJECT_ROOT / "configs" / "reranking" / "source_authority_v1.json"


def _config() -> FrozenRetrievalConfig:
    return load_frozen_retrieval_config(FROZEN_CONFIG_PATH)


def _result(
    *,
    rank: int,
    chunk_id: str,
    similarity: float = 0.8,
    chunk_type: str = "faq",
    document_id: str = "doc",
) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"content for {chunk_id}",
        source_path=f"data/02_clean_markdown/{document_id}.md",
        chunk_type=chunk_type,
        topic="topic",
        risk_level="low",
        heading=f"Heading {chunk_id}",
        heading_path=[f"Heading {chunk_id}"],
        section="Section",
        subsection="Sub",
        similarity=similarity,
        distance=1.0 - similarity,
    )


def _response(
    results: list[SearchResult],
    *,
    top_k: int = 24,
    fetch_k: int = 24,
    threshold: float = 0.0,
) -> SearchResponse:
    return SearchResponse(
        query="query",
        top_k=top_k,
        fetch_k=fetch_k,
        similarity_threshold=threshold,
        results=results,
        candidates_fetched=len(results),
        candidates_above_threshold=len(results),
        embedding_model="text-embedding-3-small",
        collection_name="customer_claims",
    )


class ControlledRetriever:
    def __init__(
        self,
        response: SearchResponse,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str) -> SearchResponse:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.response


def _reranked_candidate(
    result: SearchResult,
    *,
    baseline_rank: int,
    candidate_rank: int,
    source_authority_bonus: float = 0.0,
) -> RerankedCandidate:
    copied = deepcopy(result)
    copied.rank = candidate_rank
    return RerankedCandidate(
        result=copied,
        baseline_rank=baseline_rank,
        candidate_rank=candidate_rank,
        source_authority_bonus=source_authority_bonus,
        rerank_score=copied.similarity + source_authority_bonus,
    )


class ControlledReranker:
    def __init__(
        self,
        reranker_id: str = "source-authority-v1",
        *,
        output: list[RerankedCandidate] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.config = RerankerConfig(
            reranker_id=reranker_id,
            version="1.0.0",
            experiment_mode="production-like",
            max_source_bonus=0.03,
            source_authority_mapping={"policy": 0.03},
            tie_breaking=["rerank_score_desc"],
        )
        self.output = output or []
        self.error = error
        self.queries: list[str] = []
        self.candidates: list[list[SearchResult]] = []

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[RerankedCandidate]:
        self.queries.append(query)
        self.candidates.append(deepcopy(candidates))
        if self.error is not None:
            raise self.error
        return self.output


def _service(
    *,
    results: list[SearchResult] | None = None,
    retriever: ControlledRetriever | None = None,
    reranker: ControlledReranker | None = None,
) -> tuple[FrozenRetrievalService, ControlledRetriever, ControlledReranker]:
    response_results = results or [_result(rank=1, chunk_id="doc::1")]
    controlled_retriever = retriever or ControlledRetriever(_response(response_results))
    controlled_reranker = reranker or ControlledReranker(
        output=[
            _reranked_candidate(
                response_results[0],
                baseline_rank=response_results[0].rank,
                candidate_rank=1,
            ),
        ],
    )
    service = FrozenRetrievalService(
        retriever=controlled_retriever,
        reranker=controlled_reranker,
        config=_config(),
    )
    return service, controlled_retriever, controlled_reranker


def test_exact_query_passed_to_retriever() -> None:
    service, retriever, _ = _service()
    service.search("Упаковка была вскрыта.")
    assert retriever.queries == ["Упаковка была вскрыта."]


def test_search_response_settings_verified() -> None:
    service, _, _ = _service()
    final = service.search("query")
    assert len(final) == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("top_k", 4, "top_k mismatch"),
        ("fetch_k", 12, "fetch_k mismatch"),
        ("threshold", 0.5, "similarity_threshold mismatch"),
    ],
)
def test_misconfigured_search_response_rejected(
    field: str,
    value: int | float,
    match: str,
) -> None:
    kwargs = {"top_k": 24, "fetch_k": 24, "threshold": 0.0}
    kwargs[field] = value
    retriever = ControlledRetriever(
        _response([_result(rank=1, chunk_id="doc::1")], **kwargs),
    )
    service = FrozenRetrievalService(
        retriever=retriever,
        reranker=ControlledReranker(),
        config=_config(),
    )
    with pytest.raises(RetrievalError, match=match):
        service.search("query")


def test_results_count_greater_than_top_k_rejected() -> None:
    results = [_result(rank=index, chunk_id=f"doc::{index}") for index in range(1, 6)]
    retriever = ControlledRetriever(_response(results, top_k=4, fetch_k=4))
    service = FrozenRetrievalService(
        retriever=retriever,
        reranker=ControlledReranker(),
        config=FrozenRetrievalConfig(
            config_id="test",
            version="1.0.0",
            vector_top_k=4,
            vector_fetch_k=4,
            similarity_threshold=0.0,
            candidate_pool_k=4,
            final_top_k=2,
            reranker_id="source-authority-v1",
        ),
    )
    with pytest.raises(RetrievalError, match="more results than top_k"):
        service.search("query")


def test_candidate_pool_limited_to_config_pool_k() -> None:
    results = [
        _result(rank=index, chunk_id=f"doc::{index}", similarity=1.0 - index * 0.01)
        for index in range(1, 25)
    ]
    reranker = ControlledReranker(
        output=[
            _reranked_candidate(
                item,
                baseline_rank=item.rank,
                candidate_rank=index,
            )
            for index, item in enumerate(results, start=1)
        ],
    )
    service, _, controlled_reranker = _service(results=results, reranker=reranker)
    service.search("query")
    assert len(controlled_reranker.candidates[0]) == 24
    assert controlled_reranker.candidates[0][0].chunk_id == "doc::1"
    assert controlled_reranker.candidates[0][-1].chunk_id == "doc::24"


def test_candidate_pool_ranks_normalized_before_reranker() -> None:
    results = [
        _result(rank=3, chunk_id="doc::3"),
        _result(rank=1, chunk_id="doc::1"),
        _result(rank=2, chunk_id="doc::2"),
    ]
    service, _, controlled_reranker = _service(results=results)
    service.search("query")
    assert [item.rank for item in controlled_reranker.candidates[0]] == [1, 2, 3]


def test_input_search_results_not_mutated() -> None:
    results = [
        _result(rank=3, chunk_id="doc::3"),
        _result(rank=1, chunk_id="doc::1"),
    ]
    original = deepcopy(results)
    service, _, _ = _service(results=results)
    service.search("query")
    assert results == original


def test_reranker_receives_exact_query() -> None:
    service, _, reranker = _service()
    service.search("exact query")
    assert reranker.queries == ["exact query"]


def test_final_results_limited_to_final_top_k() -> None:
    results = [
        _result(rank=index, chunk_id=f"doc::{index}", similarity=1.0 - index * 0.01)
        for index in range(1, 16)
    ]
    reranked = [
        _reranked_candidate(
            item,
            baseline_rank=item.rank,
            candidate_rank=index,
        )
        for index, item in enumerate(results, start=1)
    ]
    service, _, _ = _service(results=results, reranker=ControlledReranker(output=reranked))
    final = service.search("query")
    assert len(final) == 12
    assert [item.chunk_id for item in final] == [f"doc::{index}" for index in range(1, 13)]


def test_final_ranks_are_contiguous() -> None:
    service, _, _ = _service()
    final = service.search("query")
    assert [item.rank for item in final] == [1]


def test_metadata_preserved_in_final_results() -> None:
    source = _result(rank=1, chunk_id="doc::1", similarity=0.91, chunk_type="policy")
    reranker = ControlledReranker(
        output=[
            _reranked_candidate(
                source,
                baseline_rank=1,
                candidate_rank=1,
                source_authority_bonus=0.03,
            ),
        ],
    )
    service, _, _ = _service(results=[source], reranker=reranker)
    final = service.search("query")[0]
    assert final.chunk_id == "doc::1"
    assert final.document_id == "doc"
    assert final.content == "content for doc::1"
    assert final.source_path == "data/02_clean_markdown/doc.md"
    assert final.heading == "Heading doc::1"
    assert final.chunk_type == "policy"
    assert final.similarity == 0.91
    assert final.distance == pytest.approx(0.09)


def test_fewer_than_pool_size_works() -> None:
    results = [_result(rank=1, chunk_id="doc::1"), _result(rank=2, chunk_id="doc::2")]
    reranker = ControlledReranker(
        output=[
            _reranked_candidate(results[1], baseline_rank=2, candidate_rank=1),
            _reranked_candidate(results[0], baseline_rank=1, candidate_rank=2),
        ],
    )
    service, _, controlled_reranker = _service(results=results, reranker=reranker)
    final = service.search("query")
    assert len(controlled_reranker.candidates[0]) == 2
    assert len(final) == 2
    assert [item.rank for item in final] == [1, 2]


def test_empty_retrieval_returns_empty_list_without_reranker_call() -> None:
    retriever = ControlledRetriever(_response([]))
    reranker = ControlledReranker()
    service = FrozenRetrievalService(
        retriever=retriever,
        reranker=reranker,
        config=_config(),
    )
    assert service.search("query") == []
    assert reranker.queries == []


def test_search_error_propagates() -> None:
    retriever = ControlledRetriever(_response([]), error=SearchError("search failed"))
    service = FrozenRetrievalService(
        retriever=retriever,
        reranker=ControlledReranker(),
        config=_config(),
    )
    with pytest.raises(SearchError, match="search failed"):
        service.search("query")


def test_reranker_runtime_error_propagates() -> None:
    service, _, _ = _service(
        reranker=ControlledReranker(error=RuntimeError("reranker bug")),
    )
    with pytest.raises(RuntimeError, match="reranker bug"):
        service.search("query")


def test_reranker_id_mismatch_rejected_at_constructor() -> None:
    with pytest.raises(ValueError, match="requires reranker 'source-authority-v1'"):
        FrozenRetrievalService(
            retriever=ControlledRetriever(_response([])),
            reranker=ControlledReranker(reranker_id="other-reranker"),
            config=_config(),
        )


def test_provenance_properties() -> None:
    service, _, _ = _service()
    assert service.retrieval_config_id == "vector-pool-expansion-v1"
    assert service.retrieval_config_version == "1.0.0"
    assert service.reranker_config_id == "source-authority-v1"


def test_deterministic_output_for_same_input() -> None:
    results = [
        _result(rank=1, chunk_id="faq::1", similarity=0.80, chunk_type="faq"),
        _result(rank=2, chunk_id="policy::1", similarity=0.79, chunk_type="policy"),
    ]
    reranker_impl = SourceAuthorityV1Reranker(load_reranker_config(RERANKER_CONFIG_PATH))
    retriever = ControlledRetriever(_response(results))
    service = FrozenRetrievalService(
        retriever=retriever,
        reranker=reranker_impl,
        config=_config(),
    )
    first = service.search("query")
    second = service.search("query")
    assert first == second
    assert [item.chunk_id for item in first] == ["policy::1", "faq::1"]


def test_service_structurally_satisfies_retrieval_port() -> None:
    service, _, _ = _service()

    def use_port(port: RetrievalPort) -> int:
        return len(port.search("query"))

    assert use_port(service) == 1


def test_original_search_result_ranks_unchanged_after_service_search() -> None:
    results = [
        _result(rank=5, chunk_id="doc::5"),
        _result(rank=2, chunk_id="doc::2"),
    ]
    original_ranks = [item.rank for item in results]
    service, _, _ = _service(results=results)
    service.search("query")
    assert [item.rank for item in results] == original_ranks


def test_non_contiguous_reranker_ranks_rejected() -> None:
    broken = _result(rank=9, chunk_id="doc::1")
    reranker = ControlledReranker(
        output=[
            _reranked_candidate(broken, baseline_rank=1, candidate_rank=9),
        ],
    )
    service, _, _ = _service(reranker=reranker)
    with pytest.raises(RetrievalError, match="final retrieval ranks"):
        service.search("query")


def test_service_module_does_not_import_evaluation() -> None:
    module = importlib.import_module("customer_claims_rag.application.frozen_retrieval")
    source_path = Path(module.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = (
        "customer_claims_rag.evaluation",
        "customer_claims_rag.cli",
        "streamlit",
        "customer_claims_rag.generation",
        "customer_claims_rag.risk",
        "customer_claims_rag.retrieval.adapters",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for item in forbidden:
                    assert item not in alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            for item in forbidden:
                assert item not in node.module
