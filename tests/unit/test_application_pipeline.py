"""Unit tests for CustomerClaimsPipeline."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.application.models import CustomerClaimsRequest
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.exceptions import GenerationValidationError, SearchError
from customer_claims_rag.generation.fallback import INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
from customer_claims_rag.generation.handoff import HIGH_HANDOFF_NOTICE
from customer_claims_rag.generation.models import (
    Citation,
    ContextPackage,
    GroundedGenerationRequest,
    GroundedGenerationResult,
)
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import RiskAssessmentRequest, RiskLevel
from customer_claims_rag.retrieval.models import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_APPLICATION_IMPORTS = (
    "customer_claims_rag.evaluation",
    "streamlit",
    "chromadb",
    "customer_claims_rag.retrieval.adapters",
    "customer_claims_rag.generation.adapters",
    "customer_claims_rag.cli",
    "langchain",
    "langchain_openai",
    "openai",
)


def _search_result(
    *,
    rank: int,
    chunk_id: str,
    document_id: str = "06_food_quality_and_packaging",
    content: str = "Policy text about packaging.",
) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        source_path=f"data/02_clean_markdown/{document_id}.md",
        chunk_type="policy",
        topic="packaging",
        risk_level="high",
        heading="Упаковка",
        heading_path=["Упаковка"],
        section="Section",
        subsection="Sub",
        similarity=0.91,
        distance=0.09,
    )


def _risk_aware_generation_result() -> RiskAwareGroundedGenerationResult:
    risk = assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Упаковка была вскрыта."),
    )
    return RiskAwareGroundedGenerationResult(
        generation=GroundedGenerationResult(
            response_mode="grounded_answer",
            customer_response="По правилам вскрытая упаковка фиксируется [S1].",
            citations=[
                Citation(
                    citation_key="S1",
                    document_id="06_food_quality_and_packaging",
                    chunk_id="06_food_quality_and_packaging::1",
                    heading="Упаковка",
                    source_path="data/02_clean_markdown/06_food_quality_and_packaging.md",
                ),
            ],
        ),
        risk_assessment=risk,
        handoff_notice=HIGH_HANDOFF_NOTICE,
        generation_outcome="grounded_answer",
    )


class ControlledRetrieval:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = [] if results is None else results
        self.error = error
        self.queries: list[str] = []
        self.call_count = 0

    def search(self, query: str) -> list[SearchResult]:
        self.call_count += 1
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return list(self.results)


class ControlledGeneration:
    def __init__(
        self,
        response: RiskAwareGroundedGenerationResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response or _risk_aware_generation_result()
        self.error = error
        self.requests: list[GroundedGenerationRequest] = []
        self.call_count = 0

    def generate(
        self,
        request: GroundedGenerationRequest,
    ) -> RiskAwareGroundedGenerationResult:
        self.call_count += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


def test_pipeline_success_with_controlled_ports() -> None:
    retrieval = ControlledRetrieval(
        [
            _search_result(rank=1, chunk_id="06_food_quality_and_packaging::1"),
        ]
    )
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(retrieval=retrieval, generator=generation)

    request = CustomerClaimsRequest(customer_query="Упаковка была вскрыта.")
    result = pipeline.handle(request)

    assert retrieval.queries == ["Упаковка была вскрыта."]
    assert generation.call_count == 1
    assert generation.requests[0].customer_query == "Упаковка была вскрыта."
    assert len(generation.requests[0].context_package.items) == 1
    assert generation.requests[0].context_package.items[0].chunk_id == (
        "06_food_quality_and_packaging::1"
    )
    assert generation.requests[0].context_package.items[0].citation_key == "S1"

    assert result.response == generation.response
    assert result.response.generation.customer_response == (
        "По правилам вскрытая упаковка фиксируется [S1]."
    )
    assert len(result.response.generation.citations) == 1
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE
    assert result.response.generation_outcome == "grounded_answer"


def test_pipeline_search_result_ordering_uses_context_builder() -> None:
    retrieval = ControlledRetrieval(
        [
            _search_result(rank=3, chunk_id="doc::3", document_id="doc-c"),
            _search_result(rank=1, chunk_id="doc::1", document_id="doc-a"),
            _search_result(rank=2, chunk_id="doc::2", document_id="doc-b"),
        ]
    )
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(retrieval=retrieval, generator=generation)

    pipeline.handle(CustomerClaimsRequest(customer_query="Упаковка была вскрыта."))

    items = generation.requests[0].context_package.items
    assert [item.citation_key for item in items] == ["S1", "S2", "S3"]
    assert [item.chunk_id for item in items] == ["doc::1", "doc::2", "doc::3"]
    assert [item.rank for item in items] == [1, 2, 3]


def test_pipeline_query_consistency_after_strip() -> None:
    retrieval = ControlledRetrieval(
        [_search_result(rank=1, chunk_id="doc::1", document_id="doc-a")]
    )
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(retrieval=retrieval, generator=generation)

    pipeline.handle(
        CustomerClaimsRequest(customer_query="  Не привезли одну позицию.  "),
    )

    assert retrieval.queries == ["Не привезли одну позицию."]
    assert generation.requests[0].customer_query == "Не привезли одну позицию."


def test_pipeline_retrieval_error_propagates() -> None:
    retrieval = ControlledRetrieval(error=SearchError("search failed"))
    context_builder = MagicMock()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=retrieval,
        generator=generation,
        context_builder=context_builder,
    )

    with pytest.raises(SearchError, match="search failed"):
        pipeline.handle(CustomerClaimsRequest(customer_query="Упаковка была вскрыта."))

    context_builder.assert_not_called()
    assert generation.call_count == 0


def test_pipeline_context_builder_error_propagates() -> None:
    retrieval = ControlledRetrieval(
        [_search_result(rank=1, chunk_id="doc::1")]
    )
    generation = ControlledGeneration()

    def failing_builder(results):
        raise GenerationValidationError("duplicate rank in context input: 1")

    pipeline = CustomerClaimsPipeline(
        retrieval=retrieval,
        generator=generation,
        context_builder=failing_builder,
    )

    with pytest.raises(GenerationValidationError, match="duplicate rank"):
        pipeline.handle(CustomerClaimsRequest(customer_query="Упаковка была вскрыта."))

    assert generation.call_count == 0


def test_pipeline_generation_programmer_error_propagates() -> None:
    retrieval = ControlledRetrieval(
        [_search_result(rank=1, chunk_id="doc::1")]
    )
    generation = ControlledGeneration(error=RuntimeError("programmer bug"))
    pipeline = CustomerClaimsPipeline(retrieval=retrieval, generator=generation)

    with pytest.raises(RuntimeError, match="programmer bug"):
        pipeline.handle(CustomerClaimsRequest(customer_query="Упаковка была вскрыта."))


def test_pipeline_empty_retrieval_calls_generator_with_empty_context() -> None:
    retrieval = ControlledRetrieval([])
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(retrieval=retrieval, generator=generation)

    pipeline.handle(CustomerClaimsRequest(customer_query="Упаковка была вскрыта."))

    assert generation.call_count == 1
    assert generation.requests[0].context_package == ContextPackage(items=[])


def test_application_package_does_not_import_forbidden_modules() -> None:
    application_root = PROJECT_ROOT / "src" / "customer_claims_rag" / "application"
    for module_info in pkgutil.walk_packages(
        [str(application_root)],
        prefix="customer_claims_rag.application.",
    ):
        module = importlib.import_module(module_info.name)
        source_path = Path(module.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in _FORBIDDEN_APPLICATION_IMPORTS:
                        assert forbidden not in alias.name
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in _FORBIDDEN_APPLICATION_IMPORTS:
                    assert forbidden not in node.module
