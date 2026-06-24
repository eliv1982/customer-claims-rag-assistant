"""Integration tests for CustomerClaimsPipeline without external services."""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.application.models import CustomerClaimsRequest
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.exceptions import LLMCallError
from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.handoff import CRITICAL_HANDOFF_NOTICE, HIGH_HANDOFF_NOTICE
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.risk.models import RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.retrieval.models import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "grounded_answer_v1.md"


def _fixed_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            rank=1,
            chunk_id="06_food_quality_and_packaging::1",
            document_id="06_food_quality_and_packaging",
            content="Правила фиксации вскрытой упаковки и качества продукции.",
            source_path="data/02_clean_markdown/06_food_quality_and_packaging.md",
            chunk_type="policy",
            topic="packaging",
            risk_level="high",
            heading="Упаковка",
            heading_path=["Упаковка"],
            section="Section",
            subsection="Sub",
            similarity=0.92,
            distance=0.08,
        ),
    ]


class FixedRetrieval:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
    ) -> None:
        self._results = _fixed_search_results() if results is None else results
        self.queries: list[str] = []

    def search(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        return list(self._results)


def _valid_grounded_json(answer: str) -> str:
    escaped = answer.replace('"', '\\"')
    return f'{{"response_mode":"grounded_answer","answer":"{escaped}"}}'


def _pipeline(
    *,
    retrieval: FixedRetrieval | None = None,
    chat_model: FakeChatModel | None = None,
) -> tuple[CustomerClaimsPipeline, FixedRetrieval, FakeChatModel]:
    fixed_retrieval = retrieval or FixedRetrieval()
    model = chat_model or FakeChatModel(
        response=_valid_grounded_json("По правилам вскрытая упаковка фиксируется [S1]."),
    )
    grounded = GroundedGenerator(
        chat_model=model,
        prompt_builder=PromptBuilder(prompt_path=PROMPT_PATH),
    )
    generator = RiskAwareGroundedGenerator(grounded_generator=grounded)
    pipeline = CustomerClaimsPipeline(retrieval=fixed_retrieval, generator=generator)
    return pipeline, fixed_retrieval, model


def test_successful_grounded_flow() -> None:
    pipeline, retrieval, _ = _pipeline()
    request = CustomerClaimsRequest(customer_query="Упаковка была вскрыта.")

    result = pipeline.handle(request)

    assert retrieval.queries == ["Упаковка была вскрыта."]
    assert result.response.generation_outcome == "grounded_answer"
    assert result.response.generation.response_mode == "grounded_answer"
    assert "[S1]" in result.response.generation.customer_response
    assert len(result.response.generation.citations) == 1
    assert (
        result.response.generation.citations[0].document_id
        == "06_food_quality_and_packaging"
    )
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH


@pytest.mark.parametrize(
    ("customer_query", "expected_level", "expected_handoff"),
    [
        (
            "Где посмотреть правила доставки?",
            RiskLevel.LOW,
            None,
        ),
        (
            "Не привезли одну позицию.",
            RiskLevel.MEDIUM,
            None,
        ),
        (
            "Упаковка была вскрыта.",
            RiskLevel.HIGH,
            HIGH_HANDOFF_NOTICE,
        ),
        (
            "После еды стало трудно дышать.",
            RiskLevel.CRITICAL,
            CRITICAL_HANDOFF_NOTICE,
        ),
    ],
)
def test_risk_tiers_preserved(
    customer_query: str,
    expected_level: RiskLevel,
    expected_handoff: str | None,
) -> None:
    pipeline, _, model = _pipeline()
    result = pipeline.handle(CustomerClaimsRequest(customer_query=customer_query))

    assert result.response.generation_outcome == "grounded_answer"
    assert result.response.risk_assessment.risk_floor is expected_level
    assert result.response.handoff_notice == expected_handoff
    if expected_level is RiskLevel.MEDIUM:
        assert RiskReasonCode.MISSING_ITEM in result.response.risk_assessment.reason_codes
    if expected_level is RiskLevel.HIGH:
        assert result.response.risk_assessment.handoff_required is True
        assert result.response.risk_assessment.priority_handoff is False
    if expected_level is RiskLevel.CRITICAL:
        assert result.response.risk_assessment.handoff_required is True
        assert result.response.risk_assessment.priority_handoff is True
    assert model.call_count == 1


def test_empty_retrieval_insufficient_context_preserves_risk() -> None:
    pipeline, _, model = _pipeline(retrieval=FixedRetrieval(results=[]))
    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="Упаковка была вскрыта."),
    )

    assert model.call_count == 0
    assert result.response.generation_outcome == "insufficient_context"
    assert result.response.generation.response_mode == "insufficient_context"
    assert result.response.generation.customer_response == INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
    assert result.response.generation.citations == []
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE


def test_generation_fallback_preserved_for_high_risk() -> None:
    pipeline, _, _ = _pipeline(
        chat_model=FakeChatModel(response="{bad-json"),
    )
    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="Упаковка была вскрыта."),
    )

    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert result.response.generation.citations == []
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE


def test_generation_llm_call_error_fallback_preserved() -> None:
    pipeline, _, _ = _pipeline(
        chat_model=FakeChatModel(error=LLMCallError("boom")),
    )
    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="Упаковка была вскрыта."),
    )

    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE


def test_query_consistency_through_real_risk_assessor() -> None:
    retrieval = FixedRetrieval()
    pipeline, _, _ = _pipeline(retrieval=retrieval)

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="  Упаковка была вскрыта.  "),
    )

    assert retrieval.queries == ["Упаковка была вскрыта."]
    assert result.response.risk_assessment.risk_floor is RiskLevel.HIGH
    assert RiskReasonCode.PACKAGE_TAMPERING in result.response.risk_assessment.reason_codes
    assert result.response.handoff_notice == HIGH_HANDOFF_NOTICE
