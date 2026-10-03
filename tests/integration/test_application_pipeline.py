"""Integration tests for CustomerClaimsPipeline without external services."""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.application.models import CustomerClaimsRequest
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.exceptions import EmbeddingError, LLMCallError
from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.handoff import (
    CRITICAL_HANDOFF_NOTICE,
    HIGH_HANDOFF_NOTICE,
    UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE,
)
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.models import RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.retrieval.models import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"


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


# ---------------------------------------------------------------------------
# Stage 2B / H4: one assessment per request, computed before retrieval, kept on failure
# ---------------------------------------------------------------------------


class FailingRetrieval:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.queries: list[str] = []

    def search(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        raise self._error


class CountingAssessor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return assess_deterministic_risk(request)


def _pipeline_with_assessors(
    *,
    retrieval,
    chat_model: FakeChatModel | None = None,
) -> tuple[CustomerClaimsPipeline, CountingAssessor, CountingAssessor, FakeChatModel]:
    pipeline_assessor = CountingAssessor()
    generator_assessor = CountingAssessor()
    model = chat_model or FakeChatModel(
        response=_valid_grounded_json("По правилам вскрытая упаковка фиксируется [S1]."),
    )
    grounded = GroundedGenerator(
        chat_model=model,
        prompt_builder=PromptBuilder(prompt_path=PROMPT_PATH),
    )
    generator = RiskAwareGroundedGenerator(
        grounded_generator=grounded,
        risk_assessor=generator_assessor,
    )
    pipeline = CustomerClaimsPipeline(
        retrieval=retrieval,
        generator=generator,
        risk_assessor=pipeline_assessor,
    )
    return pipeline, pipeline_assessor, generator_assessor, model


def test_risk_is_assessed_once_per_request_through_the_real_generator() -> None:
    pipeline, pipeline_assessor, generator_assessor, model = _pipeline_with_assessors(
        retrieval=FixedRetrieval(),
    )

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="После еды стало трудно дышать."),
    )

    assert pipeline_assessor.calls == 1
    assert generator_assessor.calls == 0  # the generator reuses the pipeline's assessment
    assert model.call_count == 1
    assert result.response.generation_outcome == "grounded_answer"
    assert result.response.risk_assessment.risk_floor is RiskLevel.CRITICAL
    assert result.response.handoff_notice == CRITICAL_HANDOFF_NOTICE


def test_critical_risk_and_handoff_survive_a_retrieval_failure() -> None:
    retrieval = FailingRetrieval(EmbeddingError("OpenAI embedding request failed"))
    pipeline, pipeline_assessor, generator_assessor, model = _pipeline_with_assessors(
        retrieval=retrieval,
    )

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="После еды стало трудно дышать."),
    )

    assert retrieval.queries == ["После еды стало трудно дышать."]
    assert pipeline_assessor.calls == 1
    assert generator_assessor.calls == 0
    assert model.call_count == 0  # no model call without context
    assert result.retrieval_failed is True
    risk = result.response.risk_assessment
    assert risk.risk_floor is RiskLevel.CRITICAL
    assert risk.handoff_required is True
    assert risk.priority_handoff is True
    assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
    assert result.response.handoff_notice == CRITICAL_HANDOFF_NOTICE
    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE


def test_ordinary_query_degrades_without_inventing_a_risk_on_retrieval_failure() -> None:
    pipeline, pipeline_assessor, _, _ = _pipeline_with_assessors(
        retrieval=FailingRetrieval(EmbeddingError("OpenAI embedding request failed")),
    )

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="Где посмотреть правила доставки?"),
    )

    risk = result.response.risk_assessment
    assert pipeline_assessor.calls == 1
    assert result.retrieval_failed is True
    assert risk.risk_floor is RiskLevel.LOW
    assert risk.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert risk.handoff_required is False
    assert result.response.handoff_notice is None


def _duplicate_rank_results() -> list[SearchResult]:
    first = _fixed_search_results()[0]
    second = first.model_copy(update={"chunk_id": "06_food_quality_and_packaging::2"})
    return [first, second]  # same rank twice: the real context builder rejects it


def test_critical_risk_and_handoff_survive_a_context_build_failure() -> None:
    pipeline, pipeline_assessor, generator_assessor, model = _pipeline_with_assessors(
        retrieval=FixedRetrieval(_duplicate_rank_results()),
    )

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="После еды стало трудно дышать."),
    )

    assert pipeline_assessor.calls == 1
    assert generator_assessor.calls == 0
    assert model.call_count == 0  # generation never ran
    assert result.context_build_failed is True
    assert result.retrieval_failed is False
    risk = result.response.risk_assessment
    assert risk.risk_floor is RiskLevel.CRITICAL
    assert risk.handoff_required is True
    assert risk.priority_handoff is True
    assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
    assert result.response.handoff_notice == CRITICAL_HANDOFF_NOTICE
    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE


def test_ordinary_query_degrades_without_inventing_a_risk_on_context_build_failure() -> None:
    pipeline, pipeline_assessor, _, model = _pipeline_with_assessors(
        retrieval=FixedRetrieval(_duplicate_rank_results()),
    )

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="Где посмотреть правила доставки?"),
    )

    risk = result.response.risk_assessment
    assert pipeline_assessor.calls == 1
    assert model.call_count == 0
    assert result.context_build_failed is True
    assert risk.risk_floor is RiskLevel.LOW
    assert risk.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert risk.handoff_required is False
    assert risk.priority_handoff is False
    assert result.response.handoff_notice is None


def test_generation_failure_keeps_the_assessment_made_before_retrieval() -> None:
    pipeline, pipeline_assessor, generator_assessor, _ = _pipeline_with_assessors(
        retrieval=FixedRetrieval(),
        chat_model=FakeChatModel(error=LLMCallError("boom")),
    )

    result = pipeline.handle(
        CustomerClaimsRequest(customer_query="После еды стало трудно дышать."),
    )

    assert pipeline_assessor.calls == 1
    assert generator_assessor.calls == 0
    assert result.retrieval_failed is False
    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.risk_assessment.risk_floor is RiskLevel.CRITICAL
    assert result.response.handoff_notice == CRITICAL_HANDOFF_NOTICE


def test_unsupported_language_query_carries_manual_review_through_the_real_stack() -> None:
    pipeline, _, _, model = _pipeline_with_assessors(retrieval=FixedRetrieval())

    result = pipeline.handle(
        CustomerClaimsRequest(
            customer_query="After eating the soup I was taken to hospital by ambulance.",
        ),
    )

    risk = result.response.risk_assessment
    assert model.call_count == 1
    assert risk.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE
    assert risk.explicit_match is False
    assert risk.handoff_required is True
    assert risk.priority_handoff is False
    assert result.response.handoff_notice == UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE


def test_standalone_generator_still_assesses_when_no_assessment_is_supplied() -> None:
    from customer_claims_rag.generation.models import ContextPackage, GroundedGenerationRequest

    assessor = CountingAssessor()
    model = FakeChatModel(response=_valid_grounded_json("Ответ [S1]."))
    generator = RiskAwareGroundedGenerator(
        grounded_generator=GroundedGenerator(
            chat_model=model,
            prompt_builder=PromptBuilder(prompt_path=PROMPT_PATH),
        ),
        risk_assessor=assessor,
    )

    result = generator.generate(
        GroundedGenerationRequest(
            customer_query="Упаковка была вскрыта.",
            context_package=ContextPackage(items=[]),
        ),
    )

    assert assessor.calls == 1
    assert result.risk_assessment.risk_floor is RiskLevel.HIGH
