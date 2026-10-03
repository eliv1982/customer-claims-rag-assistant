"""Unit tests for CustomerClaimsPipeline."""

from __future__ import annotations

import ast
import importlib
import logging
import pkgutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.application.models import CustomerClaimsRequest
from customer_claims_rag.application.pipeline import CustomerClaimsPipeline
from customer_claims_rag.exceptions import (
    GenerationParseError,
    GenerationValidationError,
    LLMCallError,
    SearchError,
)
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.handoff import (
    CRITICAL_HANDOFF_NOTICE,
    HIGH_HANDOFF_NOTICE,
    UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE,
)
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
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.models import (
    DeterministicRiskResult,
    RiskAssessmentRequest,
    RiskLevel,
)
from customer_claims_rag.risk.reason_codes import RiskReasonCode
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
        events: list[str] | None = None,
    ) -> None:
        self.results = [] if results is None else results
        self.error = error
        self.queries: list[str] = []
        self.call_count = 0
        self._events = events

    def search(self, query: str) -> list[SearchResult]:
        if self._events is not None:
            self._events.append("retrieval")
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
        self.risk_assessments: list[DeterministicRiskResult] = []
        self.call_count = 0

    def generate(
        self,
        request: GroundedGenerationRequest,
        risk_assessment: DeterministicRiskResult,
    ) -> RiskAwareGroundedGenerationResult:
        self.call_count += 1
        self.requests.append(request)
        self.risk_assessments.append(risk_assessment)
        if self.error is not None:
            raise self.error
        return self.response


class CountingAssessor:
    """Real deterministic assessor that records every call (and its place in the event order)."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[RiskAssessmentRequest] = []
        self.results: list[DeterministicRiskResult] = []
        self._events = events

    def __call__(self, request: RiskAssessmentRequest) -> DeterministicRiskResult:
        if self._events is not None:
            self._events.append("risk")
        self.calls.append(request)
        result = assess_deterministic_risk(request)
        self.results.append(result)
        return result


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


def test_pipeline_retrieval_error_degrades_instead_of_raising() -> None:
    retrieval = ControlledRetrieval(error=SearchError("search failed"))
    context_builder = MagicMock()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=retrieval,
        generator=generation,
        context_builder=context_builder,
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query="Упаковка была вскрыта."))

    context_builder.assert_not_called()
    assert generation.call_count == 0
    assert result.retrieval_failed is True
    assert result.retrieved_items == ()
    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert result.response.generation.citations == []


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


# ---------------------------------------------------------------------------
# Stage 2B / H4: deterministic risk is computed once, before retrieval, and survives failures
# ---------------------------------------------------------------------------

_CRITICAL_QUERY = "После еды стало трудно дышать."
_ORDINARY_QUERY = "Где посмотреть правила доставки?"
_UNSUPPORTED_QUERY = (
    "After eating the soup I was taken to hospital and could not breathe."
)


def test_pipeline_assesses_risk_before_retrieval() -> None:
    events: list[str] = []
    assessor = CountingAssessor(events)
    retrieval = ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")], events=events)
    pipeline = CustomerClaimsPipeline(
        retrieval=retrieval,
        generator=ControlledGeneration(),
        risk_assessor=assessor,
    )

    pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    assert events == ["risk", "retrieval"]


def test_pipeline_assesses_risk_exactly_once_per_request() -> None:
    assessor = CountingAssessor()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")]),
        generator=ControlledGeneration(),
        risk_assessor=assessor,
    )

    pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))
    assert len(assessor.calls) == 1
    pipeline.handle(CustomerClaimsRequest(customer_query=_ORDINARY_QUERY))
    assert len(assessor.calls) == 2


def test_pipeline_hands_the_same_assessment_object_to_the_generator() -> None:
    assessor = CountingAssessor()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")]),
        generator=generation,
        risk_assessor=assessor,
    )

    pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    assert len(generation.risk_assessments) == 1
    assert generation.risk_assessments[0] is assessor.results[0]
    assert generation.risk_assessments[0].risk_floor is RiskLevel.CRITICAL


def test_pipeline_assesses_the_stripped_query() -> None:
    assessor = CountingAssessor()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")]),
        generator=ControlledGeneration(),
        risk_assessor=assessor,
    )

    pipeline.handle(CustomerClaimsRequest(customer_query=f"  {_CRITICAL_QUERY}  "))

    assert [call.customer_query for call in assessor.calls] == [_CRITICAL_QUERY]


def test_retrieval_failure_keeps_critical_risk_handoff_and_reasons() -> None:
    events: list[str] = []
    assessor = CountingAssessor(events)
    retrieval = ControlledRetrieval(error=SearchError("search failed"), events=events)
    context_builder = MagicMock()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=retrieval,
        generator=generation,
        context_builder=context_builder,
        risk_assessor=assessor,
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    risk = result.response.risk_assessment
    assert risk.risk_floor is RiskLevel.CRITICAL
    assert risk.assessment_status is RiskAssessmentStatus.RULE_MATCH
    assert risk.handoff_required is True
    assert risk.priority_handoff is True
    assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
    assert [signal.rule_id for signal in risk.risk_signals] == [
        "health_symptoms_after_consumption",
    ]
    assert result.response.handoff_notice == CRITICAL_HANDOFF_NOTICE
    assert result.retrieval_failed is True
    assert result.context_build_failed is False
    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    # assessed exactly once, before the failing retrieval; nothing downstream ran
    assert events == ["risk", "retrieval"]
    assert len(assessor.calls) == 1
    assert generation.call_count == 0
    context_builder.assert_not_called()


def test_retrieval_failure_on_ordinary_query_does_not_invent_a_high_risk() -> None:
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(error=SearchError("search failed")),
        generator=ControlledGeneration(),
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_ORDINARY_QUERY))

    risk = result.response.risk_assessment
    assert result.retrieval_failed is True
    assert result.response.generation_outcome == "generation_error_fallback"
    assert risk.risk_floor is RiskLevel.LOW
    assert risk.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert risk.explicit_match is False
    assert risk.reason_codes == ()
    assert risk.handoff_required is False
    assert risk.priority_handoff is False
    assert result.response.handoff_notice is None


def test_retrieval_failure_on_unsupported_language_keeps_manual_review_handoff() -> None:
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(error=SearchError("search failed")),
        generator=ControlledGeneration(),
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_UNSUPPORTED_QUERY))

    risk = result.response.risk_assessment
    assert risk.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE
    assert risk.explicit_match is False
    assert risk.handoff_required is True
    assert risk.priority_handoff is False
    assert result.response.handoff_notice == UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE


def test_retrieval_failure_is_logged_without_the_customer_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(error=SearchError("search failed")),
        generator=ControlledGeneration(),
    )

    with caplog.at_level(logging.WARNING, logger="customer_claims_rag.application.pipeline"):
        pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    messages = [record.getMessage() for record in caplog.records]
    assert any("SearchError" in message for message in messages)
    assert all(_CRITICAL_QUERY not in message for message in messages)


def test_non_retrieval_errors_from_retrieval_still_propagate() -> None:
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(error=RuntimeError("programmer bug")),
        generator=generation,
    )

    with pytest.raises(RuntimeError, match="programmer bug"):
        pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    assert generation.call_count == 0


# ---------------------------------------------------------------------------
# Stage 2B.1: deterministic safety survives a context-building failure
# ---------------------------------------------------------------------------


def _duplicate_rank_results() -> list[SearchResult]:
    """Retrieval output the real context builder rejects with GenerationValidationError."""
    return [
        _search_result(rank=1, chunk_id="doc::1", document_id="doc-a"),
        _search_result(rank=1, chunk_id="doc::2", document_id="doc-b"),
    ]


class FailingContextBuilder:
    """Context builder stub that records its place in the event order and then raises."""

    def __init__(self, error: Exception, events: list[str] | None = None) -> None:
        self.error = error
        self.call_count = 0
        self._events = events

    def __call__(self, results):
        if self._events is not None:
            self._events.append("context")
        self.call_count += 1
        raise self.error


def test_context_build_failure_keeps_critical_risk_handoff_and_priority() -> None:
    events: list[str] = []
    assessor = CountingAssessor(events)
    retrieval = ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")], events=events)
    context_builder = FailingContextBuilder(
        GenerationValidationError("duplicate rank in context input: 1"),
        events,
    )
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=retrieval,
        generator=generation,
        context_builder=context_builder,
        risk_assessor=assessor,
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    risk = result.response.risk_assessment
    assert risk.risk_floor is RiskLevel.CRITICAL
    assert risk.assessment_status is RiskAssessmentStatus.RULE_MATCH
    assert risk.handoff_required is True
    assert risk.priority_handoff is True
    assert RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION in risk.reason_codes
    assert result.response.handoff_notice == CRITICAL_HANDOFF_NOTICE
    # degraded shape is the same as for a retrieval failure, but the source is distinguishable
    assert result.context_build_failed is True
    assert result.retrieval_failed is False
    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert result.response.generation.citations == []
    assert result.retrieved_items == ()
    # assessed exactly once, before retrieval and context building; generation never ran
    assert events == ["risk", "retrieval", "context"]
    assert len(assessor.calls) == 1
    assert context_builder.call_count == 1
    assert generation.call_count == 0


def test_real_context_builder_rejection_keeps_critical_risk_and_priority() -> None:
    assessor = CountingAssessor()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(_duplicate_rank_results()),
        generator=generation,
        risk_assessor=assessor,
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    risk = result.response.risk_assessment
    assert result.context_build_failed is True
    assert result.retrieval_failed is False
    assert risk.risk_floor is RiskLevel.CRITICAL
    assert risk.handoff_required is True
    assert risk.priority_handoff is True
    assert result.response.handoff_notice == CRITICAL_HANDOFF_NOTICE
    assert result.response.generation_outcome == "generation_error_fallback"
    assert len(assessor.calls) == 1
    assert generation.call_count == 0


def test_context_build_failure_on_ordinary_query_does_not_invent_a_risk() -> None:
    assessor = CountingAssessor()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(_duplicate_rank_results()),
        generator=generation,
        risk_assessor=assessor,
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_ORDINARY_QUERY))

    risk = result.response.risk_assessment
    assert result.context_build_failed is True
    assert result.retrieval_failed is False
    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert risk.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert risk.risk_floor is RiskLevel.LOW
    assert risk.explicit_match is False
    assert risk.reason_codes == ()
    assert risk.risk_signals == ()
    assert risk.handoff_required is False
    assert risk.priority_handoff is False
    assert result.response.handoff_notice is None
    assert len(assessor.calls) == 1
    assert generation.call_count == 0


def test_context_build_failure_on_unsupported_language_keeps_manual_review_handoff() -> None:
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(_duplicate_rank_results()),
        generator=ControlledGeneration(),
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_UNSUPPORTED_QUERY))

    risk = result.response.risk_assessment
    assert result.context_build_failed is True
    assert risk.assessment_status is RiskAssessmentStatus.UNSUPPORTED_LANGUAGE
    assert risk.handoff_required is True
    assert risk.priority_handoff is False
    assert result.response.handoff_notice == UNSUPPORTED_LANGUAGE_HANDOFF_NOTICE


def test_context_build_failure_is_logged_without_the_customer_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(_duplicate_rank_results()),
        generator=ControlledGeneration(),
    )

    with caplog.at_level(logging.WARNING, logger="customer_claims_rag.application.pipeline"):
        pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    messages = [record.getMessage() for record in caplog.records]
    assert any("GenerationValidationError" in message for message in messages)
    assert all(_CRITICAL_QUERY not in message for message in messages)


def test_successful_context_build_flags_neither_failure_source() -> None:
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")]),
        generator=ControlledGeneration(),
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    assert result.retrieval_failed is False
    assert result.context_build_failed is False
    assert result.response.generation_outcome == "grounded_answer"


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("programmer bug"),
        ValueError("programmer bug"),
        KeyError("programmer bug"),
        # siblings of GenerationValidationError under GenerationError are not context failures
        GenerationParseError("not a context failure"),
        LLMCallError("not a context failure"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_unexpected_context_builder_exceptions_are_not_swallowed(error: Exception) -> None:
    assessor = CountingAssessor()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")]),
        generator=generation,
        context_builder=FailingContextBuilder(error),
        risk_assessor=assessor,
    )

    with pytest.raises(type(error)):
        pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    assert len(assessor.calls) == 1
    assert generation.call_count == 0


def test_generation_validation_error_from_the_generator_is_not_a_context_failure() -> None:
    # The boundary is scoped to the context-builder call: the same exception type raised by the
    # generator is still not swallowed by the pipeline.
    generation = ControlledGeneration(error=GenerationValidationError("draft failed validation"))
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")]),
        generator=generation,
    )

    with pytest.raises(GenerationValidationError, match="draft failed validation"):
        pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    assert generation.call_count == 1


# ---------------------------------------------------------------------------
# Stage 2B.2: invalid ContextItem payloads reach the same context-failure boundary
# ---------------------------------------------------------------------------

_RETRIEVED_TEXT = "SENSITIVE-RETRIEVED-TEXT"

_INVALID_RETRIEVAL_PAYLOADS = [
    pytest.param({"rank": 0}, id="rank-zero"),
    pytest.param({"content": ""}, id="blank-content"),
    pytest.param({"heading": ""}, id="blank-heading"),
    pytest.param({"chunk_id": ""}, id="blank-chunk-id"),
]


def _invalid_payload_results(update: dict) -> list[SearchResult]:
    """Retrieval output with one valid and one invalid item (SearchResult does not validate)."""
    return [
        _search_result(rank=1, chunk_id="doc::1", document_id="doc-a"),
        _search_result(rank=2, chunk_id="doc::2", document_id="doc-b").model_copy(update=update),
    ]


@pytest.mark.parametrize("update", _INVALID_RETRIEVAL_PAYLOADS)
def test_invalid_context_payload_keeps_critical_risk_handoff_and_priority(update: dict) -> None:
    assessor = CountingAssessor()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(_invalid_payload_results(update)),
        generator=generation,
        risk_assessor=assessor,
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    risk = result.response.risk_assessment
    assert risk.risk_floor is RiskLevel.CRITICAL
    assert risk.assessment_status is RiskAssessmentStatus.RULE_MATCH
    assert risk.handoff_required is True
    assert risk.priority_handoff is True
    assert result.response.handoff_notice == CRITICAL_HANDOFF_NOTICE
    assert result.context_build_failed is True
    assert result.retrieval_failed is False
    assert result.response.generation_outcome == "generation_error_fallback"
    assert result.response.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert result.retrieved_items == ()
    assert len(assessor.calls) == 1
    assert generation.call_count == 0


@pytest.mark.parametrize("update", _INVALID_RETRIEVAL_PAYLOADS)
def test_invalid_context_payload_on_no_signal_query_invents_no_escalation(update: dict) -> None:
    assessor = CountingAssessor()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(_invalid_payload_results(update)),
        generator=generation,
        risk_assessor=assessor,
    )

    result = pipeline.handle(CustomerClaimsRequest(customer_query=_ORDINARY_QUERY))

    risk = result.response.risk_assessment
    assert risk.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert risk.risk_floor is RiskLevel.LOW
    assert risk.explicit_match is False
    assert risk.reason_codes == ()
    assert risk.handoff_required is False
    assert risk.priority_handoff is False
    assert result.response.handoff_notice is None
    assert result.context_build_failed is True
    assert result.retrieval_failed is False
    assert result.response.generation_outcome == "generation_error_fallback"
    assert len(assessor.calls) == 1
    assert generation.call_count == 0


def test_invalid_context_payload_is_logged_without_customer_or_retrieved_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    results = _invalid_payload_results({"heading": ""})
    results[0] = results[0].model_copy(update={"content": _RETRIEVED_TEXT})
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval(results),
        generator=ControlledGeneration(),
    )

    with caplog.at_level(logging.WARNING, logger="customer_claims_rag.application.pipeline"):
        pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    messages = [record.getMessage() for record in caplog.records]
    assert any("GenerationValidationError" in message for message in messages)
    assert all(_CRITICAL_QUERY not in message for message in messages)
    assert all(_RETRIEVED_TEXT not in message for message in messages)


@pytest.mark.parametrize(
    "error",
    [RuntimeError("programmer bug"), ValueError("programmer bug")],
    ids=lambda error: type(error).__name__,
)
def test_programming_errors_during_real_context_construction_are_not_degraded(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def explode(**kwargs: object) -> None:
        raise error

    monkeypatch.setattr("customer_claims_rag.generation.context_builder.ContextItem", explode)
    assessor = CountingAssessor()
    generation = ControlledGeneration()
    pipeline = CustomerClaimsPipeline(
        retrieval=ControlledRetrieval([_search_result(rank=1, chunk_id="doc::1")]),
        generator=generation,
        risk_assessor=assessor,
    )

    with pytest.raises(type(error), match="programmer bug"):
        pipeline.handle(CustomerClaimsRequest(customer_query=_CRITICAL_QUERY))

    assert len(assessor.calls) == 1
    assert generation.call_count == 0
