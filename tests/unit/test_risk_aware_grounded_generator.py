"""Unit tests for RiskAwareGroundedGenerator."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from customer_claims_rag.exceptions import (
    GenerationParseError,
    GenerationValidationError,
    LLMCallError,
    RiskValidationError,
)
from customer_claims_rag.generation.adapters.fake_chat import FakeChatModel
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.generator import GroundedGenerator
from customer_claims_rag.generation.handoff import CRITICAL_HANDOFF_NOTICE, HIGH_HANDOFF_NOTICE
from customer_claims_rag.generation.models import (
    ContextItem,
    ContextPackage,
    Citation,
    GroundedGenerationRequest,
    GroundedGenerationResult,
)
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation.risk_aware_generator import RiskAwareGroundedGenerator
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import RiskAssessmentRequest, RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"

_FORBIDDEN_RISK_IMPORTS = (
    "customer_claims_rag.evaluation",
    "customer_claims_rag.retrieval",
    "customer_claims_rag.generation",
    "langchain",
    "langchain_openai",
    "openai",
    "streamlit",
)


def _package() -> ContextPackage:
    return ContextPackage(
        items=[
            ContextItem(
                citation_key="S1",
                rank=1,
                document_id="doc-a",
                chunk_id="doc-a::1",
                heading="Heading A",
                source_path="data/02_clean_markdown/doc-a.md",
                content="Policy text about delivery rules.",
            ),
        ],
    )


def _request(
    customer_query: str,
    *,
    items: list[ContextItem] | None = None,
) -> GroundedGenerationRequest:
    package = ContextPackage(items=items if items is not None else _package().items)
    return GroundedGenerationRequest(
        customer_query=customer_query,
        context_package=package,
    )


def _grounded_generator(
    *,
    chat_model: FakeChatModel | None = None,
) -> GroundedGenerator:
    return GroundedGenerator(
        chat_model=chat_model or FakeChatModel(),
        prompt_builder=PromptBuilder(prompt_path=PROMPT_PATH),
    )


def _risk_aware_generator(
    *,
    chat_model: FakeChatModel | None = None,
    risk_assessor=assess_deterministic_risk,
) -> RiskAwareGroundedGenerator:
    return RiskAwareGroundedGenerator(
        grounded_generator=_grounded_generator(chat_model=chat_model),
        risk_assessor=risk_assessor,
    )


def _valid_grounded_json(answer: str) -> str:
    escaped = answer.replace('"', '\\"')
    return f'{{"response_mode":"grounded_answer","answer":"{escaped}"}}'


def test_low_risk_success() -> None:
    model = FakeChatModel(
        response=_valid_grounded_json("Правила доставки описаны в материалах [S1]."),
    )
    result = _risk_aware_generator(chat_model=model).generate(
        _request("Где посмотреть правила доставки?"),
    )

    assert result.generation_outcome == "grounded_answer"
    assert result.risk_assessment.risk_floor is RiskLevel.LOW
    assert result.risk_assessment.explicit_match is False
    assert result.handoff_notice is None
    assert result.generation.response_mode == "grounded_answer"
    assert "[S1]" in result.generation.customer_response


def test_medium_risk_success() -> None:
    model = FakeChatModel(
        response=_valid_grounded_json("По правилам частичная недостача оформляется [S1]."),
    )
    result = _risk_aware_generator(chat_model=model).generate(
        _request("Не привезли одну позицию."),
    )

    assert result.generation_outcome == "grounded_answer"
    assert result.risk_assessment.risk_floor is RiskLevel.MEDIUM
    assert RiskReasonCode.MISSING_ITEM in result.risk_assessment.reason_codes
    assert result.handoff_notice is None
    assert result.generation.citations


def test_high_risk_success_preserves_grounded_answer() -> None:
    answer = "По правилам вскрытая упаковка фиксируется [S1]."
    model = FakeChatModel(response=_valid_grounded_json(answer))
    result = _risk_aware_generator(chat_model=model).generate(
        _request("Упаковка была вскрыта."),
    )

    assert result.risk_assessment.risk_floor is RiskLevel.HIGH
    assert RiskReasonCode.PACKAGE_TAMPERING in result.risk_assessment.reason_codes
    assert result.risk_assessment.handoff_required is True
    assert result.risk_assessment.priority_handoff is False
    assert result.handoff_notice == HIGH_HANDOFF_NOTICE
    assert result.generation.customer_response == answer
    assert len(result.generation.citations) == 1


def test_critical_risk_success_preserves_grounded_answer() -> None:
    answer = "При симптомах после еды следует обратиться [S1]."
    model = FakeChatModel(response=_valid_grounded_json(answer))
    result = _risk_aware_generator(chat_model=model).generate(
        _request("После еды стало трудно дышать."),
    )

    assert result.risk_assessment.risk_floor is RiskLevel.CRITICAL
    assert (
        RiskReasonCode.HEALTH_SYMPTOMS_AFTER_CONSUMPTION
        in result.risk_assessment.reason_codes
    )
    assert result.risk_assessment.handoff_required is True
    assert result.risk_assessment.priority_handoff is True
    assert result.handoff_notice == CRITICAL_HANDOFF_NOTICE
    assert result.generation.customer_response == answer


@pytest.mark.parametrize(
    "customer_query",
    [
        "Упаковка была вскрыта.",
        "После еды стало трудно дышать.",
    ],
)
def test_empty_context_preserves_risk_and_handoff(customer_query: str) -> None:
    model = FakeChatModel(response="should-not-be-used")
    generator = _risk_aware_generator(chat_model=model)
    result = generator.generate(_request(customer_query, items=[]))

    assert model.call_count == 0
    assert result.generation_outcome == "insufficient_context"
    assert result.generation.response_mode == "insufficient_context"
    assert result.generation.customer_response == INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE
    assert result.risk_assessment.risk_floor in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert result.handoff_notice is not None


@pytest.mark.parametrize(
    "customer_query",
    [
        "Упаковка была вскрыта.",
        "После еды стало трудно дышать.",
    ],
)
def test_invalid_json_preserves_risk_and_handoff(customer_query: str) -> None:
    model = FakeChatModel(response="{bad-json")
    result = _risk_aware_generator(chat_model=model).generate(_request(customer_query))

    assert result.generation_outcome == "generation_error_fallback"
    assert result.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert result.generation.citations == []
    assert result.risk_assessment.risk_floor in {RiskLevel.HIGH, RiskLevel.CRITICAL}
    assert result.handoff_notice is not None


@pytest.mark.parametrize(
    "customer_query",
    [
        "Упаковка была вскрыта.",
        "После еды стало трудно дышать.",
    ],
)
def test_invalid_citation_preserves_risk_and_handoff(customer_query: str) -> None:
    model = FakeChatModel(
        response='{"response_mode":"grounded_answer","answer":"Без ссылок"}',
    )
    result = _risk_aware_generator(chat_model=model).generate(_request(customer_query))

    assert result.generation_outcome == "generation_error_fallback"
    assert result.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert result.handoff_notice is not None


@pytest.mark.parametrize(
    "customer_query",
    [
        "Упаковка была вскрыта.",
        "После еды стало трудно дышать.",
    ],
)
def test_llm_call_error_preserves_risk_and_handoff(customer_query: str) -> None:
    model = FakeChatModel(error=LLMCallError("boom"))
    result = _risk_aware_generator(chat_model=model).generate(_request(customer_query))

    assert result.generation_outcome == "generation_error_fallback"
    assert result.generation.customer_response == GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert result.handoff_notice is not None


def test_low_risk_generation_error_has_no_handoff() -> None:
    model = FakeChatModel(response="{bad-json")
    result = _risk_aware_generator(chat_model=model).generate(
        _request("Где посмотреть правила доставки?"),
    )

    assert result.risk_assessment.risk_floor is RiskLevel.LOW
    assert result.handoff_notice is None
    assert result.generation_outcome == "generation_error_fallback"


def test_medium_risk_generation_error_has_no_handoff() -> None:
    model = FakeChatModel(response="{bad-json")
    result = _risk_aware_generator(chat_model=model).generate(
        _request("Не привезли одну позицию."),
    )

    assert result.risk_assessment.risk_floor is RiskLevel.MEDIUM
    assert result.handoff_notice is None
    assert result.generation_outcome == "generation_error_fallback"


def test_unexpected_programmer_error_propagates() -> None:
    grounded = MagicMock()
    grounded.generate.side_effect = RuntimeError("programmer bug")

    generator = RiskAwareGroundedGenerator(
        grounded_generator=grounded,
        risk_assessor=assess_deterministic_risk,
    )

    with pytest.raises(RuntimeError, match="programmer bug"):
        generator.generate(_request("Упаковка была вскрыта."))


def test_assessor_error_propagates_without_generation() -> None:
    grounded = MagicMock()
    risk_assessor = MagicMock(side_effect=RiskValidationError("bad risk"))

    generator = RiskAwareGroundedGenerator(
        grounded_generator=grounded,
        risk_assessor=risk_assessor,
    )

    with pytest.raises(RiskValidationError, match="bad risk"):
        generator.generate(_request("Упаковка была вскрыта."))

    grounded.generate.assert_not_called()


def test_risk_assessment_runs_before_generation() -> None:
    events: list[str] = []

    def risk_assessor(request: RiskAssessmentRequest):
        events.append("risk")
        return assess_deterministic_risk(request)

    grounded = MagicMock()
    grounded.generate.side_effect = lambda request: (
        events.append("generation"),
        GroundedGenerationResult(
            response_mode="grounded_answer",
            customer_response="Ответ [S1].",
            citations=[
                Citation(
                    citation_key="S1",
                    document_id="doc-a",
                    chunk_id="doc-a::1",
                    heading="Heading A",
                    source_path="data/02_clean_markdown/doc-a.md",
                ),
            ],
        ),
    )[1]

    generator = RiskAwareGroundedGenerator(
        grounded_generator=grounded,
        risk_assessor=risk_assessor,
    )
    generator.generate(_request("Упаковка была вскрыта."))

    assert events == ["risk", "generation"]


def test_risk_assessment_is_deterministic_across_llm_wording() -> None:
    query = "Упаковка была вскрыта."
    first_model = FakeChatModel(
        response=_valid_grounded_json("Первый вариант ответа [S1]."),
    )
    second_model = FakeChatModel(
        response=_valid_grounded_json("Совсем другой текст ответа [S1]."),
    )

    first = _risk_aware_generator(chat_model=first_model).generate(_request(query))
    second = _risk_aware_generator(chat_model=second_model).generate(_request(query))

    assert first.risk_assessment == second.risk_assessment
    assert first.handoff_notice == second.handoff_notice
    assert first.generation.customer_response != second.generation.customer_response


def test_risk_package_still_does_not_import_generation() -> None:
    risk_root = PROJECT_ROOT / "src" / "customer_claims_rag" / "risk"
    for module_info in pkgutil.walk_packages([str(risk_root)], prefix="customer_claims_rag.risk."):
        module = importlib.import_module(module_info.name)
        source_path = Path(module.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in _FORBIDDEN_RISK_IMPORTS:
                        assert forbidden not in alias.name
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in _FORBIDDEN_RISK_IMPORTS:
                    assert forbidden not in node.module


def test_invalid_generation_result_from_port_propagates_validation_error() -> None:
    class InvalidGenerationPort:
        def generate(
            self,
            request: GroundedGenerationRequest,
        ) -> GroundedGenerationResult:
            return GroundedGenerationResult(
                response_mode="grounded_answer",
                customer_response="Ответ без ссылок",
                citations=[],
            )

    generator = RiskAwareGroundedGenerator(
        grounded_generator=InvalidGenerationPort(),
        risk_assessor=assess_deterministic_risk,
    )

    with pytest.raises(ValidationError, match="at least one citation"):
        generator.generate(_request("Упаковка была вскрыта."))


def test_generation_error_subclasses_are_caught() -> None:
    """Confirm parse/validation/llm errors map to safe fallback, not propagate."""
    grounded = GroundedGenerator(
        chat_model=FakeChatModel(response="{bad-json"),
        prompt_builder=PromptBuilder(prompt_path=PROMPT_PATH),
    )
    generator = RiskAwareGroundedGenerator(grounded_generator=grounded)

    result = generator.generate(_request("Упаковка была вскрыта."))
    assert result.generation_outcome == "generation_error_fallback"

    grounded_validation = GroundedGenerator(
        chat_model=FakeChatModel(
            response='{"response_mode":"grounded_answer","answer":"Без ссылок"}',
        ),
        prompt_builder=PromptBuilder(prompt_path=PROMPT_PATH),
    )
    result_validation = RiskAwareGroundedGenerator(
        grounded_generator=grounded_validation,
    ).generate(_request("Упаковка была вскрыта."))
    assert result_validation.generation_outcome == "generation_error_fallback"

    with pytest.raises(GenerationParseError):
        grounded.generate(_request("Упаковка была вскрыта."))

    with pytest.raises(GenerationValidationError):
        grounded_validation.generate(_request("Упаковка была вскрыта."))
