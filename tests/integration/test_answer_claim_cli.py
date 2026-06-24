"""Tests for the production answer-claim CLI."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from customer_claims_rag.application.models import CustomerClaimsRequest, CustomerClaimsResult
from customer_claims_rag.cli import answer_claim
from customer_claims_rag.exceptions import EmbeddingError, IndexManifestError
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.handoff import CRITICAL_HANDOFF_NOTICE, HIGH_HANDOFF_NOTICE
from customer_claims_rag.generation.models import Citation, GroundedGenerationResult
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.risk import assess_deterministic_risk
from customer_claims_rag.risk.models import RiskAssessmentRequest, RiskLevel
from customer_claims_rag.risk.reason_codes import RiskReasonCode

SECRET_API_KEY = "sk-test-secret-value-12345"
SENSITIVE_SOURCE_PATH = "data/02_clean_markdown/secret_doc.md"
SENSITIVE_REASON_CODE = RiskReasonCode.PACKAGE_TAMPERING.value


def _low_risk():
    return assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Где посмотреть правила доставки?"),
    )


def _medium_risk():
    return assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Не привезли одну позицию."),
    )


def _high_risk():
    return assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Упаковка была вскрыта."),
    )


def _critical_risk():
    return assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="После еды стало трудно дышать."),
    )


def _grounded_generation(answer: str = "Ответ по правилам [S1].") -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=answer,
        citations=[
            Citation(
                citation_key="S1",
                document_id="06_food_quality_and_packaging",
                chunk_id="06_food_quality_and_packaging::1",
                heading="Упаковка",
                source_path=SENSITIVE_SOURCE_PATH,
            ),
        ],
    )


def _insufficient_context_generation() -> GroundedGenerationResult:
    return GroundedGenerationResult(
        response_mode="insufficient_context",
        customer_response=INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
        citations=[],
    )


def _pipeline_result(
    *,
    risk_assessment,
    generation: GroundedGenerationResult,
    generation_outcome: str,
) -> CustomerClaimsResult:
    from customer_claims_rag.generation.handoff import build_handoff_notice

    response = RiskAwareGroundedGenerationResult(
        generation=generation,
        risk_assessment=risk_assessment,
        handoff_notice=build_handoff_notice(risk_assessment),
        generation_outcome=generation_outcome,  # type: ignore[arg-type]
    )
    return CustomerClaimsResult(response=response)


def _mock_pipeline(result: CustomerClaimsResult) -> MagicMock:
    pipeline = MagicMock()
    pipeline.handle.return_value = result
    return pipeline


def _mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.retrieval.openai_api_key = SECRET_API_KEY
    return settings


@pytest.fixture
def wired_run_answer(monkeypatch):
    settings = _mock_settings()
    pipeline = MagicMock()
    factory_calls: list[object] = []
    handle_calls: list[CustomerClaimsRequest] = []

    def settings_loader() -> MagicMock:
        return settings

    def pipeline_factory(loaded_settings: object) -> MagicMock:
        factory_calls.append(loaded_settings)
        return pipeline

    def invoke(
        *,
        message: str,
        result: CustomerClaimsResult,
    ) -> tuple[int, dict | None]:
        pipeline.handle.side_effect = None
        pipeline.handle.return_value = result
        pipeline.handle.reset_mock()
        factory_calls.clear()
        handle_calls.clear()

        def record_handle(request: CustomerClaimsRequest) -> CustomerClaimsResult:
            handle_calls.append(request)
            return result

        pipeline.handle.side_effect = record_handle
        code, payload = answer_claim.run_answer(
            message=message,
            settings_loader=settings_loader,
            pipeline_factory=pipeline_factory,
        )
        return code, payload, factory_calls, handle_calls, settings, pipeline

    return invoke


def test_format_answer_payload_maps_contract_fields() -> None:
    result = _pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    payload = answer_claim.format_answer_payload(result)
    assert payload == {
        "answer": "Ответ по правилам [S1].",
        "response_mode": "grounded_answer",
        "generation_outcome": "grounded_answer",
        "risk_level": "high",
        "handoff_required": True,
        "priority_handoff": False,
        "handoff_notice": HIGH_HANDOFF_NOTICE,
        "citations": [
            {
                "key": "S1",
                "heading": "Упаковка",
                "document_id": "06_food_quality_and_packaging",
            },
        ],
    }


def test_run_answer_success_grounded_with_citations(
    wired_run_answer,
    capsys,
) -> None:
    result = _pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    code, payload, _, _, _, _ = wired_run_answer(
        message="Упаковка была вскрыта.",
        result=result,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert payload is not None
    assert payload["generation_outcome"] == "grounded_answer"
    assert len(payload["citations"]) == 1
    json.loads(captured.out)
    assert captured.err == ""


def test_run_answer_insufficient_context_with_empty_citations(
    wired_run_answer,
    capsys,
) -> None:
    result = _pipeline_result(
        risk_assessment=_high_risk(),
        generation=_insufficient_context_generation(),
        generation_outcome="insufficient_context",
    )
    code, payload, _, _, _, _ = wired_run_answer(
        message="Упаковка была вскрыта.",
        result=result,
    )
    assert code == 0
    assert payload is not None
    assert payload["response_mode"] == "insufficient_context"
    assert payload["generation_outcome"] == "insufficient_context"
    assert payload["citations"] == []


def test_run_answer_high_risk_includes_handoff(wired_run_answer, capsys) -> None:
    result = _pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    code, payload, _, _, _, _ = wired_run_answer(
        message="Упаковка была вскрыта.",
        result=result,
    )
    assert code == 0
    assert payload is not None
    assert payload["risk_level"] == "high"
    assert payload["handoff_required"] is True
    assert payload["priority_handoff"] is False
    assert payload["handoff_notice"] == HIGH_HANDOFF_NOTICE


def test_run_answer_critical_priority_handoff(wired_run_answer, capsys) -> None:
    result = _pipeline_result(
        risk_assessment=_critical_risk(),
        generation=_grounded_generation("Критический ответ [S1]."),
        generation_outcome="grounded_answer",
    )
    code, payload, _, _, _, _ = wired_run_answer(
        message="После еды стало трудно дышать.",
        result=result,
    )
    assert code == 0
    assert payload is not None
    assert payload["risk_level"] == "critical"
    assert payload["handoff_required"] is True
    assert payload["priority_handoff"] is True
    assert payload["handoff_notice"] == CRITICAL_HANDOFF_NOTICE


@pytest.mark.parametrize(
    ("risk_assessment", "expected_level"),
    [
        (_low_risk(), "low"),
        (_medium_risk(), "medium"),
    ],
)
def test_run_answer_low_and_medium_without_handoff_notice(
    wired_run_answer,
    risk_assessment,
    expected_level,
    capsys,
) -> None:
    result = _pipeline_result(
        risk_assessment=risk_assessment,
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    code, payload, _, _, _, _ = wired_run_answer(
        message="Тестовый запрос.",
        result=result,
    )
    assert code == 0
    assert payload is not None
    assert payload["risk_level"] == expected_level
    assert payload["handoff_notice"] is None
    assert payload["handoff_required"] is False
    assert payload["priority_handoff"] is False


@pytest.mark.parametrize("message", ["", "   "])
def test_run_answer_invalid_message_returns_exit_code_2(message: str, capsys) -> None:
    code, payload = answer_claim.run_answer(message=message)
    captured = capsys.readouterr()
    assert code == 2
    assert payload is None
    assert captured.out == ""
    assert "empty or whitespace-only" in captured.err


def test_run_answer_startup_configuration_failure(capsys) -> None:
    def broken_factory(_settings: object) -> MagicMock:
        raise EmbeddingError("OPENAI_API_KEY is required")

    code, payload = answer_claim.run_answer(
        message="Упаковка была вскрыта.",
        settings_loader=_mock_settings,
        pipeline_factory=broken_factory,
    )
    captured = capsys.readouterr()
    assert code == 1
    assert payload is None
    assert "OPENAI_API_KEY is required" in captured.err
    assert SECRET_API_KEY not in captured.err


def test_run_answer_index_validation_failure(capsys) -> None:
    def broken_factory(_settings: object) -> MagicMock:
        raise IndexManifestError("index manifest not found")

    code, payload = answer_claim.run_answer(
        message="Упаковка была вскрыта.",
        settings_loader=_mock_settings,
        pipeline_factory=broken_factory,
    )
    captured = capsys.readouterr()
    assert code == 1
    assert payload is None
    assert "index manifest not found" in captured.err


def test_run_answer_unexpected_runtime_failure(capsys) -> None:
    pipeline = MagicMock()
    pipeline.handle.side_effect = RuntimeError(f"boom {SECRET_API_KEY}")

    code, payload = answer_claim.run_answer(
        message="Упаковка была вскрыта.",
        settings_loader=_mock_settings,
        pipeline_factory=lambda _settings: pipeline,
    )
    captured = capsys.readouterr()
    assert code == 1
    assert payload is None
    assert captured.err.strip() == answer_claim._RUNTIME_ERROR_MESSAGE
    assert SECRET_API_KEY not in captured.err
    assert "Traceback" not in captured.err


def test_run_answer_json_excludes_sensitive_fields(wired_run_answer, capsys) -> None:
    risk = _high_risk()
    assert SENSITIVE_REASON_CODE in [code.value for code in risk.reason_codes]
    assert risk.risk_signals

    result = _pipeline_result(
        risk_assessment=risk,
        generation=_grounded_generation("Ответ по правилам [S1]."),
        generation_outcome="grounded_answer",
    )
    code, payload, _, _, settings, _ = wired_run_answer(
        message="Упаковка была вскрыта.",
        result=result,
    )
    captured = capsys.readouterr()
    assert code == 0
    rendered = captured.out
    forbidden = (
        SECRET_API_KEY,
        SENSITIVE_SOURCE_PATH,
        "source_path",
        "chunk_id",
        SENSITIVE_REASON_CODE,
        "risk_signals",
        "reason_codes",
        "explanation",
        repr(settings),
    )
    for fragment in forbidden:
        assert fragment not in rendered
    assert payload is not None
    assert payload["citations"][0]["key"] == "S1"


def test_run_answer_calls_pipeline_once(wired_run_answer, capsys) -> None:
    result = _pipeline_result(
        risk_assessment=_low_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    _, _, factory_calls, handle_calls, settings, _ = wired_run_answer(
        message="  Где правила доставки?  ",
        result=result,
    )
    capsys.readouterr()
    assert len(factory_calls) == 1
    assert factory_calls[0] is settings
    assert len(handle_calls) == 1
    assert handle_calls[0].customer_query == "Где правила доставки?"


def test_run_answer_serializes_generation_error_fallback(wired_run_answer, capsys) -> None:
    from customer_claims_rag.generation.handoff import build_handoff_notice

    risk = _high_risk()
    response = RiskAwareGroundedGenerationResult(
        generation=GroundedGenerationResult(
            response_mode="insufficient_context",
            customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
            citations=[],
        ),
        risk_assessment=risk,
        handoff_notice=build_handoff_notice(risk),
        generation_outcome="generation_error_fallback",
    )
    result = CustomerClaimsResult(response=response)
    code, payload, _, _, _, _ = wired_run_answer(
        message="Упаковка была вскрыта.",
        result=result,
    )
    capsys.readouterr()
    assert code == 0
    assert payload is not None
    assert payload["generation_outcome"] == "generation_error_fallback"


def test_main_uses_production_wiring_and_returns_success(monkeypatch, capsys) -> None:
    settings = _mock_settings()
    pipeline = _mock_pipeline(
        _pipeline_result(
            risk_assessment=_low_risk(),
            generation=_grounded_generation(),
            generation_outcome="grounded_answer",
        ),
    )
    settings_calls: list[str] = []
    factory_calls: list[object] = []

    monkeypatch.setattr(answer_claim, "load_project_env", lambda: None)
    monkeypatch.setattr(
        answer_claim.ApplicationSettings,
        "from_env",
        classmethod(lambda cls: (settings_calls.append("loaded") or settings)),
    )
    monkeypatch.setattr(
        answer_claim,
        "build_customer_claims_pipeline",
        lambda loaded_settings: (factory_calls.append(loaded_settings) or pipeline),
    )

    code = answer_claim.main(["--message", "Где правила доставки?"])
    captured = capsys.readouterr()
    assert code == 0
    assert settings_calls == ["loaded"]
    assert factory_calls == [settings]
    json.loads(captured.out)
    pipeline.handle.assert_called_once()


def test_main_missing_message_exits_with_code_2() -> None:
    with pytest.raises(SystemExit) as exc_info:
        answer_claim.main([])
    assert exc_info.value.code == 2


def test_serialize_answer_payload_is_deterministic() -> None:
    result = _pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    payload = answer_claim.format_answer_payload(result)
    first = answer_claim.serialize_answer_payload(payload)
    second = answer_claim.serialize_answer_payload(payload)
    assert first == second
    parsed = json.loads(first)
    assert list(parsed.keys()) == sorted(parsed.keys())
