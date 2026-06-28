"""Unit tests for UI display helpers."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.application.models import CustomerClaimsRequest, CustomerClaimsResult
from customer_claims_rag.exceptions import IndexManifestError, ReleasePostureError
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
from customer_claims_rag.risk.models import RiskAssessmentRequest
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.ui.display import (
    INPUT_ERROR_MESSAGE,
    LOADING_MESSAGE,
    SERVICE_ERROR_MESSAGE,
    STARTUP_CONFIG_ERROR_MESSAGE,
    STARTUP_INDEX_ERROR_MESSAGE,
    STARTUP_ERROR_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
    ClaimErrorView,
    ClaimSuccessView,
    format_citation_label,
    map_result_to_display,
    process_claim,
    startup_error_view,
    startup_error_view_for_exception,
)
from customer_claims_rag.ui.pipeline_resource import create_production_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
                document_id="policy_refund",
                chunk_id="policy_refund::1",
                heading="Возврат денежных средств",
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


def test_format_citation_label_excludes_source_path_and_chunk_id() -> None:
    citation = _grounded_generation().citations[0]
    label = format_citation_label(citation)
    assert label == "[S1] Возврат денежных средств — policy_refund"
    assert SENSITIVE_SOURCE_PATH not in label
    assert "chunk_id" not in label
    assert citation.chunk_id not in label


def test_map_result_to_display_grounded_answer_and_citations() -> None:
    result = _pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    view = map_result_to_display(result)
    assert isinstance(view, ClaimSuccessView)
    # Citation markers are stripped from the customer draft
    assert "[S1]" not in view.customer_draft
    assert "Ответ по правилам" in view.customer_draft
    assert view.response_mode == "grounded_answer"
    assert view.generation_outcome == "grounded_answer"
    assert view.outcome_notice is None
    assert len(view.citations) == 1
    assert view.citations[0].label.startswith("[S1]")


def test_map_result_to_display_insufficient_context() -> None:
    result = _pipeline_result(
        risk_assessment=_high_risk(),
        generation=_insufficient_context_generation(),
        generation_outcome="insufficient_context",
    )
    view = map_result_to_display(result)
    assert view.generation_outcome == "insufficient_context"
    assert view.citations == ()
    assert view.outcome_notice is not None
    assert "недостаточно" in view.outcome_notice


def test_map_result_to_display_generation_error_fallback() -> None:
    from customer_claims_rag.generation.handoff import build_handoff_notice
    from customer_claims_rag.ui.display import _GENERIC_RISK_DRAFTS

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
    view = map_result_to_display(CustomerClaimsResult(response=response))
    assert view.generation_outcome == "generation_error_fallback"
    # generation_error_fallback → category template or generic risk draft (not the raw error constant)
    assert view.customer_draft != GENERATION_FAILURE_CUSTOMER_RESPONSE
    assert view.draft_sanitized is True
    assert view.outcome_notice is not None
    assert "отказа" in view.outcome_notice


@pytest.mark.parametrize(
    ("risk_assessment", "expected_label", "expected_tone"),
    [
        (_low_risk(), "Низкий", "neutral"),
        (_medium_risk(), "Средний", "neutral"),
    ],
)
def test_map_result_low_and_medium_without_handoff_notice(
    risk_assessment,
    expected_label,
    expected_tone,
) -> None:
    result = _pipeline_result(
        risk_assessment=risk_assessment,
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    view = map_result_to_display(result)
    assert view.handoff_notice is None
    assert view.risk_label == expected_label
    assert view.risk_tone == expected_tone


def test_map_result_high_risk_with_handoff_notice() -> None:
    result = _pipeline_result(
        risk_assessment=_high_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    view = map_result_to_display(result)
    assert view.risk_level == "high"
    assert view.risk_tone == "warning"
    assert view.handoff_notice == HIGH_HANDOFF_NOTICE
    assert view.priority_handoff is False


def test_map_result_critical_priority_handoff() -> None:
    result = _pipeline_result(
        risk_assessment=_critical_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    view = map_result_to_display(result)
    assert view.risk_level == "critical"
    assert view.risk_tone == "critical"
    assert view.priority_handoff is True
    assert view.handoff_notice == CRITICAL_HANDOFF_NOTICE


def test_display_mapping_excludes_reason_codes_and_risk_signals() -> None:
    risk = _high_risk()
    assert risk.reason_codes
    assert risk.risk_signals
    result = _pipeline_result(
        risk_assessment=risk,
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    view = map_result_to_display(result)
    rendered = repr(view)
    assert SENSITIVE_REASON_CODE not in rendered
    assert "reason_codes" not in rendered
    assert "risk_signals" not in rendered
    assert "explanation" not in rendered
    assert SENSITIVE_SOURCE_PATH not in rendered


@pytest.mark.parametrize("message", ["", "   "])
def test_process_claim_invalid_message_returns_input_error(message: str) -> None:
    pipeline = MagicMock()
    view = process_claim(pipeline, message)
    assert view == ClaimErrorView(INPUT_ERROR_MESSAGE, "input")
    pipeline.handle.assert_not_called()


def test_process_claim_startup_error_category_message() -> None:
    view = startup_error_view()
    assert view.message == STARTUP_ERROR_MESSAGE
    assert view.category == "startup"


def test_process_claim_service_failure() -> None:
    pipeline = MagicMock()
    pipeline.handle.side_effect = IndexManifestError("index manifest not found at /secret/path")
    view = process_claim(pipeline, "Упаковка была вскрыта.")
    assert view == ClaimErrorView(SERVICE_ERROR_MESSAGE, "service")
    assert "/secret/path" not in view.message


def test_process_claim_unexpected_failure() -> None:
    pipeline = MagicMock()
    pipeline.handle.side_effect = RuntimeError("boom secret")
    view = process_claim(pipeline, "Упаковка была вскрыта.")
    assert view == ClaimErrorView(UNEXPECTED_ERROR_MESSAGE, "unexpected")
    assert "boom" not in view.message


def test_process_claim_calls_pipeline_handle_once() -> None:
    result = _pipeline_result(
        risk_assessment=_low_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    pipeline = _mock_pipeline(result)
    view = process_claim(pipeline, "  Где правила доставки?  ")
    assert isinstance(view, ClaimSuccessView)
    pipeline.handle.assert_called_once()
    request = pipeline.handle.call_args.args[0]
    assert isinstance(request, CustomerClaimsRequest)
    assert request.customer_query == "Где правила доставки?"


def test_create_production_pipeline_uses_env_and_factory() -> None:
    settings = MagicMock()
    pipeline = MagicMock()
    settings_calls: list[str] = []
    factory_calls: list[object] = []

    def settings_loader():
        settings_calls.append("loaded")
        return settings

    def pipeline_factory(loaded_settings: object):
        factory_calls.append(loaded_settings)
        return pipeline

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            "customer_claims_rag.ui.pipeline_resource.load_project_env",
            lambda: None,
        )
        built = create_production_pipeline(
            settings_loader=settings_loader,
            pipeline_factory=pipeline_factory,
        )

    assert built is pipeline
    assert settings_calls == ["loaded"]
    assert factory_calls == [settings]


def test_streamlit_app_wraps_pipeline_with_cache_resource() -> None:
    source = (
        PROJECT_ROOT
        / "src"
        / "customer_claims_rag"
        / "ui"
        / "streamlit_app.py"
    ).read_text(encoding="utf-8")
    assert "@st.cache_resource" in source
    assert "create_production_pipeline" in source
    assert "get_production_pipeline" in source
    assert "get_release_diagnostics" in source
    assert "st.spinner" in source
    assert "LOADING_MESSAGE" in source
    assert "map_diagnostics_to_release_view" in source


def test_startup_error_view_for_release_posture_uses_index_category() -> None:
    view = startup_error_view_for_exception(ReleasePostureError("descriptor mismatch"))
    assert view.category == "startup_index"
    assert view.message == STARTUP_INDEX_ERROR_MESSAGE


def test_startup_error_view_for_validation_uses_config_category() -> None:
    from pydantic import ValidationError

    view = startup_error_view_for_exception(ValidationError.from_exception_data("x", []))
    assert view.category == "startup_config"
    assert view.message == STARTUP_CONFIG_ERROR_MESSAGE


_FORBIDDEN_STREAMLIT_IMPORT_PACKAGES = (
    "customer_claims_rag.application",
    "customer_claims_rag.retrieval",
    "customer_claims_rag.generation",
    "customer_claims_rag.risk",
)


def test_core_packages_do_not_import_streamlit() -> None:
    for package_name in _FORBIDDEN_STREAMLIT_IMPORT_PACKAGES:
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).resolve().parent
        prefix = f"{package_name}."
        for module_info in pkgutil.walk_packages([str(package_path)], prefix=prefix):
            module = importlib.import_module(module_info.name)
            source_path = Path(module.__file__).resolve()
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert alias.name != "streamlit"
                        assert not alias.name.startswith("streamlit.")
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module != "streamlit"
                    assert not node.module.startswith("streamlit.")


def test_streamlit_app_imports_with_ui_dependency() -> None:
    pytest.importorskip("streamlit")
    import customer_claims_rag.ui.streamlit_app as streamlit_app

    assert callable(streamlit_app.main)
    assert callable(streamlit_app.get_production_pipeline)
