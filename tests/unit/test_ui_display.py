"""Unit tests for UI display helpers."""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from customer_claims_rag.application.customer_output import NO_SIGNAL_RISK_LABEL
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
from customer_claims_rag.risk.assessment_status import RiskAssessmentStatus
from customer_claims_rag.risk.models import RiskAssessmentRequest
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
    from customer_claims_rag.application.customer_templates import GENERIC_RISK_DRAFTS

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
        # No rule matched: the LOW floor is only a neutral bound, never an affirmative "Низкий".
        (_low_risk(), NO_SIGNAL_RISK_LABEL, "undetermined"),
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
    assert view.risk_floor == "high"
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
    assert view.risk_floor == "critical"
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
    import customer_claims_rag.ui.streamlit_app as streamlit_app

    assert callable(streamlit_app.main)
    assert callable(streamlit_app.get_production_pipeline)


def test_vague_order_query_requests_clarification() -> None:
    """Regression: vague FoodFlow query must ask for clarification, not make promises."""
    vague_query = "С моим заказом что-то не так. Разберитесь."
    result = _pipeline_result(
        risk_assessment=_low_risk(),
        generation=_insufficient_context_generation(),
        generation_outcome="insufficient_context",
    )
    result = CustomerClaimsResult(
        response=result.response,
        customer_query=vague_query,
    )
    view = map_result_to_display(result)

    assert isinstance(view, ClaimSuccessView)

    draft = view.customer_draft

    # Must actively request clarification
    assert any(
        phrase in draft
        for phrase in ("уточните", "укажите", "сообщите")
    ), f"Draft must ask for clarification; got: {draft!r}"

    # Must not promise an already-started review or automatic follow-up
    assert "Мы проверим информацию" not in draft, f"Draft must not claim review started: {draft!r}"
    assert "сообщим о результате" not in draft, f"Draft must not promise follow-up: {draft!r}"

    # No citation markers
    import re
    assert not re.search(r"\[S\d+\]", draft), f"Draft must not contain citation markers: {draft!r}"

    # No internal terminology
    for forbidden in ("insufficient_context", "response_mode", "high", "critical", "low", "medium"):
        assert forbidden not in draft.lower(), (
            f"Draft must not contain internal term {forbidden!r}: {draft!r}"
        )

    # No refund or compensation promises
    assert "возврат" not in draft.lower(), f"Draft must not promise refund: {draft!r}"
    assert "компенсац" not in draft.lower(), f"Draft must not promise compensation: {draft!r}"

    # Neutral floor preserved, but no rule matched: shown as undetermined, not as "low risk"
    assert view.risk_floor == "low"
    assert view.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert view.requires_escalation is False
    assert view.priority_handoff is False

    # Category should reflect generic FoodFlow query
    assert view.claim_category == "Общий запрос по сервису FoodFlow"


def test_vague_order_grounded_answer_with_generic_promise_replaced() -> None:
    """Regression: grounded_answer with vague promise text must be replaced by clarification template.

    Defect: for 'С моим заказом что-то не так. Разберитесь.' the LLM returns
    grounded_answer mode with the generic promise text that previously passed all
    forbidden-pattern checks and was shown verbatim to the customer.
    """
    import re as _re

    from customer_claims_rag.generation.handoff import build_handoff_notice

    vague_query = "С моим заказом что-то не так. Разберитесь."

    # Simulate LLM returning grounded_answer with the defective generic promise text
    bad_llm_text = (
        "Благодарим за обращение. "
        "Мы проверим информацию и при необходимости уточним детали. "
        "После проверки сообщим о результате. [S1]"
    )

    risk = _low_risk()
    result = CustomerClaimsResult(
        response=RiskAwareGroundedGenerationResult(
            generation=_grounded_generation(bad_llm_text),
            risk_assessment=risk,
            handoff_notice=build_handoff_notice(risk),
            generation_outcome="grounded_answer",
        ),
        customer_query=vague_query,
    )
    view = map_result_to_display(result)

    assert isinstance(view, ClaimSuccessView)
    draft = view.customer_draft

    # Must actively request clarification about what happened with the order
    assert any(phrase in draft for phrase in ("уточните", "укажите", "сообщите")), (
        f"Draft must ask for clarification; got: {draft!r}"
    )

    # Must not contain the defective vague-promise phrases
    assert "Мы проверим информацию" not in draft, (
        f"Draft must not claim review started: {draft!r}"
    )
    assert "сообщим о результате" not in draft, (
        f"Draft must not promise follow-up without specifics: {draft!r}"
    )

    # No citation markers [Sx]
    assert not _re.search(r"\[S\d+\]", draft), (
        f"Draft must not contain citation markers: {draft!r}"
    )

    # No internal terminology
    for _forbidden in ("insufficient_context", "response_mode", "generation_outcome"):
        assert _forbidden not in draft.lower(), (
            f"Draft must not contain internal term {_forbidden!r}: {draft!r}"
        )

    # No refund or compensation promises
    assert "возврат" not in draft.lower(), f"Draft must not promise refund: {draft!r}"
    assert "компенсац" not in draft.lower(), f"Draft must not promise compensation: {draft!r}"

    # Neutral floor preserved, but no rule matched: shown as undetermined, not as "low risk"
    assert view.risk_floor == "low"
    assert view.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert view.requires_escalation is False
    assert view.priority_handoff is False

    # Category must reflect generic FoodFlow query (no specific issue detected)
    assert view.claim_category == "Общий запрос по сервису FoodFlow"

    # Draft was sanitized — replaced by template, not passed through raw
    assert view.draft_sanitized is True


def test_delivery_address_grounded_answer_not_replaced() -> None:
    """Informational query with clean grounded answer must not be replaced by generic fallback."""
    import re as _re

    from customer_claims_rag.generation.handoff import build_handoff_notice

    query = "Можно ли изменить адрес доставки?"
    risk = assess_deterministic_risk(RiskAssessmentRequest(customer_query=query))
    clean_llm_text = (
        "Согласно правилам FoodFlow, изменить адрес можно до начала сборки заказа. [S1]"
    )
    result = CustomerClaimsResult(
        response=RiskAwareGroundedGenerationResult(
            generation=_grounded_generation(clean_llm_text),
            risk_assessment=risk,
            handoff_notice=build_handoff_notice(risk),
            generation_outcome="grounded_answer",
        ),
        customer_query=query,
    )
    view = map_result_to_display(result)

    assert isinstance(view, ClaimSuccessView)
    assert view.draft_sanitized is False
    assert "изменить адрес" in view.customer_draft.lower()
    assert not _re.search(r"\[S\d+\]", view.customer_draft)
    assert "Мы проверим информацию" not in view.customer_draft
    assert view.risk_floor == "low"
    assert view.assessment_status is RiskAssessmentStatus.NO_SIGNAL
    assert view.requires_escalation is False


def test_non_delivery_refund_request_uses_non_delivery_category() -> None:
    """Non-delivery + refund: max-risk floor wins; non-delivery category takes precedence."""
    import re as _re

    from customer_claims_rag.generation.handoff import build_handoff_notice
    from customer_claims_rag.risk.reason_codes import RiskReasonCode
    from customer_claims_rag.application.customer_templates import CATEGORY_DRAFT_TEMPLATES

    query = "Я оплатил заказ, но его не доставили. Прошу вернуть деньги."
    risk = assess_deterministic_risk(RiskAssessmentRequest(customer_query=query))
    assert RiskReasonCode.NON_DELIVERY in risk.reason_codes
    assert RiskReasonCode.REFUND_REQUEST in risk.reason_codes
    assert risk.risk_floor.value == "high"
    assert risk.handoff_required is True

    result = CustomerClaimsResult(
        response=RiskAwareGroundedGenerationResult(
            generation=_insufficient_context_generation(),
            risk_assessment=risk,
            handoff_notice=build_handoff_notice(risk),
            generation_outcome="insufficient_context",
        ),
        customer_query=query,
    )
    view = map_result_to_display(result)

    assert isinstance(view, ClaimSuccessView)
    assert view.claim_category == "Недоставка оплаченного заказа"
    assert view.customer_draft == CATEGORY_DRAFT_TEMPLATES["Недоставка оплаченного заказа"]
    assert view.risk_floor == "high"
    assert view.requires_escalation is True
    assert view.handoff_notice is not None
    assert view.draft_sanitized is True
    assert not _re.search(r"\[S\d+\]", view.customer_draft)
    assert "вернём" not in view.customer_draft.lower()
    assert "одобрим возврат" not in view.customer_draft.lower()
    for forbidden in ("insufficient_context", "response_mode", "generation_outcome"):
        assert forbidden not in view.customer_draft.lower()
    assert view.citations == ()


# ---------------------------------------------------------------------------
# Stage 2B / H5: oversized input is rejected below the UI and reported clearly
# ---------------------------------------------------------------------------


def test_process_claim_oversized_message_returns_specific_input_error() -> None:
    from customer_claims_rag.ui.display import INPUT_TOO_LONG_MESSAGE

    pipeline = MagicMock()
    view = process_claim(pipeline, "а" * 4001)
    assert view == ClaimErrorView(INPUT_TOO_LONG_MESSAGE, "input")
    assert "4000" in INPUT_TOO_LONG_MESSAGE
    assert INPUT_TOO_LONG_MESSAGE != INPUT_ERROR_MESSAGE
    pipeline.handle.assert_not_called()


def test_process_claim_accepts_exactly_4000_characters() -> None:
    result = _pipeline_result(
        risk_assessment=_low_risk(),
        generation=_grounded_generation(),
        generation_outcome="grounded_answer",
    )
    pipeline = _mock_pipeline(result)
    view = process_claim(pipeline, "а" * 4000)
    assert isinstance(view, ClaimSuccessView)
    pipeline.handle.assert_called_once()
