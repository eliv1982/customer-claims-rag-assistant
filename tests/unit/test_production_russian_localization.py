"""Regression tests for Russian localization of the production user-facing flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_claims_rag.generation_config import DEFAULT_GENERATION_PROMPT_PATH
from customer_claims_rag.generation.fallback import (
    GENERATION_FAILURE_CUSTOMER_RESPONSE,
    INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE,
)
from customer_claims_rag.generation.handoff import CRITICAL_HANDOFF_NOTICE, HIGH_HANDOFF_NOTICE
from customer_claims_rag.generation.prompt_builder import PromptBuilder
from customer_claims_rag.generation.risk_integration_models import RiskAwareGenerationOutcome
from customer_claims_rag.risk import assess_deterministic_risk, handoff_flags_for_level
from customer_claims_rag.risk.models import RiskAssessmentRequest, RiskLevel
from customer_claims_rag.ui.display import (
    INPUT_ERROR_MESSAGE,
    LOADING_MESSAGE,
    SERVICE_ERROR_MESSAGE,
    STARTUP_CONFIG_ERROR_MESSAGE,
    STARTUP_ERROR_MESSAGE,
    STARTUP_INDEX_ERROR_MESSAGE,
    UNEXPECTED_ERROR_MESSAGE,
    _OUTCOME_NOTICES,
    _RISK_LABELS,
    generation_outcome_notice,
    startup_error_view,
    startup_error_view_for_exception,
)
from customer_claims_rag.ui.release_identity import format_release_identity_lines

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"
STREAMLIT_APP_PATH = (
    PROJECT_ROOT / "src" / "customer_claims_rag" / "ui" / "streamlit_app.py"
)

_JSON_CONTRACT_KEYS = (
    '"response_mode"',
    '"grounded_answer"',
    '"insufficient_context"',
    '"answer"',
)

_USER_VISIBLE_STREAMLIT_STRINGS = (
    "FoodFlow — обращения клиентов",
    "FoodFlow — первичная обработка обращений",
    "Текст обращения",
    "Опишите проблему клиента",
    "Проанализировать обращение",
    "Черновик ответа клиенту",
    "Служебная информация",
    "Категория обращения",
    "Рекомендуемый маршрут обработки",
    "Рекомендуемые действия сотрудника",
    "Основания и источники",
    "Источники не указаны.",
    "Идентификация рабочего релиза",
)


def _contains_cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def test_authoritative_production_prompt_path_and_file() -> None:
    assert PROMPT_PATH.is_file()
    assert DEFAULT_GENERATION_PROMPT_PATH.resolve() == PROMPT_PATH.resolve()
    builder = PromptBuilder(prompt_path=PROMPT_PATH)
    assert builder.system_message


def test_production_prompt_requires_russian_user_answers() -> None:
    prompt = PromptBuilder(prompt_path=PROMPT_PATH).system_message.lower()
    assert "русск" in prompt
    assert "только на русском" in prompt or "только на русском языке" in prompt


def test_production_prompt_forbids_medical_diagnoses() -> None:
    prompt = PromptBuilder(prompt_path=PROMPT_PATH).system_message.lower()
    assert "медицинск" in prompt
    assert "диагноз" in prompt
    assert "не ставьте медицинские диагнозы" in prompt or "не ставьте" in prompt


def test_production_prompt_forbids_false_refund_confirmation() -> None:
    prompt = PromptBuilder(prompt_path=PROMPT_PATH).system_message.lower()
    assert "возврат" in prompt
    assert "компенсац" in prompt
    assert "не могу подтвердить" in prompt or "не заявляйте" in prompt


def test_production_prompt_preserves_json_contract_keys() -> None:
    prompt = PromptBuilder(prompt_path=PROMPT_PATH).system_message
    for key in _JSON_CONTRACT_KEYS:
        assert key in prompt


@pytest.mark.parametrize("fragment", _USER_VISIBLE_STREAMLIT_STRINGS)
def test_streamlit_user_visible_strings_are_russian(fragment: str) -> None:
    source = STREAMLIT_APP_PATH.read_text(encoding="utf-8")
    assert fragment in source
    assert _contains_cyrillic(fragment)


def test_loading_message_is_russian() -> None:
    assert _contains_cyrillic(LOADING_MESSAGE)
    assert "FoodFlow" in LOADING_MESSAGE


@pytest.mark.parametrize(
    "message",
    [
        INPUT_ERROR_MESSAGE,
        SERVICE_ERROR_MESSAGE,
        UNEXPECTED_ERROR_MESSAGE,
        STARTUP_ERROR_MESSAGE,
        STARTUP_CONFIG_ERROR_MESSAGE,
        STARTUP_INDEX_ERROR_MESSAGE,
    ],
)
def test_safe_error_messages_are_russian(message: str) -> None:
    assert _contains_cyrillic(message)


def test_startup_index_message_has_no_english_release_posture_phrase() -> None:
    lowered = STARTUP_INDEX_ERROR_MESSAGE.lower()
    assert "release posture" not in lowered
    assert "posture" not in lowered


def test_generation_error_fallback_customer_text_is_russian() -> None:
    assert _contains_cyrillic(GENERATION_FAILURE_CUSTOMER_RESPONSE)
    notice = generation_outcome_notice("generation_error_fallback")
    assert notice is not None
    assert _contains_cyrillic(notice)


def test_retrieval_empty_fallback_customer_text_is_russian() -> None:
    assert _contains_cyrillic(INSUFFICIENT_CONTEXT_CUSTOMER_RESPONSE)
    notice = generation_outcome_notice("insufficient_context")
    assert notice is not None
    assert _contains_cyrillic(notice)


def test_risk_escalation_and_source_labels_are_russian() -> None:
    for label in _RISK_LABELS.values():
        assert _contains_cyrillic(label)
    assert _contains_cyrillic(HIGH_HANDOFF_NOTICE)
    assert _contains_cyrillic(CRITICAL_HANDOFF_NOTICE)
    for outcome, notice in _OUTCOME_NOTICES.items():
        if notice is not None:
            assert _contains_cyrillic(notice), outcome


def test_release_identity_labels_are_russian_except_internal_ids() -> None:
    from customer_claims_rag.release.posture import ReleasePostureDiagnostics

    diagnostics = ReleasePostureDiagnostics(
        release_posture_id="foodflow-10doc-release-v1",
        selected_target="active",
        target_status="selected_production_release",
        index_path_relative="data/04_index_backup_10docs_215chunks",
        corpus_fingerprint="bf3df0d4631f29f322760b50735382039f67d3ee7c6860b25b1ce221312074f3",
        chunk_count=215,
        document_count=10,
        collection_name="customer_claims",
        embedding_model="text-embedding-3-small",
        vector_dimension=1536,
        frozen_retrieval_config_path="configs/retrieval/vector_pool_expansion_v1.json",
        frozen_config_hash="ff53ff9721ad86b1c542bf96dce616d9057ed3b347e341fed59750b07b69e048",
        vector_fetch_k=24,
        candidate_pool_k=24,
        final_top_k=12,
        similarity_threshold=0.0,
        reranker_id="source-authority-v1",
    )
    from customer_claims_rag.ui.release_identity import map_diagnostics_to_release_view

    lines = format_release_identity_lines(
        map_diagnostics_to_release_view(diagnostics, service_ready=True),
    )
    rendered = "\n".join(lines)
    assert "Релиз:" in rendered
    assert "Реранкер:" in rendered
    assert "Reranker:" not in rendered
    assert "Контракт извлечения:" in rendered
    assert "retrieval:" not in rendered.lower()


def test_risk_escalation_invariants_unchanged() -> None:
    high = assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="Упаковка была вскрыта."),
    )
    critical = assess_deterministic_risk(
        RiskAssessmentRequest(customer_query="После еды стало трудно дышать."),
    )
    assert high.risk_floor is RiskLevel.HIGH
    assert critical.risk_floor is RiskLevel.CRITICAL
    assert handoff_flags_for_level(RiskLevel.HIGH) == (True, False)
    assert handoff_flags_for_level(RiskLevel.CRITICAL) == (True, True)


def test_generation_outcome_enum_values_remain_english() -> None:
    outcomes: tuple[RiskAwareGenerationOutcome, ...] = (
        "grounded_answer",
        "insufficient_context",
        "out_of_scope",
        "generation_error_fallback",
    )
    for outcome in outcomes:
        assert outcome.isascii()
        assert _contains_cyrillic(outcome) is False


def test_startup_error_views_use_russian_categories_not_tracebacks() -> None:
    view = startup_error_view()
    assert view.category == "startup"
    assert "traceback" not in view.message.lower()
    config_view = startup_error_view_for_exception(ValueError("secret /path/config"))
    assert config_view.message == STARTUP_CONFIG_ERROR_MESSAGE
    assert "/path" not in config_view.message
