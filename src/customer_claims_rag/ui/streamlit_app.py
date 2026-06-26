"""Streamlit MVP for FoodFlow customer claims handling."""

from __future__ import annotations

from pydantic import ValidationError

import streamlit as st

from customer_claims_rag.exceptions import GenerationError, RetrievalError
from customer_claims_rag.ui.diagnostics_resource import load_validated_release_diagnostics
from customer_claims_rag.ui.display import (
    LOADING_MESSAGE,
    ClaimErrorView,
    ClaimSuccessView,
    process_claim,
    startup_error_view_for_exception,
)
from customer_claims_rag.ui.pipeline_resource import create_production_pipeline
from customer_claims_rag.ui.release_identity import (
    format_release_identity_lines,
    map_diagnostics_to_release_view,
)


@st.cache_resource
def get_release_diagnostics():
    """Return cached validated release diagnostics for this Streamlit process."""
    return load_validated_release_diagnostics()


@st.cache_resource
def get_production_pipeline():
    """Return the cached production pipeline for this Streamlit process."""
    return create_production_pipeline()


def _render_release_identity(*, service_ready: bool) -> None:
    try:
        diagnostics = get_release_diagnostics()
        identity = map_diagnostics_to_release_view(diagnostics, service_ready=service_ready)
    except Exception:
        st.caption("Идентификация релиза недоступна.")
        return

    with st.expander("Идентификация production-релиза", expanded=False):
        for line in format_release_identity_lines(identity):
            st.caption(line)


def _render_success(view: ClaimSuccessView) -> None:
    st.subheader("Ответ")
    st.write(view.answer)

    st.subheader("Уровень риска")
    if view.risk_tone == "critical":
        st.error(view.risk_label)
    elif view.risk_tone == "warning":
        st.warning(view.risk_label)
    else:
        st.info(view.risk_label)

    if view.handoff_notice:
        st.subheader("Передача сотруднику поддержки")
        if view.priority_handoff:
            st.error(view.handoff_notice)
        else:
            st.warning(view.handoff_notice)

    if view.outcome_notice:
        st.subheader("Статус обработки")
        st.info(view.outcome_notice)

    st.subheader("Источники")
    if view.citations:
        for citation in view.citations:
            st.write(citation.label)
    else:
        st.write("Источники не указаны.")


def _render_error(view: ClaimErrorView) -> None:
    if view.category == "input":
        st.warning(view.message)
    elif view.category in {"startup", "startup_config", "startup_index"}:
        st.error(view.message)
    else:
        st.error(view.message)


def main() -> None:
    st.set_page_config(page_title="FoodFlow — обращения клиентов", layout="centered")
    st.title("FoodFlow — первичная обработка обращений")
    st.write(
        "Введите текст обращения клиента. Ассистент подготовит проект ответа "
        "на основе базы знаний, оценит уровень риска и при необходимости "
        "укажет на передачу сотруднику поддержки."
    )

    pipeline = None
    try:
        pipeline = get_production_pipeline()
    except (RetrievalError, GenerationError, ValidationError, ValueError) as exc:
        _render_release_identity(service_ready=False)
        _render_error(startup_error_view_for_exception(exc))
        return
    except Exception:
        _render_release_identity(service_ready=False)
        _render_error(startup_error_view_for_exception(RuntimeError("startup")))
        return

    _render_release_identity(service_ready=True)

    with st.form("claim_form", clear_on_submit=False):
        message = st.text_area(
            "Текст обращения",
            placeholder="Опишите проблему клиента...",
            height=160,
        )
        submitted = st.form_submit_button("Проанализировать обращение")

    if not submitted:
        return

    with st.spinner(LOADING_MESSAGE):
        view = process_claim(pipeline, message)
    if isinstance(view, ClaimSuccessView):
        _render_success(view)
    else:
        _render_error(view)


if __name__ == "__main__":
    main()
