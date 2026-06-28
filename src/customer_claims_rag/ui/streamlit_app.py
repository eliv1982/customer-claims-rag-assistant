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

# ---------------------------------------------------------------------------
# CSS visual polish
# No orphan HTML wrapper divs — all content styled via CSS selectors or
# via inline single-element st.markdown("<div>TEXT</div>") calls.
# ---------------------------------------------------------------------------

_PAGE_CSS = """
<style>
/* ── Page background ─────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main {
    background: #f0f4f8;
}

/* ── Style the Streamlit form container ──────────────────────────── */
[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #d0dcea;
    border-radius: 8px;
    padding: 8px 4px 4px 4px;
}

/* ── Calm the primary / form-submit button ───────────────────────── */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stFormSubmitButton"] > button:focus,
[data-testid="stFormSubmitButton"] > button:active {
    background-color: #2e7d9a;
    border-color: #2e7d9a;
    color: #ffffff;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #23607a;
    border-color: #23607a;
    color: #ffffff;
}

/* ── Reduce textarea visual dominance ────────────────────────────── */
[data-testid="stTextArea"] textarea {
    min-height: 110px !important;
    font-size: 0.97rem;
}

/* ── Customer draft card ─────────────────────────────────────────── */
.draft-card {
    background: #edf6ee;
    border: 1px solid #aed4b1;
    border-left: 4px solid #3a8f42;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 1rem;
    line-height: 1.7;
    color: #1a3a1d;
}
.draft-card-warning {
    background: #fdf6e3;
    border: 1px solid #e8cf8a;
    border-left: 4px solid #c4920a;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 1rem;
    line-height: 1.7;
    color: #4a3800;
}
.draft-card-oos {
    background: #eef0fa;
    border: 1px solid #9fa8da;
    border-left: 4px solid #5c6bc0;
    border-radius: 8px;
    padding: 14px 20px;
    margin-bottom: 12px;
    font-size: 1rem;
    color: #263578;
}

/* ── Risk badge ──────────────────────────────────────────────────── */
.risk-badge-neutral  { background:#388e3c; color:#fff; padding:3px 12px; border-radius:4px;
                       font-size:0.82rem; font-weight:600; display:inline-block; }
.risk-badge-warning  { background:#c0690a; color:#fff; padding:3px 12px; border-radius:4px;
                       font-size:0.82rem; font-weight:600; display:inline-block; }
.risk-badge-critical { background:#b71c1c; color:#fff; padding:3px 12px; border-radius:4px;
                       font-size:0.82rem; font-weight:600; display:inline-block; }

/* ── Section labels ──────────────────────────────────────────────── */
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #5c7a90;
    margin-bottom: 4px;
}
.divider { border-top: 1px solid #d8e5ef; margin: 16px 0; }
</style>
"""


@st.cache_resource
def get_release_diagnostics():
    return load_validated_release_diagnostics()


@st.cache_resource
def get_production_pipeline():
    return create_production_pipeline()


def _render_release_identity_expander(*, service_ready: bool) -> None:
    try:
        diagnostics = get_release_diagnostics()
        identity = map_diagnostics_to_release_view(diagnostics, service_ready=service_ready)
    except Exception:
        st.caption("Идентификация релиза недоступна.")
        return
    with st.expander("Идентификация рабочего релиза", expanded=False):
        for line in format_release_identity_lines(identity):
            st.caption(line)


def _risk_badge(label: str, tone: str) -> str:
    css_class = f"risk-badge-{tone}"
    return f'<span class="{css_class}">{label}</span>'


def _render_success(view: ClaimSuccessView) -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Card 1: Customer draft ─────────────────────────────────────
    st.markdown('<p class="section-label">Черновик ответа клиенту</p>', unsafe_allow_html=True)
    st.caption("Проверьте и при необходимости отредактируйте перед отправкой.")

    outcome = view.generation_outcome

    if outcome == "out_of_scope":
        st.markdown(
            f'<div class="draft-card-oos">{view.customer_draft}</div>',
            unsafe_allow_html=True,
        )
    elif view.draft_sanitized:
        st.markdown(
            f'<div class="draft-card-warning">{view.customer_draft}</div>',
            unsafe_allow_html=True,
        )
    else:
        draft_text = view.customer_draft or ""
        if draft_text:
            st.markdown(
                f'<div class="draft-card">{draft_text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="draft-card-warning">Текст ответа не был подготовлен. '
                "Обращение требует ручной проверки.</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Card 2: Staff information ──────────────────────────────────
    st.markdown('<p class="section-label">Служебная информация</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f"**Категория обращения:** {view.claim_category}")

        col_risk, col_route = st.columns([1, 2])

        with col_risk:
            st.markdown("**Уровень риска**")
            st.markdown(_risk_badge(view.risk_label, view.risk_tone), unsafe_allow_html=True)
            esc_text = "Да" if view.requires_escalation else "Нет"
            st.caption(f"Требуется эскалация: **{esc_text}**")

        with col_route:
            st.markdown("**Рекомендуемый маршрут обработки**")
            st.markdown(view.routing_recommendation)

    if view.handoff_notice:
        if view.priority_handoff:
            st.error(f"⚠️ {view.handoff_notice}")
        else:
            st.warning(f"ℹ️ {view.handoff_notice}")

    if view.staff_actions:
        st.markdown("**Рекомендуемые действия сотрудника**")
        for i, action in enumerate(view.staff_actions, 1):
            st.markdown(f"{i}. {action}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Card 3: Sources / retrieved materials ──────────────────────
    if outcome not in {"out_of_scope"}:
        if view.citations:
            with st.expander("Основания и источники", expanded=False):
                st.caption("Фрагменты базы знаний, на которые опирается ответ.")
                for citation in view.citations:
                    st.markdown(
                        f"**{citation.key}** — {citation.heading}  \n"
                        f"<span style='color:#78909c;font-size:0.85rem;'>"
                        f"Документ: {citation.document_id}</span>",
                        unsafe_allow_html=True,
                    )
        elif view.retrieved_materials:
            with st.expander("Найденные материалы для проверки", expanded=False):
                st.caption(
                    "Материалы найдены retrieval. Ответ на них не опирался — "
                    "используйте для ручной проверки."
                )
                for mat in view.retrieved_materials:
                    st.markdown(
                        f"**{mat.key}** — {mat.heading}  \n"
                        f"<span style='color:#78909c;font-size:0.85rem;'>"
                        f"Документ: {mat.document_id}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            with st.expander("Основания и источники", expanded=False):
                st.caption("Источники не указаны.")

    # ── Technical section (always collapsed) ──────────────────────
    with st.expander("Техническая информация", expanded=False):
        st.caption(f"Режим ответа: {view.response_mode}")
        st.caption(f"Исход генерации: {view.generation_outcome}")
        if view.draft_sanitized:
            if outcome == "generation_error_fallback":
                st.caption("Подтверждённый ответ не был получен. Используется проверенный шаблон.")
            else:
                st.caption("Используется проверенный шаблон ответа для данной категории.")
        if view.outcome_notice:
            st.caption(view.outcome_notice)
        _render_release_identity_expander(service_ready=True)


def _render_error(view: ClaimErrorView) -> None:
    if view.category == "input":
        st.warning(view.message)
    else:
        st.error(view.message)


def main() -> None:
    st.set_page_config(page_title="FoodFlow — обращения клиентов", layout="centered")
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

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
        _render_release_identity_expander(service_ready=False)
        _render_error(startup_error_view_for_exception(exc))
        return
    except Exception:
        _render_release_identity_expander(service_ready=False)
        _render_error(startup_error_view_for_exception(RuntimeError("startup")))
        return

    with st.form("claim_form", clear_on_submit=False):
        message = st.text_area(
            "Текст обращения",
            placeholder="Опишите проблему клиента...",
            height=120,
        )
        submitted = st.form_submit_button(
            "Проанализировать обращение",
            type="primary",
        )

    if not submitted:
        _render_release_identity_expander(service_ready=True)
        return

    with st.spinner(LOADING_MESSAGE):
        view = process_claim(pipeline, message)
    if isinstance(view, ClaimSuccessView):
        _render_success(view)
    else:
        _render_error(view)


if __name__ == "__main__":
    main()
