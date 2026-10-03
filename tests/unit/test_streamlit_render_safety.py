"""Streamlit rendering boundary (Stage 2C, M2): untrusted text must stay text, never markup.

Customer-visible text, model text and retrieved headings are untrusted. They must reach the page
only through native Streamlit elements fed by ``escape_markdown_text``; every element rendered with
``unsafe_allow_html`` must be static markup. These tests drive the real Streamlit script through
``streamlit.testing.v1.AppTest`` with a fake pipeline (no network, no index).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from customer_claims_rag.application.models import CustomerClaimsResult, RetrievedItemMeta
from customer_claims_rag.generation.fallback import GENERATION_FAILURE_CUSTOMER_RESPONSE
from customer_claims_rag.generation.handoff import build_handoff_notice
from customer_claims_rag.generation.models import Citation, GroundedGenerationResult
from customer_claims_rag.generation.risk_integration_models import (
    RiskAwareGroundedGenerationResult,
)
from customer_claims_rag.risk.models import RiskLevel, RiskSignal
from customer_claims_rag.risk.reason_codes import RiskReasonCode
from customer_claims_rag.risk.validator import build_deterministic_risk_result
from customer_claims_rag.ui import display
from customer_claims_rag.ui.display import ClaimSuccessView, DisplayCitation, escape_markdown_text

APP_PATH = Path(__file__).resolve().parents[2] / "src" / "customer_claims_rag" / "ui" / "streamlit_app.py"

PAYLOADS = (
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    '<a href="javascript:alert(1)">click</a>',
    "[click](javascript:alert(1))",
)
_TAG_FRAGMENTS = ("<script", "<img", "<a href", "onerror", "javascript:")

# Static markup the page is allowed to render with unsafe_allow_html.
_STATIC_HTML = (
    re.compile(r"\A\s*<style>.*</style>\s*\Z", re.DOTALL),
    re.compile(r'\A<div class="divider"></div>\Z'),
    re.compile(r'\A<p class="section-label">[А-Яа-яЁё ]+</p>\Z'),
    # label text is either plain or HTML-entity-escaped, never a raw tag
    re.compile(
        r'\A<span class="risk-badge-(?:neutral|warning|critical|undetermined)">'
        r'(?:[^<>&"\']|&(?:lt|gt|amp|quot|#x27);)+</span>\Z'
    ),
)


class _FakePipeline:
    def __init__(self, result: CustomerClaimsResult) -> None:
        self._result = result

    def handle(self, request) -> CustomerClaimsResult:  # noqa: ANN001
        return self._result


def _result(draft: str, *, heading: str = "Раздел", document_id: str = "doc-1") -> CustomerClaimsResult:
    risk = build_deterministic_risk_result(
        [RiskSignal(reason_code=RiskReasonCode.MISSING_ITEM, level=RiskLevel.MEDIUM, rule_id="missing_item")]
    )
    generation = GroundedGenerationResult(
        response_mode="grounded_answer",
        customer_response=draft,
        citations=[
            Citation(
                citation_key="S1",
                heading=heading,
                document_id=document_id,
                chunk_id="c1",
                source_path="docs/x.md",
            )
        ],
    )
    return CustomerClaimsResult(
        response=RiskAwareGroundedGenerationResult(
            generation=generation,
            risk_assessment=risk,
            handoff_notice=build_handoff_notice(risk),
            generation_outcome="grounded_answer",
        ),
        customer_query="В заказе не хватало позиции",
        retrieved_items=(
            RetrievedItemMeta(citation_key="S1", rank=1, heading=heading, document_id=document_id),
        ),
    )


def _failure_result() -> CustomerClaimsResult:
    risk = build_deterministic_risk_result(
        [RiskSignal(reason_code=RiskReasonCode.DIRECT_THREAT, level=RiskLevel.CRITICAL, rule_id="direct_threat")]
    )
    return CustomerClaimsResult(
        response=RiskAwareGroundedGenerationResult(
            generation=GroundedGenerationResult(
                response_mode="insufficient_context",
                customer_response=GENERATION_FAILURE_CUSTOMER_RESPONSE,
                citations=[],
            ),
            risk_assessment=risk,
            handoff_notice=build_handoff_notice(risk),
            generation_outcome="generation_error_fallback",
        ),
        customer_query="угроза",
        retrieved_items=(
            RetrievedItemMeta(
                citation_key="S1", rank=1, heading=PAYLOADS[0], document_id=PAYLOADS[1]
            ),
        ),
        retrieval_failed=False,
        context_build_failed=True,
    )


def _run_app(monkeypatch: pytest.MonkeyPatch, result: CustomerClaimsResult, *, view=None) -> AppTest:
    """Submit one claim through the real Streamlit script with a fake pipeline."""
    st.cache_resource.clear()
    monkeypatch.setattr(
        "customer_claims_rag.ui.pipeline_resource.create_production_pipeline",
        lambda: _FakePipeline(result),
    )

    def _no_diagnostics():  # noqa: ANN202
        raise RuntimeError("diagnostics are not part of this test")

    monkeypatch.setattr(
        "customer_claims_rag.ui.diagnostics_resource.load_validated_release_diagnostics",
        _no_diagnostics,
    )
    if view is not None:
        monkeypatch.setattr(display, "map_result_to_display", lambda _result: view)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    assert not app.exception
    app.text_area[0].input("Тестовое обращение")
    app.button[0].click()
    app.run()
    assert not app.exception
    return app


def _all_bodies(app: AppTest) -> list[str]:
    bodies = [element.value for element in app.markdown]
    bodies += [element.value for element in app.caption]
    for kind in ("success", "warning", "info", "error"):
        bodies += [element.value for element in getattr(app, kind)]
    return bodies


def _html_enabled_bodies(app: AppTest) -> list[str]:
    return [element.value for element in app.markdown if element.proto.allow_html]


def _unescaped_angle_brackets(body: str) -> bool:
    return re.search(r"(?<!\\)[<>]", body) is not None


def _view(draft: str, *, heading: str, document_id: str, provenance: str = "llm_draft") -> ClaimSuccessView:
    return ClaimSuccessView(
        customer_draft=draft,
        provenance=provenance,  # type: ignore[arg-type]
        response_mode="grounded_answer",
        generation_outcome="grounded_answer",
        failure_source=None,
        assessment_status=build_deterministic_risk_result([]).assessment_status,
        risk_floor="low",
        risk_label=PAYLOADS[0],
        risk_note=None,
        risk_tone="not-a-tone",  # type: ignore[arg-type]
        handoff_notice=None,
        priority_handoff=False,
        requires_escalation=False,
        outcome_notice=None,
        routing_recommendation="Стандартная обработка.",
        claim_category="Категория",
        staff_actions=("Проверьте заказ.",),
        citations=(
            DisplayCitation(
                key="S1", heading=heading, document_id=document_id, label="x", is_confirmed=True
            ),
        ),
        retrieved_materials=(
            DisplayCitation(
                key="S2", heading=heading, document_id=document_id, label="y", is_confirmed=False
            ),
        ),
    )


# ---------------------------------------------------------------------------
# escape_markdown_text
# ---------------------------------------------------------------------------

def test_escape_markdown_text_neutralizes_every_markup_character() -> None:
    assert escape_markdown_text("<script>alert(1)</script>") == r"\<script\>alert\(1\)\<\/script\>"
    assert escape_markdown_text("[x](javascript:alert(1))") == r"\[x\]\(javascript\:alert\(1\)\)"
    assert escape_markdown_text("**bold** _it_ `code` # h :red[x]") == (
        r"\*\*bold\*\* \_it\_ \`code\` \# h \:red\[x\]"
    )
    assert escape_markdown_text("https://evil.example/x") == r"https\:\/\/evil\.example\/x"


def test_escape_markdown_text_keeps_russian_text_and_line_breaks() -> None:
    assert escape_markdown_text("Сожалеем") == "Сожалеем"
    assert escape_markdown_text("а\nб") == "а  \nб"
    assert escape_markdown_text("а\r\nб") == "а  \nб"


@pytest.mark.parametrize("payload", PAYLOADS)
def test_escaped_payload_has_no_unescaped_angle_bracket_or_link(payload: str) -> None:
    escaped = escape_markdown_text(payload)
    assert not _unescaped_angle_brackets(escaped)
    assert re.search(r"(?<!\\)\[", escaped) is None
    assert re.search(r"(?<!\\):", escaped) is None


# ---------------------------------------------------------------------------
# The real script, real policy: hostile model text never reaches the page
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", PAYLOADS)
def test_hostile_model_draft_is_replaced_and_never_rendered(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    app = _run_app(monkeypatch, _result(f"Сожалеем. {payload} [S1]"))
    for body in _all_bodies(app):
        assert payload not in body
        for fragment in _TAG_FRAGMENTS:
            assert fragment not in body.lower()
    # A deterministic template is shown instead, in the warning box (not the green "model draft" box).
    assert not app.success
    assert app.warning
    assert "Не хватало" not in "".join(_all_bodies(app))


@pytest.mark.parametrize("payload", PAYLOADS)
def test_hostile_citation_metadata_is_escaped_in_the_sources_list(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    app = _run_app(monkeypatch, _result("Просим указать номер заказа [S1].", heading=payload, document_id=payload))
    assert app.success, "a clean model draft is shown in the success box"
    for body in _html_enabled_bodies(app):
        for fragment in _TAG_FRAGMENTS:
            assert fragment not in body.lower()
    source_bodies = [b for b in _all_bodies(app) if "S1" in b and "Раздел" not in b]
    assert source_bodies
    for body in source_bodies:
        assert not _unescaped_angle_brackets(body)


def test_retrieved_material_headings_are_escaped_in_the_degraded_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _run_app(monkeypatch, _failure_result())
    assert not app.success
    for body in _html_enabled_bodies(app):
        for fragment in _TAG_FRAGMENTS:
            assert fragment not in body.lower()
    material_bodies = [b for b in _all_bodies(app) if "S1" in b and "alert" in b]
    assert material_bodies, "retrieved material should be listed for staff"
    for body in material_bodies:
        assert not _unescaped_angle_brackets(body)
    joined = " ".join(_all_bodies(app))
    assert "Найденные материалы" in " ".join(e.label for e in app.expander)
    assert "не опирался" in joined
    assert "сбой подготовки материалов" in joined


# ---------------------------------------------------------------------------
# The renderer itself, with the policy bypassed (a hypothetical gap must still be harmless)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("provenance", ["llm_draft", "category_template", "out_of_scope"])
def test_renderer_treats_untrusted_view_fields_as_text(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    provenance: str,
) -> None:
    view = _view(payload, heading=payload, document_id=payload, provenance=provenance)
    app = _run_app(monkeypatch, _result("irrelevant"), view=view)

    box = {"llm_draft": app.success, "category_template": app.warning, "out_of_scope": app.info}[provenance]
    assert len(box) == 1, "the draft is rendered in exactly one native box"
    assert box[0].value == escape_markdown_text(payload)

    # Every HTML-enabled element is known static markup carrying no untrusted fragment ...
    html_bodies = _html_enabled_bodies(app)
    for body in html_bodies:
        assert any(pattern.match(body) for pattern in _STATIC_HTML), f"unexpected HTML-enabled body: {body!r}"
        for fragment in _TAG_FRAGMENTS:
            assert fragment not in body.lower()
    # ... and everywhere else untrusted angle brackets are escaped.
    for body in _all_bodies(app):
        assert body in html_bodies or not _unescaped_angle_brackets(body), body


def test_risk_badge_label_is_html_escaped_and_tone_is_allow_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _view("Просим указать номер заказа.", heading="h", document_id="d")
    app = _run_app(monkeypatch, _result("irrelevant"), view=view)
    badges = [b for b in _html_enabled_bodies(app) if b.startswith("<span")]
    assert badges
    for badge in badges:
        assert "<script" not in badge
        assert "&lt;script&gt;" in badge
        assert 'class="risk-badge-undetermined"' in badge, "unknown tone falls back to the neutral badge"


def test_only_static_markup_uses_unsafe_allow_html() -> None:
    """Static check on the script: no interpolated string may feed an HTML-enabled call."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    html_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "markdown"
        and any(
            kw.arg == "unsafe_allow_html" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        )
    ]
    assert html_calls
    for call in html_calls:
        argument = call.args[0]
        if isinstance(argument, ast.Name):
            assert argument.id == "_PAGE_CSS"
        elif isinstance(argument, ast.Call):
            assert ast.unparse(argument) == "_risk_badge(view.risk_label, view.risk_tone)"
        else:
            assert isinstance(argument, ast.Constant), f"interpolated HTML: {ast.unparse(argument)}"
            assert any(p.match(argument.value) for p in _STATIC_HTML[1:3]), argument.value
    # nothing else in the script enables raw HTML
    flagged = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and any(kw.arg == "unsafe_allow_html" for kw in node.keywords)
    ]
    assert len(flagged) == len(html_calls)
    source = APP_PATH.read_text(encoding="utf-8")
    assert "st.html(" not in source and "components.html" not in source
