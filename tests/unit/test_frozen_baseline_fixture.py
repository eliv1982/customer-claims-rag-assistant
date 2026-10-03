"""The committed frozen-baseline fixture is static, sanitized and internally consistent.

The A/B, pool-expansion, rebuild and reporting regressions all read
``tests/fixtures/frozen_retrieval_baseline_v1.json``. These tests make a malformed,
tampered or semantically inconsistent fixture fail loudly instead of letting those
regressions pass against bad input. ``_problems`` returns every inconsistency it finds;
the last tests prove each detector fires on a deliberately damaged copy.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from customer_claims_rag.evaluation.metrics import aggregate_case_metrics, compute_case_metrics
from customer_claims_rag.evaluation.models import EvaluationRun
from customer_claims_rag.evaluation.parser import load_evaluation_corpus
from customer_claims_rag.evaluation.result_id import compute_evaluation_result_id
from tests.frozen_fixtures import (
    FROZEN_RETRIEVAL_BASELINE,
    FROZEN_RETRIEVAL_BASELINE_PROVENANCE,
    load_frozen_retrieval_baseline,
)
from tests.local_artifacts import requires_local_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT_ROOT / "tests" / "01_test_questions.md"
EXPECTED = PROJECT_ROOT / "tests" / "02_expected_answers.md"
TRACKED_AB_ARTIFACT = PROJECT_ROOT / "data" / "05_evaluation" / "reranking_ab_source_authority_v1.json"
LOCAL_BASELINE = PROJECT_ROOT / "data" / "05_evaluation" / "retrieval_results.json"

RELEASE_10_DOCUMENTS = {
    "01_service_overview",
    "02_delivery_rules",
    "03_order_changes_and_cancellations",
    "04_refund_policy",
    "05_compensation_policy",
    "06_food_quality_and_packaging",
    "07_complaint_handling_procedure",
    "08_escalation_and_risk_rules",
    "09_response_style_and_templates",
    "10_customer_faq",
}
FALLBACK_CASES = {"T006", "T060"}
LEAK_PATTERNS = {
    "windows drive path": re.compile(r"[A-Za-z]:[\\/]"),
    "escaped windows path": re.compile(r"[A-Za-z]:\\\\"),
    "unix home path": re.compile(r"/(?:Users|home)/"),
    "api key": re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    "credential word": re.compile(r"OPENAI|api[_-]?key|bearer\s", re.IGNORECASE),
}


def _provenance() -> dict:
    return json.loads(FROZEN_RETRIEVAL_BASELINE_PROVENANCE.read_text(encoding="utf-8"))


def _problems(run: EvaluationRun, *, raw_text: str, provenance: dict) -> list[str]:
    """Every way the fixture can disagree with itself, its provenance or the evaluation corpus."""
    problems: list[str] = []
    digest = hashlib.sha256(raw_text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    if digest != provenance["content_sha256"]:
        problems.append(f"content digest {digest} != provenance {provenance['content_sha256']}")

    metadata = run.run_metadata
    for key, actual in (
        ("evaluation_result_id", metadata.evaluation_result_id),
        ("index_fingerprint", metadata.index_fingerprint),
        ("chunk_count", metadata.chunk_count),
        ("document_count", metadata.document_count),
        ("case_count", metadata.evaluation_case_count),
        ("source_git_commit", metadata.git_commit),
    ):
        if provenance[key] != actual:
            problems.append(f"provenance {key} {provenance[key]!r} != run {actual!r}")

    recomputed_id = compute_evaluation_result_id(run)
    if recomputed_id != metadata.evaluation_result_id:
        problems.append(f"evaluation_result_id {metadata.evaluation_result_id} recomputes as {recomputed_id}")

    for label, pattern in LEAK_PATTERNS.items():
        match = pattern.search(raw_text)
        if match:
            problems.append(f"{label} found: {raw_text[max(0, match.start() - 20):match.end() + 20]!r}")

    for case in run.case_results:
        recomputed = compute_case_metrics(
            test_id=case.test_id,
            query=case.query,
            expected_risk=case.expected_risk,
            expected_primary_documents=case.expected_primary_documents,
            expected_supporting_documents=case.expected_supporting_documents,
            fallback_expected=case.fallback_expected,
            category=case.category,
            retrieved_chunks=case.retrieved_chunks,
        )
        stale = sorted(
            field
            for field, value in recomputed.model_dump().items()
            if value != case.model_dump()[field]
        )
        if stale:
            problems.append(f"{case.test_id}: stored metrics disagree with chunks ({', '.join(stale)})")

        ranks = [chunk.rank for chunk in case.retrieved_chunks]
        if ranks != list(range(1, len(ranks) + 1)) or len(ranks) > run.run_metadata.top_k:
            problems.append(f"{case.test_id}: ranks {ranks} are not 1..n within top_k")
        similarities = [chunk.similarity for chunk in case.retrieved_chunks]
        if similarities != sorted(similarities, reverse=True):
            problems.append(f"{case.test_id}: similarities are not in descending rank order")
        for chunk in case.retrieved_chunks:
            if abs(chunk.similarity + chunk.distance - 1.0) > 1e-6:
                problems.append(f"{case.test_id}/{chunk.chunk_id}: similarity + distance != 1")
            if not chunk.chunk_id.startswith(f"{chunk.document_id}::"):
                problems.append(f"{case.test_id}/{chunk.chunk_id}: chunk id does not belong to {chunk.document_id}")
            if chunk.source_path != f"data/02_clean_markdown/{chunk.document_id}.md":
                problems.append(f"{case.test_id}/{chunk.chunk_id}: source_path {chunk.source_path!r} not repo-relative")
            if chunk.document_id not in RELEASE_10_DOCUMENTS:
                problems.append(f"{case.test_id}/{chunk.chunk_id}: document outside the 10-document release corpus")

    if aggregate_case_metrics(run.case_results).model_dump() != run.aggregate_metrics.model_dump():
        problems.append("aggregate_metrics do not recompute from case_results")

    corpus = load_evaluation_corpus(questions_path=QUESTIONS, expected_path=EXPECTED)
    if [case.test_id for case in run.case_results] != [case.test_id for case in corpus]:
        problems.append("case ids/order differ from tests/01_test_questions.md")
    for stored, expected in zip(run.case_results, corpus):
        for field in (
            "query",
            "expected_risk",
            "expected_primary_documents",
            "expected_supporting_documents",
            "fallback_expected",
            "category",
        ):
            if getattr(stored, field) != getattr(expected, field):
                problems.append(f"{stored.test_id}: {field} differs from the evaluation corpus")

    statuses = {case.test_id: case.status for case in run.case_results if case.status != "success"}
    if statuses != {case_id: "no_grounding" for case_id in FALLBACK_CASES}:
        problems.append(f"unexpected non-success cases: {statuses}")
    return problems


@pytest.fixture(scope="module")
def raw_text() -> str:
    return FROZEN_RETRIEVAL_BASELINE.read_text(encoding="utf-8")


def test_fixture_is_present_complete_and_consistent(raw_text: str) -> None:
    run = load_frozen_retrieval_baseline()
    assert len(run.case_results) == 60
    assert _problems(run, raw_text=raw_text, provenance=_provenance()) == []


def test_fixture_is_the_baseline_the_committed_ab_artifact_was_measured_against() -> None:
    run = load_frozen_retrieval_baseline()
    identity = json.loads(TRACKED_AB_ARTIFACT.read_text(encoding="utf-8"))["baseline_identity"]
    assert identity["frozen_evaluation_result_id"] == run.run_metadata.evaluation_result_id
    assert identity["frozen_index_fingerprint"] == run.run_metadata.index_fingerprint
    assert identity["case_count"] == len(run.case_results)


def test_digest_check_ignores_line_endings(raw_text: str) -> None:
    # A Windows checkout may hold the same content with CRLF endings.
    crlf_text = raw_text.replace("\r\n", "\n").replace("\n", "\r\n")
    problems = _problems(load_frozen_retrieval_baseline(), raw_text=crlf_text, provenance=_provenance())
    assert problems == []


# --- the detectors fire on damaged copies --------------------------------------


def _damaged(raw_text: str, transform) -> tuple[EvaluationRun, str]:
    payload = json.loads(raw_text)
    transform(payload)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return EvaluationRun.model_validate(json.loads(text)), text


@pytest.mark.parametrize(
    ("label", "transform", "expected_fragment"),
    [
        (
            "retargeted chunk id",
            lambda p: p["case_results"][0]["retrieved_chunks"][0].update(chunk_id="10_customer_faq::faq-99"),
            "recomputes as",
        ),
        (
            "edited similarity",
            lambda p: p["case_results"][3]["retrieved_chunks"][0].update(similarity=0.01),
            "similarity",
        ),
        (
            "absolute path",
            lambda p: p["case_results"][1]["retrieved_chunks"][0].update(
                source_path="C:\\Users\\someone\\repo\\data\\02_clean_markdown\\10_customer_faq.md"
            ),
            "windows drive path",
        ),
        (
            "dropped case",
            lambda p: p["case_results"].pop(),
            "case ids/order differ",
        ),
        (
            "changed expected document",
            lambda p: p["case_results"][2].update(expected_primary_documents=["02_delivery_rules"]),
            "expected_primary_documents differs from the evaluation corpus",
        ),
        (
            "flipped fallback status",
            lambda p: p["case_results"][5].update(status="success"),
            "unexpected non-success cases",
        ),
        (
            "stale aggregate",
            lambda p: p["aggregate_metrics"].update(hit_rate_at_4=0.5),
            "aggregate_metrics do not recompute",
        ),
    ],
)
def test_detectors_reject_a_damaged_fixture(raw_text: str, label: str, transform, expected_fragment: str) -> None:
    run, text = _damaged(raw_text, transform)
    problems = _problems(run, raw_text=text, provenance=_provenance())
    assert any(expected_fragment in problem for problem in problems), (label, problems)


def test_a_modified_file_no_longer_matches_the_recorded_digest(raw_text: str) -> None:
    run, text = _damaged(raw_text, lambda p: p["case_results"][0].update(category="tampered"))
    assert any("content digest" in problem for problem in _problems(run, raw_text=text, provenance=_provenance()))


# --- maintainer artifact ---------------------------------------------------------


@requires_local_artifacts(
    LOCAL_BASELINE,
    why="the maintainer's frozen baseline, compared against the committed fixture",
)
def test_local_baseline_artifact_still_equals_the_committed_fixture() -> None:
    """When the real artifact exists it must not have drifted from the fixture the tests use."""
    local = json.loads(LOCAL_BASELINE.read_text(encoding="utf-8"))
    committed = json.loads(FROZEN_RETRIEVAL_BASELINE.read_text(encoding="utf-8"))
    assert local == committed, (
        "data/05_evaluation/retrieval_results.json differs from "
        "tests/fixtures/frozen_retrieval_baseline_v1.json; either the local file is stale or "
        "the fixture must be deliberately re-frozen (see its provenance file)"
    )
