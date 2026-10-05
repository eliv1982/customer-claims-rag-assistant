"""The final-acceptance capture tool: its scenario table and its guards.

The committed evidence is the *result* of ``scripts/capture_final_acceptance.py``. Whether that result
is genuine is decided by these tests together with ``test_final_acceptance_evidence.py``: the
scenarios must cover what the acceptance claims to cover, the stubbed model replies must be what the
scenario says they are, and the tool must refuse to run with a credential.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from customer_claims_rag.application.customer_text_policy import detect_draft_violations, strip_citation_markers

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capture_final_acceptance.py"


def _load():
    spec = importlib.util.spec_from_file_location("capture_final_acceptance_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolves string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


acceptance = _load()

CLI_SCHEMA_KEYS = frozenset(
    {
        "answer",
        "answer_provenance",
        "response_mode",
        "generation_outcome",
        "failure_source",
        "assessment_status",
        "risk_floor",
        "risk_label",
        "risk_note",
        "handoff_required",
        "priority_handoff",
        "handoff_notice",
        "citations",
        "retrieved_materials",
    }
)


def _by_id() -> dict:
    return {scenario.scenario_id: scenario for scenario in acceptance.SCENARIOS}


def test_scenario_ids_are_unique_and_ordered() -> None:
    ids = [scenario.scenario_id for scenario in acceptance.SCENARIOS]
    assert ids == sorted(set(ids))


def test_every_expectation_names_a_field_of_the_current_cli_schema() -> None:
    for scenario in acceptance.SCENARIOS:
        assert scenario.expect, scenario.scenario_id
        assert set(scenario.expect) <= CLI_SCHEMA_KEYS, scenario.scenario_id
        assert "risk_level" not in scenario.expect, "the retired CLI field"


def test_the_scenarios_cover_the_behaviours_the_acceptance_claims() -> None:
    expectations = [scenario.expect for scenario in acceptance.SCENARIOS]
    provenances = {item.get("answer_provenance") for item in expectations}
    assert provenances >= {
        "llm_draft",
        "category_template",
        "safety_replacement",
        "insufficient_context",
        "failure_fallback",
        "unsupported_language",
        "out_of_scope",
    }
    assert {item.get("assessment_status") for item in expectations} >= {
        "rule_match",
        "no_signal",
        "unsupported_language",
    }
    assert {item.get("failure_source") for item in expectations} >= {"retrieval", "generation"}
    critical = [s for s in acceptance.SCENARIOS if s.expect.get("risk_floor") == "critical"]
    assert len(critical) >= 6  # health, fraud, threat, personal data, foreign object, injection + health
    assert all(s.expect.get("priority_handoff") is True for s in critical)


def test_every_critical_scenario_stubs_a_draft_the_customer_must_never_see() -> None:
    """The point of a critical scenario is that the deterministic text wins over a bad draft."""
    for scenario in acceptance.SCENARIOS:
        if scenario.expect.get("answer_provenance") != "category_template":
            continue
        assert scenario.draft is not None and scenario.note, scenario.scenario_id


def test_stubbed_replies_are_what_the_scenarios_say_they_are() -> None:
    for scenario in acceptance.SCENARIOS:
        provenance = scenario.expect.get("answer_provenance")
        if provenance == "llm_draft":
            answer = scenario.draft["answer"]
            assert "[S1]" in answer
            assert detect_draft_violations(strip_citation_markers(answer)) == [], scenario.scenario_id
        if provenance == "safety_replacement":
            answer = strip_citation_markers(scenario.draft["answer"])
            assert detect_draft_violations(answer), "the replaced draft must actually break the text policy"
        if provenance == "failure_fallback" and scenario.expect.get("failure_source") == "generation":
            assert scenario.draft is None and not scenario.retrieval_fails
        if scenario.expect.get("failure_source") == "retrieval":
            assert scenario.retrieval_fails and scenario.anchor_chunk is None
        if provenance in {"out_of_scope", "insufficient_context"}:
            assert scenario.draft["answer"] == ""


def test_scenarios_are_found_by_their_exact_message() -> None:
    first = acceptance.SCENARIOS[0]
    assert acceptance.scenario_by_message(f"  {first.message}\n") is first
    assert acceptance.scenario_by_message("an unrecorded message") is None


def test_the_capture_refuses_to_run_with_a_credential_and_never_loads_the_dotenv(monkeypatch) -> None:
    from customer_claims_rag import env_bootstrap

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-not-a-real-credential")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        acceptance.prepare_environment()

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(env_bootstrap, "_loaded", False)
    acceptance.prepare_environment()
    assert env_bootstrap._loaded is True  # load_project_env() is now a no-op: .env is not read


def test_unexpected_behaviour_fails_the_capture() -> None:
    scenario = _by_id()["S04"]
    acceptance.check_expectations(scenario, dict(scenario.expect))
    with pytest.raises(SystemExit, match="S04"):
        acceptance.check_expectations(scenario, {**scenario.expect, "risk_floor": "low"})
    with pytest.raises(SystemExit, match="S04"):
        acceptance.check_expectations(scenario, {})


def test_source_state_ignores_only_the_evidence_directory(monkeypatch, tmp_path) -> None:
    out_dir = acceptance.ROOT / "deliverables" / "evidence"
    porcelain = {"value": " M deliverables/evidence/scenarios/S01.json\n?? deliverables/evidence/ui/01.png\n"}

    def fake_git(*args):
        return porcelain["value"] if args[0] == "status" else "abc123\n"

    monkeypatch.setattr(acceptance, "_git", fake_git)
    state = acceptance.source_state(out_dir)
    assert state == {"git_commit": "abc123", "tree_clean": True, "changed_paths": []}

    porcelain["value"] += " M src/customer_claims_rag/risk/rules.py\n"
    state = acceptance.source_state(out_dir)
    assert state["tree_clean"] is False
    assert state["changed_paths"] == ["src/customer_claims_rag/risk/rules.py"]


def test_manifest_digests_do_not_depend_on_line_endings(tmp_path) -> None:
    lf, crlf = tmp_path / "a.json", tmp_path / "b.json"
    lf.write_bytes(b'{\n  "a": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert acceptance.sha256_of(lf) == acceptance.sha256_of(crlf)
    png = tmp_path / "c.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n\r\nbody")
    import hashlib

    assert acceptance.sha256_of(png) == hashlib.sha256(png.read_bytes()).hexdigest()
