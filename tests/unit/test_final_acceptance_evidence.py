"""The committed final acceptance evidence is internally consistent and matches the committed release.

Two packages live under ``deliverables/``:

* ``evidence/``: current (``scripts/capture_final_acceptance.py`` output). Checked here against the
  committed release descriptor, canonical corpus manifest and CLI schema, and (where the production
  index exists) reproduced by the real pipeline.
* ``historical/stage5c_release_v1/``: the superseded release v1 package, preserved. Only its
  integrity and its labelling are checked.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    load_canonical_corpus_manifest,
)
from customer_claims_rag.release.posture import resolve_production_release_posture
from tests.local_artifacts import requires_local_artifacts

ROOT = Path(__file__).resolve().parents[2]
DELIVERABLES = ROOT / "deliverables"
EVIDENCE = DELIVERABLES / "evidence"
HISTORICAL = DELIVERABLES / "historical" / "stage5c_release_v1"
MANIFEST = EVIDENCE / "final_acceptance_manifest.json"
PRODUCTION_INDEX = ROOT / "data" / "04_index_production"
STAGE2G = ROOT / "data" / "05_evaluation" / "stage2g_pool_expansion.json"

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"OPENAI_API_KEY\s*=\s*\S+"),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile(r"[A-Za-z]:/Users/"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"/Users/[^/\s]+/"),
)
TEXT_SUFFIXES = frozenset({".json", ".md", ".txt"})
SHA1_HEX = re.compile(r"^[0-9a-f]{40}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

CLI_KEYS = frozenset(
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
PROVENANCES = frozenset(
    {
        "llm_draft",
        "category_template",
        "safety_replacement",
        "insufficient_context",
        "failure_fallback",
        "unsupported_language",
        "out_of_scope",
    }
)


def _digests(path: Path) -> set[str]:
    """Accepted SHA-256 digests: text after line-ending normalization (LF or CRLF), binary as is."""
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return {hashlib.sha256(data).hexdigest()}
    lf = data.replace(b"\r\n", b"\n")
    return {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.is_file(), "final acceptance manifest is missing"
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def release():
    context = resolve_production_release_posture(project_root=ROOT)
    corpus = load_canonical_corpus_manifest(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE)
    return context, corpus


@pytest.fixture(scope="module")
def scenarios() -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((EVIDENCE / "scenarios").glob("S*.json"))
    }


@pytest.fixture(scope="module")
def retrieval() -> dict:
    return json.loads((EVIDENCE / "retrieval_summary.json").read_text(encoding="utf-8"))


# --- manifest: provenance and integrity ------------------------------------------------------------------


def test_manifest_states_what_kind_of_run_produced_the_evidence(manifest) -> None:
    assert manifest["schema_version"] == "2.0.0"
    assert manifest["kind"] == "final_acceptance_evidence"
    assert manifest["status"] == "current"
    assert manifest["generator"] == "scripts/capture_final_acceptance.py"
    assert manifest["run_mode"] == "deterministic_stub_run"
    assert manifest["external_requests"] == {"openai": 0, "network": 0}
    assert manifest["environment"]["openai_api_key_in_process_env"] is False
    assert manifest["environment"]["repository_dotenv_read"] is False
    assert manifest["environment"]["tokenizer_vocabulary_needed"] is False
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", manifest["generated_at_utc"])


def test_manifest_names_the_clean_code_commit_that_produced_the_evidence(manifest) -> None:
    source = manifest["source"]
    assert SHA1_HEX.fullmatch(source["git_commit"])
    assert source["source_tree_clean"] is True, "evidence must come from committed sources"


def test_every_listed_file_exists_and_matches_its_digest(manifest) -> None:
    assert manifest["files"], "the manifest lists no files"
    for relative, item in manifest["files"].items():
        path = EVIDENCE / relative
        assert path.is_file(), f"missing evidence file: {relative}"
        assert SHA256_HEX.fullmatch(item["sha256"])
        assert item["sha256"] in _digests(path), f"digest mismatch for {relative}"
        assert item["bytes"] == path.stat().st_size or path.suffix.lower() in TEXT_SUFFIXES, relative


def test_no_file_in_the_package_is_unlisted_or_stale(manifest) -> None:
    present = {
        path.relative_to(EVIDENCE).as_posix()
        for path in EVIDENCE.rglob("*")
        if path.is_file() and path.name not in {"README.md", MANIFEST.name}
    }
    assert present == set(manifest["files"])


def test_screenshots_are_real_png_files(manifest) -> None:
    shots = [name for name in manifest["files"] if name.startswith("ui/")]
    assert len(shots) == 6
    for name in shots:
        data = (EVIDENCE / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 20_000, name


# --- release identity ----------------------------------------------------------------------------------------


def test_manifest_release_identity_is_the_committed_release(manifest, release) -> None:
    context, corpus = release
    target = context.resolved_target
    identity = manifest["release"]
    assert identity == {
        "release_posture_id": context.descriptor.release_posture_id,
        "target": target.target_name,
        "canonical_corpus_id": corpus.corpus_id,
        "corpus_manifest": DEFAULT_CORPUS_MANIFEST_RELATIVE.as_posix(),
        "index_path": target.index_path_relative,
        "collection": target.collection_name,
        "embedding_model": target.embedding_model,
        "vector_dimension": target.vector_dimension,
        "chunk_count": corpus.expected.chunk_count,
        "document_count": corpus.expected.document_count,
        "corpus_fingerprint": corpus.expected.corpus_fingerprint,
        "chunk_payload_digest": corpus.expected.chunk_payload_digest,
        "frozen_config_hash": context.descriptor.expected_frozen_config_hash,
        "release_can_proceed": True,
        "index_matches_canonical_corpus": "yes",
        "index_integrity": "store_recomputed",
    }
    assert (identity["chunk_count"], identity["document_count"]) == (216, 10)


def test_release_posture_capture_states_the_same_identity(manifest, release) -> None:
    context, corpus = release
    lines = (EVIDENCE / "release_posture.txt").read_text(encoding="utf-8").splitlines()
    fields = dict(line.split("=", 1) for line in lines if "=" in line)
    assert fields["release_posture_id"] == context.descriptor.release_posture_id
    assert fields["canonical_corpus_id"] == corpus.corpus_id
    assert fields["corpus_fingerprint"] == corpus.expected.corpus_fingerprint
    assert fields["chunk_payload_digest"] == corpus.expected.chunk_payload_digest
    assert (fields["chunk_count"], fields["document_count"]) == ("216", "10")
    assert fields["index_path"] == "data/04_index_production"
    assert fields["frozen_config_hash"] == context.descriptor.expected_frozen_config_hash
    assert fields["index_integrity"] == "store_recomputed"
    assert fields["index_present"] == "yes"
    assert fields["index_matches_canonical_corpus"] == "yes"
    assert fields["canonical_corpus_sources"].startswith("verified")
    assert fields["release_can_proceed"] == "yes"
    assert fields["exit_status"] == "0"


# --- scenarios and the CLI contract --------------------------------------------------------------------------


def test_every_scenario_of_the_tool_has_a_committed_payload_and_nothing_else(scenarios) -> None:
    tool = _load_script("capture_final_acceptance")
    assert list(scenarios) == [scenario.scenario_id for scenario in tool.SCENARIOS]


def test_committed_payloads_have_exactly_the_current_cli_schema(scenarios) -> None:
    for scenario_id, payload in scenarios.items():
        assert set(payload) == CLI_KEYS, scenario_id
        assert "risk_level" not in payload, f"{scenario_id}: retired CLI field"
        assert payload["answer"].strip() and "�" not in payload["answer"], scenario_id
        assert payload["answer_provenance"] in PROVENANCES
        assert payload["assessment_status"] in {"rule_match", "no_signal", "unsupported_language"}
        assert payload["risk_floor"] in {"low", "medium", "high", "critical"}
        for source in payload["citations"] + payload["retrieved_materials"]:
            assert set(source) == {"key", "heading", "document_id"}, scenario_id


def test_committed_payloads_satisfy_the_expectations_the_tool_records(scenarios) -> None:
    tool = _load_script("capture_final_acceptance")
    for scenario in tool.SCENARIOS:
        tool.check_expectations(scenario, scenarios[scenario.scenario_id])


def test_citations_and_retrieved_materials_keep_their_distinct_meanings(scenarios) -> None:
    for scenario_id, payload in scenarios.items():
        provenance = payload["answer_provenance"]
        if provenance == "llm_draft":
            assert payload["citations"] and payload["retrieved_materials"] == [], scenario_id
        else:
            assert payload["citations"] == [], f"{scenario_id}: only an accepted model draft has citations"
        if provenance == "out_of_scope":
            assert payload["retrieved_materials"] == [], scenario_id
        assert (payload["failure_source"] is not None) == (provenance == "failure_fallback"), scenario_id


def test_no_signal_is_never_presented_as_an_affirmative_level(scenarios) -> None:
    for scenario_id, payload in scenarios.items():
        if payload["assessment_status"] != "rule_match":
            assert payload["risk_floor"] == "low", scenario_id
            assert payload["risk_label"] != "Низкий", f"{scenario_id}: a neutral bound shown as an assessment"
    assert any(p["assessment_status"] == "no_signal" for p in scenarios.values())
    assert any(p["assessment_status"] == "unsupported_language" for p in scenarios.values())


def test_the_scenarios_cover_every_answer_provenance_and_a_failure_of_each_source(scenarios) -> None:
    assert {payload["answer_provenance"] for payload in scenarios.values()} == PROVENANCES
    assert {p["failure_source"] for p in scenarios.values() if p["failure_source"]} == {"retrieval", "generation"}


def test_every_critical_floor_gets_the_fixed_text_and_a_priority_handoff(scenarios) -> None:
    critical = {sid: p for sid, p in scenarios.items() if p["risk_floor"] == "critical"}
    assert len(critical) >= 6
    for scenario_id, payload in critical.items():
        assert payload["answer_provenance"] == "category_template", scenario_id
        assert payload["priority_handoff"] and payload["handoff_required"], scenario_id
        assert payload["citations"] == [], scenario_id


def test_an_injection_around_a_health_complaint_changes_nothing(scenarios) -> None:
    assert scenarios["S09"] == scenarios["S04"]


def test_a_degraded_answer_keeps_the_safety_assessment(scenarios) -> None:
    for scenario_id in ("S10", "S11"):
        payload = scenarios[scenario_id]
        assert payload["assessment_status"] == "rule_match" and payload["risk_floor"] == "high"
        assert payload["handoff_required"] is True and payload["handoff_notice"], scenario_id


# --- retrieval summary ---------------------------------------------------------------------------------------


def test_retrieval_summary_identity_is_the_committed_release(retrieval, release, manifest) -> None:
    context, corpus = release
    identity = retrieval["identity"]
    assert identity["index_fingerprint"] == corpus.expected.corpus_fingerprint
    assert (identity["chunk_count"], identity["document_count"]) == (216, 10)
    assert identity["embedding_model"] == context.resolved_target.embedding_model
    assert identity["collection"] == context.resolved_target.collection_name
    assert identity["retrieval_config_hash"] == context.descriptor.expected_frozen_config_hash
    assert identity["contract"] == {"fetch_k": 24, "pool_k": 24, "final_top_k": 12, "threshold": 0.0}
    assert identity["case_count"] == len(retrieval["cases"]) == 60
    assert SHA256_HEX.fullmatch(identity["evaluation_dataset_fingerprint"])
    link = manifest["retrieval_evaluation"]
    assert link["index_fingerprint"] == identity["index_fingerprint"]
    assert link["retrieval_config_hash"] == identity["retrieval_config_hash"]
    assert link["evaluation_dataset_fingerprint"] == identity["evaluation_dataset_fingerprint"]
    assert link["raw_artifact_sha256"] == retrieval["derived_from"]["artifact_sha256"]


def test_retrieval_summary_records_where_it_came_from(retrieval) -> None:
    source = retrieval["derived_from"]
    assert source["tracked_in_git"] is False and SHA256_HEX.fullmatch(source["artifact_sha256"])
    assert SHA1_HEX.fullmatch(source["run_git_commit"]) and source["run_git_dirty"] is False
    assert SHA1_HEX.fullmatch(source["derived_at_commit"])
    assert not Path(source["artifact"]).is_absolute()


def test_headline_metrics_are_whole_case_counts(retrieval) -> None:
    metrics = retrieval["metrics"]
    for key in ("hit_at_1", "hit_at_4", "hit_at_12", "primary_hit_at_1", "primary_hit_at_4", "supporting_hit_at_4"):
        item = metrics[key]
        assert item["hits"] / item["cases"] == pytest.approx(item["value"], abs=1e-4), key
    rows = retrieval["cases"]
    assert sum(1 for r in rows if r["expected_primary"] and r["primary_hit_at_4"]) == metrics["primary_hit_at_4"]["hits"]
    assert sum(1 for r in rows if r["hit_at_4"]) == metrics["hit_at_4"]["hits"]
    # the figures the README quotes
    assert (round(metrics["hit_at_4"]["value"], 3), round(metrics["primary_hit_at_4"]["value"], 3)) == (0.897, 0.759)
    assert round(metrics["mrr"], 3) == 0.715


def test_pool_reachability_and_the_documented_limitations(retrieval) -> None:
    reach = retrieval["pool_reachability"]
    assert reach["pool_at_24"]["primary"] == "56/57"
    assert reach["pool_at_24"]["fully_unreachable"] == ["T004"]
    assert reach["pool_at_12"]["fully_unreachable"] == ["T004", "T040", "T047"]
    assert reach["primary_by_risk"]["critical"]["pool_at_24"] == "8/8"
    assert reach["primary_by_risk"]["high"]["pool_at_24"] == "15/15"
    limits = retrieval["limitations"]
    assert limits["primary_outside_pool_at_24"] == ["T004"]
    assert {"T040", "T044", "T047"} <= set(limits["ranking_limited_primary_in_pool_not_in_top4"])


def test_no_regression_against_the_previous_baseline_and_one_improvement(retrieval) -> None:
    comparison = retrieval["comparison_with_previous_baseline"]
    assert comparison["same_evaluation_dataset"] and comparison["same_retrieval_config"]
    assert comparison["same_reranker_config"]
    assert comparison["regressions"] == []
    assert {item["case_id"] for item in comparison["improvements"]} == {"T011"}
    assert comparison["baseline"]["tracked_in_git"] is True
    assert (ROOT / comparison["baseline"]["artifact"]).is_file()
    assert comparison["baseline"]["index_fingerprint"] != retrieval["identity"]["index_fingerprint"]


def test_deterministic_behaviour_of_the_frozen_high_and_critical_cases(retrieval) -> None:
    rows = {r["case_id"]: r for r in retrieval["cases"]}
    critical = [r for r in rows.values() if r["risk"] == "critical"]
    assert len(critical) == 8
    for row in critical:
        det = row["deterministic"]
        assert det["assessment_status"] == "rule_match" and det["risk_floor"] == "critical", row["case_id"]
        assert det["priority_handoff"] and det["answer_provenance"] == "category_template", row["case_id"]
    # safety-sensitive categories named in the portfolio claims
    assert rows["T046"]["deterministic"]["claim_category"] == "Признаки мошенничества"
    assert rows["T045"]["deterministic"]["claim_category"] == "Утечка персональных данных"
    assert rows["T044"]["deterministic"]["claim_category"] == "Утечка персональных данных"
    assert rows["T047"]["deterministic"]["claim_category"] == "Прямая угроза"
    # the ranking-limited critical cases are still covered by the fixed text
    for case_id in ("T040", "T047"):
        assert not rows[case_id]["primary_hit_at_4"]
        assert rows[case_id]["deterministic"]["answer_provenance"] == "category_template"
    # the cases the rules do not cover are reported, not hidden
    assert retrieval["limitations"]["dataset_high_or_critical_without_matching_floor"] == ["T039", "T053"]
    assert retrieval["limitations"]["deterministic_floor_two_or_more_levels_above_dataset_label"] == ["T026"]


# --- hygiene --------------------------------------------------------------------------------------------------


def _text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def test_deliverables_contain_no_secrets_and_no_workstation_paths() -> None:
    checked = 0
    for path in _text_files(DELIVERABLES):
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in (*SECRET_PATTERNS, *ABSOLUTE_PATH_PATTERNS):
            assert pattern.search(content) is None, f"{pattern.pattern} in {path.relative_to(ROOT)}"
        checked += 1
    assert checked > 20


def test_current_evidence_uses_no_retired_names() -> None:
    """Generated files name the current release only (the baseline comparison names history on purpose)."""
    for path in _text_files(EVIDENCE):
        if path.name == "README.md":
            continue  # prose; it points at the superseded package deliberately
        content = path.read_text(encoding="utf-8")
        assert "risk_level" not in content, f"retired CLI field in {path.relative_to(ROOT)}"
        stale = ["foodflow-10doc-release-v1", "04_index_backup"]
        if not path.name.startswith("retrieval_summary"):
            stale += ["bf3df0d4", "215 chunks"]
        for token in stale:
            assert token not in content, f"{token!r} in {path.relative_to(ROOT)}"


def test_markdown_links_in_the_deliverables_resolve() -> None:
    link = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    for path in sorted(DELIVERABLES.rglob("*.md")):
        for target in link.findall(path.read_text(encoding="utf-8")):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            assert resolved.exists(), f"{path.relative_to(ROOT)} links to missing {target}"


# --- the historical package -----------------------------------------------------------------------------------


def test_the_historical_package_is_intact_and_labelled() -> None:
    manifest = json.loads((HISTORICAL / "evidence" / "evidence_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "historical"
    assert manifest["project_release_id"] == "foodflow-10doc-release-v1"
    assert manifest["superseded_by"] == "deliverables/evidence/final_acceptance_manifest.json"
    scenario_ids = [entry["scenario_id"] for entry in manifest["entries"]]
    assert scenario_ids == [f"M{index:02d}" for index in range(10)]
    for entry in manifest["entries"]:
        for relative, digest in entry["sha256"].items():
            path = ROOT / relative
            assert path.is_file(), relative
            assert relative.startswith("deliverables/historical/stage5c_release_v1/evidence/")
            assert digest in _digests(path), f"historical evidence changed: {relative}"
    for name in ("manual_acceptance_report.md", "evidence/README.md"):
        text = (HISTORICAL / name).read_text(encoding="utf-8")
        assert "Historical" in text[:900] and "superseded" in text[:900], name
    readme = (HISTORICAL / "README.md").read_text(encoding="utf-8")
    assert "historical" in readme.lower() and "evidence/README.md" in readme


# --- reproduction on a machine with the local artifacts -----------------------------------------------------


@requires_local_artifacts(
    PRODUCTION_INDEX / "manifest.json",
    why="the committed scenario evidence is reproduced by the real pipeline over the production index",
)
def test_the_committed_scenarios_are_reproduced_by_the_production_pipeline() -> None:
    from customer_claims_rag.cli import answer_claim

    tool = _load_script("capture_final_acceptance")
    tool.prepare_environment()
    harness = tool.Harness()
    for scenario in tool.SCENARIOS:
        payload, _ = harness.run(scenario)
        tool.check_expectations(scenario, payload)
        committed = (EVIDENCE / "scenarios" / f"{scenario.scenario_id}.json").read_text(encoding="utf-8")
        assert answer_claim.serialize_answer_payload(payload) + "\n" == committed.replace("\r\n", "\n"), (
            f"{scenario.scenario_id}: committed evidence differs from what the production pipeline produces"
        )
    assert tool.release_identity() == json.loads(MANIFEST.read_text(encoding="utf-8"))["release"]


@requires_local_artifacts(
    STAGE2G,
    why="the committed retrieval summary is derived from the local final evaluation artifact",
)
def test_the_committed_retrieval_summary_is_what_the_local_artifact_derives() -> None:
    tool = _load_script("summarize_retrieval_evidence")
    current = json.loads(STAGE2G.read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / tool.HISTORICAL_ARTIFACT_RELATIVE).read_text(encoding="utf-8"))
    committed = json.loads((EVIDENCE / tool.SUMMARY_JSON).read_text(encoding="utf-8"))
    assert committed["derived_from"]["artifact_sha256"] == tool.sha256_raw(STAGE2G), (
        "the local evaluation artifact is not the one the committed summary was derived from"
    )
    rebuilt = tool.build_summary(
        current,
        baseline,
        overlay=tool.run_overlay(current),
        derived_from=committed["derived_from"],
        baseline_info=committed["comparison_with_previous_baseline"]["baseline"],
    )
    assert rebuilt == committed
    assert (EVIDENCE / tool.SUMMARY_MD).read_text(encoding="utf-8").replace("\r\n", "\n") == tool.render_markdown(committed)
