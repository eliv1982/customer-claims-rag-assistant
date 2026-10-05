"""The README and the documentation say what the committed configuration and evidence say.

A portfolio README is read first and trusted most. Its numbers, identities and claims are therefore
tied to the files they describe, so that a changed corpus, a regenerated evidence package or a new
metric cannot leave the front page stating something else.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from customer_claims_rag.ingestion.canonical_corpus import (
    DEFAULT_CORPUS_MANIFEST_RELATIVE,
    load_canonical_corpus_manifest,
)
from customer_claims_rag.release.posture import resolve_production_release_posture

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
EVIDENCE = ROOT / "deliverables" / "evidence"

REQUIRED_HEADINGS = (
    "What problem it solves",
    "Architecture and request flow",
    "Deterministic safety boundary",
    "Retrieval design",
    "Reproducible canonical corpus and index",
    "Evaluation results",
    "Running locally",
    "Tests and CI",
    "Docker and release workflow",
    "Known limitations",
    "Repository structure",
    "Development approach",
    "License",
)
# Names of coding assistants and tooling provenance have no place in the project's own description.
FORBIDDEN_IN_README = re.compile(
    r"claude|codex|chatgpt|copilot|cursor|gemini|anthropic|generated with|ai-generated|ai-assisted|co-authored",
    re.IGNORECASE,
)
STALE_CURRENT_STATE_CLAIMS = (
    "risk_level",
    "215 chunks",
    "215 чанк",
    "bf3df0d4",
    "04_index_backup",
    "foodflow-10doc-release-v1",
    "Functional MVP",
    "Функциональный MVP",
    "not been re-measured",
)


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def corpus():
    return load_canonical_corpus_manifest(ROOT / DEFAULT_CORPUS_MANIFEST_RELATIVE)


@pytest.fixture(scope="module")
def retrieval() -> dict:
    return json.loads((EVIDENCE / "retrieval_summary.json").read_text(encoding="utf-8"))


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.DOTALL | re.MULTILINE)
    assert match, f"README has no '## {heading}' section"
    return match.group(1)


def test_readme_has_every_planned_section_in_order(readme) -> None:
    headings = re.findall(r"^## (.+)$", readme, re.MULTILINE)
    positions = [headings.index(name) for name in REQUIRED_HEADINGS]
    assert positions == sorted(positions)


def test_readme_states_the_committed_release_identity(readme, corpus) -> None:
    descriptor = resolve_production_release_posture(project_root=ROOT).descriptor
    assert descriptor.release_posture_id in readme
    assert f"{corpus.expected.document_count} documents, {corpus.expected.chunk_count} chunks" in readme
    assert corpus.expected.corpus_fingerprint[:8] in readme
    assert corpus.expected.chunk_payload_digest[:8] in readme
    assert "data/04_index_production" in readme
    assert "configs/corpus/foodflow_production_v1.json" in readme


def test_readme_metrics_are_the_ones_in_the_committed_retrieval_summary(readme, retrieval) -> None:
    metrics = retrieval["metrics"]
    results = _section(readme, "Evaluation results")
    for key, label in (("hit_at_4", "Hit@4"), ("primary_hit_at_4", "Primary hit@4"), ("primary_hit_at_1", "Primary hit@1")):
        item = metrics[key]
        assert re.search(rf"\| {re.escape(label)}[^|]*\| {item['value']:.3f} \| {item['hits']}/{item['cases']} \|", results), label
    assert f"| MRR | {metrics['mrr']:.3f} |" in results
    reach = retrieval["pool_reachability"]["pool_at_24"]["primary"]
    assert f"| {reach} |" in results and reach in readme
    comparison = retrieval["comparison_with_previous_baseline"]
    previous = comparison["metric_deltas"]["primary_hit_at_4"]
    assert f"{previous['baseline']:.3f} to {previous['current']:.3f}" in results
    assert comparison["regressions"] == []
    assert "no per-case regression" in results


def test_readme_names_the_limitations_the_evidence_shows(readme, retrieval) -> None:
    limitations = _section(readme, "Known limitations")
    for case_id in ("T004", "T040", "T044", "T047", "T053"):
        assert case_id in limitations, case_id
    assert retrieval["limitations"]["primary_outside_pool_at_24"] == ["T004"]
    # the one rule gap the evidence still shows is named; the two closed in Stage 2I are described as closed
    assert retrieval["limitations"]["dataset_high_or_critical_without_matching_floor"] == ["T053"]
    assert retrieval["limitations"]["deterministic_floor_two_or_more_levels_above_dataset_label"] == []
    rows = {row["case_id"]: row["deterministic"] for row in retrieval["cases"]}
    assert rows["T039"]["risk_floor"] == "high" and "T039" in limitations
    assert rows["T026"]["risk_floor"] == "medium" and "T026" in limitations
    for needle in ("OpenAI credential", "cannot be proven offline", "single-shot", "Retrieval only"):
        assert needle.lower() in limitations.lower(), needle


def test_readme_positions_the_project_as_staff_assist_not_a_helpdesk(readme) -> None:
    lead = readme[: readme.index("## What problem it solves")]
    assert "staff-assist" in lead.lower()
    scope = _section(readme, "What problem it solves")
    assert "not a ticketing or helpdesk platform" in scope
    for absent in ("no database", "no authentication", "no deployment"):
        assert absent in scope
    assert "fictional" in lead


def test_readme_describes_the_deterministic_boundary_and_the_credential_rules(readme) -> None:
    safety = _section(readme, "Deterministic safety boundary")
    for needle in ("fixed customer text", "priority handoff", "no_signal", "unsupported_language", "answer_provenance", "retrieved_materials"):
        assert needle in safety, needle
    running = _section(readme, "Running locally")
    assert "process environment only" in running and "OPENAI_API_KEY" in running


def test_readme_commands_exist(readme) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    scripts = project["scripts"]
    for command in ("answer-claim", "validate-release-posture"):
        assert command in scripts and command in readme
    assert project["requires-python"] == ">=3.12" and "Python 3.12 or newer" in readme
    for path in re.findall(r"scripts/[a-z_]+\.py", readme):
        assert (ROOT / path).is_file(), path
    assert "python scripts/release_compose.py" in readme


def test_readme_carries_no_stale_current_state_claim(readme) -> None:
    for claim in STALE_CURRENT_STATE_CLAIMS:
        assert claim not in readme, claim


def test_readme_names_no_coding_assistant_or_tooling_provenance(readme) -> None:
    assert FORBIDDEN_IN_README.search(readme) is None, FORBIDDEN_IN_README.search(readme)
    approach = _section(readme, "Development approach")
    assert not re.search(r"\bAI\b|\bassistant\b|generated|prompt", approach, re.IGNORECASE)


def test_readme_names_no_absolute_workstation_path(readme) -> None:
    assert not re.search(r"[A-Za-z]:\\Users\\|/Users/[^/\s]+/|/home/[^/\s]+/", readme)


def test_relative_markdown_links_resolve() -> None:
    link = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    for path in [README, *sorted(DOCS.glob("*.md"))]:
        for target in link.findall(path.read_text(encoding="utf-8")):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            assert resolved.exists(), f"{path.relative_to(ROOT)} links to missing {target}"


def test_license_is_mit_for_the_repository_owner(readme) -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License\n")
    assert "Copyright (c) 2026 eliv1982" in text
    assert "Permission is hereby granted, free of charge" in text
    assert "[LICENSE](LICENSE)" in readme and "MIT" in _section(readme, "License")


def test_package_description_is_not_the_early_stage_one() -> None:
    description = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["description"]
    assert "ingestion layer" not in description.lower()
    assert "RAG" in description


@pytest.mark.parametrize(
    "name",
    [
        "00_project_scope.md",
        "01_data_inventory.md",
        "02_data_cleaning_report.md",
        "03_chunking_strategy.md",
        "04_metadata_schema.md",
        "05_business_rules_registry.md",
    ],
)
def test_data_preparation_records_are_labelled_as_such(name: str) -> None:
    head = (DOCS / name).read_text(encoding="utf-8")[:1500]
    assert "Статус документа:" in head
    assert "06_release_posture.md" in head


def test_the_release_posture_doc_no_longer_claims_retrieval_is_unmeasured() -> None:
    text = (DOCS / "06_release_posture.md").read_text(encoding="utf-8")
    assert "not** been re-measured" not in text and "has **not** been re-measured" not in text
    assert "deliverables/evidence/retrieval_summary.md" in text


def test_the_evidence_map_classifies_the_historical_reports() -> None:
    text = (ROOT / "deliverables" / "README.md").read_text(encoding="utf-8")
    for name in sorted((ROOT / "tests").glob("[0-1][0-9]_*.md")):
        if name.name[:2] in {"01", "02"}:
            continue  # the frozen evaluation inputs are listed under "Current"
        assert name.name in text, f"{name.name} is not classified in deliverables/README.md"
