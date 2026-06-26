"""Tests for committed manual acceptance evidence package."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELIVERABLES = PROJECT_ROOT / "deliverables"
MANIFEST_PATH = DELIVERABLES / "evidence" / "evidence_manifest.json"
REPORT_PATH = DELIVERABLES / "manual_acceptance_report.md"
EVIDENCE_ROOT = DELIVERABLES / "evidence"

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"OPENAI_API_KEY\s*=\s*\S+"),
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"/Users/[^/\s]+/"),
)

REQUIRED_SCENARIOS = tuple(f"M{index:02d}" for index in range(10))


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.is_file(), "evidence manifest is missing"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manual_acceptance_report_exists() -> None:
    assert REPORT_PATH.is_file()


def test_evidence_manifest_schema(manifest: dict) -> None:
    required_keys = {
        "schema_version",
        "project_release_id",
        "commit_tested",
        "execution_date",
        "entries",
    }
    assert required_keys.issubset(manifest.keys())
    assert manifest["project_release_id"] == "foodflow-10doc-release-v1"
    assert manifest["schema_version"] == "1.0.0"
    assert isinstance(manifest["entries"], list)
    assert manifest["entries"], "manifest must contain scenario entries"


def test_manifest_scenario_ids_complete_and_unique(manifest: dict) -> None:
    scenario_ids = [entry["scenario_id"] for entry in manifest["entries"]]
    assert scenario_ids == sorted(scenario_ids)
    assert len(scenario_ids) == len(set(scenario_ids))
    assert set(REQUIRED_SCENARIOS).issubset(set(scenario_ids))


@pytest.mark.parametrize("entry_key", ["scenario_id", "environment_category", "evidence_files", "sha256", "external_openai_request", "acceptance_result"])
def test_manifest_entry_required_fields(manifest: dict, entry_key: str) -> None:
    for entry in manifest["entries"]:
        assert entry_key in entry


def test_manifest_referenced_files_exist(manifest: dict) -> None:
    for entry in manifest["entries"]:
        for relative_path in entry["evidence_files"]:
            file_path = PROJECT_ROOT / relative_path
            assert file_path.is_file(), f"missing evidence file: {relative_path}"


def test_manifest_checksums_match_files(manifest: dict) -> None:
    import hashlib

    for entry in manifest["entries"]:
        for relative_path, expected_digest in entry["sha256"].items():
            file_path = PROJECT_ROOT / relative_path
            actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
            assert actual == expected_digest, f"checksum mismatch for {relative_path}"


def test_committed_evidence_contains_no_obvious_secrets() -> None:
    for path in EVIDENCE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".png":
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            assert pattern.search(content) is None, f"secret pattern in {path.relative_to(PROJECT_ROOT)}"


def test_committed_evidence_contains_no_local_absolute_paths() -> None:
    for path in EVIDENCE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".png":
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in ABSOLUTE_PATH_PATTERNS:
            assert pattern.search(content) is None, f"absolute path in {path.relative_to(PROJECT_ROOT)}"
