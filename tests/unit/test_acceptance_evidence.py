"""Tests for committed manual acceptance evidence package."""

from __future__ import annotations

import hashlib
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

# Digest rule: text evidence is hashed after line-ending normalization, binary
# evidence (PNG) byte for byte. Without this a Windows checkout (CRLF) and a Linux
# checkout (LF) of the same commit hash differently. The digests recorded in
# evidence_manifest.json v1.0.0 were taken from a Windows checkout, i.e. they are
# the CRLF form; a regenerated manifest may record the LF form instead (the
# repository's canonical form, see .gitattributes). Either form is accepted.
TEXT_EVIDENCE_SUFFIXES = frozenset({".json", ".md", ".txt"})


def evidence_digests(path: Path) -> set[str]:
    """SHA-256 digests under which ``path`` is accepted as matching the manifest."""
    data = path.read_bytes()
    if path.suffix.lower() not in TEXT_EVIDENCE_SUFFIXES:
        return {hashlib.sha256(data).hexdigest()}
    lf = data.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {hashlib.sha256(lf).hexdigest(), hashlib.sha256(crlf).hexdigest()}


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
    for entry in manifest["entries"]:
        for relative_path, expected_digest in entry["sha256"].items():
            file_path = PROJECT_ROOT / relative_path
            assert expected_digest in evidence_digests(file_path), (
                f"checksum mismatch for {relative_path}"
            )


def test_evidence_digest_is_line_ending_independent(tmp_path: Path) -> None:
    lf_file = tmp_path / "evidence.json"
    crlf_file = tmp_path / "evidence_crlf.json"
    lf_file.write_bytes(b'{\n  "a": 1\n}\n')
    crlf_file.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert evidence_digests(lf_file) == evidence_digests(crlf_file)
    changed = tmp_path / "changed.json"
    changed.write_bytes(b'{\n  "a": 2\n}\n')
    assert evidence_digests(lf_file).isdisjoint(evidence_digests(changed))


def test_binary_evidence_digest_is_byte_exact(tmp_path: Path) -> None:
    png = tmp_path / "shot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n\r\nbody")
    assert evidence_digests(png) == {hashlib.sha256(png.read_bytes()).hexdigest()}


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
