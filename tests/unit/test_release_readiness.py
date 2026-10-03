"""Release readiness: what is established, what is not, and the command that fixes it.

A fresh clone has a canonical corpus and no index. The validator must say exactly that, and the
instruction it prints must be one the repository can actually execute.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from customer_claims_rag import env_bootstrap
from customer_claims_rag.cli import build_index as build_index_cli
from customer_claims_rag.cli import validate_release_posture as validate_cli
from customer_claims_rag.release.posture import (
    build_instruction,
    resolve_production_release_posture,
)
from customer_claims_rag.release.readiness import assess_release_readiness
from tests.release_posture_helpers import StagedRelease, stage_consistent_release

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _context(staged: StagedRelease):
    return resolve_production_release_posture(
        descriptor_path=staged.descriptor_path,
        project_root=staged.root,
    )


@pytest.fixture
def fresh_clone(tmp_path: Path) -> StagedRelease:
    """A project whose corpus is defined but whose index was never built."""
    staged = stage_consistent_release(tmp_path)
    # Chroma keeps its files open in-process on Windows, so the built index is not deleted: the
    # descriptor is pointed at a directory that does not exist, the same state for the validator.
    payload = json.loads(staged.descriptor_path.read_text(encoding="utf-8"))
    payload["targets"]["active"]["index_path"] = "data/04_index_production"
    staged.descriptor_path.write_text(json.dumps(payload), encoding="utf-8")
    assert not (tmp_path / "data" / "04_index_production").exists()
    return staged


# --- readiness states ---------------------------------------------------------------------------


def test_fresh_clone_has_a_corpus_but_no_index_and_a_command_to_build_it(fresh_clone: StagedRelease) -> None:
    readiness = assess_release_readiness(_context(fresh_clone))
    assert readiness.index_present is False
    assert readiness.index_matches_canonical_corpus == "not_checked"
    assert readiness.release_can_proceed is False
    assert readiness.diagnostics is None
    assert "has not been built here" in (readiness.problem or "")
    lines = readiness.format_lines()
    assert "index_present=no" in lines
    assert "release_can_proceed=no" in lines
    assert "canonical_corpus_id=test-corpus" in lines
    assert any(line.startswith("next_step=python -m customer_claims_rag.cli.build_index ") for line in lines)


def test_a_verified_release_can_proceed(tmp_path: Path) -> None:
    staged = stage_consistent_release(tmp_path)
    readiness = assess_release_readiness(_context(staged))
    assert readiness.index_present is True
    assert readiness.index_matches_canonical_corpus == "yes"
    assert readiness.release_can_proceed is True
    assert readiness.problem is None and readiness.next_step is None
    assert readiness.diagnostics is not None and readiness.diagnostics.integrity == "store_recomputed"
    assert "release_can_proceed=yes" in readiness.format_status_lines()


def test_an_index_that_does_not_match_cannot_proceed_and_says_how_to_rebuild(tmp_path: Path) -> None:
    staged = stage_consistent_release(tmp_path)
    manifest_path = staged.index_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["chunk_payload_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    readiness = assess_release_readiness(_context(staged))
    assert readiness.index_present is True
    assert readiness.index_matches_canonical_corpus == "no"
    assert readiness.release_can_proceed is False
    assert "chunk payload digest mismatch" in (readiness.problem or "")
    assert readiness.next_step == build_instruction(_context(staged).resolved_target)


def test_skipping_the_store_establishes_nothing_about_the_content(tmp_path: Path) -> None:
    staged = stage_consistent_release(tmp_path)
    readiness = assess_release_readiness(_context(staged), check_index_content=False)
    assert readiness.release_can_proceed is False
    assert readiness.index_matches_canonical_corpus == "not_checked"
    assert readiness.diagnostics is not None and readiness.diagnostics.integrity == "manifest_self_attestation_only"
    assert any(
        line.startswith("release_can_proceed=not_established") for line in readiness.format_status_lines()
    )


def test_deployment_without_the_corpus_sources_reports_them_as_not_present(tmp_path: Path) -> None:
    staged = stage_consistent_release(tmp_path)
    assert assess_release_readiness(_context(staged)).corpus_sources.startswith("not_present")


def test_the_committed_corpus_sources_are_reported_verified() -> None:
    context = resolve_production_release_posture(project_root=PROJECT_ROOT)
    readiness = assess_release_readiness(context, check_index_content=False)
    assert readiness.corpus_sources == "verified (10 source files match source_sha256)"
    assert readiness.corpus_id == "foodflow-10doc-corpus-v2"
    assert readiness.corpus_document_count == 10


def test_a_source_that_drifted_from_the_manifest_is_reported(tmp_path: Path) -> None:
    staged = stage_consistent_release(tmp_path)
    sources = tmp_path / "data" / "02_clean_markdown"
    sources.mkdir(parents=True)
    (sources / "01_service_overview.md").write_text("not what the manifest approved", encoding="utf-8")
    sources_state = assess_release_readiness(_context(staged)).corpus_sources
    assert sources_state.startswith("mismatch") and "01_service_overview.md" in sources_state


# --- the instruction is executable ---------------------------------------------------------------


def test_the_build_instruction_parses_with_the_real_build_cli_and_names_the_target() -> None:
    target = resolve_production_release_posture(project_root=PROJECT_ROOT).resolved_target
    words = shlex.split(build_instruction(target))
    assert words[:3] == ["python", "-m", "customer_claims_rag.cli.build_index"]
    args = build_index_cli.build_parser().parse_args(words[3:])
    assert args.corpus_manifest == Path(target.corpus_manifest_relative)
    assert args.index_dir == Path(target.index_path_relative)
    assert args.collection == target.collection_name
    assert args.embedding_model == target.embedding_model
    assert args.rebuild is True
    assert (PROJECT_ROOT / args.corpus_manifest).is_file()


def test_the_instruction_refers_only_to_files_and_flags_in_the_repository(fresh_clone: StagedRelease) -> None:
    command = build_instruction(_context(fresh_clone).resolved_target)
    assert "--input-dir" not in command and "archive" not in command and "backup" not in command
    assert "OPENAI_API_KEY" not in command  # the key is never part of an instruction


# --- the CLI --------------------------------------------------------------------------------------


def _run_cli(monkeypatch: pytest.MonkeyPatch, staged: StagedRelease, *extra: str) -> int:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: staged.root)
    return validate_cli.main(["--descriptor", str(staged.descriptor_path), *extra])


def test_cli_on_a_fresh_clone_fails_clearly_with_the_build_command(
    fresh_clone: StagedRelease, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(monkeypatch, fresh_clone) == 1
    captured = capsys.readouterr()
    assert "index_present=no" in captured.out
    assert "release_can_proceed=no" in captured.out
    assert "release posture validation failed" in captured.err
    assert "next step: python -m customer_claims_rag.cli.build_index --corpus-manifest" in captured.err
    assert "Traceback" not in captured.err


def test_cli_on_a_verified_release_succeeds_and_reports_the_recomputed_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    staged = stage_consistent_release(tmp_path)
    assert _run_cli(monkeypatch, staged) == 0
    out = capsys.readouterr().out
    assert "index_integrity=store_recomputed" in out
    assert "index_matches_canonical_corpus=yes" in out
    assert "release_can_proceed=yes" in out
    assert f"corpus_fingerprint={staged.corpus_fingerprint}" in out


def test_cli_skip_vector_store_is_a_static_check_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    staged = stage_consistent_release(tmp_path)
    # Static validation passes, but readiness is not established, so the gate is not a success
    # (the full contract is pinned in test_release_gate_exit_status.py).
    assert _run_cli(monkeypatch, staged, "--skip-vector-store") == validate_cli.EXIT_READINESS_NOT_ESTABLISHED != 0
    out = capsys.readouterr().out
    assert "index_integrity=manifest_self_attestation_only" in out
    assert "release_can_proceed=not_established" in out


def test_cli_on_a_mismatched_index_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    staged = stage_consistent_release(tmp_path)
    manifest_path = staged.index_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["corpus_fingerprint"] = "f" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _run_cli(monkeypatch, staged) == 1
    captured = capsys.readouterr()
    assert "index_matches_canonical_corpus=no" in captured.out
    assert "corpus fingerprint mismatch" in captured.err
