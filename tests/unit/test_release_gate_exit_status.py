"""``validate-release-posture`` is a release gate: exit status 0 only when readiness is established.

Static validation (descriptor, manifests, frozen config) and release readiness are different facts.
A stale index can carry a manifest that passes every static check, so a run that never read the
store must not look like a passing gate. The contract pinned here:

======  ===================  ==========================================================
exit    release_can_proceed  meaning
======  ===================  ==========================================================
0       yes                  the index content was verified against the canonical corpus
1       no                   missing / invalid / stale index, source mismatch, bad descriptor
3       not_established      static checks passed, index content not verified
======  ===================  ==========================================================

(2 stays argparse's usage error.) ``--skip-vector-store`` can therefore only ever exit 3 or 1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import chromadb
import pytest

from customer_claims_rag import env_bootstrap
from customer_claims_rag.cli import validate_release_posture as validate_cli
from customer_claims_rag.release.posture import resolve_production_release_posture
from customer_claims_rag.release.readiness import (
    RELEASE_NO,
    RELEASE_NOT_ESTABLISHED,
    RELEASE_YES,
    assess_release_readiness,
)
from tests.release_posture_helpers import (
    StagedRelease,
    make_test_chunks,
    stage_consistent_release,
    write_consistent_index,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ("01_service_overview", "02_delivery_rules")

READY = validate_cli.EXIT_RELEASE_READY
BLOCKED = validate_cli.EXIT_RELEASE_BLOCKED
NOT_ESTABLISHED = validate_cli.EXIT_READINESS_NOT_ESTABLISHED


# --- staging helpers: a genuine release, damaged in exactly one way ---------------------------------


def _release(tmp_path: Path) -> StagedRelease:
    return stage_consistent_release(tmp_path, document_ids=DOCS, chunks_per_document=2)


def _point_descriptor_at(staged: StagedRelease, index_relative: str) -> None:
    payload = json.loads(staged.descriptor_path.read_text(encoding="utf-8"))
    payload["targets"]["active"]["index_path"] = index_relative
    staged.descriptor_path.write_text(json.dumps(payload), encoding="utf-8")


def _rewrite_manifest(index_dir: Path, **fields: object) -> None:
    path = index_dir / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(fields)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _missing_index(staged: StagedRelease) -> None:
    # Chroma keeps its files open in-process on Windows, so nothing is deleted: the descriptor
    # names a directory that does not exist, which is the state the validator sees on a fresh clone.
    _point_descriptor_at(staged, "data/04_index_production")


def _manifest_digest_wrong(staged: StagedRelease) -> None:
    _rewrite_manifest(staged.index_dir, chunk_payload_digest="0" * 64)


def _manifest_fingerprint_wrong(staged: StagedRelease) -> None:
    _rewrite_manifest(staged.index_dir, corpus_fingerprint="f" * 64)


def _honest_index_of_another_corpus(staged: StagedRelease) -> None:
    write_consistent_index(staged.root / "other-index", make_test_chunks(("04_refund_policy",), 4))
    _point_descriptor_at(staged, "other-index")


def _stored_text_edited(staged: StagedRelease) -> None:
    """The manifest still says the approved thing; only the stored records changed."""
    collection = chromadb.PersistentClient(path=str(staged.index_dir)).get_collection("customer_claims")
    chunk_id = staged.chunks[0].chunk_id
    vector = collection.get(ids=[chunk_id], include=["embeddings"])["embeddings"][0]
    collection.update(ids=[chunk_id], documents=["silently edited text"], embeddings=[list(vector)])


def _stale_index_with_forged_manifest(staged: StagedRelease) -> None:
    """Store built from other content, manifest edited to claim the approved corpus."""
    stale = [chunk.model_copy(update={"content": chunk.content + " (old wording)"}) for chunk in staged.chunks]
    write_consistent_index(staged.root / "stale-index", stale)
    _rewrite_manifest(
        staged.root / "stale-index",
        corpus_fingerprint=staged.corpus_fingerprint,
        chunk_payload_digest=staged.chunk_payload_digest,
    )
    _point_descriptor_at(staged, "stale-index")


def _source_directory_missing(staged: StagedRelease) -> None:
    source_dir = staged.root / "data" / "02_clean_markdown"
    for source in source_dir.iterdir():
        source.unlink()
    source_dir.rmdir()


def _included_source_missing(staged: StagedRelease) -> None:
    (staged.root / "data" / "02_clean_markdown" / f"{DOCS[0]}.md").unlink()


def _included_source_changed(staged: StagedRelease) -> None:
    source = staged.root / "data" / "02_clean_markdown" / f"{DOCS[0]}.md"
    source.write_text("mutated after manifest approval", encoding="utf-8")


def _canonical_selection_changed(staged: StagedRelease) -> None:
    source = staged.root / "data" / "02_clean_markdown" / "99_unexpected.md"
    source.write_text("undeclared source", encoding="utf-8")


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    staged: StagedRelease,
    *extra: str,
) -> tuple[int, str, str]:
    monkeypatch.setattr(env_bootstrap, "project_root", lambda: staged.root)
    code = validate_cli.main(["--descriptor", str(staged.descriptor_path), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# state name -> (how to damage the release, exit of the full run, exit of --skip-vector-store)
# The static check reads manifests only, so damage that lives in the store is invisible to it.
DAMAGED_STATES: dict[str, tuple[Callable[[StagedRelease], None], int, int]] = {
    "source-directory-missing": (_source_directory_missing, BLOCKED, BLOCKED),
    "included-source-missing": (_included_source_missing, BLOCKED, BLOCKED),
    "included-source-changed": (_included_source_changed, BLOCKED, BLOCKED),
    "canonical-selection-changed": (_canonical_selection_changed, BLOCKED, BLOCKED),
    "index-missing": (_missing_index, BLOCKED, BLOCKED),
    "manifest-payload-digest-wrong": (_manifest_digest_wrong, BLOCKED, BLOCKED),
    "manifest-fingerprint-wrong": (_manifest_fingerprint_wrong, BLOCKED, BLOCKED),
    "honest-index-of-another-corpus": (_honest_index_of_another_corpus, BLOCKED, BLOCKED),
    "stored-text-edited": (_stored_text_edited, BLOCKED, NOT_ESTABLISHED),
    "stale-index-forged-manifest": (_stale_index_with_forged_manifest, BLOCKED, NOT_ESTABLISHED),
}


# --- the contract itself ------------------------------------------------------------------------


def test_exit_codes_are_distinct_and_leave_argparses_usage_code_alone() -> None:
    assert (READY, BLOCKED, NOT_ESTABLISHED) == (0, 1, 3)
    assert 2 not in {READY, BLOCKED, NOT_ESTABLISHED}


def test_readiness_is_three_valued_and_only_a_verified_index_is_yes(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    context = resolve_production_release_posture(descriptor_path=staged.descriptor_path, project_root=staged.root)

    verified = assess_release_readiness(context)
    assert (verified.release_status, verified.release_can_proceed) == (RELEASE_YES, True)

    unverified = assess_release_readiness(context, check_index_content=False)
    assert unverified.release_status == RELEASE_NOT_ESTABLISHED
    assert unverified.release_can_proceed is False
    assert unverified.static_validation_passed is True  # passing static checks is not readiness

    _manifest_digest_wrong(staged)
    broken = assess_release_readiness(context)
    assert (broken.release_status, broken.release_can_proceed) == (RELEASE_NO, False)
    assert broken.static_validation_passed is False


# --- full validation ---------------------------------------------------------------------------


def test_full_validation_of_a_valid_index_is_the_only_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(monkeypatch, capsys, _release(tmp_path))
    assert code == READY == 0
    assert "release_can_proceed=yes" in out
    assert "index_matches_canonical_corpus=yes" in out
    assert "static_validation=passed" in out
    assert "index_integrity=store_recomputed" in out
    assert err == ""


@pytest.mark.parametrize("state", list(DAMAGED_STATES))
def test_a_missing_invalid_or_stale_index_blocks_the_release(
    state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    damage, expected_full, _ = DAMAGED_STATES[state]
    staged = _release(tmp_path)
    damage(staged)
    code, out, err = _run(monkeypatch, capsys, staged)
    assert code == expected_full == BLOCKED != 0
    assert "release_can_proceed=no" in out
    assert "release_can_proceed=yes" not in out
    assert "release posture validation failed" in err


# --- --skip-vector-store: static checks only, never a passing gate ------------------------------------


def test_static_validation_can_pass_while_the_gate_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _run(monkeypatch, capsys, _release(tmp_path), "--skip-vector-store")
    assert code == NOT_ESTABLISHED != 0
    # the output keeps the two facts apart
    assert "static_validation=passed" in out
    assert "index_matches_canonical_corpus=not_checked" in out
    assert "release_can_proceed=not_established" in out
    assert "release_can_proceed=yes" not in out
    assert "index_integrity=manifest_self_attestation_only" in out
    assert "release readiness not established" in err
    assert "without --skip-vector-store" in err


def test_the_static_check_passing_on_a_stale_index_is_exactly_why_skip_is_not_a_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    staged = _release(tmp_path)
    _stale_index_with_forged_manifest(staged)

    skipped_code, skipped_out, _ = _run(monkeypatch, capsys, staged, "--skip-vector-store")
    assert "static_validation=passed" in skipped_out  # the forged manifest satisfies every static check
    assert skipped_code == NOT_ESTABLISHED  # ... and the command still refuses to call that a pass

    full_code, full_out, full_err = _run(monkeypatch, capsys, staged)
    assert full_code == BLOCKED
    assert "index_matches_canonical_corpus=no" in full_out
    assert "stored chunk payload digest mismatch" in full_err


@pytest.mark.parametrize("state", ["valid", *DAMAGED_STATES])
def test_automation_can_never_read_skip_vector_store_as_a_successful_gate(
    state: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    staged = _release(tmp_path)
    expected = NOT_ESTABLISHED
    if state != "valid":
        damage, _, expected = DAMAGED_STATES[state]
        damage(staged)
    code, out, _ = _run(monkeypatch, capsys, staged, "--skip-vector-store")
    assert code == expected
    assert code != READY
    assert "release_can_proceed=yes" not in out


def test_skip_vector_store_on_a_missing_index_is_blocked_not_merely_unestablished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    staged = _release(tmp_path)
    _missing_index(staged)
    code, out, err = _run(monkeypatch, capsys, staged, "--skip-vector-store")
    assert code == BLOCKED
    assert "index_present=no" in out and "release_can_proceed=no" in out
    assert "next step: python -m customer_claims_rag.cli.build_index" in err


# --- the process exit status, as automation sees it --------------------------------------------------


def _process_exit(staged: StagedRelease, *extra: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "CUSTOMER_CLAIMS_PROJECT_ROOT": str(staged.root),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "customer_claims_rag.cli.validate_release_posture",
            "--descriptor",
            str(staged.descriptor_path),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=staged.root,
        timeout=180,
    )


def test_the_real_process_exit_status_follows_the_contract(tmp_path: Path) -> None:
    staged = _release(tmp_path)
    full = _process_exit(staged)
    skipped = _process_exit(staged, "--skip-vector-store")
    assert (full.returncode, skipped.returncode) == (0, 3), (full.stderr, skipped.stderr)
    assert "release_can_proceed=yes" in full.stdout
    assert "release_can_proceed=not_established" in skipped.stdout

    _manifest_digest_wrong(staged)
    blocked = _process_exit(staged)
    assert blocked.returncode == 1 and "release_can_proceed=no" in blocked.stdout


# --- help text states the same contract -------------------------------------------------------------


def test_help_states_the_exit_status_contract() -> None:
    help_text = " ".join(validate_cli.build_parser().format_help().split())
    assert "Exit status 0 only when release readiness is established" in help_text
    for fragment in (
        f"{READY} release can proceed",
        "(release_can_proceed=yes)",
        f"{BLOCKED} release cannot proceed",
        "(release_can_proceed=no)",
        f"{NOT_ESTABLISHED} readiness not established",
        "(release_can_proceed=not_established)",
        "2 command-line usage error",
    ):
        assert fragment in help_text, fragment
    assert f"exits {NOT_ESTABLISHED} (release_can_proceed=not_established)" in help_text
    assert "never a successful release gate" in help_text
