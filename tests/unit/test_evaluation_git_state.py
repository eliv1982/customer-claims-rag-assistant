"""Git state helper tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from customer_claims_rag.evaluation.git_state import read_git_state


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_read_git_state_reports_commit_and_dirty_flag(tmp_path: Path) -> None:
    # A throwaway repository: the test must not depend on how the project was obtained
    # (git clone, source archive, Docker build context).
    if shutil.which("git") is None:
        pytest.skip("git executable not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    tracked = repo / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "initial")
    head = _git(repo, "rev-parse", "HEAD")

    clean = read_git_state(repo)
    assert clean.commit == head
    assert clean.dirty is False
    assert clean.status_summary == "clean"

    tracked.write_text("two\n", encoding="utf-8")
    dirty = read_git_state(repo)
    assert dirty.commit == head
    assert dirty.dirty is True
    assert dirty.status_summary == "1 changed path(s)"


def test_read_git_state_unknown_outside_a_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable not available")
    # never discover a repository above the temp dir (e.g. a TEMP inside a dotfiles repo)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    state = read_git_state(tmp_path)
    assert state.commit is None
    assert state.dirty is None


def test_read_git_state_unknown_when_git_unavailable(tmp_path, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr("customer_claims_rag.evaluation.git_state.subprocess.run", fail)
    state = read_git_state(tmp_path)
    assert state.commit is None
    assert state.dirty is None
