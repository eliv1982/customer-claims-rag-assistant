"""Git working-tree metadata for evaluation runs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitState:
    commit: str | None
    dirty: bool | None
    status_summary: str | None


def read_git_state(project_root: Path | None = None) -> GitState:
    """Read commit hash and dirty flag without exposing absolute paths."""
    root = project_root.resolve() if project_root is not None else None
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root) if root is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return GitState(commit=None, dirty=None, status_summary=None)

    commit = commit_result.stdout.strip() or None
    status_lines = [line for line in status_result.stdout.splitlines() if line.strip()]
    dirty = bool(status_lines)
    if not status_lines:
        summary = "clean"
    else:
        summary = f"{len(status_lines)} changed path(s)"
    return GitState(commit=commit, dirty=dirty, status_summary=summary)
