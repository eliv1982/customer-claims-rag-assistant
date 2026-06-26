"""Controlled environment validation for authoritative evaluation artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class EnvironmentProvenanceError(RuntimeError):
    """Raised when evaluation artifacts cannot be generated in the current environment."""


def project_venv_prefix(project_root: Path) -> Path:
    return project_root.resolve() / ".venv"


def is_project_venv_interpreter(project_root: Path) -> bool:
    """Return True when the active interpreter belongs to the project virtualenv."""
    root = project_root.resolve()
    venv = project_venv_prefix(root)
    if not venv.is_dir():
        return False
    executable = Path(sys.executable).resolve()
    prefix = Path(sys.prefix).resolve()
    if prefix != venv.resolve():
        return False
    for candidate in (venv / "Scripts", venv / "bin"):
        if candidate.is_dir():
            try:
                executable.relative_to(candidate.resolve())
                return True
            except ValueError:
                continue
    return False


def validate_project_venv(project_root: Path) -> None:
    """Refuse authoritative artifact generation outside the project virtualenv."""
    root = project_root.resolve()
    venv = project_venv_prefix(root)
    if not venv.is_dir():
        raise EnvironmentProvenanceError(
            "project .venv is missing; refusing authoritative evaluation artifact generation"
        )
    if not is_project_venv_interpreter(root):
        raise EnvironmentProvenanceError(
            "active Python interpreter is not the project .venv; "
            "refusing authoritative evaluation artifact generation"
        )
    if Path.cwd().resolve() != root:
        raise EnvironmentProvenanceError(
            "working directory is not the repository root; "
            "refusing authoritative evaluation artifact generation"
        )


def capture_environment_provenance(project_root: Path) -> dict[str, object]:
    """Capture repo-safe environment metadata after venv validation."""
    validate_project_venv(project_root)
    pip_version = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    pip_line = (pip_version.stdout or pip_version.stderr or "").strip()
    pip_version_safe = pip_line.split(" from ", 1)[0].strip() if pip_line else ""
    summary = (pip_check.stdout or "").strip() or (pip_check.stderr or "").strip()
    if not summary:
        summary = "ok" if pip_check.returncode == 0 else "dependency check failed"
    return {
        "kind": "project_venv",
        "python_version": sys.version.split()[0],
        "pip_version": pip_version_safe,
        "pip_check_exit_code": pip_check.returncode,
        "dependency_check_summary": summary,
    }


__all__ = [
    "EnvironmentProvenanceError",
    "capture_environment_provenance",
    "is_project_venv_interpreter",
    "project_venv_prefix",
    "validate_project_venv",
]
