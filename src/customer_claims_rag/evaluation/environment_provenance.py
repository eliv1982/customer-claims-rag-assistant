"""Controlled environment validation for authoritative evaluation artifacts.

Authoritative artifacts must come from an isolated virtual environment on a
supported Python, started from the repository root. Where the environment lives
is irrelevant: ``<repo>/.venv`` and a virtualenv created elsewhere are equally valid.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Keep in sync with ``requires-python`` in pyproject.toml (checked by a test).
MINIMUM_PYTHON = (3, 12)


class EnvironmentProvenanceError(RuntimeError):
    """Raised when evaluation artifacts cannot be generated in the current environment."""


def is_isolated_virtualenv() -> bool:
    """Return True when the active interpreter runs inside a virtual environment."""
    return Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve()


def is_supported_python() -> bool:
    return tuple(sys.version_info[:2]) >= MINIMUM_PYTHON


def validate_project_venv(project_root: Path) -> None:
    """Refuse authoritative artifact generation outside an isolated, supported environment."""
    root = project_root.resolve()
    if not is_isolated_virtualenv():
        raise EnvironmentProvenanceError(
            "active Python interpreter is not an isolated virtual environment; "
            "refusing authoritative evaluation artifact generation"
        )
    if not is_supported_python():
        required = ".".join(str(part) for part in MINIMUM_PYTHON)
        raise EnvironmentProvenanceError(
            f"Python {required}+ is required; refusing authoritative evaluation artifact generation"
        )
    if Path.cwd().resolve() != root:
        raise EnvironmentProvenanceError(
            "working directory is not the repository root; "
            "refusing authoritative evaluation artifact generation"
        )


def capture_environment_provenance(project_root: Path) -> dict[str, object]:
    """Capture repo-safe environment metadata after environment validation."""
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
    "MINIMUM_PYTHON",
    "EnvironmentProvenanceError",
    "capture_environment_provenance",
    "is_isolated_virtualenv",
    "is_supported_python",
    "validate_project_venv",
]
