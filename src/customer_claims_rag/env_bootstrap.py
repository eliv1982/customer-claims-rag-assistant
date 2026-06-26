"""Centralized local environment bootstrap."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_loaded = False
_PROJECT_ROOT_ENV = "CUSTOMER_CLAIMS_PROJECT_ROOT"


def project_root() -> Path:
    """Return repository root (parent of ``src/``) or an explicit runtime override."""
    override = os.environ.get(_PROJECT_ROOT_ENV)
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def load_project_env(*, force: bool = False) -> None:
    """Load ``.env`` from project root once if the file exists.

    Existing process environment variables keep priority over ``.env`` values
    because ``override=False`` is used. Missing ``.env`` is not an error.
    """
    global _loaded
    if _loaded and not force:
        return

    env_path = project_root() / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)

    _loaded = True


def reset_project_env() -> None:
    """Reset bootstrap state for tests."""
    global _loaded
    _loaded = False
