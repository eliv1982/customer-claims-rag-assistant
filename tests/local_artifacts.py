"""Explicit contract for tests that verify gitignored local artifacts.

Most of ``data/04_index*`` is produced on a maintainer machine (live OpenAI embeddings) and
is not part of a clone. Tests whose *subject* is such an artifact carry the
``local_artifact`` marker:

* artifact absent   -> the test skips, with a reason that names the missing path;
* artifact present  -> the test runs, and anything wrong with it (malformed, stale,
  incompatible with the frozen expectations) is a FAILURE. An artifact that exists but is
  unusable is stale maintainer state and must be seen, never skipped.

Select or exclude them with ``-m local_artifact`` / ``-m "not local_artifact"``.

Not for tests that merely need some index or evaluation data: those build it themselves
(``tmp_path``, fakes) or read a committed fixture (``tests/frozen_fixtures.py``). Tracked
artifacts (``data/05_evaluation/*.json`` that are committed) are always present in a clone,
so tests read them directly and fail if they are missing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_T = TypeVar("_T")


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def local_artifact_marks(*paths: Path, why: str) -> list[pytest.MarkDecorator]:
    """Marks for ``pytestmark = ...``: the marker, plus a skip while a path is missing."""
    marks = [pytest.mark.local_artifact]
    missing = [_display(path) for path in paths if not path.exists()]
    if missing:
        marks.append(
            pytest.mark.skip(reason=f"local artifact missing ({', '.join(missing)}): {why}")
        )
    return marks


def requires_local_artifacts(*paths: Path, why: str) -> Callable[[_T], _T]:
    """Decorator form of ``local_artifact_marks`` for a test function or class."""
    marks = local_artifact_marks(*paths, why=why)

    def decorate(obj: _T) -> _T:
        for mark in marks:
            obj = mark(obj)
        return obj

    return decorate
