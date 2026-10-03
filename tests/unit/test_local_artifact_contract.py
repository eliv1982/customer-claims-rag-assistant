"""``local_artifact``: absent skips with the path, present-but-stale fails.

See ``tests/local_artifacts.py``. The probe session is a throwaway pytest run that loads
this suite's conftest as a plugin.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.local_artifacts import local_artifact_marks, requires_local_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_marks_for_a_missing_artifact_name_the_exact_path(tmp_path: Path) -> None:
    missing = tmp_path / "data" / "04_index" / "manifest.json"
    marks = local_artifact_marks(missing, why="a provisioned index")
    assert [mark.name for mark in marks] == ["local_artifact", "skip"]
    reason = marks[1].kwargs["reason"]
    assert str(missing) in reason
    assert "a provisioned index" in reason


def test_marks_for_an_existing_artifact_never_skip(tmp_path: Path) -> None:
    present = tmp_path / "manifest.json"
    present.write_text("{}", encoding="utf-8")
    assert [mark.name for mark in local_artifact_marks(present, why="x")] == ["local_artifact"]


def test_every_missing_path_is_listed(tmp_path: Path) -> None:
    marks = local_artifact_marks(tmp_path / "a.json", tmp_path / "b.json", why="x")
    assert "a.json" in marks[1].kwargs["reason"] and "b.json" in marks[1].kwargs["reason"]


def test_decorator_applies_the_same_marks(tmp_path: Path) -> None:
    @requires_local_artifacts(tmp_path / "absent.json", why="x")
    def probe() -> None: ...

    assert sorted(mark.name for mark in probe.pytestmark) == ["local_artifact", "skip"]


def test_absent_skips_while_present_but_stale_fails(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text('pytest_plugins = ["tests.conftest"]\n', encoding="utf-8")
    (tmp_path / "present.json").write_text('{"fingerprint": "old"}', encoding="utf-8")
    (tmp_path / "test_probe.py").write_text(
        '''
import json
from pathlib import Path
from tests.local_artifacts import requires_local_artifacts

HERE = Path(__file__).parent


@requires_local_artifacts(HERE / "absent.json", why="the maintainer's index")
def test_absent():
    raise AssertionError("must not run when the artifact is absent")


@requires_local_artifacts(HERE / "present.json", why="the maintainer's index")
def test_present_and_stale():
    assert json.loads((HERE / "present.json").read_text())["fingerprint"] == "current"


@requires_local_artifacts(HERE / "present.json", why="the maintainer's index")
def test_present_and_current():
    assert json.loads((HERE / "present.json").read_text())["fingerprint"] == "old"
''',
        encoding="utf-8",
    )
    env = {name: value for name, value in os.environ.items() if not name.startswith("PYTEST_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "src")])
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rsf", "-q", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = completed.stdout
    assert "1 failed, 1 passed, 1 skipped" in output, output
    assert f"local artifact missing ({tmp_path / 'absent.json'})" in output
    assert "FAILED test_probe.py::test_present_and_stale" in output
    assert completed.returncode == 1
