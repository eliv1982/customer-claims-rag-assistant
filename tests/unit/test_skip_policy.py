"""Every way a test can skip is accounted for, so coverage cannot erode quietly.

A skip is legitimate only when the test's precondition is a property of the machine
(no git executable, symlinks not creatable), the lane (the real-tokenizer lane is not
requested) or a declared local artifact that is absent (``tests/local_artifacts.py``).
It is never a reaction to a tracked artifact being missing or out of date: a clone always
has those, and a stale one must fail. This test fails when a skip appears anywhere not
listed below, which forces the new skip to be justified here.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
SKIP_NAMES = {"skip", "skipif", "xfail", "importorskip"}

# path under tests/  ->  (number of skip sites, why they are legitimate)
ALLOWED_SKIPS = {
    "conftest.py": (1, "real_tiktoken tests skip when the real-tokenizer lane is not requested"),
    "local_artifacts.py": (1, "the one skip behind the local_artifact marker: artifact absent"),
    "integration/test_path_security.py": (1, "symlink creation not permitted on this machine"),
    "unit/test_retrieval_path_helpers.py": (1, "symlink creation not permitted on this machine"),
    "unit/test_evaluation_git_state.py": (2, "git executable not installed"),
    "unit/test_hermetic_test_environment.py": (1, "checks the default lane; the real lane replaces its subject"),
}


def _is_pytest_skip(node: ast.AST) -> bool:
    """``pytest.skip``, ``pytest.importorskip``, ``pytest.mark.skip|skipif|xfail`` (called or bare)."""
    if not isinstance(node, ast.Attribute) or node.attr not in SKIP_NAMES:
        return False
    owner = node.value
    if isinstance(owner, ast.Name):
        return owner.id == "pytest"
    return (
        isinstance(owner, ast.Attribute)
        and owner.attr == "mark"
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "pytest"
    )


def _skip_sites(source: str) -> int:
    return sum(_is_pytest_skip(node) for node in ast.walk(ast.parse(source)))


def test_skips_exist_only_where_they_are_justified() -> None:
    found = {}
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        count = _skip_sites(path.read_text(encoding="utf-8"))
        if count:
            found[path.relative_to(TESTS_ROOT).as_posix()] = count
    expected = {path: count for path, (count, _why) in ALLOWED_SKIPS.items()}
    assert found == expected, (
        "skip sites changed; a new skip must be justified in ALLOWED_SKIPS "
        "(machine capability, lane selection or a declared local artifact), "
        "and a skip on a tracked artifact is never acceptable"
    )


def test_the_scanner_recognises_every_skip_spelling() -> None:
    assert _skip_sites("import pytest\npytest.skip('x')") == 1
    assert _skip_sites("import pytest\npytest.importorskip('x')") == 1
    assert _skip_sites("import pytest\n@pytest.mark.skipif(True, reason='x')\ndef t(): ...") == 1
    assert _skip_sites("import pytest\n@pytest.mark.skip\ndef t(): ...") == 1
    assert _skip_sites("import pytest\n@pytest.mark.xfail\ndef t(): ...") == 1
    assert _skip_sites("import pytest\n@pytest.mark.real_tiktoken\ndef t(): ...") == 0
    assert _skip_sites("skip = 1\nobj.skip()") == 0
