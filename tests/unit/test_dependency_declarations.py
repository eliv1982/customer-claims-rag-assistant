"""Every third-party package the project imports directly must be declared in pyproject.toml.

Relying on a transitive install (``numpy`` via chromadb, ``langchain_core`` via
langchain-openai) breaks as soon as the upstream package drops it. Where a
package is declared follows where it is used:

* ``src/`` -> runtime dependencies (``ui/`` -> the ``ui`` extra)
* ``scripts/`` -> an optional extra
* ``tests/`` -> runtime dependencies or the ``dev`` extra
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# import name -> distribution name. A new third-party import must be added here on purpose.
# ``pydantic_core`` maps to None: it is pydantic's own compiled core, pinned by pydantic
# to one exact version, so it cannot be declared independently of pydantic.
IMPORT_TO_DISTRIBUTION: dict[str, str | None] = {
    "chromadb": "chromadb",
    "dotenv": "python-dotenv",
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "numpy": "numpy",
    "playwright": "playwright",
    "pydantic": "pydantic",
    "pydantic_core": None,
    "pytest": "pytest",
    "streamlit": "streamlit",
    "tiktoken": "tiktoken",
    "yaml": "pyyaml",
}

FIRST_PARTY = {"customer_claims_rag", "tests"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(specs: list[str]) -> set[str]:
    return {_normalize(re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec).group(0)) for spec in specs}


def _declared() -> dict[str, set[str]]:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project.get("optional-dependencies", {})
    return {
        "runtime": _requirement_names(project["dependencies"]),
        **{f"extra:{name}": _requirement_names(specs) for name, specs in extras.items()},
    }


def _imported_top_level_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return {
        module
        for module in modules
        if module not in sys.stdlib_module_names and module not in FIRST_PARTY
    }


def _third_party_imports(area: str) -> dict[str, list[str]]:
    """module -> files that import it, for one area of the repository."""
    root = PROJECT_ROOT / area
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if "/ui/" in relative:
            continue
        for module in _imported_top_level_modules(path):
            found.setdefault(module, []).append(relative)
    return found


def _ui_imports() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted((PROJECT_ROOT / "src").rglob("ui/*.py")):
        for module in _imported_top_level_modules(path):
            found.setdefault(module, []).append(path.relative_to(PROJECT_ROOT).as_posix())
    return found


def _undeclared(imports: dict[str, list[str]], allowed: set[str]) -> list[str]:
    problems = []
    for module, files in sorted(imports.items()):
        if module not in IMPORT_TO_DISTRIBUTION:
            problems.append(f"{module}: unknown import (add it to IMPORT_TO_DISTRIBUTION) in {files[0]}")
            continue
        distribution = IMPORT_TO_DISTRIBUTION[module]
        if distribution is not None and _normalize(distribution) not in allowed:
            problems.append(f"{module}: distribution {distribution!r} is not declared (used in {files[0]})")
    return problems


def test_src_imports_are_declared_runtime_dependencies() -> None:
    declared = _declared()
    assert _undeclared(_third_party_imports("src"), declared["runtime"]) == []


def test_ui_imports_are_declared_in_the_ui_extra() -> None:
    declared = _declared()
    assert _undeclared(_ui_imports(), declared["runtime"] | declared["extra:ui"]) == []
    assert "streamlit" in declared["extra:ui"]


def test_script_imports_are_declared_in_an_extra() -> None:
    declared = _declared()
    allowed = declared["runtime"].union(*(v for k, v in declared.items() if k.startswith("extra:")))
    assert _undeclared(_third_party_imports("scripts"), allowed) == []
    assert "playwright" in declared["extra:screenshots"]
    assert "playwright" not in declared["runtime"] | declared["extra:dev"]


def test_test_imports_are_declared_runtime_or_dev() -> None:
    declared = _declared()
    assert _undeclared(_third_party_imports("tests"), declared["runtime"] | declared["extra:dev"]) == []


def test_previously_transitive_imports_are_now_direct() -> None:
    runtime = _declared()["runtime"]
    assert {"numpy", "langchain-core"} <= runtime


def test_undeclared_import_is_reported() -> None:
    problems = _undeclared({"numpy": ["a.py"], "left_pad": ["b.py"]}, {"chromadb"})
    assert len(problems) == 2
    assert any("numpy" in p and "not declared" in p for p in problems)
    assert any("left_pad" in p and "unknown import" in p for p in problems)
