"""Focused tests for Docker delivery packaging (stage 5B).

The runtime container contract (user, installed extras, exposure, hardening, credential source,
entrypoint ordering, build context) lives in test_container_runtime_contract.py; this module keeps
the delivery-level checks that predate it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"
COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
ENTRYPOINT_PATH = PROJECT_ROOT / "scripts" / "docker-entrypoint.sh"
HEALTHCHECK_PATH = PROJECT_ROOT / "scripts" / "healthcheck.py"
LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "release_compose.py"
DESCRIPTOR_PATH = PROJECT_ROOT / "configs" / "release" / "production_posture.json"

ACTIVE_INDEX_CONTAINER_PATH = "/app/data/04_index_production"
STAGE_2E_TEXT_PATHS = (
    ".dockerignore",
    ".env.example",
    ".github/workflows/ci.yml",
    "Dockerfile",
    "README.md",
    "compose.yaml",
    "docs/07_index_provisioning.md",
    "docs/06_release_posture.md",
    "docs/08_docker_runbook.md",
    "docs/09_ci_contract.md",
    "scripts/docker-entrypoint.sh",
    "scripts/healthcheck.py",
    "scripts/release_compose.py",
    "src/customer_claims_rag/release/readiness.py",
    "src/customer_claims_rag/retrieval/embedding_validation.py",
    "src/customer_claims_rag/retrieval/adapters/chroma_store.py",
    "src/customer_claims_rag/retrieval/adapters/openai_embeddings.py",
    "tests/conftest.py",
    "tests/integration/test_chroma_store.py",
    "tests/network_guard.py",
    "tests/release_posture_helpers.py",
    "tests/subprocess_guard/sitecustomize.py",
    "tests/unit/test_container_healthcheck.py",
    "tests/unit/test_container_runtime_contract.py",
    "tests/unit/test_docker_delivery.py",
    "tests/unit/test_embeddings.py",
    "tests/unit/test_hermetic_test_environment.py",
    "tests/unit/test_release_compose_launcher.py",
    "tests/unit/test_release_gate_exit_status.py",
    "tests/unit/test_release_readiness.py",
)
RELEASE_ID = "foodflow-10doc-release-v2"

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"OPENAI_API_KEY\s*=\s*['\"]?[A-Za-z0-9_-]{8,}"),
)


@pytest.mark.parametrize(
    "path",
    [
        DOCKERFILE_PATH,
        DOCKERIGNORE_PATH,
        COMPOSE_PATH,
        ENTRYPOINT_PATH,
        HEALTHCHECK_PATH,
        LAUNCHER_PATH,
    ],
)
def test_docker_runtime_files_exist(path: Path) -> None:
    assert path.is_file(), f"missing required Docker file: {path.name}"


def test_dockerignore_denies_by_default_so_secrets_and_local_indexes_cannot_enter_the_context() -> None:
    patterns = [
        line.strip()
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert patterns[0] == "*"
    for sensitive in (".env", ".venv", ".git"):
        assert not any(p.startswith("!" + sensitive) for p in patterns), sensitive
    assert {p for p in patterns if p.startswith("!data")} == {
        "!data/",
        "!data/02_clean_markdown/",
        "!data/02_clean_markdown/*.md",
    }


def test_gitignore_excludes_production_index() -> None:
    content = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/04_index_production/" in content


def test_stage_2e_text_files_are_physically_lf_under_the_repository_policy() -> None:
    attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in attributes
    for relative in STAGE_2E_TEXT_PATHS:
        content = (PROJECT_ROOT / relative).read_bytes()
        assert b"\r" not in content, f"{relative} contains a CR byte instead of repository LF"


def test_compose_mount_destination_matches_descriptor() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = compose["services"]["streamlit"]
    mounts = service["volumes"]
    assert len(mounts) == 1
    mount = mounts[0]
    assert mount["target"] == ACTIVE_INDEX_CONTAINER_PATH
    assert mount["type"] == "bind"
    assert "RAG_ACTIVE_INDEX_HOST_PATH" in mount["source"]


def test_compose_defaults_to_active_release_target() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    release_target = compose["services"]["streamlit"]["environment"]["RAG_RELEASE_TARGET"]
    assert release_target == "${RAG_RELEASE_TARGET:-active}"


def test_compose_mounts_only_the_production_index() -> None:
    """Historical local archives (data/04_index, backups) are never mounted into the container."""
    compose_text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "data/04_index" not in compose_text.replace("04_index_production", "")


def test_descriptor_active_index_path_matches_container_mount() -> None:
    import json

    descriptor = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    assert descriptor["release_posture_id"] == RELEASE_ID
    active_path = descriptor["targets"]["active"]["index_path"]
    assert active_path == "data/04_index_production"
    assert f"/app/{active_path}" == ACTIVE_INDEX_CONTAINER_PATH


def test_entrypoint_runs_release_validation_before_streamlit() -> None:
    content = ENTRYPOINT_PATH.read_text(encoding="utf-8")
    assert "validate-release-posture" in content
    assert '[ "$1" = "streamlit" ]' in content
    assert content.index("run_release_gate\nfi") < content.index('exec "$@"')
    assert f'INDEX_DIR="{ACTIVE_INDEX_CONTAINER_PATH}"' in content


def test_dockerfile_uses_the_ui_extra_and_streamlit_port() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "python:3.12-slim" in content
    assert "8501" in content
    assert 'pip install ".[ui]"' in content
    assert "OPENAI_API_KEY" not in content


@pytest.mark.parametrize(
    "path",
    [DOCKERFILE_PATH, COMPOSE_PATH, ENTRYPOINT_PATH, DOCKERIGNORE_PATH, HEALTHCHECK_PATH, LAUNCHER_PATH],
)
def test_docker_files_do_not_contain_secret_literals(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        assert pattern.search(content) is None, f"suspected secret literal in {path.name}"
