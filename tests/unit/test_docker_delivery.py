"""Focused tests for Docker delivery packaging (stage 5B)."""

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
DESCRIPTOR_PATH = PROJECT_ROOT / "configs" / "release" / "production_posture.json"

ACTIVE_INDEX_CONTAINER_PATH = "/app/data/04_index_production"
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
    ],
)
def test_docker_runtime_files_exist(path: Path) -> None:
    assert path.is_file(), f"missing required Docker file: {path.name}"


def test_dockerignore_excludes_production_index_and_secrets() -> None:
    content = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    assert "data/04_index_production" in content
    assert "data/04_index" in content
    assert ".env" in content
    assert ".venv" in content


def test_gitignore_excludes_production_index() -> None:
    content = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/04_index_production/" in content


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
    assert '["$1" = "streamlit"]' in content or '[ "$1" = "streamlit" ]' in content
    assert "exec" in content
    assert "chroma.sqlite3" in content
    assert f'INDEX_DIR="{ACTIVE_INDEX_CONTAINER_PATH}"' in content


def test_dockerfile_uses_non_root_user_and_streamlit_port() -> None:
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")
    assert "python:3.12-slim" in content
    assert "USER app" in content
    assert "8501" in content
    assert "--server.address=0.0.0.0" in content
    assert 'pip install ".[ui]"' in content
    assert "OPENAI_API_KEY" not in content


@pytest.mark.parametrize(
    "path",
    [DOCKERFILE_PATH, COMPOSE_PATH, ENTRYPOINT_PATH, DOCKERIGNORE_PATH],
)
def test_docker_files_do_not_contain_secret_literals(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        assert pattern.search(content) is None, f"suspected secret literal in {path.name}"
