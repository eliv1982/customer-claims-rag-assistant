"""Static contract of the runtime container: image, Compose stack, entrypoint and build context.

Nothing here needs Docker. These checks pin the properties the release boundary depends on (who the
process runs as, what is installed, what is writable, what is published, where the credential may
come from, what runs before the server) so that an edit cannot weaken one silently. They do not
restate every line of the files. The behaviour itself (a built image, a failing startup, a healthy
server) is verified against a real Docker daemon and recorded in docs/08_docker_runbook.md.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "compose.yaml"
ENTRYPOINT = ROOT / "scripts" / "docker-entrypoint.sh"
HEALTHCHECK = ROOT / "scripts" / "healthcheck.py"
LAUNCHER = ROOT / "scripts" / "release_compose.py"
DOCKERIGNORE = ROOT / ".dockerignore"
DESCRIPTOR = ROOT / "configs" / "release" / "production_posture.json"
CORPUS_MANIFEST = ROOT / "configs" / "corpus" / "foodflow_production_v1.json"

PRODUCTION_INDEX = "data/04_index_production"
CONTAINER_INDEX = f"/app/{PRODUCTION_INDEX}"
STREAMLIT_PORT = "8501"


def _launcher():
    spec = importlib.util.spec_from_file_location("release_compose_contract", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- Dockerfile ----------------------------------------------------------------------------------


def _instructions(text: str) -> list[tuple[str, str]]:
    """Dockerfile instructions with continuation lines joined and comments dropped."""
    logical: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#") or (not stripped and not pending):
            continue
        if raw.rstrip().endswith("\\"):
            pending += raw.rstrip()[:-1] + " "
            continue
        logical.append((pending + raw).strip())
        pending = ""
    pairs = []
    for line in logical:
        keyword, _, rest = line.partition(" ")
        pairs.append((keyword.upper(), rest.strip()))
    return pairs


@pytest.fixture(scope="module")
def dockerfile() -> list[tuple[str, str]]:
    return _instructions(DOCKERFILE.read_text(encoding="utf-8"))


def _of(instructions, keyword: str) -> list[str]:
    return [rest for kw, rest in instructions if kw == keyword]


def _image_env(instructions) -> dict[str, str]:
    env: dict[str, str] = {}
    for rest in _of(instructions, "ENV"):
        for token in shlex.split(rest):
            name, _, value = token.partition("=")
            env[name] = value
    return env


def test_the_final_user_is_a_numeric_non_root_user(dockerfile) -> None:
    users = _of(dockerfile, "USER")
    assert users, "the image must set a USER"
    assert re.fullmatch(r"[1-9]\d*:[1-9]\d*", users[-1]), users[-1]
    uid = users[-1].split(":")[0]
    runs = " ".join(_of(dockerfile, "RUN"))
    assert f"--uid {uid}" in runs, "the USER must be the dedicated account created in the image"
    assert "sudo" not in runs
    # Nothing after the last USER switches back to root.
    last_user_index = max(i for i, (kw, _) in enumerate(dockerfile) if kw == "USER")
    assert all(kw not in {"RUN"} for kw, _ in dockerfile[last_user_index + 1 :])


def test_the_image_installs_the_ui_extra_and_nothing_else(dockerfile) -> None:
    installs = [
        command.strip()
        for run in _of(dockerfile, "RUN")
        for command in run.split("&&")
        if "pip install" in command
    ]
    assert installs == ['pip install ".[ui]"'], installs
    code = _code(dockerfile).lower()
    for forbidden in ("[dev]", "screenshots", "pytest", "playwright", "pip install -e", "apt-get", "curl", "wget"):
        assert forbidden not in code, forbidden


def _code(instructions) -> str:
    """The Dockerfile without comments: what the build actually executes."""
    return "\n".join(f"{keyword} {rest}" for keyword, rest in instructions)


def test_the_application_tree_stays_root_owned_and_is_not_handed_to_the_runtime_user(dockerfile) -> None:
    code = _code(dockerfile)
    assert "chown" not in code
    assert "chmod -R" not in code and "777" not in code and "a+w" not in code


def test_only_explicit_files_are_copied_into_the_image(dockerfile) -> None:
    sources: set[str] = set()
    for rest in _of(dockerfile, "COPY"):
        assert "--from" not in rest
        parts = shlex.split(rest)
        sources.update(part.rstrip("/") for part in parts[:-1])
    assert sources == {
        "pyproject.toml",
        "README.md",
        "src",
        "configs",
        "prompts",
        "data/02_clean_markdown",
        "scripts/docker-entrypoint.sh",
        "scripts/healthcheck.py",
    }
    assert "ADD" not in {kw for kw, _ in dockerfile}


def test_the_dockerfile_carries_no_credential_and_no_build_time_secret(dockerfile) -> None:
    keywords = {kw for kw, _ in dockerfile}
    assert "ARG" not in keywords, "a build argument is a way to pass a secret into the image"
    code = _code(dockerfile)
    assert "OPENAI" not in code and "--secret" not in code and "--build-arg" not in code
    for name in _image_env(dockerfile):
        assert not re.search(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", name, re.IGNORECASE), name


def test_the_server_does_not_phone_home_and_the_ports_agree(dockerfile) -> None:
    env = _image_env(dockerfile)
    assert env["ANONYMIZED_TELEMETRY"] == "False"
    assert env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    # A set browser address stops headless Streamlit from asking a third party for the public IP.
    assert env["STREAMLIT_BROWSER_SERVER_ADDRESS"]
    assert env["STREAMLIT_SERVER_HEADLESS"] == "true"
    assert env["STREAMLIT_SERVER_PORT"] == STREAMLIT_PORT
    assert _of(dockerfile, "EXPOSE") == [f"{STREAMLIT_PORT}"]
    assert env["CUSTOMER_CLAIMS_PROJECT_ROOT"] == "/app"


def test_the_health_check_is_the_standard_library_script_and_the_entrypoint_gates_the_server(dockerfile) -> None:
    health = " ".join(_of(dockerfile, "HEALTHCHECK"))
    assert '["python", "/app/scripts/healthcheck.py"]' in health
    assert "--interval" in health and "--timeout" in health and "--retries" in health
    assert json.loads(_of(dockerfile, "ENTRYPOINT")[0]) == ["/app/scripts/docker-entrypoint.sh"]
    command = json.loads(_of(dockerfile, "CMD")[0])
    assert command[:2] == ["streamlit", "run"]
    assert (ROOT / command[2]).is_file()
    assert "STREAMLIT_SERVER_ADDRESS" in _image_env(dockerfile)


def test_the_index_mount_point_is_created_empty_and_nothing_in_the_build_fills_it(dockerfile) -> None:
    runs = " ".join(_of(dockerfile, "RUN"))
    assert f"mkdir -p {PRODUCTION_INDEX}" in runs
    copied = " ".join(_of(dockerfile, "COPY"))
    assert "data/02_clean_markdown" in copied
    assert PRODUCTION_INDEX not in copied
    assert "data/03_chunks" not in copied and "data/05_evaluation" not in copied


# --- Compose -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def service(compose) -> dict:
    return compose["services"]["streamlit"]


def test_the_ui_is_published_on_loopback_by_default_and_wider_binding_is_opt_in(service, dockerfile) -> None:
    ports = service["ports"]
    assert len(ports) == 1
    match = re.fullmatch(
        r"\$\{RAG_BIND_ADDRESS:-(?P<bind>[^}]+)\}:\$\{RAG_HOST_PORT:-(?P<host>\d+)\}:(?P<container>\d+)", ports[0]
    )
    assert match, ports[0]
    assert match["bind"] == "127.0.0.1", "the unauthenticated UI must default to loopback"
    assert match["container"] == _image_env(dockerfile)["STREAMLIT_SERVER_PORT"]


def test_compose_applies_only_hardening_that_was_verified_against_the_application(service) -> None:
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["tmpfs"] and all(entry.startswith("/tmp") for entry in service["tmpfs"])


def test_compose_grants_no_extra_privilege_and_mounts_no_host_control_surface(compose, service) -> None:
    for key in ("privileged", "cap_add", "devices", "network_mode", "pid", "ipc", "userns_mode", "env_file", "secrets"):
        assert key not in service, key
    assert service.get("user") in (None, "10001:10001")
    assert "secrets" not in compose and "configs" not in compose
    text = COMPOSE.read_text(encoding="utf-8")
    assert "docker.sock" not in text
    assert "network_mode" not in text
    for volume in service["volumes"]:
        assert volume["type"] == "bind" and volume["target"] == CONTAINER_INDEX


def test_the_only_mount_is_the_writable_production_index_and_a_missing_host_directory_is_an_error(service) -> None:
    (mount,) = service["volumes"]
    assert mount["source"] == "${RAG_ACTIVE_INDEX_HOST_PATH:-./data/04_index_production}"
    assert mount["target"] == CONTAINER_INDEX
    # Chroma opens its SQLite database read-write even to query: a read-only mount fails with
    # "attempt to write a readonly database" (verified; see docs/08_docker_runbook.md).
    assert not mount.get("read_only", False)
    # Without this, Compose creates a missing host directory: an empty "index" owned by root.
    assert mount["bind"]["create_host_path"] is False


def test_compose_refuses_to_resolve_without_the_launcher_and_the_credential(compose, service) -> None:
    launcher = _launcher()
    guard = compose["x-release-guard"]
    assert guard.startswith("${" + launcher.LAUNCHER_MARKER_ENV + ":?"), guard
    key = service["environment"]["OPENAI_API_KEY"]
    assert key.startswith("${OPENAI_API_KEY:?") and ":-" not in key.split("}")[0], key
    assert "build" in service and service["build"] == "."
    # No build argument or secret: the key is a run-time value only.
    assert "args" not in (service["build"] if isinstance(service["build"], dict) else {})


def test_a_failed_release_validation_is_not_retried_forever(service) -> None:
    assert re.fullmatch(r"on-failure:[1-5]", service["restart"]), service["restart"]


def test_compose_and_launcher_agree_on_the_names_and_the_index_path(service) -> None:
    launcher = _launcher()
    (mount,) = service["volumes"]
    assert f"${{{launcher.INDEX_HOST_PATH_ENV}:-{launcher.DEFAULT_INDEX_HOST_PATH}}}" == mount["source"]
    assert launcher.BIND_ADDRESS_ENV in service["ports"][0]
    assert launcher.CREDENTIAL_ENV in service["environment"]


def test_the_launchers_provisioning_hint_is_the_command_the_release_gate_prints() -> None:
    from customer_claims_rag.release.posture import build_instruction, resolve_production_release_posture

    target = resolve_production_release_posture(project_root=ROOT).resolved_target
    assert _launcher().PROVISIONING_COMMAND == build_instruction(target)


# --- entrypoint ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def entrypoint() -> str:
    return ENTRYPOINT.read_text(encoding="utf-8")


def _serving_branch(entrypoint: str) -> list[str]:
    match = re.search(r'if \[ "\$#" -gt 0 \] && \[ "\$1" = "streamlit" \]; then\n(.*?)\nfi\n', entrypoint, re.S)
    assert match, "the serving branch is missing"
    return [line.strip() for line in match[1].splitlines() if line.strip()]


def test_the_server_starts_only_after_configuration_the_index_check_and_the_full_release_validation(
    entrypoint,
) -> None:
    assert re.search(r"^set -eu$", entrypoint, re.M)
    assert _serving_branch(entrypoint) == ["require_configuration", "require_writable_index", "run_release_gate"]
    tail = entrypoint.split("\nfi\n", 1)[1].strip()
    assert tail == 'exec "$@"', "exec replaces the shell with the server, after the gate and nowhere else"
    assert entrypoint.count('exec "$@"') == 1


def test_the_gate_is_the_full_validation_and_a_failure_stops_the_container(entrypoint) -> None:
    gate = re.search(r"run_release_gate\(\) \{\n(.*?)\n\}\n", entrypoint, re.S)[1]
    assert re.search(r"^\s*validate-release-posture \|\| status=\$\?$", gate, re.M)
    assert "--" not in gate.split("validate-release-posture", 1)[1].split("\n", 1)[0]
    assert 'exit "$status"' in gate
    for bypass in ("|| true", "|| exit 0", "|| :"):
        assert bypass not in entrypoint


def test_the_entrypoint_never_builds_repairs_or_creates_an_index_and_has_no_fallback(entrypoint) -> None:
    code = "\n".join(line for line in entrypoint.splitlines() if not line.lstrip().startswith("#"))
    for word in ("build_index", "build-index", "mkdir", "rm -", "cp ", "chroma.sqlite3 ||"):
        assert word not in code, word
    assert re.findall(r"data/04_index\w*", code) == [PRODUCTION_INDEX]
    assert f'INDEX_DIR="{CONTAINER_INDEX}"' in entrypoint


def test_the_entrypoint_never_prints_the_credential(entrypoint) -> None:
    expansions = [line for line in entrypoint.splitlines() if re.search(r"\$\{?OPENAI_API_KEY", line)]
    assert len(expansions) == 1 and "case" in expansions[0], expansions
    for line in entrypoint.splitlines():
        if "echo" in line:
            assert not re.search(r"\$\{?OPENAI", line), line


def test_a_missing_credential_is_a_configuration_error_distinct_from_a_release_failure(entrypoint) -> None:
    config = re.search(r"require_configuration\(\) \{\n(.*?)\n\}\n", entrypoint, re.S)[1]
    assert "exit 2" in config
    assert "release_compose.py" in config


# --- build context -------------------------------------------------------------------------------

ALLOWED_CONTEXT = {
    "pyproject.toml",
    "README.md",
    "src/",
    "configs/",
    "prompts/",
    "scripts/docker-entrypoint.sh",
    "scripts/healthcheck.py",
    "data/",
    "data/02_clean_markdown/",
    "data/02_clean_markdown/*.md",
}


def _ignore_patterns() -> list[str]:
    return [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_the_build_context_is_an_allowlist_that_denies_everything_by_default() -> None:
    patterns = _ignore_patterns()
    assert patterns[0] == "*"
    assert {p[1:] for p in patterns if p.startswith("!")} == ALLOWED_CONTEXT


def test_every_path_the_dockerfile_copies_is_allowed_by_the_build_context(dockerfile) -> None:
    allowed = {entry.rstrip("/") for entry in ALLOWED_CONTEXT}
    for rest in _of(dockerfile, "COPY"):
        for source in shlex.split(rest)[:-1]:
            assert source.rstrip("/") in allowed, source


def test_local_state_inside_allowed_directories_is_excluded_after_the_allowlist() -> None:
    patterns = _ignore_patterns()
    last_allow = max(i for i, p in enumerate(patterns) if p.startswith("!"))
    after = patterns[last_allow + 1 :]
    # `**/` is required: Docker matches a bare `__pycache__/` only at the context root, which is
    # how compiled files from a developer machine once reached the image.
    for required in ("**/__pycache__", "**/*.py[cod]", "**/*.egg-info"):
        assert required in after, required


@pytest.mark.parametrize(
    "sensitive",
    [".env", ".git", ".github", ".venv", "tests", "docs", "deliverables", "experiments", ".claude", "compose", "Dockerfile"],
)
def test_no_sensitive_or_local_path_is_re_included_in_the_build_context(sensitive: str) -> None:
    for pattern in _ignore_patterns():
        if pattern.startswith("!"):
            assert not pattern[1:].startswith(sensitive), pattern


def test_only_the_canonical_source_subtree_is_reincluded_from_data() -> None:
    patterns = _ignore_patterns()
    data_allows = {pattern for pattern in patterns if pattern.startswith("!data")}
    assert data_allows == {
        "!data/",
        "!data/02_clean_markdown/",
        "!data/02_clean_markdown/*.md",
    }
    for forbidden in (
        "03_chunks",
        "04_index",
        "04_index_production",
        "04_index_experiments",
        "05_evaluation",
    ):
        assert all(forbidden not in pattern for pattern in data_allows)


def test_canonical_source_directory_contains_exactly_the_manifest_selection() -> None:
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        entry["file"]
        for field in ("documents", "excluded_documents")
        for entry in manifest[field]
    }
    actual = {path.name for path in (ROOT / "data" / "02_clean_markdown").glob("*.md")}
    assert actual == expected


# --- the production index path is one fact everywhere --------------------------------------------


def test_every_file_that_names_the_production_index_names_the_descriptors_path(service) -> None:
    descriptor = json.loads(DESCRIPTOR.read_text(encoding="utf-8"))
    assert descriptor["targets"]["active"]["index_path"] == PRODUCTION_INDEX
    launcher = _launcher()
    assert launcher.DEFAULT_INDEX_HOST_PATH == f"./{PRODUCTION_INDEX}"
    assert PRODUCTION_INDEX in launcher.PROVISIONING_COMMAND
    assert service["volumes"][0]["target"] == CONTAINER_INDEX
    for path in (DOCKERFILE, COMPOSE, ENTRYPOINT, LAUNCHER, HEALTHCHECK):
        text = path.read_text(encoding="utf-8")
        historical = re.findall(r"data/04_index(?!_production)\w*", text)
        assert historical == [], f"{path.name} points at a non-production index directory: {historical}"


# --- documentation names the supported path ------------------------------------------------------

BARE_COMPOSE = re.compile(r"^\s*(?:PS [^>]*> )?docker compose (?:up|run|build|config|down)\b", re.M)


@pytest.mark.parametrize("doc", ["README.md", "docs/07_index_provisioning.md", "docs/08_docker_runbook.md"])
def test_operator_documentation_runs_compose_through_the_launcher_never_bare(doc: str) -> None:
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert BARE_COMPOSE.findall(text) == [], (
        f"{doc} shows a bare `docker compose` command; a bare invocation stops at the guard in compose.yaml"
    )


def test_the_runbook_documents_the_credential_rule_the_exposure_model_and_the_write_requirement() -> None:
    text = (ROOT / "docs" / "08_docker_runbook.md").read_text(encoding="utf-8")
    for needle in (
        "scripts/release_compose.py",
        "127.0.0.1",
        "RAG_BIND_ADDRESS",
        "up -d --wait",
        "attempt to write a readonly database",
        "10001",
    ):
        assert needle in text, needle
    assert ".env" in text and "never" in text.lower()
