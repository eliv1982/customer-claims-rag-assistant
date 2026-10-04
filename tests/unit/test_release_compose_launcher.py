"""The release launcher decides where the container's paid credential may come from.

Compose reads a project ``.env`` for interpolation by default, so an old repository ``.env`` could
silently supply ``OPENAI_API_KEY`` to a release container. ``scripts/release_compose.py`` is the
supported entry: it needs the key in the real process environment and tells Compose to ignore the
project ``.env``. These tests drive it with an injected runner (no Docker needed) and fake keys.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "release_compose.py"

PROCESS_KEY = "sk-FAKE-PROCESS-KEY-0000000000"
DOTENV_KEY = "sk-FAKE-DOTENV-KEY-1111111111"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("release_compose_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher()


class Recorder:
    """Stands in for subprocess.run and records what the launcher handed to Compose."""

    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.calls: list[dict] = []
        self._result = subprocess.CompletedProcess([], returncode, stdout, stderr)

    def __call__(self, command, **kwargs):
        env_file = Path(command[command.index("--env-file") + 1])
        self.calls.append(
            {
                "command": list(command),
                "env": dict(kwargs.get("env") or {}),
                "capture": bool(kwargs.get("capture_output")),
                "env_file_existed": env_file.is_file(),
                "env_file_size": env_file.stat().st_size if env_file.is_file() else None,
                "env_file": env_file,
            }
        )
        return self._result


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A project root whose production index directory exists (the launcher checks existence only)."""
    (tmp_path / "data" / "04_index_production").mkdir(parents=True)
    return tmp_path


def _run(args, environ, project, runner=None, which=lambda name: "/usr/bin/docker"):
    runner = runner or Recorder()
    code = launcher.main(args, environ=environ, runner=runner, which=which, project_root=project)
    return code, runner


def _assert_no_key(capsys, *keys: str) -> None:
    captured = capsys.readouterr()
    for key in keys:
        assert key not in captured.out
        assert key not in captured.err


# --- the credential must come from the process environment -----------------------------------


@pytest.mark.parametrize(
    ("environ", "reason"),
    [
        ({}, "is not set in the process environment"),
        ({"OPENAI_API_KEY": ""}, "is set but empty"),
        ({"OPENAI_API_KEY": "   "}, "is set but empty"),
        ({"OPENAI_API_KEY": PROCESS_KEY + "\n"}, "contains whitespace"),
        ({"OPENAI_API_KEY": " " + PROCESS_KEY}, "contains whitespace"),
        ({"OPENAI_API_KEY": "sk-FAKE KEY"}, "contains whitespace"),
    ],
    ids=["absent", "empty", "blank", "trailing-newline", "leading-space", "inner-space"],
)
def test_missing_empty_or_malformed_credential_is_refused_before_compose_runs(
    environ, reason, project, capsys
) -> None:
    code, runner = _run(["up"], environ, project)
    assert code == launcher.EXIT_REFUSED == 2
    assert runner.calls == []
    err = capsys.readouterr().err
    assert reason in err
    assert "shell environment" in err
    assert "sk-FAKE" not in err


def test_a_repository_dotenv_alone_is_not_a_credential_source(project: Path, capsys) -> None:
    """The scenario this launcher exists for: a stale .env holds a key, the shell does not."""
    (project / ".env").write_text(f"OPENAI_API_KEY={DOTENV_KEY}\nOPENAI_CHAT_MODEL=x\n", encoding="utf-8")
    code, runner = _run(["up"], {"PATH": "/usr/bin"}, project)
    assert code == 2
    assert runner.calls == []
    captured = capsys.readouterr()
    assert "deliberately ignored" in captured.err
    assert DOTENV_KEY not in captured.out + captured.err


def test_a_dotenv_key_never_replaces_or_joins_the_process_key(project: Path, capsys) -> None:
    (project / ".env").write_text(f"OPENAI_API_KEY={DOTENV_KEY}\n", encoding="utf-8")
    code, runner = _run(["up"], {"OPENAI_API_KEY": PROCESS_KEY}, project)
    assert code == 0
    (call,) = runner.calls
    assert call["env"]["OPENAI_API_KEY"] == PROCESS_KEY
    assert DOTENV_KEY not in " ".join(call["command"]) + repr(call["env"])
    _assert_no_key(capsys, PROCESS_KEY, DOTENV_KEY)


def test_dotenv_detection_reads_only_the_variable_name(tmp_path: Path) -> None:
    assert launcher.dotenv_defines_credential(tmp_path) is False
    (tmp_path / ".env").write_text("# OPENAI_API_KEY=commented\nOTHER=1\n", encoding="utf-8")
    assert launcher.dotenv_defines_credential(tmp_path) is False
    (tmp_path / ".env").write_text("export OPENAI_API_KEY = x\n", encoding="utf-8")
    assert launcher.dotenv_defines_credential(tmp_path) is True
    (tmp_path / ".env").write_bytes(b"\xff\xfe\x00bad")
    assert launcher.dotenv_defines_credential(tmp_path) is False


# --- how Compose is invoked ---------------------------------------------------------------------


def test_compose_is_told_to_ignore_the_project_dotenv_and_the_file_is_pinned(project: Path) -> None:
    code, runner = _run(["up", "-d", "--wait"], {"OPENAI_API_KEY": PROCESS_KEY}, project)
    assert code == 0
    (call,) = runner.calls
    command = call["command"]
    assert command[:2] == ["/usr/bin/docker", "compose"]
    assert command[command.index("-f") + 1] == str(launcher.COMPOSE_FILE)
    assert command[command.index("--project-directory") + 1] == str(launcher.ROOT)
    assert command[-3:] == ["up", "-d", "--wait"]
    # --env-file replaces the default project .env; the replacement file exists and is empty
    # while Compose runs, and it is removed afterwards.
    assert call["env_file_existed"] is True
    assert call["env_file_size"] == 0
    assert not call["env_file"].exists()
    assert Path(call["env_file"]).resolve() != (ROOT / ".env").resolve()


def test_the_key_travels_in_the_child_environment_never_on_the_command_line(project: Path, capsys) -> None:
    code, runner = _run(["up"], {"OPENAI_API_KEY": PROCESS_KEY, "PATH": "/usr/bin"}, project)
    assert code == 0
    (call,) = runner.calls
    assert PROCESS_KEY not in " ".join(call["command"])
    assert call["env"]["OPENAI_API_KEY"] == PROCESS_KEY
    assert call["env"]["PATH"] == "/usr/bin"
    assert call["env"][launcher.LAUNCHER_MARKER_ENV] == "scripts/release_compose.py"
    _assert_no_key(capsys, PROCESS_KEY)


def test_environment_sources_that_would_bypass_the_pinned_compose_setup_are_dropped(project: Path) -> None:
    environ = {
        "OPENAI_API_KEY": PROCESS_KEY,
        "COMPOSE_ENV_FILES": ".env",
        "COMPOSE_FILE": "other.yaml",
        "COMPOSE_PATH_SEPARATOR": ";",
    }
    _, runner = _run(["up"], environ, project)
    (call,) = runner.calls
    for name in ("COMPOSE_ENV_FILES", "COMPOSE_FILE", "COMPOSE_PATH_SEPARATOR"):
        assert name not in call["env"]


@pytest.mark.parametrize("first", ["-f", "--env-file", "--project-directory", "--profile", "-p"])
def test_global_compose_options_are_not_accepted_so_they_cannot_reintroduce_an_env_source(
    first, project: Path, capsys
) -> None:
    code, runner = _run([first, ".env", "up"], {"OPENAI_API_KEY": PROCESS_KEY}, project)
    assert code == 2
    assert runner.calls == []
    assert "fixed by this launcher" in capsys.readouterr().err


def test_no_arguments_prints_usage_without_running_anything(project: Path) -> None:
    code, runner = _run([], {"OPENAI_API_KEY": PROCESS_KEY}, project)
    assert code == 2 and runner.calls == []


def test_the_exit_status_of_compose_is_passed_through(project: Path) -> None:
    code, _ = _run(["up"], {"OPENAI_API_KEY": PROCESS_KEY}, project, runner=Recorder(returncode=7))
    assert code == 7


def test_a_missing_docker_command_is_reported_distinctly(project: Path, capsys) -> None:
    code, runner = _run(["up"], {"OPENAI_API_KEY": PROCESS_KEY}, project, which=lambda name: None)
    assert code == launcher.EXIT_DOCKER_MISSING
    assert runner.calls == []
    assert "docker command was not found" in capsys.readouterr().err


# --- no secret in any output ---------------------------------------------------------------------


def test_config_output_is_redacted_because_compose_prints_the_resolved_key(project: Path, capsys) -> None:
    resolved = f"services:\n  streamlit:\n    environment:\n      OPENAI_API_KEY: {PROCESS_KEY}\n"
    runner = Recorder(stdout=resolved, stderr=f"note {PROCESS_KEY}\n")
    code, _ = _run(["config"], {"OPENAI_API_KEY": PROCESS_KEY}, project, runner=runner)
    assert code == 0
    assert runner.calls[0]["capture"] is True
    captured = capsys.readouterr()
    assert PROCESS_KEY not in captured.out + captured.err
    assert "OPENAI_API_KEY: [REDACTED]" in captured.out


def test_other_commands_stream_their_output_without_capture(project: Path) -> None:
    runner = Recorder()
    _run(["up"], {"OPENAI_API_KEY": PROCESS_KEY}, project, runner=runner)
    assert runner.calls[0]["capture"] is False


def test_redact_is_exact_and_tolerates_an_empty_secret() -> None:
    assert launcher.redact(f"a {PROCESS_KEY} b {PROCESS_KEY}", PROCESS_KEY) == "a [REDACTED] b [REDACTED]"
    assert launcher.redact("unchanged", "") == "unchanged"


# --- the index mount: existence is checked, nothing is created ----------------------------------


@pytest.mark.parametrize("command", ["up", "run", "create"])
def test_a_missing_index_directory_stops_mounting_commands_with_the_provisioning_command(
    command, tmp_path: Path, capsys
) -> None:
    code, runner = _run([command], {"OPENAI_API_KEY": PROCESS_KEY}, tmp_path)
    assert code == 2
    assert runner.calls == []
    err = capsys.readouterr().err
    assert "does not exist" in err
    assert launcher.PROVISIONING_COMMAND in err
    assert "docs/07_index_provisioning.md" in err
    assert not (tmp_path / "data").exists(), "the launcher must not create the index directory"


@pytest.mark.parametrize("command", ["config", "build", "down", "ps", "logs"])
def test_commands_that_do_not_mount_the_index_do_not_need_it(command, tmp_path: Path) -> None:
    code, runner = _run([command], {"OPENAI_API_KEY": PROCESS_KEY}, tmp_path)
    assert code == 0
    assert len(runner.calls) == 1


def test_the_host_index_path_override_is_honoured_and_relative_paths_resolve_against_the_project(
    tmp_path: Path,
) -> None:
    environ = {"OPENAI_API_KEY": PROCESS_KEY, "RAG_ACTIVE_INDEX_HOST_PATH": "elsewhere/idx"}
    assert launcher.index_problem(environ, "up", project_root=tmp_path) is not None
    (tmp_path / "elsewhere" / "idx").mkdir(parents=True)
    assert launcher.index_problem(environ, "up", project_root=tmp_path) is None
    absolute = {"RAG_ACTIVE_INDEX_HOST_PATH": str(tmp_path / "elsewhere" / "idx")}
    assert launcher.index_problem(absolute, "up", project_root=Path("/nonexistent")) is None


def test_an_empty_index_directory_is_left_to_the_containers_full_validation(tmp_path: Path) -> None:
    """The launcher checks that the mount source exists; whether it holds a valid index is the
    container's job (full release validation), so an empty directory still reaches it."""
    (tmp_path / "data" / "04_index_production").mkdir(parents=True)
    assert launcher.index_problem({}, "up", project_root=tmp_path) is None


# --- exposure warning ---------------------------------------------------------------------------


@pytest.mark.parametrize("bind", ["0.0.0.0", "192.168.1.10", "::"])
def test_binding_beyond_loopback_is_allowed_but_warned_about(bind, project: Path, capsys) -> None:
    code, runner = _run(["up"], {"OPENAI_API_KEY": PROCESS_KEY, "RAG_BIND_ADDRESS": bind}, project)
    assert code == 0 and len(runner.calls) == 1
    assert "beyond loopback" in capsys.readouterr().err


@pytest.mark.parametrize("bind", [None, "", "127.0.0.1", "localhost", "::1"])
def test_loopback_or_default_binding_is_not_warned_about(bind, project: Path, capsys) -> None:
    environ = {"OPENAI_API_KEY": PROCESS_KEY}
    if bind is not None:
        environ["RAG_BIND_ADDRESS"] = bind
    _run(["up"], environ, project)
    assert "beyond loopback" not in capsys.readouterr().err


# --- the real script, as an operator runs it ------------------------------------------------------


def test_the_script_itself_refuses_without_a_process_credential_whatever_a_dotenv_holds(tmp_path: Path) -> None:
    environment = {name: value for name, value in os.environ.items() if name != "OPENAI_API_KEY"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "config"],
        env=environment,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=60,
    )
    assert result.returncode == 2
    assert "refused" in result.stderr
    assert result.stdout == ""
