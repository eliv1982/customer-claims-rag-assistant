#!/usr/bin/env python3
"""Supported launcher for the release Compose stack (compose.yaml).

    python scripts/release_compose.py up --build
    python scripts/release_compose.py config
    python scripts/release_compose.py run --rm streamlit validate-release-posture

Why it exists: Compose reads a project ``.env`` for variable interpolation by default, so a stale
repository ``.env`` could silently provide the paid OpenAI credential of a release container. This
launcher makes the credential an intentional act of the operator's shell:

* ``OPENAI_API_KEY`` must be set, non-empty and free of whitespace in the real process environment;
  a repository ``.env`` is never read for its value (its mere presence only improves the message);
* Compose is told to ignore the project ``.env`` (``--env-file`` an empty file) and the compose file
  is pinned, so no other environment source or override file is picked up;
* the key travels to Compose in the environment of the child process, never on a command line, and
  is not printed. ``config`` prints the resolved model, so its output is redacted.

compose.yaml refuses to resolve without the marker this launcher sets, so a bare ``docker compose``
stops with an instruction instead of using ``.env``. Standard library only. Exit status: 2 for a
refusal by this launcher, otherwise Compose's own.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "compose.yaml"

CREDENTIAL_ENV = "OPENAI_API_KEY"
LAUNCHER_MARKER_ENV = "RELEASE_COMPOSE_LAUNCHER"
BIND_ADDRESS_ENV = "RAG_BIND_ADDRESS"
INDEX_HOST_PATH_ENV = "RAG_ACTIVE_INDEX_HOST_PATH"
DEFAULT_INDEX_HOST_PATH = "./data/04_index_production"

# Commands that mount the index. compose.yaml does not create a missing host directory (an empty
# directory is not an index, and Docker would create it as root), so Compose alone would fail with
# a bare "bind source path does not exist". The check below says what to do instead.
INDEX_MOUNTING_COMMANDS = frozenset({"up", "run", "create"})
# Kept equal to release.posture.build_instruction(active target) by a unit test.
PROVISIONING_COMMAND = (
    "python -m customer_claims_rag.cli.build_index "
    "--corpus-manifest configs/corpus/foodflow_production_v1.json "
    "--index-dir data/04_index_production --collection customer_claims "
    "--embedding-model text-embedding-3-small --rebuild"
)

EXIT_REFUSED = 2
EXIT_DOCKER_MISSING = 127

LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
# Compose variables that would add an environment or compose-file source of their own.
UNPINNED_COMPOSE_VARIABLES = ("COMPOSE_ENV_FILES", "COMPOSE_FILE", "COMPOSE_PATH_SEPARATOR")
REDACTED = "[REDACTED]"

_DOTENV_CREDENTIAL = re.compile(rf"^\s*(?:export\s+)?{CREDENTIAL_ENV}\s*=", re.MULTILINE)


def dotenv_defines_credential(project_root: Path) -> bool:
    """Whether a repository .env mentions the credential. Only the name is looked at, never kept."""
    try:
        return bool(_DOTENV_CREDENTIAL.search((project_root / ".env").read_text(encoding="utf-8")))
    except OSError:
        return False
    except UnicodeDecodeError:
        return False


def credential_problem(environ: Mapping[str, str], *, project_root: Path = ROOT) -> str | None:
    """Why the process environment cannot supply the credential, or None. Never includes the value."""
    value = environ.get(CREDENTIAL_ENV)
    if value is None:
        reason = f"{CREDENTIAL_ENV} is not set in the process environment"
    elif not value.strip():
        reason = f"{CREDENTIAL_ENV} is set but empty"
    elif value != value.strip() or any(ch.isspace() for ch in value):
        reason = f"{CREDENTIAL_ENV} contains whitespace (a paste error: re-export it without spaces or a newline)"
    else:
        return None
    message = (
        f"{reason}. The release stack takes its credential from your shell environment only; "
        f"export {CREDENTIAL_ENV} in the shell that runs this command."
    )
    if dotenv_defines_credential(project_root):
        message += (
            f" A repository .env defines {CREDENTIAL_ENV}, but it is deliberately ignored here "
            "(it may be stale)."
        )
    return message


def index_problem(
    environ: Mapping[str, str], command: str, *, project_root: Path = ROOT
) -> str | None:
    """Why the host side of the index mount cannot be used, or None. Existence only: whether the
    index is valid is decided by the container's full release validation, not here."""
    if command not in INDEX_MOUNTING_COMMANDS:
        return None
    configured = environ.get(INDEX_HOST_PATH_ENV) or DEFAULT_INDEX_HOST_PATH
    path = Path(configured)
    if not path.is_absolute():
        path = project_root / path  # Compose resolves relative sources against the project directory
    if path.is_dir():
        return None
    return (
        f"the production index directory does not exist: {path}. The index is a build artifact; "
        "Compose will not create it. Build it on the host first (docs/07_index_provisioning.md), "
        f"then start again: {PROVISIONING_COMMAND}"
    )


def compose_command(args: Sequence[str], *, env_file: Path, docker: str = "docker") -> list[str]:
    return [
        docker,
        "compose",
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(env_file),
        "-f",
        str(COMPOSE_FILE),
        *args,
    ]


def redact(text: str, secret: str) -> str:
    return text.replace(secret, REDACTED) if secret else text


def child_environment(environ: Mapping[str, str]) -> dict[str, str]:
    env = {name: value for name, value in environ.items() if name not in UNPINNED_COMPOSE_VARIABLES}
    env[LAUNCHER_MARKER_ENV] = "scripts/release_compose.py"
    return env


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
    project_root: Path = ROOT,
) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    environ = os.environ if environ is None else environ

    if not args or args[0].startswith("-"):
        print(
            "usage: python scripts/release_compose.py <compose command> [args...]   "
            "(for example: up --build | config | down)\n"
            "Compose's global options (-f, --env-file, --project-directory) are fixed by this launcher.",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    problem = credential_problem(environ, project_root=project_root)
    if problem is not None:
        print(f"release_compose: refused: {problem}", file=sys.stderr)
        return EXIT_REFUSED

    missing_index = index_problem(environ, args[0], project_root=project_root)
    if missing_index is not None:
        print(f"release_compose: refused: {missing_index}", file=sys.stderr)
        return EXIT_REFUSED

    docker = which("docker")
    if docker is None:
        print("release_compose: the docker command was not found on PATH", file=sys.stderr)
        return EXIT_DOCKER_MISSING

    bind_address = environ.get(BIND_ADDRESS_ENV, "").strip()
    if bind_address and bind_address not in LOOPBACK_ADDRESSES:
        print(
            f"release_compose: warning: {BIND_ADDRESS_ENV}={bind_address} publishes the UI beyond loopback. "
            "It has no authentication: put an authenticating reverse proxy or firewall in front of it.",
            file=sys.stderr,
        )

    print(
        f"release_compose: credential source=process environment ({CREDENTIAL_ENV}, value not shown); "
        "project .env is not used",
        file=sys.stderr,
    )

    descriptor, empty_env_name = tempfile.mkstemp(prefix="release-compose-empty-", suffix=".env")
    os.close(descriptor)
    try:
        command = compose_command(args, env_file=Path(empty_env_name), docker=docker)
        env = child_environment(environ)
        if args[0] == "config":
            # `config` prints the resolved environment, which contains the key in clear text.
            result = runner(command, env=env, capture_output=True, text=True)
            secret = environ[CREDENTIAL_ENV]
            sys.stdout.write(redact(result.stdout or "", secret))
            sys.stderr.write(redact(result.stderr or "", secret))
            return result.returncode
        return runner(command, env=env).returncode
    finally:
        try:
            os.unlink(empty_env_name)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
