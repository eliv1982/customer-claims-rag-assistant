"""The CI workflow states the repository's test contract; these checks keep it from drifting.

They read ``.github/workflows/ci.yml`` as data and pin the properties other documents and the
audit depend on (which suite runs where, that the real-tokenizer lane cannot fall back to the
stand-in, that the default lane runs in a fresh virtual environment isolated from the runner's own
packages, that no credential, deployment or upload exists). They do not restate every step.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
import venv
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

from tests.tokenizer_lanes import REAL_LANE_COMMAND

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PROVISION_SCRIPT = ROOT / "scripts" / "provision_tiktoken_cache.py"
PROVISION_COMMAND = "python scripts/provision_tiktoken_cache.py"
INSTALL_COMMAND = 'python -m pip install -e ".[dev]"'
PIP_CHECK_COMMAND = "python -m pip check"
VENV_CREATE_STEP = "[venv] create"
VENV_VERIFY_STEP = "[venv] verify"
RUNNER_TEMP_EXPRESSION = "${{ runner.temp }}"

DEFAULT_JOB = "default-suite"
REAL_JOB = "real-tokenizer"


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _jobs(workflow: dict) -> dict[str, dict]:
    return workflow["jobs"]


def _steps(job: dict) -> list[dict]:
    return job["steps"]


def _commands(job: dict) -> list[str]:
    return [step["run"].strip() for step in _steps(job) if "run" in step]


def _step_index(job: dict, predicate) -> int:
    for index, step in enumerate(_steps(job)):
        if predicate(step):
            return index
    raise AssertionError("no step matches")


def _environments(workflow: dict) -> list[dict]:
    envs = [workflow.get("env") or {}]
    for job in _jobs(workflow).values():
        envs.append(job.get("env") or {})
        envs.extend(step.get("env") or {} for step in _steps(job))
    return envs


def test_workflow_runs_on_push_to_main_and_on_pull_requests(workflow: dict) -> None:
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare ``on`` as True
    assert "pull_request" in triggers
    assert triggers["push"]["branches"] == ["main"]
    # These two would run workflow code from a fork with the base repository's privileges.
    assert "pull_request_target" not in triggers and "workflow_run" not in triggers
    assert set(_jobs(workflow)) == {DEFAULT_JOB, REAL_JOB}


def test_permissions_are_read_only_contents_everywhere(workflow: dict) -> None:
    assert workflow["permissions"] == {"contents": "read"}
    for name, job in _jobs(workflow).items():
        assert "permissions" not in job, f"job {name} widens the workflow permissions"
        assert "environment" not in job, f"job {name} targets a deployment environment"


def test_nothing_uses_secrets_credentials_deployments_or_uploads(workflow: dict) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "secrets." not in text and "${{ secrets" not in text
    for env in _environments(workflow):
        assert not [name for name in env if name.upper().startswith(("OPENAI", "ANTHROPIC"))]
    for step in (step for job in _jobs(workflow).values() for step in _steps(job)):
        action = step.get("uses", "")
        assert "upload-artifact" not in action and "deploy" not in action
        assert "login" not in action and "publish" not in action


def test_every_checkout_drops_the_token_and_every_action_is_official_and_versioned(workflow: dict) -> None:
    uses = [step["uses"] for job in _jobs(workflow).values() for step in _steps(job) if "uses" in step]
    assert uses
    for reference in uses:
        assert re.fullmatch(r"actions/[\w.-]+@(v\d+|[0-9a-f]{40})", reference), reference
    for job in _jobs(workflow).values():
        for step in _steps(job):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False


def test_python_is_the_projects_minimum_supported_version(workflow: dict) -> None:
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
    floor = re.fullmatch(r">=(\d+\.\d+)", requires)
    assert floor, f"unexpected requires-python {requires!r}"
    versions = {
        str(step["with"]["python-version"])
        for job in _jobs(workflow).values()
        for step in _steps(job)
        if step.get("uses", "").startswith("actions/setup-python@")
    }
    assert versions == {floor.group(1)}


@pytest.mark.parametrize("job_name", [DEFAULT_JOB, REAL_JOB])
def test_each_job_installs_only_the_declared_dev_environment_then_checks_it(workflow: dict, job_name: str) -> None:
    commands = _commands(_jobs(workflow)[job_name])
    assert INSTALL_COMMAND in commands
    assert commands.index(PIP_CHECK_COMMAND) == commands.index(INSTALL_COMMAND) + 1
    for command in commands:
        if "pip install" in command:
            assert command == INSTALL_COMMAND, "undeclared packages must not be installed ad hoc"
        assert "playwright" not in command and "screenshots" not in command


def _named_step(job: dict, prefix: str) -> dict:
    matches = [step for step in _steps(job) if step.get("name", "").startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one step named {prefix!r}"
    return matches[0]


def test_default_suite_builds_its_own_environment_under_runner_temp_before_installing(workflow: dict) -> None:
    job = _jobs(workflow)[DEFAULT_JOB]
    create = _named_step(job, VENV_CREATE_STEP)
    verify = _named_step(job, VENV_VERIFY_STEP)

    order = [
        _step_index(job, lambda step: step.get("uses", "").startswith("actions/setup-python@")),
        _step_index(job, lambda step: step is create),
        _step_index(job, lambda step: step is verify),
        _step_index(job, lambda step: step.get("run", "").strip() == INSTALL_COMMAND),
        _step_index(job, lambda step: step.get("run", "").strip() == PIP_CHECK_COMMAND),
        _step_index(job, lambda step: "[hermetic] precondition" in step.get("name", "")),
        _step_index(job, lambda step: "python -m pytest" in step.get("run", "")),
    ]
    assert order == sorted(set(order)), "venv before install, install before pip check, all before the suite runs"
    before_verify = [step for step in _steps(job)[: order[2]] if "run" in step]
    assert before_verify == [create], "nothing but the venv step may run code before the isolation check"

    for step in (create, verify):
        assert step["shell"] == "python"
    assert create["env"]["PROJECT_VENV"].startswith(RUNNER_TEMP_EXPRESSION), "outside the checkout, in runner.temp"
    assert "system_site_packages=False" in create["run"] and "clear=True" in create["run"]
    assert "GITHUB_PATH" in create["run"], "the venv's interpreter must lead PATH for the remaining steps"
    assert "GITHUB_ENV" in create["run"] and "VIRTUAL_ENV" in create["run"]


def test_default_suite_has_no_route_back_to_runner_global_packages(workflow: dict) -> None:
    job = _jobs(workflow)[DEFAULT_JOB]
    for command in _commands(job):
        for escape in ("--system-site-packages", "system_site_packages=True", "--user", "--break-system-packages",
                       "--target", "--prefix", "pythonLocation"):
            assert escape not in command, f"{escape} reaches outside the project environment"
    for env in _environments(workflow):
        for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PIP_USER", "PIP_TARGET", "PIP_PREFIX"):
            assert name not in env, f"{name} redirects Python away from the project environment"
    # The verification is not advisory: it must stop the job and must hold only bootstrap packages.
    verify = _named_step(job, VENV_VERIFY_STEP)["run"]
    assert "raise SystemExit" in verify
    assert '("pip", "setuptools", "wheel")' in verify and "include-system-site-packages" in verify
    assert "RUNNER_TEMP" in verify and "VIRTUAL_ENV" in verify


class IsolatedEnvironment(NamedTuple):
    runner_temp: Path
    venv: Path
    interpreter: Path
    github_path: Path
    github_env: Path
    verify_script: Path
    base_env: dict[str, str]


@pytest.fixture(scope="module")
def isolated_environment(workflow: dict, tmp_path_factory: pytest.TempPathFactory) -> IsolatedEnvironment:
    """Run the workflow's own venv-creation step the way a runner would, in a scratch runner.temp."""
    job = _jobs(workflow)[DEFAULT_JOB]
    create = _named_step(job, VENV_CREATE_STEP)
    verify = _named_step(job, VENV_VERIFY_STEP)
    work = tmp_path_factory.mktemp("runner")
    runner_temp = work / "_temp"
    runner_temp.mkdir()
    github_path = work / "github_path"
    github_env = work / "github_env"
    create_script = work / "create_venv.py"
    verify_script = work / "verify_venv.py"
    create_script.write_text(create["run"], encoding="utf-8")
    verify_script.write_text(verify["run"], encoding="utf-8")
    for command_file in (github_path, github_env):
        command_file.write_text("", encoding="utf-8")

    base_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV"}
    }
    base_env.update(RUNNER_TEMP=str(runner_temp), GITHUB_PATH=str(github_path), GITHUB_ENV=str(github_env))
    venv_dir = Path(create["env"]["PROJECT_VENV"].replace(RUNNER_TEMP_EXPRESSION, str(runner_temp)))
    result = subprocess.run(
        [sys.executable, str(create_script)],
        env={**base_env, "PROJECT_VENV": str(venv_dir)},
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")
    return IsolatedEnvironment(
        runner_temp, venv_dir, scripts / ("python.exe" if os.name == "nt" else "python"),
        github_path, github_env, verify_script, base_env,
    )


def _run_verify(
    state: IsolatedEnvironment, interpreter: Path | str, **overrides: str
) -> subprocess.CompletedProcess[str]:
    env = {**state.base_env, "VIRTUAL_ENV": str(state.venv), **overrides}
    return subprocess.run(
        [str(interpreter), str(state.verify_script)],
        env=env,
        cwd=state.runner_temp,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_venv_step_creates_an_activatable_environment_outside_the_checkout(
    isolated_environment: IsolatedEnvironment,
) -> None:
    state = isolated_environment
    assert state.interpreter.is_file()
    assert state.venv.is_relative_to(state.runner_temp) and not state.venv.is_relative_to(ROOT)
    settings = dict(
        (key.strip(), value.strip())
        for key, _, value in (line.partition("=") for line in (state.venv / "pyvenv.cfg").read_text(encoding="utf-8").splitlines())
    )
    assert settings["include-system-site-packages"] == "false"
    # What later steps inherit: the venv's script directory is put on PATH and VIRTUAL_ENV is exported.
    assert state.github_path.read_text(encoding="utf-8").splitlines() == [str(state.interpreter.parent)]
    assert state.github_env.read_text(encoding="utf-8").splitlines() == [f"VIRTUAL_ENV={state.venv}"]


def test_isolation_check_accepts_the_fresh_environment_and_lists_only_bootstrap_packages(
    isolated_environment: IsolatedEnvironment,
) -> None:
    result = _run_verify(isolated_environment, isolated_environment.interpreter)
    assert result.returncode == 0, result.stdout + result.stderr
    visible = result.stdout.split("visible packages: ")[1].strip().split(", ")
    assert "pip" in visible and set(visible) <= {"pip", "setuptools", "wheel"}


@pytest.mark.parametrize(
    "case, expected",
    [
        ("runner_interpreter", "not a virtual environment"),
        ("outside_runner_temp", "is not under RUNNER_TEMP"),
        ("wrong_virtual_env", "VIRTUAL_ENV"),
        ("system_site_packages", "can see the runner's system site-packages"),
        ("leaked_package", "packages visible before any install: leaky-pkg"),
    ],
)
def test_isolation_check_stops_the_job_when_the_environment_is_not_isolated(
    isolated_environment: IsolatedEnvironment, tmp_path: Path, case: str, expected: str
) -> None:
    state = isolated_environment
    interpreter: Path | str = state.interpreter
    overrides: dict[str, str] = {}
    if case == "runner_interpreter":
        interpreter = getattr(sys, "_base_executable", sys.executable)  # the interpreter a venv is built from
    elif case == "outside_runner_temp":
        overrides["RUNNER_TEMP"] = str(tmp_path)
    elif case == "wrong_virtual_env":
        overrides["VIRTUAL_ENV"] = str(tmp_path)
    elif case == "system_site_packages":
        shared = state.runner_temp / "shared-venv"
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(shared)
        interpreter = shared / state.interpreter.relative_to(state.venv)
        overrides["VIRTUAL_ENV"] = str(shared)
    else:
        dist_info = tmp_path / "leaky_pkg-1.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: leaky-pkg\nVersion: 1.0\n", encoding="utf-8")
        overrides["PYTHONPATH"] = str(tmp_path)

    result = _run_verify(state, interpreter, **overrides)
    assert result.returncode != 0, result.stdout
    assert "project environment is not isolated" in result.stderr
    assert expected in result.stderr, result.stderr


def test_default_suite_runs_the_whole_offline_suite_on_linux_and_windows(workflow: dict) -> None:
    job = _jobs(workflow)[DEFAULT_JOB]
    assert {"ubuntu-latest", "windows-latest"} <= set(job["strategy"]["matrix"]["os"])
    assert job["strategy"]["fail-fast"] is False  # one platform's failure must not hide the other's
    pytest_steps = [step for step in _steps(job) if "python -m pytest" in step.get("run", "")]
    assert len(pytest_steps) == 1
    command = pytest_steps[0]["run"].strip()
    assert command.startswith("python -m pytest")
    arguments = command.split()[3:]
    assert "no:cacheprovider" in " ".join(arguments)
    for argument in arguments:
        assert argument not in {"--real-tiktoken", "-m", "-k", "-x", "--lf"}, f"the default lane must run everything: {argument}"
        assert not argument.startswith(("--ignore", "--deselect", "--maxfail", "--co")), argument
    env = pytest_steps[0]["env"]
    assert "runner.temp" in env["TIKTOKEN_CACHE_DIR"]  # an empty, unprovisioned cache directory
    assert not any("provision" in step.get("run", "") for step in _steps(job))


def test_real_tokenizer_lane_runs_the_supported_command_after_a_separate_provisioning_step(workflow: dict) -> None:
    job = _jobs(workflow)[REAL_JOB]
    assert job["runs-on"] == "ubuntu-latest"
    cache_dir = job["env"]["TIKTOKEN_CACHE_DIR"]
    assert cache_dir.startswith("/"), "the provisioned cache directory must be absolute and fixed"

    provision = _step_index(job, lambda step: step.get("run", "").strip() == PROVISION_COMMAND)
    lane = _step_index(
        job,
        lambda step: REAL_LANE_COMMAND in step.get("run", "") and "negative control" not in step.get("name", ""),
    )
    assert provision < lane, "the vocabulary must be provisioned before the lane runs"
    assert "--real-tiktoken" in REAL_LANE_COMMAND and "-m real_tiktoken" in REAL_LANE_COMMAND

    lane_step = _steps(job)[lane]
    assert PROVISION_COMMAND not in lane_step["run"], "the lane step itself must not provision"
    assert "TIKTOKEN_CACHE_DIR" not in lane_step.get("env", {}), "the lane reads the provisioned cache"


def test_real_tokenizer_job_proves_the_lane_cannot_run_on_an_empty_cache(workflow: dict) -> None:
    job = _jobs(workflow)[REAL_JOB]
    controls = [step for step in _steps(job) if "negative control" in step.get("name", "")]
    assert len(controls) == 1
    control = controls[0]
    assert REAL_LANE_COMMAND in control["run"] and "-ne 4" in control["run"]
    assert "runner.temp" in control["env"]["TIKTOKEN_CACHE_DIR"]  # not the provisioned directory


@pytest.mark.parametrize("job_name", [DEFAULT_JOB, REAL_JOB])
def test_test_execution_is_shielded_from_the_network_beyond_the_in_process_guard(workflow: dict, job_name: str) -> None:
    job = _jobs(workflow)[job_name]
    test_steps = [step for step in _steps(job) if "python -m pytest" in step.get("run", "")]
    assert test_steps
    for step in test_steps:
        if "negative control" in step.get("name", ""):
            continue
        for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            assert step["env"][variable].startswith("http://127.0.0.1:"), "must be a dead local address"
        assert "127.0.0.1" in step["env"]["NO_PROXY"]  # loopback stays reachable for local test servers


@pytest.mark.parametrize("setting", [None, "", "relative/cache"])
def test_provisioning_script_refuses_an_unknown_cache_location(setting: str | None, tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "TIKTOKEN_CACHE_DIR"}
    if setting is not None:
        env["TIKTOKEN_CACHE_DIR"] = setting
    result = subprocess.run(
        [sys.executable, str(PROVISION_SCRIPT)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "TIKTOKEN_CACHE_DIR must be set to an absolute path" in result.stderr
    assert not any(tmp_path.iterdir()), "nothing may be created for a rejected setting"
