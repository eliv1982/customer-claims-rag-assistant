"""The CI workflow states the repository's test contract; these checks keep it from drifting.

They read ``.github/workflows/ci.yml`` as data and pin the properties other documents and the
audit depend on (which suite runs where, that the real-tokenizer lane cannot fall back to the
stand-in, that no credential, deployment or upload exists). They do not restate every step.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from tests.tokenizer_lanes import REAL_LANE_COMMAND

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PROVISION_SCRIPT = ROOT / "scripts" / "provision_tiktoken_cache.py"
PROVISION_COMMAND = "python scripts/provision_tiktoken_cache.py"
INSTALL_COMMAND = 'python -m pip install -e ".[dev]"'
PIP_CHECK_COMMAND = "python -m pip check"

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
    import os

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
