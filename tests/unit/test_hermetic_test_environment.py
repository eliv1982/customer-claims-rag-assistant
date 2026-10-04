"""The default suite's own contract: offline, key-free, collectable either way."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from customer_claims_rag.token_counter import TiktokenCounter
from tests.network_guard import NetworkAccessBlocked, NetworkGuard
from tests.offline_tiktoken import OfflineEncoding

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean_subprocess_env() -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PYTEST_", "OPENAI_", "RAG_", "GENERATION_"))
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


# --- tokenizer ---------------------------------------------------------------


def test_default_tokenizer_is_the_offline_stand_in(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("--real-tiktoken"):
        pytest.skip("--real-tiktoken selects the real vocabulary on purpose")
    counter = TiktokenCounter()
    assert isinstance(counter._encoding, OfflineEncoding)
    assert counter.count("two words") == 2
    assert counter.count("") == 0


def test_offline_encoding_is_deterministic_and_cuts_long_words() -> None:
    encoding = OfflineEncoding()
    assert encoding.encode("Привет, мир!") == ["Привет", ",", "мир", "!"]
    assert encoding.encode("а" * 25) == ["а" * 12, "а" * 12, "а"]
    assert encoding.encode("same text") == encoding.encode("same text")


# --- environment -------------------------------------------------------------


def test_no_app_environment_leaks_into_tests() -> None:
    leaked = [
        name for name in os.environ if name.startswith(("OPENAI_", "RAG_", "GENERATION_"))
    ]
    assert leaked == []


def test_dummy_key_fixture_provides_a_fake_key(dummy_openai_api_key: str) -> None:
    assert os.environ["OPENAI_API_KEY"] == dummy_openai_api_key
    assert dummy_openai_api_key.startswith("sk-test-")


# --- network guard -----------------------------------------------------------


def test_guard_blocks_and_records_external_dns_lookup() -> None:
    guard = NetworkGuard()
    guard.install()
    try:
        with pytest.raises(NetworkAccessBlocked):
            socket.getaddrinfo("openaipublic.blob.core.windows.net", 443)
    finally:
        guard.uninstall()
    assert guard.attempts == ["getaddrinfo 'openaipublic.blob.core.windows.net'"]


def test_guard_blocks_external_ip_connect() -> None:
    guard = NetworkGuard()
    guard.install()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            with pytest.raises(NetworkAccessBlocked):
                client.connect(("93.184.216.34", 80))
    finally:
        guard.uninstall()
    assert guard.attempts == ["connect ('93.184.216.34', 80)"]


def test_guard_allows_loopback() -> None:
    guard = NetworkGuard()
    guard.install()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with socket.create_connection(server.getsockname(), timeout=5):
                pass
    finally:
        guard.uninstall()
    assert guard.attempts == []


def test_guard_uninstall_restores_socket_functions() -> None:
    before = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
        socket.gethostbyname,
        socket.gethostbyname_ex,
    )
    guard = NetworkGuard()
    guard.install()
    guard.uninstall()
    assert (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
        socket.gethostbyname,
        socket.gethostbyname_ex,
    ) == before


def test_swallowed_network_attempt_still_fails_the_session(tmp_path: Path) -> None:
    """Code under test that catches the error must not be able to hide the attempt."""
    (tmp_path / "conftest.py").write_text(
        'pytest_plugins = ["tests.conftest"]\n', encoding="utf-8"
    )
    (tmp_path / "test_probe.py").write_text(
        textwrap.dedent(
            """
            import socket

            def test_code_that_swallows_the_error():
                try:
                    socket.getaddrinfo("example.invalid", 80)
                except OSError:
                    pass  # e.g. a retry loop or a broad except in production code
            """
        ),
        encoding="utf-8",
    )
    env = _clean_subprocess_env()
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "src")])
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "1 passed" in completed.stdout
    assert "blocked network access" in completed.stdout
    assert "getaddrinfo 'example.invalid'" in completed.stdout
    assert completed.returncode == 1


def test_ordinary_child_python_inherits_dns_and_socket_guard_but_allows_loopback(
    tmp_path: Path,
) -> None:
    """Negative control for the old in-process-only implementation.

    A nested guarded pytest session launches an ordinary Python child with the environment it
    inherited from pytest. The child catches both failures, so only the shared child-attempt log can
    make the nested session fail. A real loopback connection in that same child remains permitted.
    """
    (tmp_path / "conftest.py").write_text(
        'pytest_plugins = ["tests.conftest"]\n', encoding="utf-8"
    )
    (tmp_path / "test_child_guard.py").write_text(
        textwrap.dedent(
            '''
            import socket
            import subprocess
            import sys

            def test_inherited_child_guard_and_loopback_allowance():
                probe = r"""
            import socket

            blocked = []
            try:
                socket.getaddrinfo("example.invalid", 443)
            except OSError as exc:
                blocked.append(type(exc).__name__)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as external:
                try:
                    external.connect(("93.184.216.34", 80))
                except OSError as exc:
                    blocked.append(type(exc).__name__)
            assert blocked == ["NetworkAccessBlocked", "NetworkAccessBlocked"]

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.bind(("127.0.0.1", 0))
                server.listen(1)
                with socket.create_connection(server.getsockname(), timeout=5):
                    pass
            print("child guard blocked DNS and raw socket; loopback allowed")
            """
                child = subprocess.run(
                    [sys.executable, "-c", probe],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert child.returncode == 0, child.stdout + child.stderr
                assert "child guard blocked DNS and raw socket; loopback allowed" in child.stdout
            '''
        ),
        encoding="utf-8",
    )
    env = _clean_subprocess_env()
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "src")])
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = completed.stdout + completed.stderr
    assert "1 passed" in output
    assert "blocked network access" in output
    assert "child getaddrinfo 'example.invalid'" in output
    assert "child connect ('93.184.216.34', 80)" in output
    assert completed.returncode == 1


# --- collection --------------------------------------------------------------


def _collected_node_ids(command: list[str]) -> list[str]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_clean_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return sorted(line for line in completed.stdout.splitlines() if "::" in line)


def test_plain_pytest_and_python_dash_m_pytest_collect_the_same_tests() -> None:
    """``tests.*`` imports must not depend on how pytest was launched.

    ``python -m pytest`` puts the current directory on ``sys.path``; the ``pytest``
    console script does not. Isolated mode (``-I``) reproduces the latter.
    """
    # a module that imports from ``tests.*`` and one from a subpackage
    targets = [
        "tests/unit/test_exact_evaluation_oracle.py",
        "tests/integration/test_evaluate_pool_expansion.py",
    ]
    flags = ["--collect-only", "-q", "-p", "no:cacheprovider", *targets]
    with_dash_m = _collected_node_ids([sys.executable, "-m", "pytest", *flags])
    like_console_script = _collected_node_ids(
        [sys.executable, "-I", "-B", "-c", "import sys, pytest; sys.exit(pytest.main(sys.argv[1:]))", *flags]
    )
    assert with_dash_m, "nothing collected"
    assert like_console_script == with_dash_m


# --- tokenizer lanes ---------------------------------------------------------
#
# Each probe below is a throwaway pytest session that loads this suite's conftest as a
# plugin, so the lane contract is checked end to end without touching the real tests.

_LANE_CONFTEST = '''
pytest_plugins = ["tests.conftest"]
{extra}

def pytest_configure(config):
    config.addinivalue_line("markers", "real_tiktoken: probe marker")
'''
_PROBE_TESTS = '''
import pytest

def test_plain():
    pass

@pytest.mark.real_tiktoken
def test_needs_real_vocabulary():
    pass
'''


def _run_probe_session(
    directory: Path,
    *pytest_args: str,
    tests: str = _PROBE_TESTS,
    conftest_extra: str = "",
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    (directory / "conftest.py").write_text(_LANE_CONFTEST.format(extra=conftest_extra), encoding="utf-8")
    (directory / "test_probe.py").write_text(tests, encoding="utf-8")
    env = _clean_subprocess_env()
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), str(PROJECT_ROOT / "src")])
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *pytest_args, str(directory)],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _empty_tiktoken_cache(tmp_path: Path) -> dict[str, str]:
    cache = tmp_path / "empty-tiktoken-cache"
    cache.mkdir()
    return {"TIKTOKEN_CACHE_DIR": str(cache), "DATA_GYM_CACHE_DIR": str(cache)}


def test_default_lane_states_that_production_tokenization_is_not_verified(tmp_path: Path) -> None:
    completed = _run_probe_session(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    header = [line for line in completed.stdout.splitlines() if line.startswith("tokenizer:")]
    assert len(header) == 1
    assert "offline substitute active" in header[0]
    assert "not cl100k_base" in header[0]
    assert "python -m pytest --real-tiktoken -m real_tiktoken" in header[0]


def test_default_lane_note_survives_quiet_mode_exactly_once(tmp_path: Path) -> None:
    # -q hides pytest's header, so the note must still reach the end of the output.
    completed = _run_probe_session(tmp_path, "-q")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("offline substitute active") == 1
    assert "1 passed, 1 skipped" in completed.stdout


def test_default_lane_skip_reason_points_at_the_real_lane(tmp_path: Path) -> None:
    completed = _run_probe_session(tmp_path, "-rs")
    assert "run the real-tokenizer lane: python -m pytest --real-tiktoken -m real_tiktoken" in completed.stdout


def test_selecting_only_real_tests_without_the_flag_is_refused_not_green(tmp_path: Path) -> None:
    completed = _run_probe_session(tmp_path, "-m", "real_tiktoken")
    assert completed.returncode == int(pytest.ExitCode.USAGE_ERROR)
    assert "real-tokenizer lane was not requested" in completed.stdout + completed.stderr
    listing = _run_probe_session(tmp_path, "-m", "real_tiktoken", "--collect-only", "-q")
    assert listing.returncode == 0  # listing the real tests stays possible


def test_real_lane_with_an_empty_cache_fails_at_setup_and_never_touches_the_network(
    tmp_path: Path,
) -> None:
    completed = _run_probe_session(
        tmp_path,
        "--real-tiktoken",
        "-m",
        "real_tiktoken",
        env_overrides=_empty_tiktoken_cache(tmp_path),
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == int(pytest.ExitCode.USAGE_ERROR), output
    assert "Tests never download it" in output
    assert "Provision the cache once" in output
    assert "python -m pytest --real-tiktoken -m real_tiktoken" in output
    # nothing ran, nothing was skipped into a green result, nothing reached for the network
    assert "passed" not in output and "skipped" not in output
    assert "blocked network access" not in output
    assert not list((tmp_path / "empty-tiktoken-cache").iterdir())


def test_real_lane_without_cache_does_not_fall_back_to_the_stand_in_for_plain_tests(
    tmp_path: Path,
) -> None:
    # --real-tiktoken applies to the whole session: no cache means no run, not a quiet downgrade
    completed = _run_probe_session(tmp_path, "--real-tiktoken", env_overrides=_empty_tiktoken_cache(tmp_path))
    assert completed.returncode == int(pytest.ExitCode.USAGE_ERROR)
    assert "passed" not in completed.stdout


def test_a_real_tiktoken_test_that_skips_in_the_real_lane_is_a_failure(tmp_path: Path) -> None:
    tests = '''
import pytest

@pytest.mark.real_tiktoken
@pytest.mark.skip(reason="gap in coverage")
def test_would_hide_a_gap():
    pass
'''
    completed = _run_probe_session(
        tmp_path,
        "--real-tiktoken",
        "-m",
        "real_tiktoken",
        tests=tests,
        # pretend the cache is provisioned: this probe is about skipping, not provisioning
        conftest_extra="import tests.conftest as lane\nlane.require_provisioned_cl100k = lambda: None",
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, output
    assert "1 error" in output  # a skip is reported during setup, so it surfaces as a setup error
    assert "skipped in the real-tokenizer lane" in output and "gap in coverage" in output
    assert "1 skipped" not in output


def test_real_lane_header_names_the_real_vocabulary(tmp_path: Path) -> None:
    completed = _run_probe_session(
        tmp_path,
        "--real-tiktoken",
        "-m",
        "not no_such_marker",
        tests=_PROBE_TESTS.replace("def test_needs_real_vocabulary", "def test_real"),
        conftest_extra="import tests.conftest as lane\nlane.require_provisioned_cl100k = lambda: None",
    )
    assert "tokenizer: real cl100k_base from the local tiktoken cache" in completed.stdout
    assert "offline substitute" not in completed.stdout
