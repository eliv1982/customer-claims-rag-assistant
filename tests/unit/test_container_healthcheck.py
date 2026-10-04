"""The container health probe: standard library only, loopback only, strict about the answer.

It runs inside the image with nothing but the Python interpreter, and it is a liveness signal, not
a release gate (the entrypoint refuses to start the server for an invalid index).
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "healthcheck.py"
HEALTH_PATH = "/_stcore/health"


def _load():
    spec = importlib.util.spec_from_file_location("healthcheck_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


healthcheck = _load()


@contextmanager
def serving(status: int, body: bytes) -> Iterator[int]:
    """A loopback HTTP server answering ``status``/``body`` on the health path. Yields its port."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path != HEALTH_PATH:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _unused_port() -> int:
    with HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler) as probe:
        return probe.server_port


def _url(port: int) -> str:
    return f"http://127.0.0.1:{port}{HEALTH_PATH}"


def test_a_200_ok_answer_is_healthy() -> None:
    with serving(200, b"ok") as port:
        assert healthcheck.probe(_url(port)) is True


@pytest.mark.parametrize(
    ("status", "body"),
    [(200, b"not ok"), (200, b""), (503, b"ok"), (500, b"error")],
    ids=["wrong-body", "empty-body", "503", "500"],
)
def test_anything_other_than_a_200_ok_is_unhealthy(status: int, body: bytes) -> None:
    with serving(status, body) as port:
        assert healthcheck.probe(_url(port)) is False


def test_a_refused_connection_is_unhealthy() -> None:
    assert healthcheck.probe(_url(_unused_port()), timeout=2) is False


def test_the_probe_never_goes_through_a_configured_proxy(monkeypatch) -> None:
    """A container may be given HTTP(S)_PROXY for its outbound traffic; loopback must bypass it."""
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    with serving(200, b"ok") as port:
        assert healthcheck.probe(_url(port)) is True


def test_the_url_is_local_and_follows_the_configured_streamlit_port() -> None:
    assert healthcheck.health_url({}) == f"http://127.0.0.1:8501{HEALTH_PATH}"
    assert healthcheck.health_url({"STREAMLIT_SERVER_PORT": "9000"}) == f"http://127.0.0.1:9000{HEALTH_PATH}"
    assert healthcheck.health_url({"STREAMLIT_SERVER_PORT": ""}) == f"http://127.0.0.1:8501{HEALTH_PATH}"


def _clean_environment() -> dict[str, str]:
    """The current environment without any proxy variable (Windows needs SYSTEMROOT to open sockets)."""
    return {name: value for name, value in os.environ.items() if "proxy" not in name.lower()}


def _run_script(port: int, cwd: Path) -> int:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**_clean_environment(), "STREAMLIT_SERVER_PORT": str(port)},
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
    ).returncode


def test_the_script_exit_status_reflects_health(tmp_path: Path) -> None:
    assert _run_script(_unused_port(), tmp_path) == 1
    with serving(200, b"ok") as port:
        assert _run_script(port, tmp_path) == 0
    with serving(503, b"ok") as port:
        assert _run_script(port, tmp_path) == 1


def test_the_probe_uses_only_the_standard_library() -> None:
    """The runtime image has no curl and installs nothing for the health check."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    }
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)
