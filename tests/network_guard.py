"""Block outbound network access for the whole test session.

The README promises that automated tests run offline. This makes the promise
enforceable: any non-loopback DNS lookup or socket connect raises
``NetworkAccessBlocked`` and is also recorded, so an attempt that production
code swallows (for example a retry loop) still fails the session at the end.
"""

from __future__ import annotations

import socket
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

_LOOPBACK_HOSTS = frozenset({"", "localhost", "::1", "0.0.0.0"})
_CHILD_GUARD_ENABLED = "CUSTOMER_CLAIMS_TEST_NETWORK_GUARD"
_CHILD_GUARD_LOG = "CUSTOMER_CLAIMS_TEST_NETWORK_GUARD_LOG"


class NetworkAccessBlocked(ConnectionError):
    """Raised for an outbound network attempt during the test session."""


def _is_loopback(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    text = str(host)
    return text in _LOOPBACK_HOSTS or text.startswith("127.")


class NetworkGuard:
    """Patches ``socket`` so only loopback traffic is possible."""

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._originals: dict[str, Callable] | None = None

    def install(self) -> None:
        if self._originals is not None:
            return
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        original_getaddrinfo = socket.getaddrinfo
        original_gethostbyname = socket.gethostbyname
        original_gethostbyname_ex = socket.gethostbyname_ex
        self._originals = {
            "connect": original_connect,
            "connect_ex": original_connect_ex,
            "getaddrinfo": original_getaddrinfo,
            "gethostbyname": original_gethostbyname,
            "gethostbyname_ex": original_gethostbyname_ex,
        }

        def block(kind: str, target: object) -> NetworkAccessBlocked:
            self.attempts.append(f"{kind} {target!r}")
            return NetworkAccessBlocked(f"outbound network access is disabled in tests: {kind} {target!r}")

        def connect(sock: socket.socket, address: object) -> None:
            if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):  # type: ignore[index]
                raise block("connect", address)
            return original_connect(sock, address)

        def connect_ex(sock: socket.socket, address: object) -> int:
            if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):  # type: ignore[index]
                raise block("connect_ex", address)
            return original_connect_ex(sock, address)

        def getaddrinfo(host: object, *args: object, **kwargs: object):
            if not _is_loopback(host):
                raise block("getaddrinfo", host)
            return original_getaddrinfo(host, *args, **kwargs)

        def gethostbyname(host: str) -> str:
            if not _is_loopback(host):
                raise block("gethostbyname", host)
            return original_gethostbyname(host)

        def gethostbyname_ex(host: str):
            if not _is_loopback(host):
                raise block("gethostbyname_ex", host)
            return original_gethostbyname_ex(host)

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
        socket.gethostbyname = gethostbyname  # type: ignore[assignment]
        socket.gethostbyname_ex = gethostbyname_ex  # type: ignore[assignment]

    def uninstall(self) -> None:
        if self._originals is None:
            return
        socket.socket.connect = self._originals["connect"]  # type: ignore[method-assign]
        socket.socket.connect_ex = self._originals["connect_ex"]  # type: ignore[method-assign]
        socket.getaddrinfo = self._originals["getaddrinfo"]  # type: ignore[assignment]
        socket.gethostbyname = self._originals["gethostbyname"]  # type: ignore[assignment]
        socket.gethostbyname_ex = self._originals["gethostbyname_ex"]  # type: ignore[assignment]
        self._originals = None


class PythonSubprocessNetworkGuard:
    """Arrange for ordinary child Python interpreters to install the socket guard at startup."""

    def __init__(self) -> None:
        self._log_path: Path | None = None
        self._saved_environment: dict[str, str | None] | None = None

    def install(self) -> None:
        if self._saved_environment is not None:
            return
        guard_dir = Path(__file__).resolve().parent / "subprocess_guard"
        descriptor, log_name = tempfile.mkstemp(prefix="customer-claims-network-", suffix=".log")
        os.close(descriptor)
        self._log_path = Path(log_name)
        self._saved_environment = {
            _CHILD_GUARD_ENABLED: os.environ.get(_CHILD_GUARD_ENABLED),
            _CHILD_GUARD_LOG: os.environ.get(_CHILD_GUARD_LOG),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        }
        existing = os.environ.get("PYTHONPATH", "")
        paths = [str(guard_dir), *[part for part in existing.split(os.pathsep) if part]]
        os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
        os.environ[_CHILD_GUARD_ENABLED] = "1"
        os.environ[_CHILD_GUARD_LOG] = str(self._log_path)

    @property
    def attempts(self) -> list[str]:
        if self._log_path is None or not self._log_path.is_file():
            return []
        return [line for line in self._log_path.read_text(encoding="utf-8").splitlines() if line]

    def uninstall(self) -> None:
        if self._saved_environment is None:
            return
        for name, value in self._saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved_environment = None
        if self._log_path is not None:
            self._log_path.unlink(missing_ok=True)
            self._log_path = None
