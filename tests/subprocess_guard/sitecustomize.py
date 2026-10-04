"""Test-only startup hook that blocks external sockets in inherited Python subprocesses.

Pytest prepends this directory to ``PYTHONPATH`` only while a guarded session is running. Python
automatically imports ``sitecustomize`` during ordinary interpreter startup, so child interpreters
receive the same boundary without modifying the system installation or the production package.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

_ENABLED = "CUSTOMER_CLAIMS_TEST_NETWORK_GUARD"
_LOG = "CUSTOMER_CLAIMS_TEST_NETWORK_GUARD_LOG"
_LOOPBACK_HOSTS = frozenset({"", "localhost", "::1", "0.0.0.0"})


class NetworkAccessBlocked(ConnectionError):
    """Raised when a guarded child Python process attempts external network access."""


def _is_loopback(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    text = str(host)
    return text in _LOOPBACK_HOSTS or text.startswith("127.")


def _record(kind: str, target: object) -> NetworkAccessBlocked:
    attempt = f"child {kind} {target!r}"
    log_name = os.environ.get(_LOG)
    if log_name:
        try:
            with Path(log_name).open("a", encoding="utf-8") as stream:
                stream.write(attempt + "\n")
        except OSError:
            pass
    return NetworkAccessBlocked(f"outbound network access is disabled in tests: {attempt}")


def _install() -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def connect(sock: socket.socket, address: object) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):  # type: ignore[index]
            raise _record("connect", address)
        return original_connect(sock, address)

    def connect_ex(sock: socket.socket, address: object) -> int:
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):  # type: ignore[index]
            raise _record("connect_ex", address)
        return original_connect_ex(sock, address)

    def getaddrinfo(host: object, *args: object, **kwargs: object):
        if not _is_loopback(host):
            raise _record("getaddrinfo", host)
        return original_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host: str) -> str:
        if not _is_loopback(host):
            raise _record("gethostbyname", host)
        return original_gethostbyname(host)

    def gethostbyname_ex(host: str):
        if not _is_loopback(host):
            raise _record("gethostbyname_ex", host)
        return original_gethostbyname_ex(host)

    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]
    socket.gethostbyname = gethostbyname  # type: ignore[assignment]
    socket.gethostbyname_ex = gethostbyname_ex  # type: ignore[assignment]


if os.environ.get(_ENABLED) == "1":
    _install()
