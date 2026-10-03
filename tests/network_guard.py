"""Block outbound network access for the whole test session.

The README promises that automated tests run offline. This makes the promise
enforceable: any non-loopback DNS lookup or socket connect raises
``NetworkAccessBlocked`` and is also recorded, so an attempt that production
code swallows (for example a retry loop) still fails the session at the end.
"""

from __future__ import annotations

import socket
from collections.abc import Callable

_LOOPBACK_HOSTS = frozenset({"", "localhost", "::1", "0.0.0.0"})


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
        self._originals = {
            "connect": original_connect,
            "connect_ex": original_connect_ex,
            "getaddrinfo": original_getaddrinfo,
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

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = getaddrinfo  # type: ignore[assignment]

    def uninstall(self) -> None:
        if self._originals is None:
            return
        socket.socket.connect = self._originals["connect"]  # type: ignore[method-assign]
        socket.socket.connect_ex = self._originals["connect_ex"]  # type: ignore[method-assign]
        socket.getaddrinfo = self._originals["getaddrinfo"]  # type: ignore[assignment]
        self._originals = None
