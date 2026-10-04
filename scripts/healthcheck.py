"""Container health probe: succeeds only when Streamlit's own health endpoint answers ``ok``.

Standard library only (the runtime image has no curl) and local only: it talks to the loopback
address inside the container and never goes through a proxy configured for outbound traffic.

Health is not release readiness. A container whose production index is missing or stale never
reaches this probe because the entrypoint refuses to start the server (docs/08_docker_runbook.md).
"""

from __future__ import annotations

import os
import sys
import urllib.request
from collections.abc import Mapping

HEALTH_PATH = "/_stcore/health"
DEFAULT_PORT = "8501"
TIMEOUT_SECONDS = 5


def health_url(environ: Mapping[str, str] = os.environ) -> str:
    port = environ.get("STREAMLIT_SERVER_PORT") or DEFAULT_PORT
    return f"http://127.0.0.1:{port}{HEALTH_PATH}"


def probe(url: str, *, timeout: float = TIMEOUT_SECONDS) -> bool:
    # An empty ProxyHandler ignores HTTP_PROXY/NO_PROXY: loopback must never be sent to a proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            return response.status == 200 and response.read(16).strip() == b"ok"
    except Exception:  # refused, timed out, non-2xx, malformed: all mean "not healthy"
        return False


def main() -> int:
    return 0 if probe(health_url()) else 1


if __name__ == "__main__":
    sys.exit(main())
