"""Capture screenshots of the current Streamlit UI for the acceptance evidence.

Starts ``scripts/acceptance_ui_app.py`` (the real UI over the real release gate and production index,
deterministic stubs for the query embedding and the model reply, no credential, no network) on a local
port, drives it with headless Chromium and writes PNG files. Needs the optional tooling::

    python -m pip install -e ".[screenshots]"
    python -m playwright install chromium

Usually run through ``python scripts/capture_final_acceptance.py --ui``.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_UI_DIR = ROOT / "deliverables" / "evidence" / "ui"

# file name -> (scenario id, accordion to open before the screenshot)
SHOTS: dict[str, tuple[str | None, str | None]] = {
    "01_release_identity.png": (None, "Идентификация рабочего релиза"),
    "02_claim_with_sources.png": ("S01", "Основания и источники"),
    "03_critical_health_template.png": ("S04", "Техническая информация"),
    "04_unsafe_draft_replaced.png": ("S12", "Техническая информация"),
    "05_retrieval_unavailable.png": ("S10", "Техническая информация"),
    "06_unsupported_language.png": ("S03", "Найденные материалы (не подтверждают ответ)"),
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_healthy(url: str, *, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("the evidence UI did not become healthy in time")


def capture(
    ui_dir: Path = DEFAULT_UI_DIR, *, base_url: str | None = None, chromium_executable: str | None = None
) -> list[Path]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "playwright is required for UI screenshots: "
            'pip install -e ".[screenshots]" && python -m playwright install chromium'
        ) from exc
    sys.path.insert(0, str(SCRIPTS))
    import capture_final_acceptance as acceptance

    messages = {scenario.scenario_id: scenario.message for scenario in acceptance.SCENARIOS}
    server: subprocess.Popen | None = None
    if base_url is None:
        if os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("refusing to start the evidence UI with OPENAI_API_KEY in the environment")
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        server = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run", str(SCRIPTS / "acceptance_ui_app.py"),
                "--server.headless", "true", "--server.port", str(port), "--server.address", "127.0.0.1",
                "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    written: list[Path] = []
    try:
        _wait_healthy(base_url)
        ui_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            # An already installed Chromium can be named when the one matching this Playwright
            # version is not downloaded (PLAYWRIGHT_CHROMIUM_EXECUTABLE or --chromium-executable).
            executable = chromium_executable or os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
            browser = playwright.chromium.launch(headless=True, executable_path=executable or None)
            # Streamlit scrolls inside its own container, so ``full_page`` would not see the whole
            # result: use a tall viewport and clip to the rendered content.
            page = browser.new_page(viewport={"width": 1000, "height": 2400})
            for filename, (scenario_id, accordion) in SHOTS.items():
                page.goto(base_url, wait_until="networkidle")
                page.get_by_label("Текст обращения").wait_for(timeout=60000)
                if scenario_id is not None:
                    page.get_by_label("Текст обращения").fill(messages[scenario_id])
                    page.get_by_role("button", name="Проанализировать обращение").click()
                    page.get_by_text("Черновик ответа клиенту").wait_for(timeout=60000)
                page.get_by_text(accordion, exact=True).first.click()
                page.wait_for_timeout(800)
                target = ui_dir / filename
                content = page.locator('[data-testid="stMainBlockContainer"]').bounding_box()
                height = 2400 if content is None else min(2400, int(content["y"] + content["height"]) + 32)
                page.screenshot(path=str(target), clip={"x": 0, "y": 0, "width": 1000, "height": height})
                written.append(target)
                print("captured", filename)
            browser.close()
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture acceptance screenshots of the current UI.")
    parser.add_argument("--ui-dir", type=Path, default=DEFAULT_UI_DIR)
    parser.add_argument("--base-url", default=None, help="use an already running acceptance UI")
    parser.add_argument("--chromium-executable", default=None, help="path of an installed Chromium to use")
    args = parser.parse_args()
    capture(args.ui_dir.resolve(), base_url=args.base_url, chromium_executable=args.chromium_executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
