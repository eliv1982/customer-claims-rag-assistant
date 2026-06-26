"""Capture Streamlit UI screenshots for manual acceptance evidence."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "deliverables" / "evidence" / "ui"

SCENARIOS = {
    "ui_release_identity_empty_form": None,
    "ui_M01_delivery_faq": "Сколько обычно идет доставка и где посмотреть статус заказа?",
    "ui_M03_opened_container": (
        "Курьер привез заказ, и один контейнер был вскрыт. Я не хочу это есть."
    ),
    "ui_M04_refund_compensation": (
        "Заказ сильно опоздал. Я хочу вернуть деньги и понять, положена ли компенсация."
    ),
    "ui_M06_critical_escalation": (
        "После еды стало трудно дышать, состояние ухудшается."
    ),
    "ui_M07_out_of_scope": "Какой сегодня курс биткоина и стоит ли его покупать?",
}


def _submit_claim(page, message: str, *, base_url: str, timeout_ms: int) -> None:
    page.goto(base_url, wait_until="networkidle")
    page.get_by_label("Текст обращения").fill(message)
    page.get_by_role("button", name="Проанализировать обращение").click()
    page.wait_for_timeout(timeout_ms)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Streamlit acceptance screenshots.")
    parser.add_argument("--base-url", default="http://localhost:8501")
    parser.add_argument("--wait-ms", type=int, default=45000)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "playwright is required for UI screenshots: pip install playwright && playwright install chromium"
        ) from exc

    UI_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        for filename, message in SCENARIOS.items():
            target = UI_DIR / f"{filename}.png"
            if message is None:
                page.goto(args.base_url, wait_until="networkidle")
                page.wait_for_timeout(2000)
            else:
                _submit_claim(page, message, base_url=args.base_url, timeout_ms=args.wait_ms)
            page.screenshot(path=str(target), full_page=True)
            print("captured", target.name)
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
