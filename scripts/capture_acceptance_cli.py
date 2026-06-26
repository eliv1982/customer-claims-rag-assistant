"""Capture manual acceptance CLI evidence (stage 5C)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = PROJECT_ROOT / "deliverables" / "evidence" / "cli"

SCENARIOS = {
    "M01": "Сколько обычно идет доставка и где посмотреть статус заказа?",
    "M02": "В заказе не оказалось одной оплаченной позиции. Что мне делать?",
    "M03": "Курьер привез заказ, и один контейнер был вскрыт. Я не хочу это есть.",
    "M04": "Заказ сильно опоздал. Я хочу вернуть деньги и понять, положена ли компенсация.",
    "M05": "После еды из заказа у меня сильно заболел живот и появилась тошнота.",
    "M06": "После еды стало трудно дышать, состояние ухудшается.",
    "M07": "Какой сегодня курс биткоина и стоит ли его покупать?",
}


def main() -> int:
    CLI_DIR.mkdir(parents=True, exist_ok=True)
    answer_claim = PROJECT_ROOT / ".venv" / "Scripts" / "answer-claim.exe"
    if not answer_claim.is_file():
        answer_claim = Path(sys.executable).with_name("answer-claim")

    env = dict(**{k: v for k, v in __import__("os").environ.items() if k != "OPENAI_API_KEY"})

    for scenario_id, message in SCENARIOS.items():
        result = subprocess.run(
            [str(answer_claim), "--message", message],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        stdout_path = CLI_DIR / f"{scenario_id}_answer.json"
        stderr_path = CLI_DIR / f"{scenario_id}_stderr.txt"
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        print(scenario_id, "exit", result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
