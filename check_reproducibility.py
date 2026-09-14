"""Проверка воспроизводимости: обучаем дважды и сравниваем метрики.

Запуск: poetry run python check_reproducibility.py
"""
import json
import subprocess
import sys
from pathlib import Path

METRICS_PATH = Path("data/08_reporting/metrics.json")


def run_training() -> dict:
    subprocess.run(
        [sys.executable, "-m", "kedro", "run", "--pipeline", "training"],
        check=True,
    )
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    print(">>> Прогон 1")
    m1 = run_training()
    print(">>> Прогон 2")
    m2 = run_training()

    print("\nметрики прогон 1:", m1)
    print("метрики прогон 2:", m2)

    if m1 == m2:
        print("\nOK: метрики совпали побитово — воспроизводимо.")
    else:
        print("\nВНИМАНИЕ: метрики отличаются!")
        sys.exit(1)


if __name__ == "__main__":
    main()
