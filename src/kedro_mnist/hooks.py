"""Проектные хуки Kedro."""
import logging
import subprocess
from pathlib import Path

import mlflow
from kedro.framework.hooks import hook_impl

logger = logging.getLogger(__name__)

# корень проекта: src/kedro_mnist/hooks.py -> ../../.. = папка репозитория
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True
    ).strip()


class MLflowGitHook:
    """Пишет в активный MLflow-ран коммит гита, ветку и флаг 'грязного' дерева.

    kedro-mlflow открывает ран в своём хуке before_pipeline_run; проектные хуки
    вызываются после плагиновых, поэтому здесь ран уже активен.
    """

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog) -> None:
        try:
            commit = _git("rev-parse", "HEAD")
            branch = _git("rev-parse", "--abbrev-ref", "HEAD")
            dirty = bool(_git("status", "--porcelain"))

            mlflow.set_tag("git_commit", commit)
            mlflow.set_tag("git_branch", branch)
            mlflow.set_tag("git_dirty", str(dirty).lower())

            if dirty:
                logger.warning(
                    "Git-дерево содержит незакоммиченные изменения (git_dirty=true): "
                    "запуск не воспроизводится по одному коммиту."
                )
            logger.info("Залогирован git-коммит в MLflow: %s (%s)", commit[:8], branch)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось залогировать git-инфо в MLflow: %s", exc)
