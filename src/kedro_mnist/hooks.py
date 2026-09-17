"""Проектные хуки Kedro."""
import logging
import subprocess
from pathlib import Path

import mlflow
from kedro.framework.hooks import hook_impl

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


class MLflowGitHook:
    """Логирует git-коммит запуска в активный MLflow-ран.

    kedro-mlflow открывает ран раньше (плагиновые хуки идут до проектных),
    поэтому здесь ран уже активен.
    """

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog) -> None:
        try:
            dirty = bool(_git("status", "--porcelain"))
            mlflow.set_tag("git_commit", _git("rev-parse", "HEAD"))
            mlflow.set_tag("git_branch", _git("rev-parse", "--abbrev-ref", "HEAD"))
            mlflow.set_tag("git_dirty", str(dirty).lower())
            if dirty:
                logger.warning("git_dirty=true: есть незакоммиченные изменения.")
        except Exception as exc:  # git может быть недоступен — запуск не рушим
            logger.warning("git-инфо не записана в MLflow: %s", exc)
