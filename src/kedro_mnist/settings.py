"""Настройки проекта. Только отличия от дефолтов Kedro."""
from kedro_mnist.hooks import MLflowGitHook

# kedro-mlflow подключается сам как плагин; наш хук дописывает git-коммит в ран
HOOKS = (MLflowGitHook(),)

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}
