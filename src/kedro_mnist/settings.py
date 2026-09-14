"""Настройки проекта. Меняем только то, что отличается от дефолтов Kedro.
См. https://docs.kedro.org/en/stable/configure/configuration_basics/"""
from kedro_mnist.hooks import MLflowGitHook

# Проектные хуки. kedro-mlflow подключается автоматически как плагин;
# наш хук доописывает в ран git-коммит (и вызывается уже при активном ране).
HOOKS = (MLflowGitHook(),)

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}
