# kedro_mnist — пайплайн Kedro + MLflow на MNIST

Два пайплайна Kedro (сбор данных и обучение модели), управление зависимостями через poetry,
логирование в MLflow и строгая воспроизводимость.

Модель - XGBoost на пикселях MNIST (каждая картинка 28×28 = 784
признака)

---

## Структура

```
kedro_mnist/
├── conf/base/                          # общий конфиг проекта: коммитится в git и шарится между всеми
│   ├── catalog.yml                     # где какой датасет лежит
│   ├── parameters_data_collection.yml  # параметры сбора данных
│   ├── parameters_training.yml         # гиперпараметры модели
│   └── mlflow.yml                      # настройки MLflow (kedro-mlflow)
├── conf/local/                         # параметры под конкретную машину/юзера: креды, секретные ключи, IDE-настройки. Игнорируется гитом
├── data/                               # локальные артефакты (в .gitignore)
│   ├── 01_raw/            mnist_raw.parquet
│   ├── 02_intermediate/  train/test после сплита
│   ├── 05_model_input/   X_train/y_train/X_test/y_test
│   ├── 06_models/        xgb_model.pkl
│   └── 08_reporting/      metrics.json, confusion_matrix.png
├── src/kedro_mnist/
│   ├── pipelines/
│   │   ├── data_collection/   # скачать MNIST -> сплит -> подготовка
│   │   └── training/          # обучение XGBoost + метрики
│   ├── hooks.py               # логирование git-коммита в MLflow
│   └── settings.py
├── mlflow.db                  # трекинг-стор MLflow, sqlite (в .gitignore)
├── mlruns/                    # артефакты MLflow (в .gitignore)
├── check_reproducibility.py   # прогон обучения дважды + сверка метрик
└── pyproject.toml             # poetry конфиг
```

Два пайплайна (`kedro registry list`):
- **`data_collection`** — скачивает MNIST из OpenML (фикс. версия), делает
  каноничный сплит 60000/10000, нормализует пиксели, сохраняет все
  преобразованные датасеты локально в `data/`.
- **`training`** — обучает XGBoost, считает метрики (accuracy, macro-F1),
  рисует матрицу ошибок; всё логируется в MLflow.

---

## Установка

Нужен Python 3.10–3.12 и [poetry](https://python-poetry.org/).

```bash
cd kedro_mnist
poetry install
```

Poetry создаёт локальный `.venv` в папке проекта. Точные версии всех
пакетов зафиксированы в `poetry.lock` — это основа воспроизводимости.

---

## Запуск

```bash
# 1) собрать данные
poetry run kedro run --pipeline data_collection

# 2) обучить модель и посчитать метрики
poetry run kedro run --pipeline training

# или всё сразу (data_collection + training)
poetry run kedro run
```

Метрики после обучения:
- локально — `data/08_reporting/metrics.json` и `confusion_matrix.png`;
- в MLflow — параметры, метрики, модель и матрица ошибок как артефакты.

Посмотреть эксперименты:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
# затем открыть http://127.0.0.1:5000
```

---

## MLflow

Подключён через плагин **kedro-mlflow** (`conf/base/mlflow.yml`):
- локальный трекинг-стор — БД `mlflow.db` (sqlite) в корне проекта;
  артефакты (модель, матрица ошибок) — локально в `mlruns/`;
- параметры пайплайнов логируются автоматически (kedro-mlflow), метрики — из
  ноды `evaluate_model` через `mlflow.log_metrics`;
- модель и матрица ошибок логируются как артефакты (обёртка
  `MlflowArtifactDataset` в каталоге);
- **git-коммит запуска** пишется в теги рана (`git_commit`, `git_branch`,
  `git_dirty`) хуком `MLflowGitHook` — видно, на каком состоянии кода обучались.
  Если дерево «грязное» (есть незакоммиченные правки), тег `git_dirty=true`.

---

## Воспроизводимость

Перезапуск даёт те же метрики. Как это обеспечено:
- источник данных зафиксирован (`fetch_openml('mnist_784', version=1)`),
  порядок строк канонический, сплит без рандома;
- XGBoost: фиксированный `random_state`, `n_jobs=1`, без стохастического
  сэмплирования (`subsample=1.0`, `colsample_bytree=1.0`);
- все гиперпараметры — в `conf/base/parameters_training.yml`;
- окружение зафиксировано в `poetry.lock`.

Проверка (обучение дважды, сравнение метрик):

```bash
poetry run python check_reproducibility.py
```

---

## Как адаптировать под другую модель

1. Заменить ноды в `src/kedro_mnist/pipelines/training/nodes.py`
   (`train_model`, `evaluate_model`) на свою модель.
2. Поправить гиперпараметры в `conf/base/parameters_training.yml`.
3. При необходимости поменять источник данных в
   `pipelines/data_collection/nodes.py` и пути в `catalog.yml`.

Инфраструктура (poetry, MLflow, git-логирование, воспроизводимость) остаётся
без изменений.
