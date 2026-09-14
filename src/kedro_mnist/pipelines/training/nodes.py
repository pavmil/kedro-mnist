"""Ноды пайплайна обучения (XGBoost на пикселях MNIST)."""
import matplotlib

matplotlib.use("Agg")  # без GUI, чтобы рисовать в файл
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict) -> XGBClassifier:
    """Обучаем XGBoost-классификатор на 10 классов.

    Для воспроизводимости: фиксированный random_state и n_jobs=1
    (многопоточность в XGBoost может давать чуть разные результаты).
    """
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        tree_method=params["tree_method"],
        objective="multi:softprob",
        num_class=10,
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
        verbosity=0,
    )
    model.fit(X_train, y_train.values.ravel())
    return model


def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> tuple[dict, dict, plt.Figure]:
    """Считаем метрики и рисуем матрицу ошибок.

    Возвращаем:
    - metrics_mlflow: в формате для kedro-mlflow (name -> {value, step});
    - metrics_report: плоский dict для локального json;
    - fig: матрица ошибок (png).
    """
    y_true = y_test.values.ravel()
    preds = model.predict(X_test)

    accuracy = float(accuracy_score(y_true, preds))
    f1_macro = float(f1_score(y_true, preds, average="macro"))

    metrics_report = {"accuracy": accuracy, "f1_macro": f1_macro}
    metrics_mlflow = {
        "accuracy": {"value": accuracy, "step": 0},
        "f1_macro": {"value": f1_macro, "step": 0},
    }

    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return metrics_mlflow, metrics_report, fig
