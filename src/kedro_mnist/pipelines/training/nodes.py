"""Ноды обучения: XGBoost на пикселях MNIST."""
import matplotlib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

matplotlib.use("Agg")  # headless: график пишем в файл, окно не нужно
import matplotlib.pyplot as plt  # noqa: E402


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict) -> XGBClassifier:
    # seed и n_jobs=1 берём из params -> результат воспроизводим при перезапуске
    model = XGBClassifier(objective="multi:softprob", num_class=10, verbosity=0, **params)
    model.fit(X_train, y_train.values.ravel())
    return model


def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.DataFrame
) -> tuple[dict, plt.Figure]:
    y_true = y_test.values.ravel()
    preds = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
    }
    if mlflow.active_run():
        mlflow.log_metrics(metrics)

    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(title="Матрица ошибок", xlabel="Предсказано", ylabel="Истинный класс",
           xticks=range(10), yticks=range(10))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return metrics, fig
