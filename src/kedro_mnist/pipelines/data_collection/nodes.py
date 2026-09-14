"""Ноды пайплайна сбора данных (MNIST)."""
import pandas as pd
from sklearn.datasets import fetch_openml


def download_mnist() -> pd.DataFrame:
    """Скачиваем MNIST (784 пикселя + метка) из OpenML.

    Версия датасета зафиксирована (version=1), порядок строк канонический,
    поэтому выгрузка воспроизводима: одинаковые данные при каждом запуске.
    """
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=True,
        parser="auto",
    )
    df = X.copy()
    df["label"] = y.astype("int64")
    return df


def split_data(mnist_raw: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Каноничный сплит MNIST: первые 60000 — train, последние 10000 — test.

    Без рандома: сплит детерминированный по позиции строк.
    """
    n_test = params["n_test"]
    train_df = mnist_raw.iloc[:-n_test].reset_index(drop=True)
    test_df = mnist_raw.iloc[-n_test:].reset_index(drop=True)
    return train_df, test_df


def preprocess(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Лёгкая подготовка: пиксели -> float32 и деление на 255, метка отдельно."""
    label_col = params["label_col"]
    scale = params["scale"]
    y = df[[label_col]].astype("int64")
    X = df.drop(columns=[label_col]).astype("float32") / scale
    return X, y
