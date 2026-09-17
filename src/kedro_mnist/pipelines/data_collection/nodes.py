"""Ноды сбора данных: скачивание и подготовка MNIST."""
import pandas as pd
from sklearn.datasets import fetch_openml


def download_mnist() -> pd.DataFrame:
    # version=1 фиксирует датасет -> одинаковые данные при каждом запуске
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=True)
    return X.assign(label=y.astype("int64"))


def split_data(mnist_raw: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    # каноничный сплит MNIST по позиции строк: первые 60k train, последние 10k test
    n_test = params["n_test"]
    train = mnist_raw.iloc[:-n_test].reset_index(drop=True)
    test = mnist_raw.iloc[-n_test:].reset_index(drop=True)
    return train, test


def preprocess(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Пиксели -> float32 в [0, 1], метка отдельным датафреймом."""
    label = params["label_col"]
    y = df[[label]].astype("int64")
    X = df.drop(columns=[label]).astype("float32") / params["scale"]
    return X, y
