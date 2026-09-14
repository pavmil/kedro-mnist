"""Пайплайн сбора данных: скачивание MNIST -> сплит -> подготовка."""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_mnist, preprocess, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_mnist,
                inputs=None,
                outputs="mnist_raw",
                name="download_mnist",
            ),
            node(
                func=split_data,
                inputs=["mnist_raw", "params:split"],
                outputs=["mnist_train_raw", "mnist_test_raw"],
                name="split_data",
            ),
            node(
                func=preprocess,
                inputs=["mnist_train_raw", "params:preprocess"],
                outputs=["X_train", "y_train"],
                name="preprocess_train",
            ),
            node(
                func=preprocess,
                inputs=["mnist_test_raw", "params:preprocess"],
                outputs=["X_test", "y_test"],
                name="preprocess_test",
            ),
        ]
    )
