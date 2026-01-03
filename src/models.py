"""Model definitions (placeholders).

Define model architecture and serialization helpers.
"""
from typing import Any


class Model:
    def __init__(self):
        pass

    def fit(self, X: Any, y: Any):
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    @staticmethod
    def load(path: str) -> "Model":
        raise NotImplementedError
