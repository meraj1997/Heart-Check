"""Inference helpers (placeholders)."""
from .models import Model


def predict(model_path: str, input_data):
    model = Model.load(model_path)
    return model.predict(input_data)
