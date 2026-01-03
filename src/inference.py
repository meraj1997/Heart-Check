import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/model_rf.joblib"  # choose best from metrics

_model = None

def load_model(path: str = MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model

def predict_single(features: dict) -> dict:
    model = load_model()
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": float(proba)}
