from fastapi import FastAPI
from pydantic import BaseModel
import logging
from src.inference import predict_single

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Heart Disease API")

class HeartFeatures(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HeartFeatures):
    data = features.dict()
    logger.info(f"Request: {data}")
    result = predict_single(data)
    logger.info(f"Response: {result}")
    return result
