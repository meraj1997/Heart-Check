from fastapi import FastAPI
from pydantic import BaseModel
import logging
from datetime import datetime
from src.inference import predict_single

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Heart Disease API")

# Simple metrics counter
metrics = {
    "total_requests": 0,
    "predictions_positive": 0,
    "predictions_negative": 0,
    "start_time": datetime.now().isoformat()
}

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
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
def get_metrics():
    """Expose application metrics"""
    uptime = (datetime.now() - datetime.fromisoformat(metrics["start_time"])).total_seconds()
    return {
        **metrics,
        "uptime_seconds": uptime
    }

@app.post("/predict")
def predict(features: HeartFeatures):
    data = features.dict()
    logger.info(f"[REQUEST] Received prediction request: {data}")
    
    try:
        result = predict_single(data)
        metrics["total_requests"] += 1
        
        if result["prediction"] == 1:
            metrics["predictions_positive"] += 1
        else:
            metrics["predictions_negative"] += 1
            
        logger.info(f"[RESPONSE] Prediction: {result['prediction']}, Probability: {result['probability']:.3f}")
        return result
    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {str(e)}")
        raise
