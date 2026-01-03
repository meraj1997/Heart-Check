from src.train import train_and_log
from src.inference import predict_single

def test_training_runs():
    model_path, metrics = train_and_log("logreg")
    assert "accuracy" in metrics
    assert metrics["accuracy"] > 0.5

def test_inference_output():
    sample = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    result = predict_single(sample)
    assert "prediction" in result
    assert "probability" in result
