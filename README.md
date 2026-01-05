# Heart Disease Prediction - MLOps Pipeline

Complete end-to-end machine learning pipeline for predicting heart disease risk using MLOps best practices.

## 🎯 Project Overview

- **Dataset**: UCI Heart Disease Dataset (303 patients, 13 features)
- **Models**: Logistic Regression & Random Forest
- **Best Model**: Random Forest (88% accuracy, 0.92 ROC-AUC)
- **Tech Stack**: Python, FastAPI, Docker, Kubernetes, MLflow, GitHub Actions

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop
- Kubernetes enabled

### Installation

1. Clone repository:

git clone https://github.com/meraj1997/Heart-Check.git
cd Heart-Check

2. Setup Environment

python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

3. Run Pipeline:

python -m src.download_data
python -m src.data_processing
python -m src.train
pytest

##  Features

✅ Automated data acquisition from UCI

✅ EDA with visualizations

✅ MLflow experiment tracking

✅ CI/CD with GitHub Actions

✅ Docker containerization

✅ Kubernetes deployment

✅ API monitoring & logging

## API Testing

### Local Testing
uvicorn api.main:app --reload

### Docker
docker build -t heart-diagnos:latest .
docker run -p 8000:8000 heart-diagnos:latest

### Kubernetes
kubectl apply -f deploy/

### Make Predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

## Project Structure

Heart-Check/
├── data/                  # Datasets
├── notebooks/             # EDA notebooks
├── src/                   # Source code
│   ├── download_data.py
│   ├── data_processing.py
│   ├── train.py
│   └── inference.py
├── api/                   # FastAPI application
├── tests/                 # Unit tests
├── deploy/                # Kubernetes manifests
├── .github/workflows/     # CI/CD
├── Dockerfile
├── requirements.txt
└── report.docx            # Complete documentation

## Testing

pytest -v

## Model Performance

| Model               | Accuracy | Precision | Recall | ROC-AUC |
| ------------------- | -------- | --------- | ------ | ------- |
| Logistic Regression | 0.85     | 0.83      | 0.82   | 0.89    |
| Random Forest       | 0.88     | 0.87      | 0.85   | 0.92    |

## Monitoring

Health: GET /health

Metrics: GET /metrics

Logs: kubectl logs <pod-name>
