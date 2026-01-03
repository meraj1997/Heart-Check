import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from .data_processing import load_raw, clean_data, train_test_split_data

mlflow.set_experiment("heart_mlops")

def build_pipeline(model):
    # Assuming all features except target are numeric in heart dataset
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", model),
    ])

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    return metrics

def train_and_log(model_name="logreg"):
    df_raw = load_raw()
    df_clean = clean_data(df_raw)
    X_train, X_test, y_train, y_test = train_test_split_data(df_clean)

    if model_name == "logreg":
        base_model = LogisticRegression(max_iter=1000)
    elif model_name == "rf":
        base_model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError("Unknown model_name")

    pipe = build_pipeline(base_model)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        os.makedirs("models", exist_ok=True)
        model_path = f"models/model_{model_name}.joblib"
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(model_path)
        return model_path, metrics

if __name__ == "__main__":
    m1_path, m1_metrics = train_and_log("logreg")
    m2_path, m2_metrics = train_and_log("rf")
    print("LogReg:", m1_metrics)
    print("RF:", m2_metrics)
