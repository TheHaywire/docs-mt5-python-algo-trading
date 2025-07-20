"""
MLflow Experiment Tracking Example
Logs parameters, metrics, and artifacts for reproducible research
"""

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Generate dummy data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("random_state", 42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and log metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    
    # Save and log model artifact
    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")
    
    print(f"Logged run with accuracy: {acc:.4f}") 