"""
train.py
--------
Phase 2 - Modularize ML Pipeline

Trains a Voting Classifier (Ensemble) and logs:
  - Parameters (model config)
  - Metrics (Recall, F1, Accuracy)
  - Artifacts (trained model .pkl)
to MLflow Tracking Server.

TODO: Full implementation — to be refactored from notebook.
"""

import logging
import os
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "telco-churn-prediction"
MODEL_NAME = "telco-churn-voting-classifier"


def train(X_train, y_train, X_test, y_test):
    """Train Voting Classifier and log to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # TODO: Full implementation — refactor from notebook
        raise NotImplementedError("Full implementation in Phase 2")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Train script placeholder — Phase 2 implementation pending.")
