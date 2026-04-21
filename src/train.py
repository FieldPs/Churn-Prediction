"""
train.py
--------
Phase 2 — Train a Voting Classifier (Ensemble) and log everything to MLflow.

Model: VotingClassifier (soft voting)
  - GradientBoostingClassifier
  - LogisticRegression
  - AdaBoostClassifier

Logged to MLflow:
  - Parameters: model type, test_size, random_state
  - Metrics: Accuracy, Recall (primary), Precision, F1
  - Artifacts: trained model (.pkl)
  - MLflow Model: registered as "telco-churn-voting-classifier"
"""

import logging
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import (
    VotingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
)

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "telco-churn-prediction"
MODEL_NAME = "telco-churn-voting-classifier"
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "voting_classifier.pkl")

RECALL_THRESHOLD = 0.80
DECISION_THRESHOLD = 0.45  # Tuned to achieve Recall >= 0.80


def build_voting_classifier() -> VotingClassifier:
    """Build the soft-voting ensemble (mirrors notebook Cell 26)."""
    clf1 = GradientBoostingClassifier(random_state=42)
    clf2 = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf3 = AdaBoostClassifier(random_state=42)
    model = VotingClassifier(
        estimators=[("gbc", clf1), ("lr", clf2), ("abc", clf3)],
        voting="soft",
    )
    return model


def compute_metrics(y_true, y_pred) -> dict:
    """Compute all classification metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
    }


def train(X_train, y_train, X_test, y_test) -> dict:
    """
    Train the Voting Classifier, log to MLflow, save model artifact.

    Returns:
        dict with metrics and the local model path.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        # ── Apply SMOTE ────────────────────────────────────────────────────
        if SMOTE is not None:
            logger.info("Applying SMOTE to handle class imbalance...")
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            logger.warning("imblearn not installed. Skipping SMOTE.")
            X_train_res, y_train_res = X_train, y_train

        # ── Build & Train ──────────────────────────────────────────────────
        model = build_voting_classifier()
        logger.info("Training VotingClassifier (GBC + LR + AdaBoost)...")
        model.fit(X_train_res, y_train_res)

        # ── Predict & Evaluate ─────────────────────────────────────────────
        # Use predict_proba for threshold tuning
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
        metrics = compute_metrics(y_test, predictions)

        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Recall:   {metrics['recall']:.4f}  (threshold: {RECALL_THRESHOLD})")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info("\n" + classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))

        # ── Log Parameters ─────────────────────────────────────────────────
        mlflow.log_params({
            "model_type": "VotingClassifier",
            "voting":     "soft",
            "estimators": "GradientBoosting, LogisticRegression, AdaBoost",
            "test_size":  0.30,
            "random_state": 40,
            "smote_applied": SMOTE is not None,
            "decision_threshold": DECISION_THRESHOLD,
        })

        # ── Log Metrics ────────────────────────────────────────────────────
        mlflow.log_metrics(metrics)

        # ── Save & Log Model Artifact ──────────────────────────────────────
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        mlflow.log_artifact(MODEL_OUTPUT_PATH, artifact_path="model_pkl")

        logger.info(f"Model saved to: {MODEL_OUTPUT_PATH}")

        # ── Tag run with pass/fail ─────────────────────────────────────────
        passed = metrics["recall"] >= RECALL_THRESHOLD
        mlflow.set_tag("recall_threshold_passed", str(passed))
        mlflow.set_tag("run_type", "training")

    return {**metrics, "model_path": MODEL_OUTPUT_PATH, "run_id": run.info.run_id}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_ingestion import ingest
    from preprocessing import preprocess

    df = ingest()
    X_train, X_test, y_train, y_test, _ = preprocess(df)
    results = train(X_train, y_train, X_test, y_test)
    print("\n=== Training Complete ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
