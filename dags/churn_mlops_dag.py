"""
churn_mlops_dag.py
------------------
Phase 3 - Airflow DAG

Defines the full MLOps pipeline DAG:
  Extract → Transform → Train → Validate (Recall ≥ 0.80) → Deploy

Schedule: Monthly (for billing cycle batch prediction)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import logging
import os

import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

RECALL_THRESHOLD = 0.80
ARTIFACT_DIR = "/opt/airflow/data/processed"
MODEL_NAME = "telco-churn-voting-classifier"

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ─── Task Functions ────────────────────────────────────────────────────────────

def extract(**context):
    """Load raw data, clean it, save to disk, push path via XCom."""
    logger.info("=== EXTRACT: Loading raw data ===")
    from data_ingestion import ingest

    df = ingest()

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, "cleaned_data.pkl")
    df.to_pickle(path)

    logger.info(f"Cleaned data ({df.shape[0]} rows) saved to: {path}")
    context["ti"].xcom_push(key="cleaned_data_path", value=path)


def transform(**context):
    """Preprocess cleaned data: encode, scale, split. Save artifacts to disk."""
    logger.info("=== TRANSFORM: Preprocessing ===")
    from preprocessing import preprocess

    path = context["ti"].xcom_pull(task_ids="extract", key="cleaned_data_path")
    if not path:
        raise ValueError("Missing XCom key 'cleaned_data_path' from extract task")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned data artifact not found: {path}")

    df = pd.read_pickle(path)

    X_train, X_test, y_train, y_test, _ = preprocess(df)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    train_path = os.path.join(ARTIFACT_DIR, "train_data.pkl")
    test_path = os.path.join(ARTIFACT_DIR, "test_data.pkl")

    joblib.dump({"X_train": X_train, "y_train": y_train}, train_path)
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)

    logger.info(f"Train artifacts saved to: {train_path}")
    logger.info(f"Test  artifacts saved to: {test_path}")

    context["ti"].xcom_push(key="train_data_path", value=train_path)
    context["ti"].xcom_push(key="test_data_path", value=test_path)


def train_model(**context):
    """Train Voting Classifier, log to MLflow, push metrics via XCom."""
    logger.info("=== TRAIN: Training Voting Classifier ===")
    from train import train as train_fn

    train_path = context["ti"].xcom_pull(task_ids="transform", key="train_data_path")
    test_path = context["ti"].xcom_pull(task_ids="transform", key="test_data_path")
    if not train_path or not test_path:
        raise ValueError("Missing train/test artifact paths from transform task XCom")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train artifact not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test artifact not found: {test_path}")

    train_data = joblib.load(train_path)
    test_data = joblib.load(test_path)

    results = train_fn(
        train_data["X_train"], train_data["y_train"],
        test_data["X_test"], test_data["y_test"],
    )

    context["ti"].xcom_push(key="recall", value=float(results["recall"]))
    context["ti"].xcom_push(key="run_id", value=results["run_id"])
    context["ti"].xcom_push(key="model_path", value=results["model_path"])

    logger.info(
        f"Training complete — Recall: {results['recall']:.4f} | "
        f"Run ID: {results['run_id']}"
    )


def validate(**context):
    """Check if Recall >= threshold. Branch accordingly."""
    recall = context["ti"].xcom_pull(task_ids="train", key="recall")
    if recall is None:
        logger.error("No recall score received from training task.")
        return "validation_failed"
    recall = float(recall)
    logger.info(f"Recall score: {recall:.4f} | Threshold: {RECALL_THRESHOLD}")
    if recall >= RECALL_THRESHOLD:
        return "deploy"
    logger.warning(f"Recall {recall:.4f} below threshold — skipping deploy.")
    return "validation_failed"


def deploy(**context):
    """Promote model to 'Production' stage in MLflow Model Registry."""
    logger.info("=== DEPLOY: Promoting model to Production ===")
    run_id = context["ti"].xcom_pull(task_ids="train", key="run_id")
    if not run_id:
        raise ValueError("Missing XCom key 'run_id' from train task")

    client = MlflowClient()

    # Find the model version created in this run
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    target = next((v for v in versions if v.run_id == run_id), None)
    if target is None:
        raise ValueError(f"No model version found for run_id={run_id}")

    # Archive any existing Production versions first
    for v in versions:
        if v.current_stage == "Production" and v.version != target.version:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=v.version,
                stage="Archived",
            )
            logger.info(f"Archived previous Production version: {v.version}")

    # Promote the new version to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=target.version,
        stage="Production",
    )
    logger.info(f"Model version {target.version} promoted to Production.")


# ─── DAG Definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="churn_mlops_pipeline",
    default_args=default_args,
    description="Monthly Telco Churn Prediction MLOps Pipeline",
    schedule="0 0 1 * *",   # 1st of every month
    catchup=False,
    tags=["mlops", "churn", "telco"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract",
        python_callable=extract,
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=train_model,
    )

    validate_task = BranchPythonOperator(
        task_id="validate",
        python_callable=validate,
    )

    deploy_task = PythonOperator(
        task_id="deploy",
        python_callable=deploy,
    )

    validation_failed = EmptyOperator(task_id="validation_failed")
    pipeline_end = EmptyOperator(task_id="pipeline_end", trigger_rule="none_failed_min_one_success")

    # ─── Task Dependencies ──────────────────────────────────────────────────────
    extract_task >> transform_task >> train_task >> validate_task
    validate_task >> [deploy_task, validation_failed]
    deploy_task >> pipeline_end
    validation_failed >> pipeline_end
