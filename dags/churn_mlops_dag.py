"""
churn_mlops_dag.py
------------------
Phase 3 - Airflow DAG

Defines the full MLOps pipeline DAG:
  Extract → Transform → Train → Validate (Recall ≥ 0.80) → Deploy

Schedule: Monthly (for billing cycle batch prediction)

Runtime contract:
  - PYTHONPATH must include /opt/airflow/src (set in docker-compose)
  - /opt/airflow/data/processed and /opt/airflow/models must be writable
  - MLFLOW_TRACKING_URI must point to the MLflow server
"""

import sys
import os
import logging

# ── Path safety: ensure /opt/airflow/src is importable regardless of env ──
_SRC_DIR = "/opt/airflow/src"
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────
RECALL_THRESHOLD = 0.80
AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
ARTIFACT_DIR = os.path.join(AIRFLOW_HOME, "data", "processed")
RAW_DATA_DIR = os.path.join(AIRFLOW_HOME, "data", "raw")
MODEL_NAME = "telco-churn-voting-classifier"

# Default raw CSV path — matches src/data_ingestion.py default
DEFAULT_RAW_CSV = os.path.join(
    RAW_DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

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

    # Guard: verify raw data file exists before attempting ingestion
    if not os.path.exists(DEFAULT_RAW_CSV):
        raise FileNotFoundError(
            f"Raw data CSV not found at {DEFAULT_RAW_CSV}. "
            "Ensure DVC pull or data mount is configured."
        )

    df = ingest()

    # Guard: sanity check on output
    if df.empty:
        raise ValueError("Ingest returned an empty DataFrame — aborting pipeline.")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, "cleaned_data.pkl")
    df.to_pickle(path)

    logger.info(f"Cleaned data ({df.shape[0]} rows, {df.shape[1]} cols) saved to: {path}")
    context["ti"].xcom_push(key="cleaned_data_path", value=path)


def transform(**context):
    """Preprocess cleaned data: encode, scale, split. Save artifacts to disk."""
    logger.info("=== TRANSFORM: Preprocessing ===")
    from preprocessing import preprocess

    # Guard: pull and validate XCom value
    path = context["ti"].xcom_pull(task_ids="extract", key="cleaned_data_path")
    if not path:
        raise ValueError("Missing XCom key 'cleaned_data_path' from extract task")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned data artifact not found: {path}")

    df = pd.read_pickle(path)
    logger.info(f"Loaded cleaned data: {df.shape}")

    X_train, X_test, y_train, y_test, _ = preprocess(df)

    # Guard: verify split output shapes
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError(
            f"Train/test split returned empty arrays — "
            f"X_train: {X_train.shape}, X_test: {X_test.shape}"
        )

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    train_path = os.path.join(ARTIFACT_DIR, "train_data.pkl")
    test_path = os.path.join(ARTIFACT_DIR, "test_data.pkl")

    joblib.dump({"X_train": X_train, "y_train": y_train}, train_path)
    joblib.dump({"X_test": X_test, "y_test": y_test}, test_path)

    logger.info(f"Train artifacts saved to: {train_path} (shape: {X_train.shape})")
    logger.info(f"Test  artifacts saved to: {test_path} (shape: {X_test.shape})")

    context["ti"].xcom_push(key="train_data_path", value=train_path)
    context["ti"].xcom_push(key="test_data_path", value=test_path)


def train_model(**context):
    """Train Voting Classifier, log to MLflow, push metrics via XCom."""
    logger.info("=== TRAIN: Training Voting Classifier ===")
    from train import train as train_fn

    # Guard: pull and validate XCom values
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

    # Guard: verify critical return values
    if "recall" not in results:
        raise ValueError("train() did not return 'recall' key in results dict")
    if "run_id" not in results:
        raise ValueError("train() did not return 'run_id' key in results dict")

    recall_val = float(results["recall"])
    context["ti"].xcom_push(key="recall", value=recall_val)
    context["ti"].xcom_push(key="run_id", value=results["run_id"])
    context["ti"].xcom_push(key="model_path", value=results["model_path"])

    logger.info(
        f"Training complete — Recall: {recall_val:.4f} | "
        f"Run ID: {results['run_id']}"
    )


def validate(**context):
    """Check if Recall > currently deployed Production model. Branch accordingly."""
    recall = context["ti"].xcom_pull(task_ids="train_model", key="recall")
    if recall is None:
        logger.error("No recall score received from training task.")
        return "validation_failed"
    recall = float(recall)
    
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    prod_recall = 0.0
    has_prod_model = False
    
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        prod_version = next((v for v in versions if v.current_stage == "Production"), None)
        if prod_version:
            has_prod_model = True
            run_id = prod_version.run_id
            prod_run = client.get_run(run_id)
            prod_recall = prod_run.data.metrics.get("recall", 0.0)
            logger.info(f"Current Production model (v{prod_version.version}) Recall: {prod_recall:.4f}")
        else:
            logger.info("No Production model found. Falling back to default RECALL_THRESHOLD.")
            prod_recall = RECALL_THRESHOLD
    except Exception as e:
        logger.warning(f"Could not fetch Production model metrics: {e}. Falling back to default RECALL_THRESHOLD.")
        prod_recall = RECALL_THRESHOLD
        
    logger.info(f"New model Recall score: {recall:.4f} | Target to beat: {prod_recall:.4f}")
    
    if has_prod_model:
        if recall >= prod_recall:
            logger.info("New model is better than Production! Proceeding to deploy.")
            return "deploy"
        else:
            logger.warning(f"New model Recall ({recall:.4f}) is not better than Production ({prod_recall:.4f}) — skipping deploy.")
            return "validation_failed"
    else:
        if recall >= prod_recall:
            logger.info("New model meets default threshold! Proceeding to deploy.")
            return "deploy"
        else:
            logger.warning(f"New model Recall ({recall:.4f}) below default threshold ({prod_recall:.4f}) — skipping deploy.")
            return "validation_failed"


def deploy(**context):
    """Promote model to 'Production' stage in MLflow Model Registry."""
    logger.info("=== DEPLOY: Promoting model to Production ===")
    run_id = context["ti"].xcom_pull(task_ids="train_model", key="run_id")
    if not run_id:
        raise ValueError("Missing XCom key 'run_id' from train_model task")

    client = MlflowClient()

    # Find the model version created in this run
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    target = next((v for v in versions if v.run_id == run_id), None)
    if target is None:
        raise ValueError(
            f"No model version found for run_id={run_id} "
            f"in registered model '{MODEL_NAME}'. "
            "Ensure the training step completed and logged the model to MLflow."
        )

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
    
    import requests
    try:
        # ใช้ service name ของ Docker Compose เพื่อให้ใช้ได้ทั้ง Linux และ macOS
        reload_url = "http://backend:8000/reload-model"
        response = requests.post(reload_url, timeout=10)
        logger.info(f"Notification sent to Backend: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not notify Backend: {e}")


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
        task_id="train_model",
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