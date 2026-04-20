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
import sys
import os

sys.path.insert(0, "/opt/airflow/src")

logger = logging.getLogger(__name__)

RECALL_THRESHOLD = 0.80

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
    """Load raw data and push path to XCom."""
    logger.info("=== EXTRACT: Loading raw data ===")
    # TODO Phase 3: from data_ingestion import load_raw_data
    raise NotImplementedError("Phase 3 implementation pending")


def transform(**context):
    """Preprocess data."""
    logger.info("=== TRANSFORM: Preprocessing ===")
    # TODO Phase 3: from preprocessing import preprocess
    raise NotImplementedError("Phase 3 implementation pending")


def train_model(**context):
    """Train model and push Recall score to XCom."""
    logger.info("=== TRAIN: Training Voting Classifier ===")
    # TODO Phase 3: from train import train
    raise NotImplementedError("Phase 3 implementation pending")


def validate(**context):
    """Check if Recall >= threshold. Branch accordingly."""
    recall = context["ti"].xcom_pull(task_ids="train", key="recall")
    logger.info(f"Recall score: {recall} | Threshold: {RECALL_THRESHOLD}")
    if recall >= RECALL_THRESHOLD:
        return "deploy"
    return "validation_failed"


def deploy(**context):
    """Promote model to 'Production' in MLflow Model Registry."""
    logger.info("=== DEPLOY: Promoting model to Production ===")
    # TODO Phase 3: promote via MLflow client
    raise NotImplementedError("Phase 3 implementation pending")


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
