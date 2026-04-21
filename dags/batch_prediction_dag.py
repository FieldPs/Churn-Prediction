"""
batch_prediction_dag.py
-----------------------
Monthly Batch Prediction DAG for Telco Churn

Executes the batch_predict.py script to generate predictions for current customers.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Path safety: ensure /opt/airflow/src is importable regardless of env ──
_SRC_DIR = "/opt/airflow/src"
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

logger = logging.getLogger(__name__)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def run_batch_prediction(**context):
    """Execute the batch prediction script."""
    logger.info("=== BATCH PREDICTION START ===")
    from batch_predict import batch_predict
    
    try:
        result_df = batch_predict()
        logger.info(f"Batch prediction completed successfully. Processed {len(result_df)} records.")
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise e

with DAG(
    dag_id="monthly_churn_batch_prediction",
    default_args=default_args,
    description="Monthly Telco Churn Batch Prediction",
    schedule="0 0 1 * *",   # 1st of every month
    catchup=False,
    tags=["mlops", "churn", "telco", "inference"],
) as dag:

    predict_task = PythonOperator(
        task_id="run_batch_prediction",
        python_callable=run_batch_prediction,
    )

    predict_task
