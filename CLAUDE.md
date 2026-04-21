# Developer Guidelines for Telco Customer Churn Prediction

## Overview
This is an MLOps project for telecom customer churn prediction utilizing a production-ready pipeline powered by Airflow, MLflow, DVC, and Docker.

## Project Structure
- `dags/` - Apache Airflow DAG definitions.
- `src/` - Core ML pipeline scripts (`data_ingestion.py`, `preprocessing.py`, `train.py`, `evaluate.py`).
- `tests/` - Unit tests (run via `pytest`).
- `data/raw/` - Raw datasets pulled via DVC from DAGsHub.
- `models/` - Saved model artifacts.

## Technology Stack
- **Languages:** Python
- **Machine Learning:** Scikit-Learn (Voting Classifier: GBC, LR, AdaBoost)
- **MLOps:** MLflow (tracking, model registry), Airflow (orchestration), DVC (data versioning)
- **Infrastructure:** Docker & Docker Compose

## Coding Standards & Best Practices
- **Modularity:** Ensure ML logic is decoupled from orchestration. Airflow DAGs should call modular Python functions from the `src/` directory.
- **Logging:** Use `logging` module to log process steps. Log model parameters, metrics (Accuracy, Recall, Precision, F1), and artifacts using MLflow.
- **Paths:** Use reliable relative paths (e.g., `os.path.join(os.path.dirname(__file__), ...)`). Airflow container mounts `src/` to `/opt/airflow/src`.
- **Target Metric:** Optimize for **Recall (≥0.80)** to minimize undetected churners.

## Common Commands
- **Run local pipeline manually:** `python src/train.py`
- **Start Docker services:** `docker compose up -d`
- **Stop Docker services:** `docker compose down`
- **Pull data via DVC:** `dvc pull`
- **Run tests:** `pytest tests/`

## Current Implementation Status
- Basic project structure and Docker setup are present.
- ML pipeline scripts in `src/` are implemented.
- Airflow DAG `churn_mlops_dag.py` structure exists but currently has `NotImplementedError` placeholders for its tasks.
