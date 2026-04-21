# AI Agent Workflow & Roadmap

## Project Context
The goal of this project is to build a production-ready MLOps pipeline for Telco Churn Prediction. 
The system trains a Voting Classifier and evaluates it on Recall (target threshold ≥ 0.80) to minimize the number of undetected churners.

## Agent Roles & Responsibilities
As an AI Agent, your role is to assist the developer in completing the roadmap outlined in `projectTRD.md`. You should focus on making the ML pipeline robust, production-ready, and well-orchestrated.

## Implementation Roadmap Status
**Phase 1: Environment & Infrastructure** (Mostly Complete)
- Docker Compose setup for Airflow & MLflow is done.
- Project structure (`dags`, `src`, `tests`, `data`) is initialized.
- DVC is initialized for tracking raw data.

**Phase 2: Modularizing the ML Pipeline** (Complete)
- `src/data_ingestion.py`, `src/preprocessing.py`, `src/train.py`, and `src/evaluate.py` have been implemented.

**Phase 3: Airflow DAG Development** (In Progress)
- **ACTION REQUIRED:** `dags/churn_mlops_dag.py` exists but its task functions (`extract`, `transform`, `train_model`, `deploy`) currently raise `NotImplementedError`. You need to wire these Airflow PythonOperator tasks to correctly call the module functions implemented in `/src/`.

**Phase 4: Monitoring & Retraining Logic** (Pending)
- **ACTION REQUIRED:** Implement a Drift Detection script (for feature drift, e.g., MonthlyCharges, and concept drift).
- **ACTION REQUIRED:** Configure automated retraining logic in the Airflow DAG based on drift detection or performance decay.

**Phase 5: GitHub Integration (CI/CD)** (Pending)
- **ACTION REQUIRED:** Setup GitHub Actions to run `pytest` on every push.
- **ACTION REQUIRED:** Create continuous deployment workflows to sync the `/dags` folder to the Airflow environment upon merging to `main`.

**Phase 6: Responsible AI & Testing** (Pending)
- **ACTION REQUIRED:** Implement Bias Checks (ensure demographic features like Gender are excluded or checked for fairness).
- **ACTION REQUIRED:** Improve robust logging for prediction scores in batch runs.

## Guidelines for Next Steps
1. Always review `projectTRD.md` to ensure your steps align with the project roadmap.
2. Keep Airflow code (`dags/`) and ML logic (`src/`) cleanly separated.
3. If asked to fix DAGs, check `/opt/airflow/src` path configurations.
4. When writing code, add sufficient inline comments explaining MLflow integrations and Airflow context handling (XComs).
