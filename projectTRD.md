Project Brief: MLOps for Telco Churn Prediction

Source : https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/notebook

1. Current Scope & Context

    Business Goal: Predict customer churn for a telecom company to enable targeted retention campaigns.

    ML Task: Binary Classification using a Voting Classifier (Ensemble).

    Primary Metric: Recall (Target: >0.80) to minimize undetected churners.

    Current Status: Offline evaluation is complete. A preprocessing pipeline is designed but needs to be modularized for production.

    Deployment Strategy: Primarily Batch Prediction for monthly billing cycles, with an optional REST API for individual lookups.

2. Technical Stack

    OS: Fedora/Ubuntu Linux.

    Orchestration: Apache Airflow (running on Docker).

    Tracking & Registry: MLflow.

    Version Control: Git (GitHub) with DVC (Data Version Control).

    CI/CD: GitHub Actions.

    Environment: Docker & Docker Compose.

3. Implementation Roadmap (Action Items for AI Agent)
Phase 1: Environment & Infrastructure

    [ ] Dockerize Airflow: Create a docker-compose.yaml to run Airflow (Webserver, Scheduler, Postgres) and an MLflow Tracking Server.

    [ ] Project Structure: Organize the repository into /dags, /src (scripts), /tests, and /data.

    [ ] DVC Setup: Initialize DVC to track the Kaggle Telco Churn dataset and model artifacts.

Phase 2: Modularizing the ML Pipeline

Refactor the existing Jupyter Notebook into production-ready Python scripts:

    [ ] data_ingestion.py: Script to load raw data from the source.

    [ ] preprocessing.py: Implement the Scikit-learn Pipeline (Handling missing values, Encoding, and Scaling) .

    [ ] train.py: Train the Voting Classifier, log parameters and metrics (Recall, F1) to MLflow, and save the model .

    [ ] evaluate.py: Perform Cross-group Evaluation and Churn Risk Score distribution analysis.

Phase 3: Airflow DAG Development

    [ ] Create churn_mlops_dag.py: Define a DAG with the following task sequence:

        Extract: Run data_ingestion.py.

        Transform: Run preprocessing.py.

        Train: Run train.py.

        Validate: Check if Recall≥0.80.

        Deploy: If validation passes, move the model to the "Production" stage in the MLflow Model Registry.

Phase 4: Monitoring & Retraining Logic

    [ ] Drift Detection: Implement a script to detect Feature Drift (e.g., changes in MonthlyCharges) and Concept Drift .

    [ ] Automated Retraining: Configure the Airflow DAG to trigger a retrain if the system detects significant drift or if model performance degrades below the threshold.

Phase 5: GitHub Integration (CI/CD)

    [ ] Continuous Integration: Set up GitHub Actions to run pytest on every push to ensure preprocessing and API logic remain intact.

    [ ] Continuous Deployment: Create a workflow to sync the /dags folder from GitHub to the local Airflow environment upon a successful merge to the main branch.

Phase 6: Responsible AI & Testing

    [ ] Bias Check: Ensure demographic features (like Gender) are excluded from training to maintain fairness.

    [ ] Logging: Implement comprehensive logging for every batch run to track Churn Risk Scores and system errors.

Next Step for AI Agent: "Please start by generating the docker-compose.yaml file for Airflow and MLflow, then create the basic project directory structure."