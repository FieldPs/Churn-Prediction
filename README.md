# Telco Customer Churn Prediction (MLOps)

An MLOps project for telecom customer churn prediction utilizing a production-ready pipeline powered by **Airflow**, **MLflow**, **DVC**, and **Docker**.

---

## 🚀 Quick Start Guide

### 1. Clone & Environment Setup
```bash
git clone https://github.com/FieldPs/Churn-Prediction.git
cd Churn-Prediction

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Data (DVC)
Datasets are tracked via DVC on **DAGsHub**.
```bash
dvc pull
```

### 3. Start the MLOps Stack (Docker)
Start **Airflow**, **MLflow**, and the **Backend API** using Docker Compose:
```bash
docker compose up -d --build
```
> [!NOTE]
> Please allow 1-2 minutes for the database and services to initialize on the first run.

**🔗 Service Endpoints:**
- **Airflow UI:** [http://localhost:8080](http://localhost:8080) (`admin`/`admin`)
- **MLflow UI:** [http://localhost:5000](http://localhost:5000)
- **Backend API:** [http://localhost:8001](http://localhost:8001)

### 4. Deploy the Model (Airflow Pipeline)
The system uses **MLflow Model Registry** for deployment.
1. Access **Airflow** at port 8080.
2. Unpause and **Trigger** the `churn_mlops_pipeline` DAG.
3. This pipeline will train the model, log metrics to MLflow, and promote the best model to the **"Production"** stage.

### 5. Run the Frontend
The frontend is a Next.js application that communicates with the Backend API.
```bash
cd frontend/churn-frontend
npm install
npm run dev
```
Access the **UI** at: [http://localhost:3000](http://localhost:3000)

---

## 🔄 MLOps Workflow
1. **Training (Airflow):** The pipeline automates Data Ingestion -> Preprocessing -> Training -> Validation.
2. **Registry (MLflow):** If the model meets the Recall threshold (≥ 0.80), it is registered and tagged as `Production`.
3. **Serving (Backend):** The FastAPI backend pulls the `Production` model directly from the MLflow Tracking Server.
4. **Consumption (Frontend):** Users interact with the model via a modern web interface.

---

## 📁 Project Structure
- `/dags/` - Airflow DAGs (MLOps pipeline definitions).
- `/src/` - Core ML logic (Ingestion, Preprocessing, Training).
- `/backend/` - FastAPI service that serves models from MLflow.
- `/frontend/` - Next.js web application.
- `/data/` - Dataset storage (managed by DVC).
- `/models/` - Local model fallbacks.
- `projectTRD.md` - Technical roadmap and requirements.
- `AGENTS.md` - AI Agent instructions and phase status.
