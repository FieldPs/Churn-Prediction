# Telco Customer Churn Prediction (MLOps)

An MLOps project for telecom customer churn prediction utilizing a production-ready pipeline powered by **Airflow**, **MLflow**, **DVC**, and **Docker**.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
Start by cloning the project to your local machine:
```bash
git clone https://github.com/FieldPs/Churn-Prediction.git
cd Churn-Prediction
```

### 2. Set Up the Virtual Environment
Create an isolated Python environment to avoid package conflicts:
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate    # For macOS / Linux
# .venv\Scripts\activate     # For Windows

# Install dependencies and DVC Google Drive extension
pip install --upgrade pip
pip install -r requirements.txt
pip install "dvc[gdrive]"
```

### 3. Pull Data with DVC
To keep our repository lightweight, large datasets are tracked via DVC and stored securely on Google Drive. 

Pull the dataset by running:
```bash
dvc pull
```
> **Note (First-time setup only):** 
Running this command will open your web browser. You must log in using the Google Account that has access to our shared project drive. Follow the prompts to authenticate, and DVC will automatically download the dataset into the `data/raw/` directory.

### 4. Start the Application Stack
Ensure you have Docker and Docker Compose installed. Start all necessary services (Airflow, MLflow, Postgres) by running:

```bash
docker compose up -d
```
> Please allow 1-2 minutes on the first run for the Airflow metadata database to initialize.

**🔗 Access the Services:**
- **Airflow Web UI:** [http://localhost:8080](http://localhost:8080)
  - **Username:** `admin`
  - **Password:** `admin`
- **MLflow UI:** [http://localhost:5000](http://localhost:5000)

*(To stop the services later, run `docker compose down`)*

---

## 📁 Project Structure
- `/dags/` - Airflow DAG definitions.
- `/src/` - Core ML pipeline scripts (`data_ingestion.py`, `preprocessing.py`, `train.py`).
- `/tests/` - Unit tests (run via `pytest tests/`).
- `/data/raw/` - Raw datasets pulled via DVC.
- `/models/` - Saved model artifacts.
- `projectTRD.md` - Technical Requirements Document and task roadmap.
