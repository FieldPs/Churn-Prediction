from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from fastapi.responses import Response
import io
import numpy as np
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os
import joblib
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# 1. ตั้งค่าการเชื่อมต่อกับ MLflow (Docker)
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

app = FastAPI(title="Churn Prediction API (MLOps Ready)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ตัวแปรระดับ Global สำหรับเก็บโมเดลที่โหลดมาแล้ว
model = None
preprocessor = None
model_metadata = {
    "source": None,
    "run_id": None,
    "model_version": None,
}
MODEL_NAME = "telco-churn-voting-classifier"
FALLBACK_MODEL_PATH = os.getenv("MODEL_FALLBACK_PATH", "/app/models/voting_classifier.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "/app/models/preprocessor.pkl")

DEMOGRAPHIC_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents"]

LABEL_ENCODE_MAPS = {
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

OHE_REVERSE_MAPS = {
    "InternetService": {0: "DSL", 1: "Fiber optic", 2: "No"},
    "Contract": {0: "Month-to-month", 1: "One year", 2: "Two year"},
    "PaymentMethod": {
        0: "Bank transfer (automatic)",
        1: "Credit card (automatic)",
        2: "Electronic check",
        3: "Mailed check",
    },
}


def assign_risk(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def _load_preprocessor() -> None:
    global preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"✅ Loaded preprocessor: {PREPROCESSOR_PATH}")


def _map_label_column(series: pd.Series, mapping: dict, col_name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    mapped = series.astype(str).str.strip().map(mapping)
    if mapped.isna().any():
        unknown = sorted(series[mapped.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unsupported values in column '{col_name}': {unknown[:5]}")
    return mapped


def _normalize_ohe_column(series: pd.Series, reverse_map: dict, col_name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        mapped = pd.to_numeric(series, errors="coerce").map(reverse_map)
        if mapped.isna().any():
            bad = sorted(series[mapped.isna()].dropna().astype(int).unique().tolist())
            raise ValueError(f"Unsupported encoded values in column '{col_name}': {bad[:5]}")
        return mapped

    numeric_try = pd.to_numeric(series, errors="coerce")
    if numeric_try.notna().all():
        mapped = numeric_try.astype(int).map(reverse_map)
        if mapped.isna().any():
            bad = sorted(numeric_try[mapped.isna()].dropna().astype(int).unique().tolist())
            raise ValueError(f"Unsupported encoded values in column '{col_name}': {bad[:5]}")
        return mapped

    return series.astype(str)


def _prepare_features(input_df: pd.DataFrame) -> pd.DataFrame:
    if preprocessor is None:
        raise ValueError("Preprocessor is not initialized.")

    X = input_df.copy()
    X = X.drop(columns=["customerID", "Churn"], errors="ignore")

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        median_val = X["TotalCharges"].median()
        X["TotalCharges"] = X["TotalCharges"].fillna(0.0 if pd.isna(median_val) else median_val)

    for col, mapping in LABEL_ENCODE_MAPS.items():
        if col in X.columns:
            X[col] = _map_label_column(X[col], mapping, col)

    for col, reverse_map in OHE_REVERSE_MAPS.items():
        if col in X.columns:
            X[col] = _normalize_ohe_column(X[col], reverse_map, col)

    X = X.drop(columns=DEMOGRAPHIC_COLS, errors="ignore")

    if hasattr(preprocessor, "feature_names_in_"):
        expected_cols = list(preprocessor.feature_names_in_)
        missing_cols = [col for col in expected_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        X = X[expected_cols]

    if X.isna().any().any():
        missing_info = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN values found after feature preparation in columns: {missing_info}")

    return X

def load_production_model():
    global model, model_metadata, preprocessor
    try:
        # กำหนด URI ไปที่ Production
        model_uri = f"models:/{MODEL_NAME}/Production"
        
        print(f"🔄 กำลังดึงโมเดลล่าสุดจาก MLflow: {model_uri}")
        
        # โหลดโมเดลผ่าน Tracking Server
        model = mlflow.sklearn.load_model(model_uri)
        _load_preprocessor()
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        production_version = next((version for version in versions if version.current_stage == "Production"), None)
        model_metadata = {
            "source": "mlflow-production",
            "run_id": production_version.run_id if production_version else None,
            "model_version": production_version.version if production_version else None,
        }
        
        print(f"✅ Successfully loaded Production model from MLflow")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # หากโหลดจาก MLflow ไม่ได้ ให้ลองโหลดไฟล์สำรองในเครื่อง
        try:
            model = joblib.load(FALLBACK_MODEL_PATH)
            _load_preprocessor()
            model_metadata = {
                "source": "local-pickle",
                "run_id": None,
                "model_version": None,
            }
            print(f"⚠️ Loaded fallback local model ({FALLBACK_MODEL_PATH})")
        except Exception as fallback_error:
            model = None
            preprocessor = None
            model_metadata = {
                "source": None,
                "run_id": None,
                "model_version": None,
            }
            print(f"🚨 No usable fallback model/preprocessor found: {fallback_error}")

# สั่งให้โหลดโมเดลทันทีเมื่อเริ่มรัน API
@app.on_event("startup")
async def startup_event():
    load_production_model()

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is not initialized. Please check MLflow connection.")

    # แปลงข้อมูลเป็น DataFrame
    input_df = pd.DataFrame([data.dict()])
    prepared_df = _prepare_features(input_df)
    transformed_input = preprocessor.transform(prepared_df)
    
    # ทำนายผล
    prediction = model.predict(transformed_input)
    probability = model.predict_proba(transformed_input)
    
    is_churn = bool(prediction[0] == 1)
    churn_prob = float(probability[0][1])
    
    risk_level = assign_risk(churn_prob)

    return {
        "churn_prediction": "Yes" if is_churn else "No",
        "churn_probability": round(churn_prob, 2),
        "risk_level": risk_level,
        "model_source": model_metadata["source"],
        "model_run_id": model_metadata["run_id"],
        "model_version": model_metadata["model_version"],
    }

# 🚀 Endpoint พิเศษสำหรับให้ Airflow ยิงมาสั่ง Update Model
@app.post("/reload-model")
async def reload_model():
    load_production_model()
    return {"message": "Backend has reloaded the latest Production model from MLflow."}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "mlflow_uri": os.environ.get("MLFLOW_TRACKING_URI"),
        "model_source": model_metadata["source"],
        "model_run_id": model_metadata["run_id"],
        "model_version": model_metadata["model_version"],
    }
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        result_df = df.copy()
        prepared_df = _prepare_features(df)
        transformed_input = preprocessor.transform(prepared_df)
        
        # ทำนายผล
        probs = model.predict_proba(transformed_input)[:, 1]
        preds = (probs >= 0.50).astype(int)

        result_df["churn_probability"] = np.round(probs, 4)
        result_df["churn_prediction"] = ["Yes" if p == 1 else "No" for p in preds]
        result_df["risk_level"] = result_df["churn_probability"].apply(assign_risk)

        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        
        response = Response(content=stream.getvalue(), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=predicted_{file.filename}"
        
        return response
        
    except Exception as e:
        print(f"Batch Predict Error: {str(e)}") # พิมพ์ Error จริงๆ ลง Terminal
        raise HTTPException(status_code=400, detail=f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {str(e)}")