from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os
import joblib  # 👈 เพิ่มตัวนี้เข้ามาแล้ว
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import mlflow.pyfunc 

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
MODEL_NAME = "telco-churn-voting-classifier"

def load_production_model():
    global model
    try:
        # กำหนด URI ไปที่ Production
        model_uri = f"models:/{MODEL_NAME}/Production"
        
        print(f"🔄 กำลังดึงโมเดลล่าสุดจาก MLflow: {model_uri}")
        
        # โหลดโมเดลผ่าน Tracking Server
        model = mlflow.sklearn.load_model(model_uri)
        
        print(f"✅ Successfully loaded Production model from MLflow")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # หากโหลดจาก MLflow ไม่ได้ ให้ลองโหลดไฟล์สำรองในเครื่อง
        try:
            model = joblib.load("churn_model.pkl")
            print("⚠️ Loaded fallback local model (churn_model.pkl)")
        except:
            print("🚨 No local fallback model found.")

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
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized. Please check MLflow connection.")

    # แปลงข้อมูลเป็น DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # ทำนายผล
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    is_churn = bool(prediction[0] == 1)
    churn_prob = float(probability[0][1])
    
    if churn_prob >= 0.7:
        risk_level = "High"
    elif churn_prob >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_prediction": "Yes" if is_churn else "No",
        "churn_probability": round(churn_prob, 2),
        "risk_level": risk_level,
        "model_source": "MLflow Production"
    }

# 🚀 Endpoint พิเศษสำหรับให้ Airflow ยิงมาสั่ง Update Model
@app.post("/reload-model")
async def reload_model():
    load_production_model()
    return {"message": "Backend has reloaded the latest Production model from MLflow."}

@app.get("/")
def read_root():
    return {"status": "online", "mlflow_uri": os.environ.get("MLFLOW_TRACKING_URI")}