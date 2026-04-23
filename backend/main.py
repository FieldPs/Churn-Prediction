from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from fastapi.responses import Response
import io
import numpy as np
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
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        result_df = df.copy()
        X = df.drop(columns=["customerID", "Churn"], errors="ignore")
        
        # 💡 เพิ่มส่วนนี้: คู่มือแปลภาษา แปลงข้อความ (Text) เป็นตัวเลข (Numeric)
        mapping = {
            "gender": {"Female": 0, "Male": 1},
            "Partner": {"No": 0, "Yes": 1},
            "Dependents": {"No": 0, "Yes": 1},
            "PhoneService": {"No": 0, "Yes": 1},
            "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
            "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
            "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
            "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
            "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
            "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
            "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
            "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
            "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
            "PaperlessBilling": {"No": 0, "Yes": 1},
            "PaymentMethod": {
                "Bank transfer (automatic)": 0,
                "Credit card (automatic)": 1,
                "Electronic check": 2,
                "Mailed check": 3
            }
        }
        
        # สั่งเปลี่ยนคำตามคู่มือด้านบน
        X = X.replace(mapping)
        
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
            X["TotalCharges"] = X["TotalCharges"].fillna(X["TotalCharges"].median())

        # ลอกป้ายชื่อคอลัมน์ออก (แปลงเป็น NumPy array โล้นๆ) 
        input_array = X.values
        
        # ทำนายผล
        probs = model.predict_proba(input_array)[:, 1]
        preds = (probs >= 0.50).astype(int)

        result_df["churn_probability"] = np.round(probs, 4)
        result_df["churn_prediction"] = ["Yes" if p == 1 else "No" for p in preds]

        def assign_risk(prob):
            if prob >= 0.70: return "High"
            elif prob >= 0.40: return "Medium"
            else: return "Low"

        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        
        response = Response(content=stream.getvalue(), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=predicted_{file.filename}"
        
        return response
        
    except Exception as e:
        print(f"Batch Predict Error: {str(e)}") # พิมพ์ Error จริงๆ ลง Terminal
        raise HTTPException(status_code=400, detail=f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {str(e)}")