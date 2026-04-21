from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

# 1. สร้างตัวแอป FastAPI
app = FastAPI(title="Churn Prediction API")
# อนุญาตให้หน้าเว็บ Next.js (Port 3000) เข้าถึง API นี้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. โหลดโมเดล Voting Classifier (eclf1) ที่เซฟไว้
model = joblib.load("churn_model.pkl")

# 3. กำหนดโครงสร้างข้อมูล (Input Schema) ตาม Progress Report 3.2
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

# 4. สร้าง Endpoint '/predict' 
@app.post("/predict")
def predict_churn(data: CustomerData):
    # แปลงข้อมูล JSON ที่รับมา ให้เป็น DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # หมายเหตุ: ถ้าโมเดลใน Colab ของคุณต้องผ่าน Data Preprocessing (เช่น One-hot encoding) 
    # คุณต้องนำโค้ดแปลงข้อมูลมาใส่ตรงนี้ก่อนเข้า model.predict() 
    # แต่ถ้าตัว Voting Classifier ของคุณมัดรวม Pipeline ไว้แล้ว ก็รันคำสั่งด้านล่างได้เลย
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # ดึงค่าทำนาย (สมมติ 1 = Churn, 0 = Not Churn)
    is_churn = bool(prediction[0] == 1)
    churn_prob = float(probability[0][1])
    
    # จัดกลุ่มความเสี่ยง ตาม Progress Report 3.4 ขั้นที่ 6
    if churn_prob >= 0.7:
        risk_level = "High"
    elif churn_prob >= 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # 5. ส่งผลลัพธ์กลับตามรูปแบบที่ออกแบบไว้ใน Progress Report 3.3
    return {
        "churn_prediction": "Yes" if is_churn else "No",
        "churn_probability": round(churn_prob, 2),
        "risk_level": risk_level
    }

@app.get("/")
def read_root():
    return {"message": "API is running! Go to http://127.0.0.1:8000/docs to test."}