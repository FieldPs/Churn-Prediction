"use client";
import { useState } from "react";

export default function Home() {
  // สร้าง State เก็บข้อมูลทั้ง 19 ฟิลด์ (ใส่ค่า Default ไว้ให้ทดสอบง่ายๆ)
  const [formData, setFormData] = useState({
    gender: 1, SeniorCitizen: 0, Partner: 1, Dependents: 0, tenure: 12.0,
    PhoneService: 1, MultipleLines: 0, InternetService: 1, OnlineSecurity: 0,
    OnlineBackup: 1, DeviceProtection: 0, TechSupport: 1, StreamingTV: 1,
    StreamingMovies: 0, Contract: 0, PaperlessBilling: 1, PaymentMethod: 2,
    MonthlyCharges: 59.9, TotalCharges: 700.5
  });

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // ฟังก์ชันอัปเดตข้อมูลเมื่อมีการพิมพ์ในช่อง Input
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: Number(e.target.value) });
  };

  // ฟังก์ชันกดปุ่มเพื่อยิงข้อมูลไปหา FastAPI
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      alert("ไม่สามารถเชื่อมต่อ API ได้ ตรวจสอบว่ารัน FastAPI อยู่หรือไม่");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold text-gray-800">Customer Churn Prediction</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* ฝั่งซ้าย: ฟอร์มกรอกข้อมูล */}
          <div className="md:col-span-2 bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">ข้อมูลลูกค้า</h2>
            <form onSubmit={handleSubmit} className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {/* ใช้ Object.keys เพื่อสร้างช่อง Input อัตโนมัติทั้ง 19 ช่อง จะได้โค้ดไม่ยาวเกินไป */}
              {Object.keys(formData).map((key) => (
                <div key={key} className="flex flex-col">
                  <label className="text-xs font-medium text-gray-500 mb-1">{key}</label>
                  <input
                    type="number"
                    step="any"
                    name={key}
                    value={formData[key as keyof typeof formData]}
                    onChange={handleChange}
                    className="border border-gray-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 text-black"
                  />
                </div>
              ))}
              
              <div className="col-span-full mt-4">
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors disabled:bg-gray-400"
                >
                  {loading ? "กำลังวิเคราะห์ข้อมูล..." : "ประเมินความเสี่ยง (Predict)"}
                </button>
              </div>
            </form>
          </div>

          {/* ฝั่งขวา: แสดงผลลัพธ์ */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col justify-center items-center text-center">
            <h2 className="text-xl font-semibold mb-6 text-gray-700">ผลการประเมิน</h2>
            
            {result ? (
              <div className="space-y-4 w-full">
                <div className={`p-6 rounded-lg border-2 ${result.churn_prediction === "Yes" ? "bg-red-50 border-red-200" : "bg-green-50 border-green-200"}`}>
                  <p className="text-sm text-gray-600 mb-1">แนวโน้มการยกเลิกบริการ</p>
                  <p className={`text-4xl font-bold ${result.churn_prediction === "Yes" ? "text-red-600" : "text-green-600"}`}>
                    {result.churn_prediction}
                  </p>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-xs text-gray-500">ความน่าจะเป็น</p>
                    <p className="text-xl font-semibold text-gray-800">
                      {(result.churn_probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-xs text-gray-500">ระดับความเสี่ยง</p>
                    <p className="text-xl font-semibold text-gray-800">{result.risk_level}</p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-400">กรอกข้อมูลและกดปุ่มเพื่อดูผลลัพธ์</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}