"use client";
import { useState } from "react";

export default function Home() {
  const [formData, setFormData] = useState({
    gender: 0, SeniorCitizen: 0, Partner: 0, Dependents: 0, tenure: 1,
    PhoneService: 1, MultipleLines: 0, InternetService: 1, OnlineSecurity: 0,
    OnlineBackup: 0, DeviceProtection: 0, TechSupport: 0, StreamingTV: 0,
    StreamingMovies: 0, Contract: 0, PaperlessBilling: 1, PaymentMethod: 2,
    MonthlyCharges: 50.0, TotalCharges: 50.0
  });

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  
  // --- State สำหรับส่วนอัปโหลดไฟล์ ---
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  // ฟังก์ชันอัปเดตข้อมูลแบบเดี่ยว
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: Number(e.target.value) });
  };

  // ยิง API ทำนายแบบเดี่ยว
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8001/predict", {
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

  // --- ฟังก์ชันจัดการอัปโหลดไฟล์ Batch Predict ---
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return alert("กรุณาเลือกไฟล์ CSV ก่อนครับ");
    
    setUploading(true);
    const formPayload = new FormData();
    formPayload.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8001/predict/batch", {
        method: "POST",
        body: formPayload, // ไม่ต้องใส่ Header Content-Type เดี๋ยว Browser จัดการให้
      });

      if (!res.ok) throw new Error("การทำนายแบบกลุ่มล้มเหลว");

      // แปลงข้อมูลที่ได้กลับมาเป็นไฟล์ แล้วสั่งให้ Browser ดาวน์โหลด
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `predicted_${file.name}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      
      alert("ดาวน์โหลดไฟล์ผลลัพธ์สำเร็จ!");
    } catch (error) {
      alert("เกิดข้อผิดพลาดในการประมวลผลไฟล์ ลองตรวจสอบโครงสร้าง CSV ของคุณดูครับ");
    }
    setUploading(false);
  };

  const formSections = [
    // ... (ส่วนฟอร์มของคุณยาวๆ คงไว้เหมือนเดิมเลยครับ)
    {
      title: "1. ข้อมูลส่วนตัวลูกค้า (Demographics)",
      fields: [
        { key: "gender", label: "เพศ", options: [{ v: 0, t: "หญิง (Female)" }, { v: 1, t: "ชาย (Male)" }] },
        { key: "SeniorCitizen", label: "เป็นผู้สูงอายุหรือไม่", options: [{ v: 0, t: "ไม่ใช่ (No)" }, { v: 1, t: "ใช่ (Yes)" }] },
        { key: "Partner", label: "มีคู่สมรสหรือไม่", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "มี (Yes)" }] },
        { key: "Dependents", label: "มีผู้อยู่ในอุปการะหรือไม่", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "มี (Yes)" }] },
      ]
    },
    {
      title: "2. บริการที่ใช้งาน (Services)",
      fields: [
        { key: "tenure", label: "ระยะเวลาที่เป็นลูกค้า (เดือน)", isNumber: true },
        { key: "PhoneService", label: "บริการโทรศัพท์", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "มี (Yes)" }] },
        { key: "MultipleLines", label: "โทรศัพท์หลายคู่สาย", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "ไม่มีบริการโทรศัพท์" }, { v: 2, t: "มี (Yes)" }] },
        { key: "InternetService", label: "ประเภทอินเทอร์เน็ต", options: [{ v: 0, t: "DSL" }, { v: 1, t: "Fiber optic" }, { v: 2, t: "ไม่ได้ใช้เน็ต (No)" }] },
        { key: "OnlineSecurity", label: "บริการความปลอดภัยออนไลน์", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "ไม่ได้ใช้เน็ต" }, { v: 2, t: "มี (Yes)" }] },
        { key: "OnlineBackup", label: "บริการสำรองข้อมูล", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "ไม่ได้ใช้เน็ต" }, { v: 2, t: "มี (Yes)" }] },
        { key: "DeviceProtection", label: "บริการป้องกันอุปกรณ์", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "ไม่ได้ใช้เน็ต" }, { v: 2, t: "มี (Yes)" }] },
        { key: "TechSupport", label: "บริการสนับสนุนทางเทคนิค", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "ไม่ได้ใช้เน็ต" }, { v: 2, t: "มี (Yes)" }] },
        { key: "StreamingTV", label: "บริการสตรีมทีวี", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "ไม่ได้ใช้เน็ต" }, { v: 2, t: "มี (Yes)" }] },
        { key: "StreamingMovies", label: "บริการสตรีมภาพยนตร์", options: [{ v: 0, t: "ไม่มี (No)" }, { v: 1, t: "ไม่ได้ใช้เน็ต" }, { v: 2, t: "มี (Yes)" }] },
      ]
    },
    {
      title: "3. ข้อมูลบัญชีและการชำระเงิน (Billing & Payment)",
      fields: [
        { key: "Contract", label: "รูปแบบสัญญา", options: [{ v: 0, t: "จ่ายรายเดือน (Month-to-month)" }, { v: 1, t: "สัญญา 1 ปี (One year)" }, { v: 2, t: "สัญญา 2 ปี (Two year)" }] },
        { key: "PaperlessBilling", label: "รับบิลแบบไร้กระดาษ", options: [{ v: 0, t: "ไม่ใช่ (No)" }, { v: 1, t: "ใช่ (Yes)" }] },
        { key: "PaymentMethod", label: "ช่องทางการชำระเงิน", options: [{ v: 0, t: "โอนผ่านธนาคารอัตโนมัติ" }, { v: 1, t: "บัตรเครดิตอัตโนมัติ" }, { v: 2, t: "เช็คอิเล็กทรอนิกส์" }, { v: 3, t: "ส่งเช็คทางไปรษณีย์" }] },
        { key: "MonthlyCharges", label: "ค่าบริการเฉลี่ยรายเดือน (Monthly Charges)", isNumber: true },
        { key: "TotalCharges", label: "ค่าบริการรวมทั้งหมด (Total Charges)", isNumber: true },
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-8">
      <div className="max-w-6xl mx-auto flex flex-col lg:flex-row gap-6">
        
        {/* ฝั่งซ้าย: ฟอร์มกรอกข้อมูล (โค้ดเดิมของคุณ) */}
        <div className="flex-1 bg-white p-6 rounded-xl shadow-sm border border-gray-100">
           {/* ... (โค้ดฟอร์มซ้ายมือเหมือนเดิมทุกประการ) ... */}
           <h1 className="text-2xl font-bold text-blue-900 border-b pb-4 mb-6">ระบบประเมินความเสี่ยงลูกค้า (Churn Prediction)</h1>
          
          <form onSubmit={handleSubmit} className="space-y-8">
            {formSections.map((section, idx) => (
              <div key={idx} className="bg-gray-50 p-4 rounded-lg border border-gray-100">
                <h2 className="text-lg font-semibold text-gray-700 mb-4">{section.title}</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {section.fields.map((field) => (
                    <div key={field.key} className="flex flex-col">
                      <label className="text-sm font-medium text-gray-600 mb-1">{field.label}</label>
                      
                      {field.isNumber ? (
                        <input
                          type="number"
                          step="any"
                          name={field.key}
                          value={formData[field.key as keyof typeof formData]}
                          onChange={handleChange}
                          className="border border-gray-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 text-black outline-none bg-white"
                          required
                        />
                      ) : (
                        <select
                          name={field.key}
                          value={formData[field.key as keyof typeof formData]}
                          onChange={handleChange}
                          className="border border-gray-300 rounded-md p-2 text-sm focus:ring-2 focus:ring-blue-500 text-black outline-none bg-white"
                        >
                          {field.options?.map((opt) => (
                            <option key={opt.v} value={opt.v}>{opt.t}</option>
                          ))}
                        </select>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
            
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 rounded-lg transition-colors shadow-md disabled:bg-gray-400 text-lg"
            >
              {loading ? "กำลังวิเคราะห์ข้อมูลด้วย AI..." : "🔎 เริ่มต้นประเมินความเสี่ยง"}
            </button>
          </form>
        </div>

        {/* ฝั่งขวา: แสดงผลลัพธ์เดี่ยว และ Batch Upload */}
        <div className="lg:w-1/3 flex flex-col gap-6">
          
          {/* กล่องบน: ผลลัพธ์จากการกรอกฟอร์ม (โค้ดเดิมของคุณ) */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h2 className="text-xl font-bold mb-6 text-gray-800 border-b pb-4">ผลการประเมินรายบุคคล</h2>
            
            {result ? (
              <div className="space-y-4">
                <div className={`p-6 rounded-xl border-2 text-center shadow-inner ${result.churn_prediction === "Yes" ? "bg-red-50 border-red-200" : "bg-green-50 border-green-200"}`}>
                  <p className="text-sm text-gray-600 mb-2 font-medium">โอกาสที่ลูกค้าจะยกเลิกบริการ</p>
                  <p className={`text-4xl font-black ${result.churn_prediction === "Yes" ? "text-red-600" : "text-green-600"}`}>
                    {result.churn_prediction === "Yes" ? "สูง (Yes)" : "ต่ำ (No)"}
                  </p>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg border border-gray-100 text-center">
                    <p className="text-xs text-gray-500 mb-1">ความมั่นใจของโมเดล</p>
                    <p className="text-2xl font-bold text-gray-800">
                      {(result.churn_probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg border border-gray-100 text-center">
                    <p className="text-xs text-gray-500 mb-1">ระดับความเสี่ยง</p>
                    <p className={`text-xl font-bold ${result.risk_level === 'High' ? 'text-red-500' : result.risk_level === 'Medium' ? 'text-orange-500' : 'text-green-500'}`}>
                      {result.risk_level}
                    </p>
                  </div>
                </div>
                
                {result.churn_prediction === "Yes" && (
                  <div className="mt-6 bg-orange-50 text-orange-800 p-4 rounded-lg text-sm border border-orange-200">
                    <span className="font-bold">💡 ข้อเสนอแนะ:</span> ลูกค้ารายนี้จัดอยู่ในกลุ่มเสี่ยงสูง ควรส่งมอบโปรโมชันพิเศษหรือให้พนักงานติดต่อกลับโดยด่วน
                  </div>
                )}
              </div>
            ) : (
              <div className="h-64 flex flex-col justify-center items-center text-gray-400 border-2 border-dashed border-gray-200 rounded-xl">
                <span className="text-4xl mb-3">📊</span>
                <p>กรอกข้อมูลและกดปุ่ม</p>
                <p className="text-sm">เพื่อดูผลการประเมินจาก AI</p>
              </div>
            )}
          </div>

          {/* กล่องล่าง: ฟีเจอร์อัปโหลดไฟล์ CSV (ทำนายแบบกลุ่ม) */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <h2 className="text-xl font-bold mb-4 text-gray-800 border-b pb-4">ทำนายผลแบบกลุ่ม (Batch)</h2>
            <p className="text-sm text-gray-500 mb-4">อัปโหลดไฟล์ CSV ที่มีข้อมูลลูกค้าหลายรายการ ระบบจะประมวลผลและส่งไฟล์ที่มีผลลัพธ์กลับมาให้ดาวน์โหลด</p>
            
            <form onSubmit={handleFileUpload} className="space-y-4">
              <input 
                type="file" 
                accept=".csv" 
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              <button
                type="submit"
                disabled={uploading || !file}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 rounded-lg transition-colors shadow-sm disabled:bg-gray-400"
              >
                {uploading ? "กำลังประมวลผลไฟล์..." : "📂 อัปโหลดและดาวน์โหลดผลลัพธ์"}
              </button>
            </form>
          </div>

        </div>

      </div>
    </div>
  );
}