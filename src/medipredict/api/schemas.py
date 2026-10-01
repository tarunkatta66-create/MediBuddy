from pydantic import BaseModel, Field
from typing import List, Dict, Any

class HeartInputSchema(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = Male, 0 = Female)")
    cp: int = Field(..., ge=0, le=3, description="Chest Pain Type (0-3)")
    trestbps: float = Field(..., ge=60, le=260, description="Resting Blood Pressure (mmHg)")
    chol: float = Field(..., ge=80, le=600, description="Serum Cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG Results (0-2)")
    thalach: float = Field(..., ge=60, le=220, description="Maximum Heart Rate Achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise Induced Angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST Depression Induced by Exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of Peak Exercise ST Segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of Major Vessels Colored by Fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

class DiabetesInputSchema(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., ge=40, le=400, description="Plasma glucose concentration (mg/dl)")
    BloodPressure: float = Field(..., ge=30, le=200, description="Diastolic blood pressure (mmHg)")
    SkinThickness: float = Field(..., ge=5, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., ge=5, le=900, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., ge=10.0, le=70.0, description="Body mass index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(..., ge=0.05, le=3.0, description="Diabetes pedigree function score")
    Age: int = Field(..., ge=1, le=120, description="Age in years")

class LiverInputSchema(BaseModel):
    Age: int = Field(..., ge=1, le=120, description="Age in years")
    Gender: int = Field(..., ge=0, le=1, description="Gender (1 = Male, 0 = Female)")
    Total_Bilirubin: float = Field(..., ge=0.1, le=80.0, description="Total Bilirubin (mg/dl)")
    Direct_Bilirubin: float = Field(..., ge=0.01, le=30.0, description="Direct Bilirubin (mg/dl)")
    Alkaline_Phosphotase: float = Field(..., ge=10, le=3000, description="Alkaline Phosphotase (IU/L)")
    Alamine_Aminotransferase: float = Field(..., ge=5, le=2000, description="Alamine Aminotransferase (IU/L)")
    Aspartate_Aminotransferase: float = Field(..., ge=5, le=2000, description="Aspartate Aminotransferase (IU/L)")
    Total_Protiens: float = Field(..., ge=2.0, le=15.0, description="Total Proteins (g/dl)")
    Albumin: float = Field(..., ge=1.0, le=10.0, description="Albumin (g/dl)")
    Albumin_and_Globulin_Ratio: float = Field(..., ge=0.1, le=5.0, description="Albumin and Globulin Ratio")

class ShapContribution(BaseModel):
    feature: str
    shap_value: float
    feature_value: float

class PredictionResponse(BaseModel):
    id: int
    condition: str
    probability: float
    risk_label: str
    model_used: str
    top_shap_contributions: List[ShapContribution]
    timestamp: str
