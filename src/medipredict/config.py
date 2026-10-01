from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, DOCS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Global Configuration
RANDOM_SEED = 42

# Clinical Feature Schemas & Descriptions
DATASET_SCHEMAS = {
    "heart": {
        "raw_file": RAW_DATA_DIR / "heart_cleveland.csv",
        "processed_file": PROCESSED_DATA_DIR / "heart_clean.csv",
        "target": "target",
        "features": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "continuous": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    },
    "diabetes": {
        "raw_file": RAW_DATA_DIR / "diabetes_pima.csv",
        "processed_file": PROCESSED_DATA_DIR / "diabetes_clean.csv",
        "target": "Outcome",
        "features": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "continuous": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "categorical": []
    },
    "liver": {
        "raw_file": RAW_DATA_DIR / "liver_ilpd.csv",
        "processed_file": PROCESSED_DATA_DIR / "liver_clean.csv",
        "target": "Dataset",
        "features": [
            "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
            "Alkaline_Phosphotase", "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase", "Total_Protiens",
            "Albumin", "Albumin_and_Globulin_Ratio"
        ],
        "continuous": [
            "Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
            "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
            "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio"
        ],
        "categorical": ["Gender"]
    }
}
