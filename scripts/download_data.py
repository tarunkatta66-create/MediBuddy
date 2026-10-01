import sys
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from medipredict.config import RAW_DATA_DIR

HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
DIABETES_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
LIVER_COLS = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"]

def download_heart():
    target_path = RAW_DATA_DIR / "heart_cleveland.csv"
    if target_path.exists():
        print(f"Heart dataset already exists at {target_path}")
        return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    try:
        df = pd.read_csv(url, header=None, names=HEART_COLS)
        df.to_csv(target_path, index=False)
        print(f"Downloaded Heart dataset directly to {target_path}")
    except Exception as e:
        print(f"Direct URL download failed for Heart dataset: {e}. Trying OpenML fallback...")
        try:
            bunch = fetch_openml(data_id=492, as_frame=True, parser="auto")
            df = bunch.frame
            df.to_csv(target_path, index=False)
            print(f"Downloaded Heart dataset via OpenML to {target_path}")
        except Exception as ex:
            print(f"OpenML fallback failed: {ex}. Please manually place processed.cleveland.data as csv at {target_path}")

def download_diabetes():
    target_path = RAW_DATA_DIR / "diabetes_pima.csv"
    if target_path.exists():
        print(f"Diabetes dataset already exists at {target_path}")
        return

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    try:
        df = pd.read_csv(url, header=None, names=DIABETES_COLS)
        df.to_csv(target_path, index=False)
        print(f"Downloaded Diabetes dataset directly to {target_path}")
    except Exception as e:
        print(f"Direct URL download failed for Diabetes dataset: {e}. Trying OpenML fallback...")
        try:
            bunch = fetch_openml(name="diabetes", version=1, as_frame=True, parser="auto")
            df = bunch.frame
            if "class" in df.columns:
                df["Outcome"] = df["class"].map({"tested_positive": 1, "tested_negative": 0})
                df.drop(columns=["class"], inplace=True)
            df.to_csv(target_path, index=False)
            print(f"Downloaded Diabetes dataset via OpenML to {target_path}")
        except Exception as ex:
            print(f"OpenML fallback failed: {ex}. Please manually place pima-indians-diabetes.csv at {target_path}")

def download_liver():
    target_path = RAW_DATA_DIR / "liver_ilpd.csv"
    if target_path.exists():
        print(f"Liver dataset already exists at {target_path}")
        return

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    try:
        df = pd.read_csv(url, header=None, names=LIVER_COLS)
        df.to_csv(target_path, index=False)
        print(f"Downloaded Liver dataset directly to {target_path}")
    except Exception as e:
        print(f"Direct URL download failed for Liver dataset: {e}. Trying OpenML fallback...")
        try:
            bunch = fetch_openml(name="ILPD", version=1, as_frame=True, parser="auto")
            df = bunch.frame
            df.to_csv(target_path, index=False)
            print(f"Downloaded Liver dataset via OpenML to {target_path}")
        except Exception as ex:
            print(f"OpenML fallback failed: {ex}. Please manually place ILPD.csv at {target_path}")

if __name__ == "__main__":
    download_heart()
    download_diabetes()
    download_liver()
