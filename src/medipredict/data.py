import numpy as np
import pandas as pd
from medipredict.config import DATASET_SCHEMAS

def load_raw_data(condition: str) -> pd.DataFrame:
    schema = DATASET_SCHEMAS[condition]
    file_path = schema["raw_file"]
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file for '{condition}' not found at {file_path}. Run scripts/download_data.py first.")
    return pd.read_csv(file_path)

def clean_heart_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = (df["target"] > 0).astype(int)
    return df

def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df

def clean_liver_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().map({"Male": 1, "Female": 0})
    if "Dataset" in df.columns:
        df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})
    return df

def load_clean_data(condition: str) -> pd.DataFrame:
    df = load_raw_data(condition)
    if condition == "heart":
        df_clean = clean_heart_data(df)
    elif condition == "diabetes":
        df_clean = clean_diabetes_data(df)
    elif condition == "liver":
        df_clean = clean_liver_data(df)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    schema = DATASET_SCHEMAS[condition]
    df_clean.to_csv(schema["processed_file"], index=False)
    return df_clean

def get_features_and_target(df: pd.DataFrame, condition: str):
    schema = DATASET_SCHEMAS[condition]
    target_col = schema["target"]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y
