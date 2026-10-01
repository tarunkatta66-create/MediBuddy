import json
import logging
import datetime
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from medipredict.config import MODELS_DIR, REPORTS_DIR, DATASET_SCHEMAS
from medipredict.data import load_clean_data, get_features_and_target
from medipredict.explain import get_shap_explainer, explain_prediction
from medipredict.api.schemas import (
    HeartInputSchema, DiabetesInputSchema, LiverInputSchema, PredictionResponse
)
from medipredict.api.db import init_db, get_db, PredictionHistory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("medipredict.api")

MODELS_CACHE: Dict[str, Any] = {}
EXPLAINERS_CACHE: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()

    logger.info("Loading pre-trained models into memory...")
    for condition in ["heart", "diabetes", "liver"]:
        model_path = MODELS_DIR / f"{condition}_best.joblib"
        if not model_path.exists():
            logger.warning(f"Best model for {condition} not found at {model_path}. Run medipredict.train first!")
            continue
        pipeline = joblib.load(model_path)
        MODELS_CACHE[condition] = pipeline

        # Load background data for SHAP explainer
        df_clean = load_clean_data(condition)
        X, _ = get_features_and_target(df_clean, condition)
        explainer = get_shap_explainer(pipeline, X)
        EXPLAINERS_CACHE[condition] = (explainer, list(X.columns))

    logger.info("Models loaded successfully.")
    yield
    MODELS_CACHE.clear()
    EXPLAINERS_CACHE.clear()

app = FastAPI(title="MediPredict API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "loaded_conditions": list(MODELS_CACHE.keys())}

@app.get("/conditions")
def get_conditions_schema():
    return DATASET_SCHEMAS

@app.get("/models/{condition}/metrics")
def get_model_metrics(condition: str):
    if condition not in ["heart", "diabetes", "liver"]:
        raise HTTPException(status_code=400, detail=f"Invalid condition: {condition}")
    json_path = MODELS_DIR / f"{condition}_best.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics for condition '{condition}' not found.")
    with open(json_path) as f:
        return json.load(f)

@app.get("/models/{condition}/compare")
def get_model_comparison(condition: str):
    if condition not in ["heart", "diabetes", "liver"]:
        raise HTTPException(status_code=400, detail=f"Invalid condition: {condition}")
    metrics_path = REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Global metrics report not found.")
    with open(metrics_path) as f:
        data = json.load(f)
    if condition not in data:
        raise HTTPException(status_code=404, detail=f"No comparison data for {condition}")
    return data[condition]

@app.post("/predict/{condition}", response_model=PredictionResponse)
def predict_risk(condition: str, payload: dict = Body(...), db: Session = Depends(get_db)):
    if condition not in MODELS_CACHE:
        raise HTTPException(status_code=400, detail=f"Condition '{condition}' is not loaded or invalid.")

    # Validate input schema dynamically based on condition
    try:
        if condition == "heart":
            validated = HeartInputSchema(**payload)
        elif condition == "diabetes":
            validated = DiabetesInputSchema(**payload)
        elif condition == "liver":
            validated = LiverInputSchema(**payload)
        else:
            raise HTTPException(status_code=400, detail="Invalid condition.")
    except Exception as err:
        raise HTTPException(status_code=422, detail=str(err))

    pipeline = MODELS_CACHE[condition]
    explainer, feature_names = EXPLAINERS_CACHE[condition]

    raw_dict = validated.model_dump()
    input_df = pd.DataFrame([raw_dict], columns=feature_names)

    # Predict probability & label
    if hasattr(pipeline, "predict_proba"):
        prob = float(pipeline.predict_proba(input_df)[0, 1])
    else:
        prob = float(pipeline.predict(input_df)[0])
    
    risk_label = "High Risk" if prob >= 0.5 else "Low Risk"

    # Get best model metadata name
    json_path = MODELS_DIR / f"{condition}_best.json"
    model_used_name = "Best Estimator"
    if json_path.exists():
        with open(json_path) as f:
            model_used_name = json.load(f).get("best_model_name", "Best Estimator")

    # Generate SHAP explanation
    shap_contribs = explain_prediction(pipeline, explainer, input_df.iloc[0], feature_names)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Persist in SQLite
    record = PredictionHistory(
        condition=condition,
        input_data=json.dumps(raw_dict),
        probability=round(prob, 4),
        risk_label=risk_label,
        model_used=model_used_name,
        timestamp=now_str
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "condition": condition,
        "probability": round(prob, 4),
        "risk_label": risk_label,
        "model_used": model_used_name,
        "top_shap_contributions": shap_contribs,
        "timestamp": now_str
    }

@app.get("/history")
def get_prediction_history(db: Session = Depends(get_db)):
    records = db.query(PredictionHistory).order_by(PredictionHistory.id.desc()).all()
    results = []
    for r in records:
        results.append({
            "id": r.id,
            "condition": r.condition,
            "input_data": json.loads(r.input_data),
            "probability": r.probability,
            "risk_label": r.risk_label,
            "model_used": r.model_used,
            "timestamp": r.timestamp
        })
    return results
