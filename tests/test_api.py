from fastapi.testclient import TestClient
from medipredict.api.app import app

client = TestClient(app)

def test_api_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_api_conditions_schema():
    with TestClient(app) as client:
        response = client.get("/conditions")
        assert response.status_code == 200
        data = response.json()
        assert "heart" in data
        assert "diabetes" in data
        assert "liver" in data

def test_api_predict_heart_valid():
    sample_heart = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2
    }
    with TestClient(app) as client:
        response = client.post("/predict/heart", json=sample_heart)
        assert response.status_code == 200
        res = response.json()
        assert res["condition"] == "heart"
        assert 0.0 <= res["probability"] <= 1.0
        assert res["risk_label"] in ["High Risk", "Low Risk"]
        assert len(res["top_shap_contributions"]) > 0

def test_api_predict_heart_invalid_range():
    # Out of bounds age (150 > 120 max limit)
    invalid_sample = {
        "age": 150, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.2, "slope": 1, "ca": 0, "thal": 2
    }
    with TestClient(app) as client:
        response = client.post("/predict/heart", json=invalid_sample)
        assert response.status_code == 422

def test_api_predict_diabetes_valid():
    sample_diabetes = {
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
        "Insulin": 80, "BMI": 25.5, "DiabetesPedigreeFunction": 0.45, "Age": 40
    }
    with TestClient(app) as client:
        response = client.post("/predict/diabetes", json=sample_diabetes)
        assert response.status_code == 200
        res = response.json()
        assert res["condition"] == "diabetes"

def test_api_predict_liver_valid():
    sample_liver = {
        "Age": 45, "Gender": 1, "Total_Bilirubin": 1.2, "Direct_Bilirubin": 0.4,
        "Alkaline_Phosphotase": 180, "Alamine_Aminotransferase": 30,
        "Aspartate_Aminotransferase": 35, "Total_Protiens": 6.8,
        "Albumin": 3.4, "Albumin_and_Globulin_Ratio": 1.0
    }
    with TestClient(app) as client:
        response = client.post("/predict/liver", json=sample_liver)
        assert response.status_code == 200
        res = response.json()
        assert res["condition"] == "liver"

def test_api_metrics_and_compare():
    with TestClient(app) as client:
        res_met = client.get("/models/heart/metrics")
        assert res_met.status_code == 200
        assert "best_model_name" in res_met.json()

        res_comp = client.get("/models/heart/compare")
        assert res_comp.status_code == 200
        assert "models" in res_comp.json()

def test_api_history():
    with TestClient(app) as client:
        response = client.get("/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
