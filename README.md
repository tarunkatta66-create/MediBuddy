# MediPredict

MediPredict is a clinical decision-support prototype built for the Sem VII CSC701 Machine Learning curriculum (University of Mumbai, R-2019 C-Scheme). It estimates risk for three medical conditions (heart disease, diabetes, and liver disease) from routine patient parameters, compares multiple machine learning algorithms side by side, and provides SHAP explanations for individual predictions.

Educational prototype. Not a medical device.

---

## Workspace Layout

- `data/`: Raw and processed dataset files (gitignored).
- `src/medipredict/`: Core machine learning package, preprocessing pipelines, SHAP explainers, from-scratch algorithms, and FastAPI backend.
- `src/medipredict/scratch/`: From-scratch implementations of Linear Regression (Gradient Descent & Normal Equation), Logistic Regression, and CART Decision Tree.
- `frontend/`: React dashboard built with Vite and Recharts.
- `reports/`: `metrics.json`, `results_summary.md`, and generated figures (learning curves, ROC curves, clustering projections).
- `docs/`: `decisions.md` (architecture log) and `viva_notes.md` (viva preparation guide).
- `tests/`: Automated unit tests for scratch implementations, zero-leakage pipeline verification, and API endpoints.

---

## Datasets and Preprocessing

We evaluated three public clinical datasets:
1. **UCI Heart Disease (Cleveland)**: 303 rows, 13 features. We replaced missing "?" values with NaNs and mapped the multiclass target to binary (0 = absent, 1 = present).
2. **PIMA Indians Diabetes**: 768 rows, 8 features. We identified zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI as physiologically invalid missing values and replaced them with NaNs for median imputation.
3. **ILPD (Indian Liver Patient Dataset)**: 583 rows, 10 features. We encoded Gender as binary (Male = 1, Female = 0) and mapped Dataset to binary (1 = Patient, 0 = Non-patient).

All imputations, scaling, and PCA transformations are fitted strictly inside cross-validation folds to ensure zero target leakage.

---

## Empirical Results Summary

Primary model selection criterion was Sensitivity (Recall), because missing a high-risk patient carries greater clinical cost than a false positive.

### Heart Disease Risk (Cleveland)
| Model | Sensitivity (Recall) | Specificity | Precision | F1 Score | Cohen's Kappa | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9286 | 0.8182 | 0.8125 | 0.8667 | 0.7388 | 0.9502 |
| Decision Tree (CART) | 0.8214 | 0.6364 | 0.6571 | 0.7302 | 0.4493 | 0.8528 |
| AdaBoost (Stumps) | 0.9643 | 0.6970 | 0.7297 | 0.8308 | 0.6377 | 0.9069 |
| XGBoost | 0.8571 | 0.7576 | 0.7500 | 0.8000 | 0.6038 | 0.8874 |
| Random Forest | 0.8929 | 0.8485 | 0.8333 | 0.8621 | 0.7352 | 0.9329 |
| SVM (RBF) | 0.9286 | 0.7879 | 0.7879 | 0.8519 | 0.7078 | 0.9426 |

### Diabetes Risk (PIMA)
| Model | Sensitivity (Recall) | Specificity | Precision | F1 Score | Cohen's Kappa | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.6667 | 0.7800 | 0.6207 | 0.6429 | 0.4355 | 0.8037 |
| Random Forest | 0.6852 | 0.7400 | 0.5873 | 0.6325 | 0.4124 | 0.7933 |
| XGBoost | 0.6852 | 0.6800 | 0.5362 | 0.6016 | 0.3506 | 0.7711 |

### Liver Disease Risk (ILPD)
| Model | Sensitivity (Recall) | Specificity | Precision | F1 Score | Cohen's Kappa | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.6341 | 0.7143 | 0.8387 | 0.7222 | 0.3168 | 0.7507 |
| Random Forest | 0.9024 | 0.2857 | 0.7475 | 0.8177 | 0.2223 | 0.7301 |

---

## Setup and How to Run on Windows PowerShell

### 1. Prerequisites and Installation
Open Windows PowerShell inside the project directory:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Download Datasets and Train Models
Run the dataset acquisition script and training pipeline:

```powershell
python scripts/download_data.py
python -m medipredict.train
python scripts/generate_results_summary.py
```

### 3. Run Test Suite
Run automated unit and integration tests:

```powershell
pytest tests/
```

### 4. Launch Backend API
Start the FastAPI backend server on port 8000:

```powershell
uvicorn medipredict.api.app:app --reload --host 127.0.0.1 --port 8000
```

### 5. Launch Frontend Dashboard
Open a separate PowerShell terminal, navigate to `frontend`, install packages, and start Vite dev server:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your web browser.

---

## Limitations

- **Small Sample Sizes**: Cleveland contains 303 patient records and ILPD contains 583 records. These sample sizes restrict model generalization to broader clinical populations.
- **Population Bias**: The PIMA dataset is collected exclusively from a single demographic subgroup (Pima Native American women), limiting extrapolation to general patient demographics.
- **Academic Scope**: This system is built strictly as an educational machine learning prototype and has not undergone formal clinical trial validation or regulatory compliance evaluation.
