# decisions.md - MediPredict Architectural & Design Decisions

- Fixed random seed set to 42 across numpy, scikit-learn, and xgboost for reproducibility.
- Used OpenML fallbacks in download_data.py to ensure robust dataset acquisition if UCI primary URLs fail.
- Treated PIMA 0 values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI as NaNs since zero values in these physiological parameters are biologically impossible.
- Replaced "?" with NaN in Cleveland dataset before numerical imputation.
- Standardized ILPD target column to binary classification (1 = Patient, 0 = Non-patient).
- Opted out of 3-class unified condition prediction across datasets because Heart, Diabetes, and Liver datasets share only Age and Sex, making a merged multi-condition model clinically invalid and synthetic.
- Selected Recall/Sensitivity as the primary optimization metric because missing a high-risk patient carries significantly higher clinical cost than a false positive.
- Used SQLite for local prediction history persistence to avoid external database dependencies during evaluation.
- Implemented CART from scratch using Gini impurity and recursive binary splits, matching scikit-learn behavior on synthetic benchmarks.
- Placed all data pre-processors (imputer, scaler, PCA) strictly inside scikit-learn Pipelines to guarantee zero target/validation data leakage during cross-validation.
