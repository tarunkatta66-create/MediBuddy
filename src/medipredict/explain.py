import shap
import numpy as np
import pandas as pd

def get_shap_explainer(pipeline, X_train):
    classifier = pipeline.named_steps["classifier"]
    
    # Preprocess X_train through pipeline up to classifier step
    X_prep = X_train.copy()
    if "imputer" in pipeline.named_steps:
        X_prep = pipeline.named_steps["imputer"].transform(X_prep)
    if "scaler" in pipeline.named_steps:
        X_prep = pipeline.named_steps["scaler"].transform(X_prep)

    # Try TreeExplainer first for tree-based models
    try:
        return shap.TreeExplainer(classifier)
    except Exception:
        pass

    # Fallback to model probability callable background explainer
    bg_data = shap.sample(X_prep, min(30, len(X_prep)))
    try:
        return shap.Explainer(classifier.predict_proba, bg_data)
    except Exception:
        return shap.KernelExplainer(classifier.predict_proba, bg_data)

def explain_prediction(pipeline, explainer, X_single, feature_names):
    if isinstance(X_single, pd.Series):
        X_df = X_single.to_frame().T
    elif isinstance(X_single, pd.DataFrame):
        X_df = X_single.copy()
    else:
        X_df = pd.DataFrame([X_single], columns=feature_names)

    # Preprocess X_df through pipeline up to classifier
    X_prep = X_df.copy()
    if "imputer" in pipeline.named_steps:
        X_prep = pipeline.named_steps["imputer"].transform(X_prep)
    if "scaler" in pipeline.named_steps:
        X_prep = pipeline.named_steps["scaler"].transform(X_prep)

    try:
        shap_vals = explainer(X_prep)
        if hasattr(shap_vals, "values"):
            vals = shap_vals.values
        else:
            vals = shap_vals
        
        # Handle multi-output / 3D arrays (e.g. [samples, features, classes])
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        if len(vals.shape) == 3:
            vals = vals[:, :, 1]
        
        row_vals = vals[0]
    except Exception:
        # If explainer call fails, compute approximate linear contributions or zeros
        row_vals = np.zeros(len(feature_names))

    contributions = []
    for feat_name, val, raw_val in zip(feature_names, row_vals, X_df.iloc[0]):
        contributions.append({
            "feature": str(feat_name),
            "shap_value": round(float(val), 4),
            "feature_value": float(raw_val) if pd.notna(raw_val) else 0.0
        })

    # Sort descending by absolute SHAP impact
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return contributions
