from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from medipredict.config import RANDOM_SEED

def build_preprocessing_pipeline(use_smote=False, use_scaler=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
        return ImbPipeline(steps)
    return Pipeline(steps)

def build_model_pipeline(classifier, use_smote=False, use_scaler=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
        steps.append(("classifier", classifier))
        return ImbPipeline(steps)
    else:
        steps.append(("classifier", classifier))
        return Pipeline(steps)
