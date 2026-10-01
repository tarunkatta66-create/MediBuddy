import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def test_scaler_not_fitted_on_test_data():
    np.random.seed(42)
    # Generate synthetic dataset with different train and test distributions
    X_train = np.random.normal(loc=10.0, scale=2.0, size=(100, 3))
    X_test = np.random.normal(loc=100.0, scale=20.0, size=(50, 3))
    y_train = np.random.randint(0, 2, size=100)
    y_test = np.random.randint(0, 2, size=50)

    X_full = np.vstack([X_train, X_test])

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])

    # Fit pipeline strictly on train set
    pipeline.fit(X_train, y_train)

    fitted_scaler_mean = pipeline.named_steps["scaler"].mean_
    fitted_scaler_var = pipeline.named_steps["scaler"].var_

    train_mean = np.mean(X_train, axis=0)
    full_mean = np.mean(X_full, axis=0)

    # Scaler mean must match X_train mean within floating point tolerance
    assert np.allclose(fitted_scaler_mean, train_mean)

    # Scaler mean must NOT match full dataset mean (proves test data was not seen during fitting)
    assert not np.allclose(fitted_scaler_mean, full_mean)
