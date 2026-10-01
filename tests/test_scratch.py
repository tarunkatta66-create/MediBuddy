import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score

from medipredict.scratch.linear_regression import LinearRegressionGD, LinearRegressionNormalEq
from medipredict.scratch.logistic_regression import LogisticRegressionGD
from medipredict.scratch.cart import CARTClassifier

def test_linear_regression_scratch():
    X, y = make_regression(n_samples=200, n_features=3, noise=1.0, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scikit-learn
    sk_model = LinearRegression()
    sk_model.fit(X_scaled, y)
    sk_preds = sk_model.predict(X_scaled)

    # Scratch Normal Eq
    norm_model = LinearRegressionNormalEq()
    norm_model.fit(X_scaled, y)
    norm_preds = norm_model.predict(X_scaled)
    assert r2_score(y, norm_preds) > 0.95
    assert np.allclose(sk_preds, norm_preds, atol=1e-3)

    # Scratch Gradient Descent
    gd_model = LinearRegressionGD(lr=0.05, n_iter=2000)
    gd_model.fit(X_scaled, y)
    gd_preds = gd_model.predict(X_scaled)
    assert r2_score(y, gd_preds) > 0.95
    assert np.allclose(sk_preds, gd_preds, atol=1e-1)

def test_logistic_regression_scratch():
    X, y = make_classification(n_samples=300, n_features=4, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scikit-learn
    sk_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    sk_model.fit(X_scaled, y)
    sk_acc = accuracy_score(y, sk_model.predict(X_scaled))

    # Scratch Logistic Regression
    scratch_model = LogisticRegressionGD(lr=0.1, n_iter=3000)
    scratch_model.fit(X_scaled, y)
    scratch_acc = accuracy_score(y, scratch_model.predict(X_scaled))

    assert scratch_acc > 0.85
    assert abs(sk_acc - scratch_acc) < 0.1

def test_cart_classifier_scratch():
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=42)

    # Scikit-learn
    sk_cart = DecisionTreeClassifier(max_depth=3, random_state=42)
    sk_cart.fit(X, y)
    sk_acc = accuracy_score(y, sk_cart.predict(X))

    # Scratch CART
    scratch_cart = CARTClassifier(max_depth=3, min_samples_leaf=2)
    scratch_cart.fit(X, y)
    scratch_acc = accuracy_score(y, scratch_cart.predict(X))

    assert scratch_acc > 0.80
    assert abs(sk_acc - scratch_acc) < 0.15
