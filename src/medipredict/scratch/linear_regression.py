import numpy as np

class LinearRegressionNormalEq:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept_ = weights[0, 0]
        self.coef_ = weights[1:, 0]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

class LinearRegressionGD:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        weights = np.zeros((n + 1, 1))

        for _ in range(self.n_iter):
            gradients = (2 / m) * X_b.T @ (X_b @ weights - y)
            weights -= self.lr * gradients

        self.intercept_ = weights[0, 0]
        self.coef_ = weights[1:, 0]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
