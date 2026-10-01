import numpy as np

class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iter=2000):
        self.lr = lr
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = np.array([0, 1])

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]
        weights = np.zeros((n + 1, 1))

        for _ in range(self.n_iter):
            z = X_b @ weights
            p = self._sigmoid(z)
            gradient = (1.0 / m) * X_b.T @ (p - y)
            weights -= self.lr * gradient

        self.intercept_ = weights[0, 0]
        self.coef_ = weights[1:, 0]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.coef_ + self.intercept_
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
