import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba

    def is_leaf(self):
        return self.value is not None

class CARTClassifier:
    def __init__(self, max_depth=5, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.classes_ = np.array([0, 1])

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y == 1)
        p0 = 1.0 - p1
        return 1.0 - (p0 ** 2 + p1 ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (np.sum(left_mask) / n_samples) * gini_left + (np.sum(right_mask) / n_samples) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_labels = len(np.unique(y))
        p1 = np.mean(y == 1) if n_samples > 0 else 0.0

        if (depth >= self.max_depth or n_labels <= 1 or n_samples < 2 * self.min_samples_leaf):
            val = 1 if p1 >= 0.5 else 0
            return Node(value=val, proba=[1.0 - p1, p1])

        feat, thresh = self._best_split(X, y)
        if feat is None:
            val = 1 if p1 >= 0.5 else 0
            return Node(value=val, proba=[1.0 - p1, p1])

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=feat, threshold=thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.root = self._build_tree(X, y)
        return self

    def _predict_sample(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(node.left, x)
        return self._predict_sample(node.right, x)

    def _predict_proba_sample(self, node, x):
        if node.is_leaf():
            return node.proba
        if x[node.feature] <= node.threshold:
            return self._predict_proba_sample(node.left, x)
        return self._predict_proba_sample(node.right, x)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_sample(self.root, x) for x in X])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_proba_sample(self.root, x) for x in X])
