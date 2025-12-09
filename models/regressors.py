import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class CustomLinearRegression(BaseEstimator, RegressorMixin):
    """
    Simple linear regression using the closed-form solution (pseudo-inverse).
    Supports fit_intercept and works with multi-output targets.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y[:, None]

        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X

        # Use pseudo-inverse to handle non full-rank cases
        w = np.linalg.pinv(X_aug) @ y

        if self.fit_intercept:
            self.intercept_ = w[0].ravel()
            self.coef_ = w[1:].T
        else:
            self.intercept_ = np.zeros(y.shape[1])
            self.coef_ = w.T

        # Align shapes with sklearn: (n_targets,) for intercept_ when single target
        if self.coef_.shape[0] == 1:
            self.intercept_ = self.intercept_[0]
            self.coef_ = self.coef_[0]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.coef_.ndim == 1:
            preds = X @ self.coef_ + (self.intercept_ if self.fit_intercept else 0.0)
        else:
            preds = X @ self.coef_.T
            if self.fit_intercept:
                preds += self.intercept_
        return preds


class CustomLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Basic binary logistic regression with optional L1/L2 regularization.
    Implements fit, predict, predict_proba and accepts the parameters used
    in the notebooks (penalty, C, solver, class_weight, max_iter, tol).
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        class_weight=None,
        max_iter: int = 100,
        tol: float = 1e-4,
        learning_rate: float = 0.1,
    ):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_, y_bin = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("CustomLogisticRegression supports binary classification only.")

        n_samples, n_features = X.shape
        X_aug = np.hstack([np.ones((n_samples, 1)), X])

        sample_weights = self._compute_sample_weights(y_bin)

        w = np.zeros(n_features + 1, dtype=float)
        reg_strength = 1.0 / self.C if self.C != 0 else 0.0

        for _ in range(self.max_iter):
            logits = X_aug @ w
            probs = 1.0 / (1.0 + np.exp(-logits))

            error = (probs - y_bin) * sample_weights
            grad = X_aug.T @ error / n_samples

            if self.penalty == "l2":
                grad[1:] += reg_strength * w[1:]
            elif self.penalty == "l1":
                grad[1:] += reg_strength * np.sign(w[1:])

            w_new = w - self.learning_rate * grad

            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                break
            w = w_new

        self.intercept_ = w[0]
        self.coef_ = w[1:][None, :]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        logits = X_aug @ np.hstack([self.intercept_, self.coef_.ravel()])
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        probs = np.column_stack([1 - probs_pos, probs_pos])
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        labels = (probs[:, 1] >= 0.5).astype(int)
        return self.classes_[labels]

    def _compute_sample_weights(self, y_bin):
        n_samples = len(y_bin)
        if self.class_weight is None:
            return np.ones(n_samples, dtype=float)
        if self.class_weight == "balanced":
            counts = np.bincount(y_bin)
            weights = np.zeros_like(y_bin, dtype=float)
            for cls_idx, cnt in enumerate(counts):
                weights[y_bin == cls_idx] = n_samples / (len(counts) * cnt)
            return weights
        if isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(cls, 1.0) for cls in y_bin], dtype=float)
        return np.ones(n_samples, dtype=float)
