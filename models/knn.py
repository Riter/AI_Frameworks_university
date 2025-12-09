import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy import sparse


class CustomKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple k-NN classifier implemented with numpy.

    Supports the parameters used in the notebook: n_neighbors, weights
    ('uniform' or 'distance'), and Minkowski distance with parameter p.

    Distances are computed in batches to avoid allocating gigantic
    intermediate arrays that can kill the kernel on larger datasets.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", p: int = 2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        X = self._to_numpy(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X should be 2D array-like")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors must not exceed number of samples")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        if self.p <= 0:
            raise ValueError("p must be positive")

        self._X = X.astype(float)
        self._train_norms = np.sum(self._X ** 2, axis=1)
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self._y_encoded = y_encoded
        self._n_classes = len(self.classes_)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = proba.argmax(axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        if not hasattr(self, "_X"):
            raise ValueError("This CustomKNeighborsClassifier instance is not fitted yet.")

        X = self._to_numpy(X).astype(float)
        neighbor_dists, neighbor_idx = self._kneighbors(X)
        weights = self._compute_weights(neighbor_dists)

        proba = np.zeros((X.shape[0], self._n_classes), dtype=float)
        for i in range(X.shape[0]):
            labels = self._y_encoded[neighbor_idx[i]]
            class_weights = np.bincount(
                labels, weights=weights[i], minlength=self._n_classes
            )
            proba[i] = class_weights / class_weights.sum()

        return proba

    def _kneighbors(self, X):
        n_samples = X.shape[0]
        n_train = self._X.shape[0]
        k = self.n_neighbors

        batch_size = self._resolve_batch_size(n_train)

        neighbor_indices = np.empty((n_samples, k), dtype=int)
        neighbor_dists = np.empty((n_samples, k), dtype=float)

        start = 0
        while start < n_samples:
            end = min(start + batch_size, n_samples)
            X_chunk = X[start:end]

            dists_chunk = self._pairwise_distances(X_chunk)
            idx_chunk = np.argpartition(dists_chunk, k - 1, axis=1)[:, :k]
            dists_neighbors = np.take_along_axis(dists_chunk, idx_chunk, axis=1)

            # Sort neighbors by distance for stability
            order = np.argsort(dists_neighbors, axis=1)
            idx_chunk = np.take_along_axis(idx_chunk, order, axis=1)
            dists_neighbors = np.take_along_axis(dists_neighbors, order, axis=1)

            neighbor_indices[start:end] = idx_chunk
            neighbor_dists[start:end] = dists_neighbors
            start = end

        return neighbor_dists, neighbor_indices

    def _pairwise_distances(self, X_chunk):
        if self.p == 2:
            return self._euclidean_distances(X_chunk)

        distances = np.zeros((X_chunk.shape[0], self._X.shape[0]), dtype=float)

        if self.p == 1:
            for j in range(self._X.shape[1]):
                distances += np.abs(X_chunk[:, None, j] - self._X[None, :, j])
            return distances

        for j in range(self._X.shape[1]):
            distances += np.abs(X_chunk[:, None, j] - self._X[None, :, j]) ** self.p
        return distances ** (1.0 / self.p)

    def _euclidean_distances(self, X_chunk):
        chunk_norms = np.sum(X_chunk ** 2, axis=1)[:, None]
        dist_sq = np.maximum(
            chunk_norms + self._train_norms[None, :] - 2 * X_chunk @ self._X.T, 0.0
        )
        return np.sqrt(dist_sq)

    def _resolve_batch_size(self, n_train):
        # Target roughly ~80MB for the distance matrix per chunk
        target_floats = 1e7
        suggested = int(target_floats // max(1, n_train))
        return max(1, min(512, suggested))

    def _compute_weights(self, neighbor_dists):
        if self.weights == "uniform":
            return np.ones_like(neighbor_dists, dtype=float)
        # Higher weight for closer neighbors; avoid division by zero
        weights = 1.0 / (neighbor_dists + 1e-9)
        weights[neighbor_dists == 0] = 1e12
        return weights

    @staticmethod
    def _to_numpy(X):
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X)


class CustomKNeighborsRegressor(BaseEstimator, RegressorMixin):
    """
    Simple k-NN regressor implemented with numpy.

    Mirrors the scikit-learn interface for the notebook usage: n_neighbors,
    weights ('uniform' or 'distance'), and Minkowski distance with parameter p.
    Uses batched distance computation to avoid excessive memory consumption.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", p: int = 2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        X = self._to_numpy(X)
        y = self._to_numpy(y)

        if X.ndim != 2:
            raise ValueError("X should be 2D array-like")
        if y.ndim == 1:
            y = y[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if self.n_neighbors > X.shape[0]:
            raise ValueError("n_neighbors must not exceed number of samples")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        if self.p <= 0:
            raise ValueError("p must be positive")

        self._X = X.astype(float)
        self._train_norms = np.sum(self._X ** 2, axis=1)
        self._y = y.astype(float)
        self._n_outputs = self._y.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "_X"):
            raise ValueError("This CustomKNeighborsRegressor instance is not fitted yet.")

        X = self._to_numpy(X).astype(float)
        neighbor_dists, neighbor_idx = self._kneighbors(X)
        weights = self._compute_weights(neighbor_dists)

        preds = np.zeros((X.shape[0], self._n_outputs), dtype=float)
        for i in range(X.shape[0]):
            neighbor_targets = self._y[neighbor_idx[i]]
            w = weights[i][:, None]
            weighted = (neighbor_targets * w).sum(axis=0) / w.sum()
            preds[i] = weighted

        if preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def _kneighbors(self, X):
        n_samples = X.shape[0]
        n_train = self._X.shape[0]
        k = self.n_neighbors

        batch_size = self._resolve_batch_size(n_train)

        neighbor_indices = np.empty((n_samples, k), dtype=int)
        neighbor_dists = np.empty((n_samples, k), dtype=float)

        start = 0
        while start < n_samples:
            end = min(start + batch_size, n_samples)
            X_chunk = X[start:end]

            dists_chunk = self._pairwise_distances(X_chunk)
            idx_chunk = np.argpartition(dists_chunk, k - 1, axis=1)[:, :k]
            dists_neighbors = np.take_along_axis(dists_chunk, idx_chunk, axis=1)

            order = np.argsort(dists_neighbors, axis=1)
            idx_chunk = np.take_along_axis(idx_chunk, order, axis=1)
            dists_neighbors = np.take_along_axis(dists_neighbors, order, axis=1)

            neighbor_indices[start:end] = idx_chunk
            neighbor_dists[start:end] = dists_neighbors
            start = end

        return neighbor_dists, neighbor_indices

    def _pairwise_distances(self, X_chunk):
        if self.p == 2:
            return self._euclidean_distances(X_chunk)

        distances = np.zeros((X_chunk.shape[0], self._X.shape[0]), dtype=float)

        if self.p == 1:
            for j in range(self._X.shape[1]):
                distances += np.abs(X_chunk[:, None, j] - self._X[None, :, j])
            return distances

        for j in range(self._X.shape[1]):
            distances += np.abs(X_chunk[:, None, j] - self._X[None, :, j]) ** self.p
        return distances ** (1.0 / self.p)

    def _euclidean_distances(self, X_chunk):
        chunk_norms = np.sum(X_chunk ** 2, axis=1)[:, None]
        dist_sq = np.maximum(
            chunk_norms + self._train_norms[None, :] - 2 * X_chunk @ self._X.T, 0.0
        )
        return np.sqrt(dist_sq)

    def _resolve_batch_size(self, n_train):
        target_floats = 1e7
        suggested = int(target_floats // max(1, n_train))
        return max(1, min(512, suggested))

    def _compute_weights(self, neighbor_dists):
        if self.weights == "uniform":
            return np.ones_like(neighbor_dists, dtype=float)
        weights = 1.0 / (neighbor_dists + 1e-9)
        weights[neighbor_dists == 0] = 1e12
        return weights

    @staticmethod
    def _to_numpy(X):
        if sparse.issparse(X):
            return X.toarray()
        return np.asarray(X)
