import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy import sparse


def _to_numpy(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


class _TreeNode:
    __slots__ = ["feature", "threshold", "left", "right", "value", "proba"]

    def __init__(self, value=None, proba=None, feature=None, threshold=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba


class CustomDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple decision tree classifier with Gini impurity.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        class_weight=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def fit(self, X, y):
        X = _to_numpy(X).astype(float)
        y = np.asarray(y)

        self.classes_, y_int = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)
        self._tree = self._build_tree(
            X, y_int, depth=0, rng=rng, n_features=n_features, sample_weight=self._sample_weights(y_int)
        )
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        class_idx = proba.argmax(axis=1)
        return self.classes_[class_idx]

    def predict_proba(self, X):
        X = _to_numpy(X).astype(float)
        proba = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i, x in enumerate(X):
            node = self._tree
            while node.left is not None and node.right is not None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            proba[i] = node.proba
        return proba

    # Tree construction helpers
    def _build_tree(self, X, y, depth, rng, n_features, sample_weight):
        num_samples = X.shape[0]
        value = self._leaf_value(y, sample_weight)
        proba = self._leaf_proba(y, sample_weight)

        if self._should_stop(depth, num_samples):
            return _TreeNode(value=value, proba=proba)

        feature_subset = self._choose_features(n_features, rng)
        best = self._best_split(X, y, sample_weight, feature_subset)

        if best is None:
            return _TreeNode(value=value, proba=proba)

        left_idx, right_idx, feature, threshold, gain = best
        if gain <= max(self.min_impurity_decrease, self.ccp_alpha):
            return _TreeNode(value=value, proba=proba)

        left_node = self._build_tree(X[left_idx], y[left_idx], depth + 1, rng, n_features, sample_weight[left_idx])
        right_node = self._build_tree(
            X[right_idx], y[right_idx], depth + 1, rng, n_features, sample_weight[right_idx]
        )

        return _TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node, value=value, proba=proba)

    def _should_stop(self, depth, num_samples):
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if num_samples < self.min_samples_split:
            return True
        return False

    def _choose_features(self, n_features, rng):
        mf = self.max_features
        if mf is None:
            k = n_features
        elif isinstance(mf, str):
            if mf == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            else:
                k = n_features
        elif isinstance(mf, float):
            k = max(1, int(np.ceil(mf * n_features)))
        else:
            k = int(mf)
            k = max(1, min(k, n_features))
        if k == n_features:
            return np.arange(n_features)
        return rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y, sample_weight, feature_subset):
        n_samples = X.shape[0]
        best_gain = -np.inf
        best_split = None

        parent_impurity = self._gini(y, sample_weight)

        for feat in feature_subset:
            sorted_idx = np.argsort(X[:, feat])
            x_sorted = X[sorted_idx, feat]
            y_sorted = y[sorted_idx]
            w_sorted = sample_weight[sorted_idx]

            unique_mask = np.diff(x_sorted) != 0
            if not np.any(unique_mask):
                continue

            cum_w = np.cumsum(w_sorted)
            total_w = cum_w[-1]

            # cumulative class weights
            class_w = np.zeros((len(x_sorted), self.n_classes_))
            for cls in range(self.n_classes_):
                class_w[:, cls] = np.cumsum(w_sorted * (y_sorted == cls))

            for i in np.where(unique_mask)[0]:
                left_count = cum_w[i]
                right_count = total_w - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                left_classes = class_w[i]
                right_classes = class_w[-1] - class_w[i]

                left_imp = self._gini_counts(left_classes, left_count)
                right_imp = self._gini_counts(right_classes, right_count)

                gain = parent_impurity - (left_count / total_w) * left_imp - (right_count / total_w) * right_imp

                if gain > best_gain + 1e-12 and gain > self.min_impurity_decrease:
                    threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
                    best_gain = gain
                    left_mask = X[:, feat] <= threshold
                    right_mask = ~left_mask
                    best_split = (left_mask, right_mask, feat, threshold, gain)

        if best_split is None:
            return None

        left_mask, right_mask, feat, threshold, gain = best_split
        return np.where(left_mask)[0], np.where(right_mask)[0], feat, threshold, gain

    def _gini(self, y, sample_weight):
        total_w = sample_weight.sum()
        counts = np.zeros(self.n_classes_)
        for cls in range(self.n_classes_):
            counts[cls] = sample_weight[y == cls].sum()
        return self._gini_counts(counts, total_w)

    @staticmethod
    def _gini_counts(counts, total):
        if total == 0:
            return 0.0
        probs = counts / total
        return 1.0 - np.sum(probs ** 2)

    def _leaf_value(self, y, sample_weight):
        counts = np.zeros(self.n_classes_)
        for cls in range(self.n_classes_):
            counts[cls] = sample_weight[y == cls].sum()
        return np.argmax(counts)

    def _leaf_proba(self, y, sample_weight):
        counts = np.zeros(self.n_classes_)
        for cls in range(self.n_classes_):
            counts[cls] = sample_weight[y == cls].sum()
        if counts.sum() == 0:
            return np.ones(self.n_classes_) / self.n_classes_
        return counts / counts.sum()

    def _sample_weights(self, y_int):
        n = len(y_int)
        weights = np.ones(n, dtype=float)
        if self.class_weight is None:
            return weights
        if self.class_weight == "balanced":
            counts = np.bincount(y_int, minlength=self.n_classes_)
            weights = np.array([1.0 / counts[cls] if counts[cls] > 0 else 0.0 for cls in y_int], dtype=float)
            weights *= n / weights.sum()
            return weights
        if isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(cls, 1.0) for cls in y_int], dtype=float)
        return weights


class CustomDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Simple decision tree regressor using variance reduction.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        random_state=None,
        max_leaf_nodes=None,
        min_weight_fraction_leaf=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_weight_fraction_leaf = min_weight_fraction_leaf

    def fit(self, X, y):
        X = _to_numpy(X).astype(float)
        y = np.asarray(y, dtype=float)
        if y.ndim > 1:
            y = y.ravel()

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self._tree = self._build_tree(X, y, depth=0, rng=rng, n_features=n_features)
        return self

    def predict(self, X):
        X = _to_numpy(X).astype(float)
        preds = np.zeros(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            node = self._tree
            while node.left is not None and node.right is not None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            preds[i] = node.value
        return preds

    def _build_tree(self, X, y, depth, rng, n_features):
        num_samples = X.shape[0]
        value = y.mean()

        if self._should_stop(depth, num_samples):
            return _TreeNode(value=value)

        feature_subset = self._choose_features(n_features, rng)
        best = self._best_split(X, y, feature_subset)

        if best is None:
            return _TreeNode(value=value)

        left_idx, right_idx, feature, threshold, gain = best
        if gain <= max(self.min_impurity_decrease, self.ccp_alpha):
            return _TreeNode(value=value)

        left_node = self._build_tree(X[left_idx], y[left_idx], depth + 1, rng, n_features)
        right_node = self._build_tree(X[right_idx], y[right_idx], depth + 1, rng, n_features)
        return _TreeNode(feature=feature, threshold=threshold, left=left_node, right=right_node, value=value)

    def _should_stop(self, depth, num_samples):
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if num_samples < self.min_samples_split:
            return True
        if self.max_leaf_nodes is not None and depth >= self.max_leaf_nodes:
            return True
        return False

    def _choose_features(self, n_features, rng):
        mf = self.max_features
        if mf is None:
            k = n_features
        elif isinstance(mf, str):
            if mf == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            else:
                k = n_features
        elif isinstance(mf, float):
            k = max(1, int(np.ceil(mf * n_features)))
        else:
            k = int(mf)
            k = max(1, min(k, n_features))
        if k == n_features:
            return np.arange(n_features)
        return rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y, feature_subset):
        n_samples = X.shape[0]
        best_gain = -np.inf
        best_split = None

        parent_impurity = self._variance(y)

        for feat in feature_subset:
            sorted_idx = np.argsort(X[:, feat])
            x_sorted = X[sorted_idx, feat]
            y_sorted = y[sorted_idx]

            unique_mask = np.diff(x_sorted) != 0
            if not np.any(unique_mask):
                continue

            cum_sum = np.cumsum(y_sorted)
            cum_sq = np.cumsum(y_sorted ** 2)

            total_sum = cum_sum[-1]
            total_sq = cum_sq[-1]

            for i in np.where(unique_mask)[0]:
                left_count = i + 1
                right_count = n_samples - left_count
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                left_sum = cum_sum[i]
                left_sq = cum_sq[i]

                right_sum = total_sum - left_sum
                right_sq = total_sq - left_sq

                left_imp = self._variance_parts(left_sum, left_sq, left_count)
                right_imp = self._variance_parts(right_sum, right_sq, right_count)

                gain = parent_impurity - (left_count / n_samples) * left_imp - (right_count / n_samples) * right_imp

                if gain > best_gain + 1e-12 and gain > self.min_impurity_decrease:
                    threshold = (x_sorted[i] + x_sorted[i + 1]) / 2.0
                    best_gain = gain
                    left_mask = X[:, feat] <= threshold
                    right_mask = ~left_mask
                    best_split = (np.where(left_mask)[0], np.where(right_mask)[0], feat, threshold, gain)

        if best_split is None:
            return None
        return best_split

    @staticmethod
    def _variance(y):
        if len(y) == 0:
            return 0.0
        mean = y.mean()
        return np.mean((y - mean) ** 2)

    @staticmethod
    def _variance_parts(sum_y, sum_sq, count):
        if count == 0:
            return 0.0
        mean = sum_y / count
        return sum_sq / count - mean ** 2


class CustomRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Basic random forest classifier that reuses CustomDecisionTreeClassifier.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        bootstrap=True,
        n_jobs=None,
        random_state=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y):
        X = _to_numpy(X).astype(float)
        y = np.asarray(y)
        n_samples = X.shape[0]

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")

        rng = np.random.default_rng(self.random_state)
        self.classes_, _ = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        self.estimators_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n_samples, size=n_samples)
            else:
                idx = np.arange(n_samples)

            tree = CustomDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                random_state=rng.integers(0, 1_000_000_000),
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X):
        X = _to_numpy(X).astype(float)
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        return all_proba.mean(axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        return self.classes_[class_idx]


class CustomRandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    Basic random forest regressor reusing CustomDecisionTreeRegressor.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        n_jobs=None,
        random_state=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        max_samples=None,
        n_bins=32,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.n_bins = n_bins

    def fit(self, X, y):
        X = _to_numpy(X).astype(float)
        X = self._apply_binning_fit(X)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            sample_size = self._resolve_max_samples(n_samples)
            if self.bootstrap:
                idx = rng.integers(0, n_samples, size=sample_size)
            else:
                idx = rng.choice(n_samples, size=sample_size, replace=False)

            tree = CustomDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                random_state=rng.integers(0, 1_000_000_000),
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        X = _to_numpy(X).astype(float)
        X = self._apply_binning_predict(X)
        preds = np.array([tree.predict(X) for tree in self.estimators_])
        return preds.mean(axis=0)

    def _resolve_max_samples(self, n_samples):
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, float):
            if not (0 < self.max_samples <= 1):
                raise ValueError("max_samples as float must be in (0, 1]")
            return max(1, int(np.ceil(self.max_samples * n_samples)))
        if isinstance(self.max_samples, int):
            return max(1, min(self.max_samples, n_samples))
        raise ValueError("max_samples must be None, float, or int")

    def _apply_binning_fit(self, X):
        # Quantile binning for high-cardinality features to speed up tree splits.
        self._bin_edges_ = []
        X_binned = np.empty_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) <= 10:
                self._bin_edges_.append(None)
                X_binned[:, j] = col
                continue
            edges = np.quantile(col, np.linspace(0, 1, self.n_bins + 1))
            edges = np.unique(edges)
            if len(edges) <= 2:
                self._bin_edges_.append(None)
                X_binned[:, j] = col
                continue
            bins = edges[1:-1]
            X_binned[:, j] = np.digitize(col, bins, right=False)
            self._bin_edges_.append(bins)
        return X_binned


class CustomGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    Simple gradient boosting for regression with squared error loss.
    Uses CustomDecisionTreeRegressor as base learners.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        max_samples=None,
        n_bins: int = 32,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_samples = max_samples
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(self, X, y):
        X = _to_numpy(X).astype(float)
        X = self._apply_binning_fit(X)
        y = np.asarray(y, dtype=float)

        self.init_ = y.mean()
        self.estimators_ = []
        rng = np.random.default_rng(self.random_state)

        y_pred = np.full_like(y, self.init_, dtype=float)
        for _ in range(self.n_estimators):
            residual = y - y_pred

            idx = self._bootstrap_indices(len(y), rng)
            tree = CustomDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.integers(0, 1_000_000_000),
            )
            tree.fit(X[idx], residual[idx])
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X = _to_numpy(X).astype(float)
        X = self._apply_binning_predict(X)
        pred = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.estimators_:
            pred += self.learning_rate * tree.predict(X)
        return pred

    def _bootstrap_indices(self, n_samples, rng):
        target = self._resolve_sample_size(n_samples)
        if target >= n_samples:
            return np.arange(n_samples)
        return rng.choice(n_samples, size=target, replace=False)

    def _resolve_sample_size(self, n_samples):
        if self.max_samples is not None:
            if isinstance(self.max_samples, float):
                if not (0 < self.max_samples <= 1):
                    raise ValueError("max_samples as float must be in (0, 1]")
                return max(1, int(np.ceil(self.max_samples * n_samples)))
            if isinstance(self.max_samples, int):
                return max(1, min(self.max_samples, n_samples))
            raise ValueError("max_samples must be None, float, or int")
        if self.subsample >= 1.0:
            return n_samples
        return max(1, int(np.ceil(self.subsample * n_samples)))

    def _apply_binning_fit(self, X):
        self._bin_edges_ = []
        X_binned = np.empty_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) <= 10:
                self._bin_edges_.append(None)
                X_binned[:, j] = col
                continue
            edges = np.quantile(col, np.linspace(0, 1, self.n_bins + 1))
            edges = np.unique(edges)
            if len(edges) <= 2:
                self._bin_edges_.append(None)
                X_binned[:, j] = col
                continue
            bins = edges[1:-1]
            X_binned[:, j] = np.digitize(col, bins, right=False)
            self._bin_edges_.append(bins)
        return X_binned

    def _apply_binning_predict(self, X):
        if not hasattr(self, "_bin_edges_"):
            return X
        X_binned = np.empty_like(X)
        for j in range(X.shape[1]):
            bins = self._bin_edges_[j]
            col = X[:, j]
            if bins is None:
                X_binned[:, j] = col
            else:
                X_binned[:, j] = np.digitize(col, bins, right=False)
        return X_binned


class CustomGradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Gradient boosting for binary classification with logistic loss.
    Reuses CustomDecisionTreeRegressor as base learners.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth=None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        X = _to_numpy(X).astype(float)
        y = np.asarray(y)

        self.classes_, y_bin = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("CustomGradientBoostingClassifier supports binary classification only.")

        # initial log-odds
        pos_ratio = np.clip(y_bin.mean(), 1e-6, 1 - 1e-6)
        self.init_ = np.log(pos_ratio / (1 - pos_ratio))

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        raw_scores = np.full_like(y_bin, self.init_, dtype=float)

        for _ in range(self.n_estimators):
            prob = self._sigmoid(raw_scores)
            residual = y_bin - prob

            idx = self._bootstrap_indices(len(y_bin), rng)
            tree = CustomDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.integers(0, 1_000_000_000),
            )
            tree.fit(X[idx], residual[idx])
            update = tree.predict(X)
            raw_scores += self.learning_rate * update
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X):
        X = _to_numpy(X).astype(float)
        raw = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.estimators_:
            raw += self.learning_rate * tree.predict(X)
        prob_pos = self._sigmoid(raw)
        prob_neg = 1 - prob_pos
        return np.column_stack([prob_neg, prob_pos])

    def predict(self, X):
        proba = self.predict_proba(X)
        labels = (proba[:, 1] >= 0.5).astype(int)
        return self.classes_[labels]

    def _bootstrap_indices(self, n_samples, rng):
        if self.subsample >= 1.0:
            return np.arange(n_samples)
        size = max(1, int(np.ceil(self.subsample * n_samples)))
        return rng.choice(n_samples, size=size, replace=False)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def _apply_binning_predict(self, X):
        if not hasattr(self, "_bin_edges_"):
            return X
        X_binned = np.empty_like(X)
        for j in range(X.shape[1]):
            bins = self._bin_edges_[j]
            col = X[:, j]
            if bins is None:
                X_binned[:, j] = col
            else:
                X_binned[:, j] = np.digitize(col, bins, right=False)
        return X_binned
