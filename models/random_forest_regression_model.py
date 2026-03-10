import numpy as np
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.decision_tree_regression_model import DecisionTreeRegressionModel


class RandomForestRegressionModel:

    def __init__(
        self,
        n_estimators:      int = 100,
        max_depth:         int = 10,
        min_samples_split: int = 5,
        min_samples_leaf:  int = 3,
        max_features=None,
        bootstrap:         bool = True,
        oob_score:         bool = True,
        n_jobs:            int = 4,
        random_state:      int = 42,
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.bootstrap         = bootstrap
        self.oob_score         = oob_score
        self.n_jobs            = n_jobs
        self.random_state      = random_state

        self._trees        = []
        self._oob_indices  = []
        self._n_features   = None
        self._n_samples    = None
        self.oob_score_    = None
        self.is_trained    = False

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return max(1, int(np.sqrt(n_features)))
        return min(self.max_features, n_features)

    def _build_single_tree(self, X, y, seed):
        """Build one tree on a bootstrap sample. Returns (tree, oob_indices)."""
        rng = np.random.default_rng(seed)
        n   = len(X)

        if self.bootstrap:
            in_bag   = rng.integers(0, n, size=n)
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[in_bag] = False
            oob_idx  = np.where(oob_mask)[0]
            X_boot, y_boot = X[in_bag], y[in_bag]
        else:
            oob_idx        = np.array([], dtype=int)
            X_boot, y_boot = X, y

        mf   = self._resolve_max_features(X.shape[1])
        tree = DecisionTreeRegressionModel(
            max_depth         = self.max_depth,
            min_samples_split = self.min_samples_split,
            min_samples_leaf  = self.min_samples_leaf,
            max_features      = mf,
        )
        tree.fit(X_boot, y_boot)
        return tree, oob_idx

    # Public API 

    def fit(self, X, y):

        X = X.astype(np.float64)
        y = y.astype(np.float64)

        self._n_samples  = len(X)
        self._n_features = X.shape[1]
        self._trees      = []
        self._oob_indices = []

        rng   = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=self.n_estimators)

        mf_str = self._resolve_max_features(self._n_features)
        print(f"  Building {self.n_estimators} trees "
              f"(max_features={mf_str}, max_depth={self.max_depth}) ...")

        results = [None] * self.n_estimators

        def _task(i):
            return i, *self._build_single_tree(X, y, int(seeds[i]))

        with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
            futures = {ex.submit(_task, i): i for i in range(self.n_estimators)}
            done = 0
            for fut in as_completed(futures):
                i, tree, oob_idx = fut.result()
                results[i] = (tree, oob_idx)
                done += 1
                if done % 20 == 0 or done == self.n_estimators:
                    print(f"    {done}/{self.n_estimators} trees done", end="\r")

        print()

        for tree, oob_idx in results:
            self._trees.append(tree)
            self._oob_indices.append(oob_idx)

        self.is_trained = True

        if self.oob_score and self.bootstrap:
            self.oob_score_ = self._compute_oob_score(X, y)
            print(f"  OOB R² = {self.oob_score_:.6f}")

    def predict(self, X):

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        X = np.atleast_2d(X).astype(np.float64)
        all_preds = np.vstack([t.predict(X) for t in self._trees])
        return all_preds.mean(axis=0)

    def predict_std(self, X):

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        X = np.atleast_2d(X).astype(np.float64)
        all_preds = np.vstack([t.predict(X) for t in self._trees])
        return all_preds.std(axis=0)

    def evaluate(self, X, y):

        preds  = self.predict(X)
        y      = y.astype(np.float64)
        mse    = float(np.mean((preds - y) ** 2))
        mae    = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((y - preds)       ** 2))
        ss_tot = float(np.sum((y - y.mean())    ** 2))
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"mse": mse, "mae": mae, "r2": r2}

    # OOB score 

    def _compute_oob_score(self, X, y):
        n          = self._n_samples
        oob_preds  = np.zeros(n, dtype=np.float64)
        oob_counts = np.zeros(n, dtype=np.int32)

        for tree, oob_idx in zip(self._trees, self._oob_indices):
            if len(oob_idx) == 0:
                continue
            preds = tree.predict(X[oob_idx])
            oob_preds[oob_idx]  += preds
            oob_counts[oob_idx] += 1

        valid       = oob_counts > 0
        oob_preds   = np.where(valid, oob_preds / np.maximum(oob_counts, 1), 0)
        y_v         = y[valid]
        p_v         = oob_preds[valid]
        if len(y_v) == 0:
            return 0.0
        ss_res = np.sum((y_v - p_v)       ** 2)
        ss_tot = np.sum((y_v - y_v.mean()) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-8))

    # Feature importances 

    @property
    def feature_importances(self):
        """Average feature importances across all trees."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        imp_matrix = np.vstack([t.feature_importances for t in self._trees])
        return imp_matrix.mean(axis=0)

    # Serialisation 

    def save(self, path="random_forest_model.pkl"):
        payload = {
            "trees":             self._trees,
            "oob_indices":       self._oob_indices,
            "n_features":        self._n_features,
            "n_samples":         self._n_samples,
            "oob_score_":        self.oob_score_,
            "n_estimators":      self.n_estimators,
            "max_depth":         self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf":  self.min_samples_leaf,
            "max_features":      self.max_features,
            "bootstrap":         self.bootstrap,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Model saved -> {path}")

    def load(self, path="random_forest_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._trees            = payload["trees"]
        self._oob_indices      = payload["oob_indices"]
        self._n_features       = payload["n_features"]
        self._n_samples        = payload["n_samples"]
        self.oob_score_        = payload["oob_score_"]
        self.n_estimators      = payload["n_estimators"]
        self.max_depth         = payload["max_depth"]
        self.min_samples_split = payload["min_samples_split"]
        self.min_samples_leaf  = payload["min_samples_leaf"]
        self.max_features      = payload["max_features"]
        self.bootstrap         = payload["bootstrap"]
        self.is_trained        = True
        print(f"Model loaded <- {path}")