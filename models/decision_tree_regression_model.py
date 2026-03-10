import numpy as np
import pickle
import os

# Internal node / leaf representation 
class _Node:

    __slots__ = (
        "feature_idx", "threshold",   # split parameters (internal nodes)
        "left", "right",              # child _Node references
        "value",                      # leaf prediction (None for internal)
        "mse_reduction",              # MSE gain from this split
        "n_samples",                  # number of training samples here
    )

    def __init__(self):
        self.feature_idx   = None
        self.threshold     = None
        self.left          = None
        self.right         = None
        self.value         = None
        self.mse_reduction = 0.0
        self.n_samples     = 0

    @property
    def is_leaf(self):
        return self.value is not None

class DecisionTreeRegressionModel:

    def __init__(
        self,
        max_depth:          int        = 8,
        min_samples_split:  int        = 10,
        min_samples_leaf:   int        = 5,
        max_features:       int | None = None,
    ):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.min_samples_leaf   = min_samples_leaf
        self.max_features       = max_features

        self._root              = None
        self._n_features        = None
        self._feature_importances = None
        self.is_trained         = False

    @staticmethod
    def _mse(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.mean((y - y.mean()) ** 2))

    def _best_split(self, X: np.ndarray, y: np.ndarray):

        n, n_feat = X.shape
        parent_mse = self._mse(y)
        best_gain  = 0.0
        best_feat  = None
        best_thr   = None

        # Optionally subsample features (for Random Forest compatibility)
        if self.max_features is not None:
            feat_indices = np.random.choice(n_feat,
                                            min(self.max_features, n_feat),
                                            replace=False)
        else:
            feat_indices = np.arange(n_feat)

        for fi in feat_indices:
            col        = X[:, fi]
            thresholds = np.unique(col)

            # For efficiency, only test midpoints between consecutive values
            if len(thresholds) > 50:
                thresholds = np.percentile(col, np.linspace(0, 100, 50))

            for thr in thresholds:
                left_mask  = col <= thr
                right_mask = ~left_mask

                n_l = left_mask.sum()
                n_r = right_mask.sum()

                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                mse_l = self._mse(y[left_mask])
                mse_r = self._mse(y[right_mask])

                weighted_child_mse = (n_l * mse_l + n_r * mse_r) / n
                gain = parent_mse - weighted_child_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feat = fi
                    best_thr  = thr

        return best_feat, best_thr, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:

        node           = _Node()
        node.n_samples = len(y)

        # Leaf conditions 
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or np.std(y) < 1e-9):          # all targets identical
            node.value = float(y.mean())
            return node

        feat, thr, gain = self._best_split(X, y)

        # No useful split found → leaf
        if feat is None:
            node.value = float(y.mean())
            return node

        # Internal node 
        node.feature_idx   = feat
        node.threshold     = thr
        node.mse_reduction = gain

        # Accumulate feature importances (weighted by n_samples)
        self._feature_importances[feat] += gain * len(y)

        mask          = X[:, feat] <= thr
        node.left     = self._build(X[mask],  y[mask],  depth + 1)
        node.right    = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_one(self, x: np.ndarray) -> float:

        node = self._root
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    # Public API 

    def fit(self, X: np.ndarray, y: np.ndarray):

    #Train the decision tree.
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        self._n_features          = X.shape[1]
        self._feature_importances = np.zeros(self._n_features, dtype=np.float64)

        self._root      = self._build(X, y, depth=0)
        self.is_trained = True

        # Normalise importances to sum to 1
        total = self._feature_importances.sum()
        if total > 0:
            self._feature_importances /= total

        self._print_fit_summary(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        X = np.atleast_2d(X).astype(np.float64)
        return np.array([self._predict_one(x) for x in X])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:

        preds  = self.predict(X)
        y      = y.astype(np.float64)
        mse    = float(np.mean((preds - y) ** 2))
        ss_res = float(np.sum((y - preds)    ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"mse": mse, "r2": r2}

    @property
    def feature_importances(self) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        return self._feature_importances

    # Serialisation 

    def save(self, path: str = "decision_tree_model.pkl"):
        payload = {
            "root":                 self._root,
            "n_features":          self._n_features,
            "feature_importances": self._feature_importances,
            "max_depth":           self.max_depth,
            "min_samples_split":   self.min_samples_split,
            "min_samples_leaf":    self.min_samples_leaf,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Model saved → {path}")

    def load(self, path: str = "decision_tree_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._root                = payload["root"]
        self._n_features          = payload["n_features"]
        self._feature_importances = payload["feature_importances"]
        self.max_depth            = payload["max_depth"]
        self.min_samples_split    = payload["min_samples_split"]
        self.min_samples_leaf     = payload["min_samples_leaf"]
        self.is_trained           = True
        print(f"Model loaded ← {path}")

    # Diagnostics 

    def _print_fit_summary(self, X, y):
        depth = self._tree_depth(self._root)
        leaves = self._count_leaves(self._root)
        print(f"\nTree built  →  depth={depth}  leaves={leaves}  "
              f"samples={len(y)}")

    def _tree_depth(self, node) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._tree_depth(node.left),
                       self._tree_depth(node.right))

    def _count_leaves(self, node) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def print_tree(self, feature_names=None, max_display_depth=4):
        """Print an ASCII representation of the tree (up to max_display_depth)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(self._n_features)]

        print(f"\n{'─'*55}")
        print(f"  Decision Tree  (showing up to depth {max_display_depth})")
        print(f"{'─'*55}")
        self._print_node(self._root, feature_names, "", True, 0,
                         max_display_depth)
        print(f"{'─'*55}")

    def _print_node(self, node, feat_names, prefix, is_left, depth,
                    max_depth):
        if node is None:
            return
        connector = "├── " if is_left else "└── "
        if node.is_leaf:
            print(f"{prefix}{connector}LEAF  val={node.value:+.4f}"
                  f"  n={node.n_samples}")
        else:
            fname = feat_names[node.feature_idx]
            print(f"{prefix}{connector}[{fname} ≤ {node.threshold:.4f}]"
                  f"  gain={node.mse_reduction:.5f}"
                  f"  n={node.n_samples}")
            if depth < max_depth:
                ext = "│   " if is_left else "    "
                self._print_node(node.left,  feat_names, prefix + ext,
                                 True,  depth + 1, max_depth)
                self._print_node(node.right, feat_names, prefix + ext,
                                 False, depth + 1, max_depth)
            else:
                ext = "│   " if is_left else "    "
                print(f"{prefix}{ext}    ... (truncated)")