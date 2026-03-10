import numpy as np
import pickle
import os

class MultipleRegressionModel:

    def __init__(self):
        self.weights = None   # shape: (n_features + 1,) including bias
        self.is_trained = False

    def _add_bias(self, X):
        """Prepend a column of ones for the bias term."""
        ones = np.ones((X.shape[0], 1), dtype=np.float32)
        return np.hstack([ones, X])

    def fit(self, X, y):
 
        X_b = self._add_bias(X)

        # Closed-form OLS: w = (X^T X)^{-1} X^T y
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y

        # Use pseudoinverse for numerical stability
        self.weights = np.linalg.pinv(XtX) @ Xty
        self.is_trained = True

        print(f"Model trained. Weights: {self.weights}")

    def predict(self, X):
        
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_b = self._add_bias(X)
        return X_b @ self.weights

    def evaluate(self, X, y):
        
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {"mse": float(mse), "r2": float(r2)}

    def save(self, path="multiple_regression_model.pkl"):
        """Save model weights to disk."""
        with open(path, "wb") as f:
            pickle.dump(self.weights, f)
        print(f"Model saved to {path}")

    def load(self, path="multiple_regression_model.pkl"):
        """Load model weights from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at: {path}")
        with open(path, "rb") as f:
            self.weights = pickle.load(f)
        self.is_trained = True
        print(f"Model loaded from {path}")