import numpy as np
import pickle
import os

class SVRModel:

    def __init__(
        self,
        C:        float = 100.0,
        epsilon:  float = 0.001,
        kernel:   str   = "rbf",
        gamma           = "scale",
        degree:   int   = 3,
        coef0:    float = 1.0,
        max_iter: int   = 2000,
        tol:      float = 1e-3,
    ):
        self.C        = C
        self.epsilon  = epsilon
        self.kernel   = kernel
        self.gamma    = gamma
        self.degree   = degree
        self.coef0    = coef0
        self.max_iter = max_iter
        self.tol      = tol

        self._alpha_sv  = None   # (αᵢ - αᵢ*) for support vectors only
        self._X_sv      = None   # support vector feature matrix (normalised)
        self._b         = 0.0
        self._gamma_val = None   # resolved gamma float
        self._feat_mean = None   # Z-score stats
        self._feat_std  = None
        self.is_trained = False

    # Kernel 

    def _K(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:

        if self.kernel == "linear":
            return A @ B.T

        elif self.kernel == "rbf":
            # ||a-b||² = ||a||² + ||b||² - 2 a·b
            aa = np.sum(A ** 2, axis=1, keepdims=True)
            bb = np.sum(B ** 2, axis=1, keepdims=True)
            d2 = np.maximum(aa + bb.T - 2.0 * (A @ B.T), 0.0)
            return np.exp(-self._gamma_val * d2)

        elif self.kernel == "poly":
            return (self._gamma_val * (A @ B.T) + self.coef0) ** self.degree

        else:
            raise ValueError(f"Unknown kernel: '{self.kernel}'")

    # Normalisation 

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        std = np.where(self._feat_std == 0, 1.0, self._feat_std)
        return (X - self._feat_mean) / std

    # SMO solver 

    def _smo(self, X: np.ndarray, y: np.ndarray):

        n   = len(y)
        w   = np.zeros(n, dtype=np.float64)
        b   = 0.0
        eps = self.epsilon
        C   = self.C

        # Pre-compute full kernel matrix once  O(n²)
        K = self._K(X, X)                       # (n, n)

        # f[i] = Σⱼ w[j] K[j,i] + b  (prediction for each training sample)
        f = K @ w + b                            # (n,)

        for _pass in range(self.max_iter):
            n_changed = 0

            for i in range(n):
                Ei = f[i] - y[i]                # prediction error at i

                # Check KKT violation
                kkt_upper = (Ei >  eps + self.tol) and (w[i] <  C)
                kkt_lower = (Ei < -eps - self.tol) and (w[i] > -C)
                if not (kkt_upper or kkt_lower):
                    continue

                # Heuristic: choose j with maximum |Ej - Ei|
                E = f - y
                diff = np.abs(E - Ei)
                diff[i] = -1.0
                j = int(np.argmax(diff))

                Ej  = E[j]
                eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
                if eta <= 1e-10:
                    continue

                wi_old = w[i]
                wj_old = w[j]

                # Compute update step
                if kkt_upper:
                    step = (Ei - eps - (Ej + eps)) / eta
                else:
                    step = (Ei + eps - (Ej - eps)) / eta

                # Apply and project into [-C, C]
                w[i] = float(np.clip(wi_old - step, -C, C))
                w[j] = float(np.clip(wj_old + step, -C, C))

                di = w[i] - wi_old
                dj = w[j] - wj_old

                if abs(di) < 1e-10:
                    continue

                # Incremental update of f
                f += di * K[:, i] + dj * K[:, j]

                # Update bias from newly free support vectors
                b_cands = []
                for idx, widx, Eidx in [(i, w[i], f[i] - y[i]),
                                        (j, w[j], f[j] - y[j])]:
                    if -C < widx < C:
                        target = eps if widx > 0 else -eps
                        b_cands.append(b + target - Eidx)
                if b_cands:
                    b_new = float(np.mean(b_cands))
                    f    += (b_new - b)
                    b     = b_new

                n_changed += 1

            if n_changed == 0:
                break   # converged

        # Store only support vectors  (|w| > tol)
        sv               = np.abs(w) > self.tol
        self._X_sv       = X[sv]
        self._alpha_sv   = w[sv]
        self._b          = b

    # Public API 

    def fit(self, X: np.ndarray, y: np.ndarray):

        X = X.astype(np.float64)
        y = y.astype(np.float64)

        # Fit Z-score normalisation on training data
        self._feat_mean = X.mean(axis=0)
        self._feat_std  = X.std(axis=0)
        X_norm = self._normalise(X)

        # Resolve gamma
        if self.gamma == "scale":
            var = float(X_norm.var())
            self._gamma_val = 1.0 / (X_norm.shape[1] * var + 1e-10)
        elif self.gamma == "auto":
            self._gamma_val = 1.0 / X_norm.shape[1]
        else:
            self._gamma_val = float(self.gamma)

        print(f"\n  SVR training:  kernel={self.kernel}  C={self.C}  "
              f"epsilon={self.epsilon}  gamma={self._gamma_val:.5f}")

        self._smo(X_norm, y)
        self.is_trained = True

        pct = 100.0 * len(self._alpha_sv) / len(X)
        print(f"  Support vectors: {len(self._alpha_sv)} / {len(X)} "
              f"({pct:.1f}%)")

    def predict(self, X: np.ndarray) -> np.ndarray:

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        X      = np.atleast_2d(X).astype(np.float64)
        X_norm = self._normalise(X)
        # K(sv, test) → (n_sv, n_test);  alpha_sv @ K → (n_test,)
        K = self._K(self._X_sv, X_norm)
        return self._alpha_sv @ K + self._b

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:

        preds  = self.predict(X)
        y      = y.astype(np.float64)
        mse    = float(np.mean((preds - y) ** 2))
        mae    = float(np.mean(np.abs(preds - y)))
        ss_res = float(np.sum((y - preds)       ** 2))
        ss_tot = float(np.sum((y - y.mean())    ** 2))
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"mse": mse, "mae": mae, "r2": r2}

    @property
    def n_support_vectors(self) -> int:
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        return len(self._alpha_sv)

    # Serialisation 

    def save(self, path: str = "svr_model.pkl"):
        payload = {
            "alpha_sv":  self._alpha_sv,
            "X_sv":      self._X_sv,
            "b":         self._b,
            "gamma_val": self._gamma_val,
            "feat_mean": self._feat_mean,
            "feat_std":  self._feat_std,
            "C":         self.C,
            "epsilon":   self.epsilon,
            "kernel":    self.kernel,
            "degree":    self.degree,
            "coef0":     self.coef0,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Model saved -> {path}")

    def load(self, path: str = "svr_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model at: {path}")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._alpha_sv  = payload["alpha_sv"]
        self._X_sv      = payload["X_sv"]
        self._b         = payload["b"]
        self._gamma_val = payload["gamma_val"]
        self._feat_mean = payload["feat_mean"]
        self._feat_std  = payload["feat_std"]
        self.C          = payload["C"]
        self.epsilon    = payload["epsilon"]
        self.kernel     = payload["kernel"]
        self.degree     = payload["degree"]
        self.coef0      = payload["coef0"]
        self.is_trained = True
        print(f"Model loaded <- {path}")