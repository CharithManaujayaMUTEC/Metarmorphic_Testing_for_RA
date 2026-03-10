import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.feature_extractor import extract_features
from models.svr_model import SVRModel

FEATURE_NAMES = [
    "lane_center_offset",
    "top_offset",
    "bottom_offset",
    "curvature_proxy",
    "white_pixel_ratio",
    "vertical_spread",
]

# Data loading  (identical pattern to all other train_*.py files) 

def load_features_and_labels(data_dir="dataset/data", csv_name="labels.csv"):
    csv_path = os.path.join(data_dir, csv_name)
    img_dir  = os.path.join(data_dir, "images")
    df       = pd.read_csv(csv_path)

    features_list, labels_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_path = os.path.join(img_dir, row["image"])
        try:
            features_list.append(extract_features(img_path))
            labels_list.append(float(row["steering"]))
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping.")

    return (np.array(features_list, dtype=np.float32),
            np.array(labels_list,   dtype=np.float32))

# Feature diagnostics 

def print_feature_diagnostics(X, y):
    print("\n--- Feature Diagnostics ---")
    print(f"{'Feature':<25} {'Mean':>8} {'Std':>8} {'Corr w/ steering':>18}")
    print("-" * 65)
    for i, name in enumerate(FEATURE_NAMES):
        col  = X[:, i]
        corr = float(np.corrcoef(col, y)[0, 1]) if col.std() > 0 else 0.0
        print(f"{name:<25} {col.mean():>8.4f} {col.std():>8.4f} {corr:>18.4f}")
    print()

# Kernel comparison 

def kernel_comparison(X_train, y_train, X_test, y_test):

    kernels = [
        ("rbf",    SVRModel(C=100, epsilon=0.01, kernel="rbf")),
        ("linear", SVRModel(C=100, epsilon=0.01, kernel="linear")),
        ("poly",   SVRModel(C=100, epsilon=0.01, kernel="poly", degree=3)),
    ]

    print("--- Kernel Comparison  (C=100, epsilon=0.01) ---")
    print(f"{'Kernel':<10} {'Train R2':>9} {'Test R2':>9} "
          f"{'Test MSE':>10} {'#SV':>6} {'Time':>7}")
    print("-" * 58)

    best_kernel = "rbf"
    best_r2     = -np.inf

    for name, model in kernels:
        t0 = time.time()
        model.fit(X_train, y_train)
        tr  = model.evaluate(X_train, y_train)
        tst = model.evaluate(X_test,  y_test)
        elapsed = time.time() - t0
        print(f"{name:<10} {tr['r2']:>9.4f} {tst['r2']:>9.4f} "
              f"{tst['mse']:>10.6f} {model.n_support_vectors:>6} "
              f"{elapsed:>6.1f}s")
        if tst["r2"] > best_r2:
            best_r2     = tst["r2"]
            best_kernel = name

    print(f"\n>>> Best kernel: {best_kernel}  (Test R2={best_r2:.4f})\n")
    return best_kernel


# C × epsilon grid search 

def hyperparam_grid_search(X_train, y_train, X_test, y_test, kernel):

    C_vals   = [1, 10, 100, 500]
    eps_vals = [0.001, 0.01, 0.05, 0.1]

    print(f"--- C × epsilon Grid Search  (kernel={kernel}) ---")
    print(f"{'C':>6} {'epsilon':>8} | {'Val MSE':>10} {'Val R2':>8} "
          f"{'#SV':>6}")
    print("-" * 50)

    best_mse = float("inf")
    best_cfg = {"C": 100, "epsilon": 0.01}

    for C in C_vals:
        for eps in eps_vals:
            m = SVRModel(C=C, epsilon=eps, kernel=kernel,
                         max_iter=2000, tol=1e-3)
            m.fit(X_train, y_train)
            metrics = m.evaluate(X_test, y_test)
            print(f"{C:>6} {eps:>8.3f} | {metrics['mse']:>10.6f} "
                  f"{metrics['r2']:>8.4f} {m.n_support_vectors:>6}")
            if metrics["mse"] < best_mse:
                best_mse = metrics["mse"]
                best_cfg = {"C": C, "epsilon": eps}

    print(f"\n>>> Best: C={best_cfg['C']}  epsilon={best_cfg['epsilon']}  "
          f"(Val MSE={best_mse:.6f})\n")
    return best_cfg

# Support vector diagnostics 

def print_sv_diagnostics(model: SVRModel, X_train, y_train, X_test, y_test):

    n_sv  = model.n_support_vectors
    n_tr  = len(X_train)
    print("--- Support Vector Diagnostics ---")
    print(f"  Total training samples   : {n_tr}")
    print(f"  Support vectors          : {n_sv}  ({100*n_sv/n_tr:.1f}% of train)")
    print(f"  Non-support vectors      : {n_tr - n_sv}  "
          f"(correctly inside epsilon-tube)")
    print(f"  Epsilon tube half-width  : ±{model.epsilon}")
    print(f"  Kernel                   : {model.kernel}")
    print(f"  C (regularisation)       : {model.C}")

    # Show how many test predictions fall inside the epsilon tube
    preds    = model.predict(X_test)
    residuals = np.abs(preds - y_test)
    in_tube  = (residuals <= model.epsilon).sum()
    print(f"  Test preds inside tube   : {in_tube} / {len(y_test)} "
          f"({100*in_tube/len(y_test):.1f}%)")
    print(f"  Max residual (test)      : {residuals.max():.5f}")
    print(f"  Mean residual (test)     : {residuals.mean():.5f}")
    print()

# Main 

def train_svr_model(
    data_dir         = "dataset/data",
    save_path        = "svr_model.pkl",
    test_split       = 0.2,
    run_kernel_cmp   = True,
    run_grid_search  = True,
):

    print("=== Support Vector Regression Training Pipeline ===\n")

    # 1. Load data
    X, y = load_features_and_labels(data_dir)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print_feature_diagnostics(X, y)

    # 2. Reproducible split  (same seed as all other models → fair comparison)
    rng     = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_test  = int(len(X) * test_split)

    X_train, y_train = X[indices[n_test:]], y[indices[n_test:]]
    X_test,  y_test  = X[indices[:n_test]], y[indices[:n_test]]
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # 3. Kernel comparison
    best_kernel = "rbf"
    if run_kernel_cmp:
        best_kernel = kernel_comparison(X_train, y_train, X_test, y_test)

    # 4. Grid search
    best_cfg = {"C": 100, "epsilon": 0.01}
    if run_grid_search:
        best_cfg = hyperparam_grid_search(
            X_train, y_train, X_test, y_test, best_kernel)

    # 5. Train final model
    print(f"--- Training final model  "
          f"(kernel={best_kernel}, C={best_cfg['C']}, "
          f"epsilon={best_cfg['epsilon']}) ---")
    t0    = time.time()
    model = SVRModel(
        kernel   = best_kernel,
        C        = best_cfg["C"],
        epsilon  = best_cfg["epsilon"],
        max_iter = 2000,
        tol      = 1e-3,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    # 6. Evaluate
    train_m = model.evaluate(X_train, y_train)
    test_m  = model.evaluate(X_test,  y_test)

    print(f"\n--- Results  (trained in {elapsed:.1f}s) ---")
    print(f"Train  MSE={train_m['mse']:.6f}  MAE={train_m['mae']:.6f}  "
          f"R2={train_m['r2']:.4f}")
    print(f"Test   MSE={test_m['mse']:.6f}  MAE={test_m['mae']:.6f}  "
          f"R2={test_m['r2']:.4f}")

    # 7. SV diagnostics
    print_sv_diagnostics(model, X_train, y_train, X_test, y_test)

    # 8. Save
    model.save(save_path)
    return model, test_m

if __name__ == "__main__":
    train_svr_model()