import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.feature_extractor import extract_features
from models.random_forest_regression_model import RandomForestRegressionModel


FEATURE_NAMES = [
    "lane_center_offset",
    "top_offset",
    "bottom_offset",
    "curvature_proxy",
    "white_pixel_ratio",
    "vertical_spread",
]

# Data loading 

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

# n_estimators convergence check 

def convergence_check(X_train, y_train, X_test, y_test,
                      tree_counts=(10, 25, 50, 75, 100)):
    print("\n--- n_estimators Convergence Check ---")
    print(f"{'n_trees':>8} | {'Train R2':>9} {'Test R2':>9} "
          f"{'OOB R2':>9} {'Time':>7}")
    print("-" * 50)

    for n in tree_counts:
        t0 = time.time()
        rf = RandomForestRegressionModel(
            n_estimators=n, max_depth=10,
            min_samples_split=5, min_samples_leaf=3,
            oob_score=True, n_jobs=4, random_state=42,
        )
        rf.fit(X_train, y_train)
        tr  = rf.evaluate(X_train, y_train)
        tst = rf.evaluate(X_test,  y_test)
        oob = rf.oob_score_ if rf.oob_score_ is not None else float("nan")
        print(f"{n:>8} | {tr['r2']:>9.4f} {tst['r2']:>9.4f} "
              f"{oob:>9.4f} {time.time()-t0:>6.1f}s")
    print()


# Hyper-parameter grid search 

def hyperparam_grid_search(X_train, y_train, X_test, y_test):
    depths  = [6, 8, 10, 12]
    mf_opts = [1, 2, 3, None]   # None -> sqrt(6) ~ 2

    print("--- Hyper-parameter Grid Search  (n_estimators=50) ---")
    print(f"{'max_depth':>10} {'max_feat':>9} | {'Val MSE':>10} "
          f"{'Val R2':>8} {'OOB R2':>8}")
    print("-" * 55)

    best_val_mse = float("inf")
    best_cfg     = {"max_depth": 10, "max_features": None}

    for d in depths:
        for mf in mf_opts:
            rf = RandomForestRegressionModel(
                n_estimators=50, max_depth=d, max_features=mf,
                min_samples_split=5, min_samples_leaf=3,
                oob_score=True, n_jobs=4, random_state=42,
            )
            rf.fit(X_train, y_train)
            m   = rf.evaluate(X_test, y_test)
            oob = rf.oob_score_ if rf.oob_score_ is not None else float("nan")
            mf_str = str(mf) if mf is not None else "sqrt"
            print(f"{d:>10} {mf_str:>9} | {m['mse']:>10.6f} "
                  f"{m['r2']:>8.4f} {oob:>8.4f}")

            if m["mse"] < best_val_mse:
                best_val_mse = m["mse"]
                best_cfg     = {"max_depth": d, "max_features": mf}

    print(f"\n>>> Best: max_depth={best_cfg['max_depth']}  "
          f"max_features={best_cfg['max_features']}  "
          f"(Val MSE={best_val_mse:.6f})\n")
    return best_cfg

# Feature importance table 

def print_feature_importances(model):
    imp   = model.feature_importances
    order = np.argsort(imp)[::-1]
    print("\n--- Feature Importances (avg MSE reduction across all trees) ---")
    print(f"{'Rank':<6} {'Feature':<25} {'Importance':>12}  Bar")
    print("-" * 65)
    for rank, idx in enumerate(order, 1):
        bar = "█" * int(imp[idx] * 40)
        print(f"{rank:<6} {FEATURE_NAMES[idx]:<25} {imp[idx]:>12.4f}  {bar}")
    print()

# Uncertainty diagnostics 

def print_uncertainty_stats(model, X_test, y_test):
    stds  = model.predict_std(X_test)
    preds = model.predict(X_test)
    errs  = np.abs(preds - y_test)

    print("--- Prediction Uncertainty Statistics ---")
    print(f"  Mean std (test set)           : {stds.mean():.5f}")
    print(f"  Max  std (test set)           : {stds.max():.5f}")
    print(f"  Uncertain predictions (>0.05) : "
          f"{(stds > 0.05).sum()} / {len(stds)}")
    corr = float(np.corrcoef(stds, errs)[0, 1])
    note = "good - uncertainty tracks error" if corr > 0.3 else "weak calibration"
    print(f"  Corr(uncertainty, |error|)    : {corr:.4f}  ({note})")
    print()

# Main 

def train_random_forest_model(
    data_dir        = "dataset/data",
    save_path       = "random_forest_model.pkl",
    test_split      = 0.2,
    n_estimators    = 100,
    run_convergence = True,
    run_grid_search = True,
):

    print("=== Random Forest Regression Training Pipeline ===\n")

    X, y = load_features_and_labels(data_dir)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print_feature_diagnostics(X, y)

    rng     = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_test  = int(len(X) * test_split)

    X_train, y_train = X[indices[n_test:]], y[indices[n_test:]]
    X_test,  y_test  = X[indices[:n_test]], y[indices[:n_test]]
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    if run_convergence:
        convergence_check(X_train, y_train, X_test, y_test)

    best_cfg = {"max_depth": 10, "max_features": None}
    if run_grid_search:
        best_cfg = hyperparam_grid_search(X_train, y_train, X_test, y_test)

    print(f"--- Training final model  "
          f"(n_estimators={n_estimators}, "
          f"max_depth={best_cfg['max_depth']}, "
          f"max_features={best_cfg['max_features']}) ---")

    t0    = time.time()
    model = RandomForestRegressionModel(
        n_estimators      = n_estimators,
        max_depth         = best_cfg["max_depth"],
        max_features      = best_cfg["max_features"],
        min_samples_split = 5,
        min_samples_leaf  = 3,
        oob_score         = True,
        n_jobs            = 4,
        random_state      = 42,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_m = model.evaluate(X_train, y_train)
    test_m  = model.evaluate(X_test,  y_test)

    print(f"\n--- Results  (trained in {elapsed:.1f}s) ---")
    print(f"Train  MSE={train_m['mse']:.6f}  MAE={train_m['mae']:.6f}  "
          f"R2={train_m['r2']:.4f}")
    print(f"Test   MSE={test_m['mse']:.6f}  MAE={test_m['mae']:.6f}  "
          f"R2={test_m['r2']:.4f}")
    if model.oob_score_ is not None:
        print(f"OOB    R2={model.oob_score_:.4f}")

    print_feature_importances(model)
    print_uncertainty_stats(model, X_test, y_test)
    model.save(save_path)
    return model, test_m

if __name__ == "__main__":
    train_random_forest_model()