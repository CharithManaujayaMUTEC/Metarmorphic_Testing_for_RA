import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.feature_extractor import extract_features
from models.decision_tree_regression_model import DecisionTreeRegressionModel

FEATURE_NAMES = [
    "lane_center_offset",
    "top_offset",
    "bottom_offset",
    "curvature_proxy",
    "white_pixel_ratio",
    "vertical_spread",
]

# Data loading (mirrors train_regression.py) 

def load_features_and_labels(data_dir: str = "dataset/data",
                              csv_name: str = "labels.csv"):
 
    csv_path = os.path.join(data_dir, csv_name)
    img_dir  = os.path.join(data_dir, "images")

    df = pd.read_csv(csv_path)
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

def print_feature_diagnostics(X: np.ndarray, y: np.ndarray):
    print("\n--- Feature Diagnostics ---")
    print(f"{'Feature':<25} {'Mean':>8} {'Std':>8} {'Corr w/ steering':>18}")
    print("-" * 65)
    for i, name in enumerate(FEATURE_NAMES):
        col  = X[:, i]
        corr = float(np.corrcoef(col, y)[0, 1]) if col.std() > 0 else 0.0
        print(f"{name:<25} {col.mean():>8.4f} {col.std():>8.4f} {corr:>18.4f}")
    print()

# Depth grid search (pure numpy — no sklearn) 

def depth_grid_search(X_train, y_train, X_val, y_val,
                      depths=(3, 4, 5, 6, 7, 8, 10, 12)):

    print("--- Depth Grid Search ---")
    print(f"{'max_depth':>10} | {'Train MSE':>10} {'Train R²':>9} "
          f"| {'Val MSE':>9} {'Val R²':>8}")
    print("-" * 58)

    best_val_mse = float("inf")
    best_depth   = depths[0]
    results      = []

    for d in depths:
        m = DecisionTreeRegressionModel(
            max_depth=d, min_samples_split=10, min_samples_leaf=5)
        m.fit(X_train, y_train)

        tr = m.evaluate(X_train, y_train)
        vl = m.evaluate(X_val,   y_val)

        print(f"{d:>10} | {tr['mse']:>10.6f} {tr['r2']:>9.4f} "
              f"| {vl['mse']:>9.6f} {vl['r2']:>8.4f}")

        results.append((d, tr, vl))
        if vl["mse"] < best_val_mse:
            best_val_mse = vl["mse"]
            best_depth   = d

    print(f"\n>>> Best max_depth = {best_depth}  "
          f"(Val MSE = {best_val_mse:.6f})\n")
    return best_depth, results

# Feature importance table 

def print_feature_importances(model: DecisionTreeRegressionModel):
    imp = model.feature_importances
    order = np.argsort(imp)[::-1]

    print("\n--- Feature Importances (normalised MSE reduction) ---")
    print(f"{'Rank':<6} {'Feature':<25} {'Importance':>12} {'Bar'}")
    print("-" * 65)
    for rank, idx in enumerate(order, 1):
        bar = "█" * int(imp[idx] * 40)
        print(f"{rank:<6} {FEATURE_NAMES[idx]:<25} {imp[idx]:>12.4f}  {bar}")
    print()

# Main training function 

def train_decision_tree_model(
    data_dir:   str   = "dataset/data",
    save_path:  str   = "decision_tree_model.pkl",
    test_split: float = 0.2,
    run_grid_search: bool = True,
):

    print("=== Decision Tree Regression Training Pipeline ===\n")

    # 1. Load data
    X, y = load_features_and_labels(data_dir)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print_feature_diagnostics(X, y)

    # 2. Reproducible train / test split
    rng     = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_test  = int(len(X) * test_split)

    X_train, y_train = X[indices[n_test:]], y[indices[n_test:]]
    X_test,  y_test  = X[indices[:n_test]], y[indices[:n_test]]
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # 3. Optional grid search over max_depth
    best_depth = 8
    if run_grid_search:
        best_depth, _ = depth_grid_search(
            X_train, y_train, X_test, y_test,
            depths=(3, 4, 5, 6, 7, 8, 10, 12)
        )

    # 4. Train final model
    print(f"--- Training final model  (max_depth={best_depth}) ---")
    model = DecisionTreeRegressionModel(
        max_depth         = best_depth,
        min_samples_split = 10,
        min_samples_leaf  = 5,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    train_m = model.evaluate(X_train, y_train)
    test_m  = model.evaluate(X_test,  y_test)
    print(f"\n--- Results ---")
    print(f"Train  MSE={train_m['mse']:.6f}  R²={train_m['r2']:.4f}")
    print(f"Test   MSE={test_m['mse']:.6f}  R²={test_m['r2']:.4f}")

    # 6. Feature importances + tree structure
    print_feature_importances(model)
    model.print_tree(feature_names=FEATURE_NAMES, max_display_depth=3)

    # 7. Save
    model.save(save_path)
    return model, test_m

if __name__ == "__main__":
    train_decision_tree_model()