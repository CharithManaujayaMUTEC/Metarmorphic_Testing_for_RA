import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.feature_extractor import extract_features
from models.multiple_regression_model import MultipleRegressionModel


def load_features_and_labels(data_dir="dataset/data", csv_name="labels.csv"):

    csv_path = os.path.join(data_dir, csv_name)
    img_dir  = os.path.join(data_dir, "images")

    df = pd.read_csv(csv_path)

    features_list, labels_list = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_path = os.path.join(img_dir, row["image"])
        try:
            features = extract_features(img_path)
            features_list.append(features)
            labels_list.append(float(row["steering"]))
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping.")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list,   dtype=np.float32)
    return X, y

def print_feature_diagnostics(X, y):
    """Print per-feature mean, std and correlation with steering label."""
    names = [
        "lane_center_offset",
        "top_offset",
        "bottom_offset",
        "curvature_proxy",
        "white_pixel_ratio",
        "vertical_spread",
    ]
    print("\n--- Feature Diagnostics ---")
    print(f"{'Feature':<25} {'Mean':>8} {'Std':>8} {'Corr w/ steering':>18}")
    print("-" * 65)
    for i, name in enumerate(names):
        col  = X[:, i]
        corr = float(np.corrcoef(col, y)[0, 1]) if col.std() > 0 else 0.0
        print(f"{name:<25} {col.mean():>8.4f} {col.std():>8.4f} {corr:>18.4f}")
    print()


def train_regression_model(
    data_dir="dataset/data",
    save_path="multiple_regression_model.pkl",
    test_split=0.2,
):
    
    print("=== Multiple Regression Training Pipeline ===\n")

    X, y = load_features_and_labels(data_dir)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

    print_feature_diagnostics(X, y)

    # Reproducible split
    rng     = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    n_test  = int(len(X) * test_split)

    X_train, y_train = X[indices[n_test:]], y[indices[n_test:]]
    X_test,  y_test  = X[indices[:n_test]], y[indices[:n_test]]

    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    model = MultipleRegressionModel()
    model.fit(X_train, y_train)

    train_m = model.evaluate(X_train, y_train)
    test_m  = model.evaluate(X_test,  y_test)

    print(f"\n--- Results ---")
    print(f"Train  MSE={train_m['mse']:.6f}  R²={train_m['r2']:.4f}")
    print(f"Test   MSE={test_m['mse']:.6f}  R²={test_m['r2']:.4f}")

    model.save(save_path)
    return model, test_m

if __name__ == "__main__":
    train_regression_model()