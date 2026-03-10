import os
import numpy as np

from dataset.generate_data      import generate_synthetic_dataset
from training.train_decision_tree import (
    train_decision_tree_model,
    load_features_and_labels,
    FEATURE_NAMES,
)
from models.decision_tree_regression_model import DecisionTreeRegressionModel

from metamorphic.decision_tree_metamorphic_tests import (
    dt_horizontal_flip_test,
    dt_brightness_invariance_test,
    dt_translation_consistency_test,
    dt_vertical_crop_test,
    dt_monotonicity_test,
    dt_symmetry_consistency_test,
)

DT_TESTS = [
    dt_horizontal_flip_test,
    dt_brightness_invariance_test,
    dt_translation_consistency_test,
    dt_vertical_crop_test,
    dt_monotonicity_test,
    dt_symmetry_consistency_test,
]

DATA_DIR      = "dataset/data"
DT_MODEL_PATH = "decision_tree_model.pkl"

# Metamorphic test runner 

def run_dt_metamorphic_tests(model, data_dir: str = DATA_DIR,
                             n_samples: int = 10):
    img_dir = os.path.join(data_dir, "images")
    images  = sorted(os.listdir(img_dir))[:n_samples]

    total_runs   = 0
    total_passed = 0

    print("\n" + "=" * 70)
    print("       DECISION TREE REGRESSION — METAMORPHIC TESTING RESULTS")
    print("=" * 70)

    for img_name in images:
        img_path = os.path.join(img_dir, img_name)

        # Show the model's prediction for context
        from dataset.feature_extractor import extract_features
        feats = extract_features(img_path).reshape(1, -1)
        pred  = float(model.predict(feats)[0])
        print(f"\nImage: {img_name}  |  Predicted steering: {pred:+.4f}")

        for test_fn in DT_TESTS:
            result = test_fn(model, img_path)
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            total_runs   += 1
            total_passed += int(result["passed"])
            print(
                f"  [{status}] {result['test']:<35} "
                f"diff={result['difference']:+.4f}  "
                f"(tol={result['tolerance']})"
            )

    print("\n" + "-" * 70)
    pct = 100 * total_passed / total_runs if total_runs else 0
    print(f"  DT Pass rate : {total_passed}/{total_runs}  ({pct:.1f}%)")
    print("=" * 70)
    return total_passed, total_runs

# 3-model comparison 

def compare_all_models(dt_model, data_dir: str = DATA_DIR):
    """
    Evaluate Decision Tree, Multiple Regression, and CNN on the same
    held-out test split and print a comparison table.
    """
    print("\n" + "=" * 62)
    print("           3-MODEL COMPARISON  (Test Split)")
    print("=" * 62)
    print(f"{'Model':<30} {'MSE':>10} {'MAE':>10} {'R²':>8}")
    print("-" * 62)

    # shared test data
    rng     = np.random.default_rng(42)
    X, y    = load_features_and_labels(data_dir)
    indices = rng.permutation(len(X))
    n_test  = int(len(X) * 0.2)
    X_test  = X[indices[:n_test]]
    y_test  = y[indices[:n_test]]

    # Decision Tree 
    m = dt_model.evaluate(X_test, y_test)
    _print_row("DecisionTreeRegression", m["mse"], m["r2"], y_test,
               dt_model.predict(X_test))

    # Multiple Regression 
    MR_PATH = "multiple_regression_model.pkl"
    if os.path.exists(MR_PATH):
        from models.multiple_regression_model import MultipleRegressionModel
        mr = MultipleRegressionModel()
        mr.load(MR_PATH)
        preds = mr.predict(X_test)
        mse   = float(np.mean((preds - y_test) ** 2))
        mae   = float(np.mean(np.abs(preds - y_test)))
        ss_r  = np.sum((y_test - preds)        ** 2)
        ss_t  = np.sum((y_test - y_test.mean()) ** 2)
        r2    = float(1 - ss_r / (ss_t + 1e-8))
        print(f"{'MultipleRegression':<30} {mse:>10.6f} {mae:>10.6f} {r2:>8.4f}")
    else:
        print("  MultipleRegression model not found — run main_regression.py first")

    # CNN 
    CNN_PATH = "cnn_regression_model.pth"
    if os.path.exists(CNN_PATH):
        try:
            import torch
            from models.cnn_regression_model  import CNNRegressionModel
            from dataset.cnn_dataset_loader   import CNNDrivingDataset
            from torch.utils.data             import DataLoader

            cnn = CNNRegressionModel()
            ckpt = torch.load(CNN_PATH, map_location="cpu")
            cnn.load_state_dict(ckpt["model_state"])
            cnn.eval()

            ds     = CNNDrivingDataset(root_dir=data_dir, mode="eval")
            _, vds = ds.split(train_ratio=0.8, seed=42)
            loader = DataLoader(vds, batch_size=64, shuffle=False)

            cnn_preds, cnn_targets = [], []
            with torch.no_grad():
                for imgs, angles in loader:
                    cnn_preds.append(cnn(imgs).squeeze())
                    cnn_targets.append(angles.squeeze())

            cp = torch.cat(cnn_preds).numpy()
            ct = torch.cat(cnn_targets).numpy()
            mse = float(np.mean((cp - ct) ** 2))
            mae = float(np.mean(np.abs(cp - ct)))
            ss_r = np.sum((ct - cp) ** 2)
            ss_t = np.sum((ct - ct.mean()) ** 2)
            r2   = float(1 - ss_r / (ss_t + 1e-8))
            print(f"{'CNNRegression':<30} {mse:>10.6f} {mae:>10.6f} {r2:>8.4f}")
        except Exception as e:
            print(f"  CNN comparison skipped: {e}")
    else:
        print("  CNN model not found — run main_cnn.py first")

    print("=" * 62)


def _print_row(name, mse, r2, y_true, preds):
    mae = float(np.mean(np.abs(preds - y_true)))
    print(f"{name:<30} {mse:>10.6f} {mae:>10.6f} {r2:>8.4f}")

if __name__ == "__main__":

    # 1. Dataset
    if not os.path.exists(os.path.join(DATA_DIR, "labels.csv")):
        print("Generating synthetic dataset …")
        generate_synthetic_dataset(output_dir=DATA_DIR, num_samples=3000)

    # 2. Train
    model, metrics = train_decision_tree_model(
        data_dir=DATA_DIR,
        save_path=DT_MODEL_PATH,
        run_grid_search=True,
    )
    print(f"\n>>> Final Test  R²={metrics['r2']:.4f}  "
          f"MSE={metrics['mse']:.6f}")

    # 3. Metamorphic tests
    run_dt_metamorphic_tests(model, data_dir=DATA_DIR, n_samples=10)

    # 4. 3-model comparison
    compare_all_models(model, data_dir=DATA_DIR)