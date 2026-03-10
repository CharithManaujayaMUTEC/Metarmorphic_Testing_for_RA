import os
import numpy as np

from dataset.generate_data        import generate_synthetic_dataset
from training.train_random_forest import (
    train_random_forest_model,
    load_features_and_labels,
)
from models.random_forest_regression_model import RandomForestRegressionModel

from metamorphic.random_forest_metamorphic_tests import (
    rf_horizontal_flip_test,
    rf_brightness_invariance_test,
    rf_translation_consistency_test,
    rf_vertical_crop_test,
    rf_monotonicity_test,
    rf_ensemble_consistency_test,
    rf_uncertainty_bounds_test,
)

RF_TESTS = [
    rf_horizontal_flip_test,
    rf_brightness_invariance_test,
    rf_translation_consistency_test,
    rf_vertical_crop_test,
    rf_monotonicity_test,
    rf_ensemble_consistency_test,
    rf_uncertainty_bounds_test,
]

DATA_DIR      = "dataset/data"
RF_MODEL_PATH = "random_forest_model.pkl"

# Metamorphic test runner 

def run_rf_metamorphic_tests(model, data_dir=DATA_DIR, n_samples=10):
    img_dir = os.path.join(data_dir, "images")
    images  = sorted(os.listdir(img_dir))[:n_samples]

    total_runs   = 0
    total_passed = 0

    print("\n" + "=" * 72)
    print("        RANDOM FOREST REGRESSION — METAMORPHIC TESTING RESULTS")
    print("=" * 72)

    for img_name in images:
        img_path = os.path.join(img_dir, img_name)

        from dataset.feature_extractor import extract_features
        feats = extract_features(img_path).reshape(1, -1)
        pred  = float(model.predict(feats)[0])
        std   = float(model.predict_std(feats)[0])
        print(f"\nImage: {img_name}  |  pred={pred:+.4f}  std={std:.4f}")

        for test_fn in RF_TESTS:
            result = test_fn(model, img_path)
            status = "PASS" if result["passed"] else "FAIL"
            emoji  = "✅" if result["passed"] else "❌"
            total_runs   += 1
            total_passed += int(result["passed"])
            print(
                f"  [{emoji} {status}] {result['test']:<38} "
                f"diff={result['difference']:+.4f}  "
                f"(tol={result['tolerance']})"
            )

    print("\n" + "-" * 72)
    pct = 100 * total_passed / total_runs if total_runs else 0
    print(f"  RF Pass rate : {total_passed}/{total_runs}  ({pct:.1f}%)")
    print("=" * 72)
    return total_passed, total_runs

# 4-model comparison 

def _print_np_row(name, preds, y_true):
    mse  = float(np.mean((preds - y_true) ** 2))
    mae  = float(np.mean(np.abs(preds - y_true)))
    ss_r = np.sum((y_true - preds)         ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    print(f"{name:<32} {mse:>10.6f} {mae:>10.6f} {r2:>8.4f}")

def compare_all_models(rf_model, data_dir=DATA_DIR):
    print("\n" + "=" * 66)
    print("           4-MODEL COMPARISON  (Test Split)")
    print("=" * 66)
    print(f"{'Model':<32} {'MSE':>10} {'MAE':>10} {'R2':>8}")
    print("-" * 66)

    rng     = np.random.default_rng(42)
    X, y    = load_features_and_labels(data_dir)
    indices = rng.permutation(len(X))
    n_test  = int(len(X) * 0.2)
    X_test  = X[indices[:n_test]]
    y_test  = y[indices[:n_test]]

    # Random Forest
    m = rf_model.evaluate(X_test, y_test)
    print(f"{'RandomForestRegression':<32} "
          f"{m['mse']:>10.6f} {m['mae']:>10.6f} {m['r2']:>8.4f}")

    # Decision Tree
    DT_PATH = "decision_tree_model.pkl"
    if os.path.exists(DT_PATH):
        from models.decision_tree_regression_model import DecisionTreeRegressionModel
        dt = DecisionTreeRegressionModel()
        dt.load(DT_PATH)
        _print_np_row("DecisionTreeRegression", dt.predict(X_test), y_test)
    else:
        print("  Decision Tree not found — run main_decision_tree.py first")

    # Multiple Regression
    MR_PATH = "multiple_regression_model.pkl"
    if os.path.exists(MR_PATH):
        from models.multiple_regression_model import MultipleRegressionModel
        mr = MultipleRegressionModel()
        mr.load(MR_PATH)
        _print_np_row("MultipleRegression", mr.predict(X_test), y_test)
    else:
        print("  Multiple Regression not found — run main_regression.py first")

    # CNN
    CNN_PATH = "cnn_regression_model.pth"
    if os.path.exists(CNN_PATH):
        try:
            import torch
            from models.cnn_regression_model import CNNRegressionModel
            from dataset.cnn_dataset_loader  import CNNDrivingDataset
            from torch.utils.data            import DataLoader

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

            _print_np_row("CNNRegression",
                          torch.cat(cnn_preds).numpy(),
                          torch.cat(cnn_targets).numpy())
        except Exception as e:
            print(f"  CNN comparison skipped: {e}")
    else:
        print("  CNN not found — run main_cnn.py first")

    print("=" * 66)

if __name__ == "__main__":

    if not os.path.exists(os.path.join(DATA_DIR, "labels.csv")):
        print("Generating synthetic dataset ...")
        generate_synthetic_dataset(output_dir=DATA_DIR, num_samples=3000)

    model, metrics = train_random_forest_model(
        data_dir        = DATA_DIR,
        save_path       = RF_MODEL_PATH,
        n_estimators    = 100,
        run_convergence = True,
        run_grid_search = True,
    )
    print(f"\n>>> Final Test  R2={metrics['r2']:.4f}  "
          f"MSE={metrics['mse']:.6f}")

    run_rf_metamorphic_tests(model, data_dir=DATA_DIR, n_samples=10)
    compare_all_models(model, data_dir=DATA_DIR)