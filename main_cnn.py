import os
import torch
from torch.utils.data import DataLoader

from dataset.generate_data       import generate_synthetic_dataset
from dataset.cnn_dataset_loader  import CNNDrivingDataset, get_eval_transform
from models.cnn_regression_model import CNNRegressionModel
from training.train_cnn          import train_cnn_model

from metamorphic.cnn_metamorphic_tests import (
    cnn_horizontal_flip_test,
    cnn_brightness_invariance_test,
    cnn_rotation_consistency_test,
    cnn_blur_invariance_test,
    cnn_translation_test,
    cnn_contrast_invariance_test,
)

CNN_TESTS = [
    cnn_horizontal_flip_test,
    cnn_brightness_invariance_test,
    cnn_rotation_consistency_test,
    cnn_blur_invariance_test,
    cnn_translation_test,
    cnn_contrast_invariance_test,
]

DATA_DIR        = "dataset/data"
CNN_MODEL_PATH  = "cnn_regression_model.pth"

# Metamorphic test runner 

def run_cnn_metamorphic_tests(model, data_dir=DATA_DIR, n_samples=10):

    dataset  = CNNDrivingDataset(root_dir=data_dir, mode="eval")
    loader   = DataLoader(dataset, batch_size=1, shuffle=False)

    total_runs   = 0
    total_passed = 0

    print("\n" + "=" * 68)
    print("           CNN REGRESSION — METAMORPHIC TESTING RESULTS")
    print("=" * 68)

    for i, (image_tensor, true_angle) in enumerate(loader):
        if i >= n_samples:
            break

        image = image_tensor.squeeze(0)          # (3, 66, 200)
        print(f"\nSample {i+1:>2}  |  True steering: {true_angle.item():+.4f}")

        for test_fn in CNN_TESTS:
            result = test_fn(model, image)
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            total_runs   += 1
            total_passed += int(result["passed"])

            print(
                f"  [{status}] {result['test']:<35} "
                f"pred={result['original_pred']:+.4f}  "
                f"diff={result['difference']:.4f}  "
                f"(tol={result['tolerance']})"
            )

    print("\n" + "-" * 68)
    pct = 100 * total_passed / total_runs if total_runs else 0
    print(f"  CNN Pass rate: {total_passed}/{total_runs}  ({pct:.1f}%)")
    print("=" * 68)

    return total_passed, total_runs


# Optional: compare CNN vs Multiple Regression 

def compare_models(cnn_model, data_dir=DATA_DIR):

    from models.multiple_regression_model import MultipleRegressionModel
    from dataset.feature_extractor        import extract_features
    import numpy as np

    MR_PATH = "multiple_regression_model.pkl"

    print("\n" + "=" * 50)
    print("        MODEL COMPARISON (Validation Set)")
    print("=" * 50)

    # CNN evaluation 
    dataset    = CNNDrivingDataset(root_dir=data_dir, mode="eval")
    _, val_ds  = dataset.split(train_ratio=0.8, seed=42)
    loader     = DataLoader(val_ds, batch_size=64, shuffle=False)

    cnn_model.eval()
    preds_cnn, targets = [], []
    with torch.no_grad():
        for imgs, angles in loader:
            preds_cnn.append(cnn_model(imgs).squeeze())
            targets.append(angles.squeeze())

    preds_cnn = torch.cat(preds_cnn).numpy()
    targets   = torch.cat(targets).numpy()

    cnn_mse = float(((preds_cnn - targets) ** 2).mean())
    cnn_mae = float(abs(preds_cnn - targets).mean())
    ss_res  = ((targets - preds_cnn) ** 2).sum()
    ss_tot  = ((targets - targets.mean()) ** 2).sum()
    cnn_r2  = float(1 - ss_res / (ss_tot + 1e-8))

    print(f"\n{'Model':<28} {'MSE':>10} {'MAE':>10} {'R²':>8}")
    print("-" * 60)
    print(f"{'CNNRegressionModel':<28} {cnn_mse:>10.6f} {cnn_mae:>10.6f} {cnn_r2:>8.4f}")

    # Multiple Regression evaluation 
    if os.path.exists(MR_PATH):
        import pandas as pd
        from tqdm import tqdm

        mr_model = MultipleRegressionModel()
        mr_model.load(MR_PATH)

        csv_path = os.path.join(data_dir, "labels.csv")
        img_dir  = os.path.join(data_dir, "images")
        df       = pd.read_csv(csv_path)

        # Use the same val indices as CNN split (approximate via last 20 %)
        n_val = int(len(df) * 0.2)
        val_df = df.tail(n_val)

        feat_list, lbl_list = [], []
        for _, row in tqdm(val_df.iterrows(), total=len(val_df),
                           desc="Extracting MR features", leave=False):
            try:
                feat_list.append(
                    extract_features(os.path.join(img_dir, row["image"])))
                lbl_list.append(float(row["steering"]))
            except Exception:
                pass

        import numpy as np
        X_val = np.array(feat_list)
        y_val = np.array(lbl_list)

        preds_mr = mr_model.predict(X_val)
        mr_mse   = float(((preds_mr - y_val) ** 2).mean())
        mr_mae   = float(abs(preds_mr - y_val).mean())
        ss_res   = ((y_val - preds_mr) ** 2).sum()
        ss_tot   = ((y_val - y_val.mean()) ** 2).sum()
        mr_r2    = float(1 - ss_res / (ss_tot + 1e-8))

        print(f"{'MultipleRegressionModel':<28} {mr_mse:>10.6f} {mr_mae:>10.6f} {mr_r2:>8.4f}")
    else:
        print("  (Multiple Regression model not found — skipping comparison)")

    print("=" * 50)

if __name__ == "__main__":

    # 1. Generate dataset if needed
    if not os.path.exists(os.path.join(DATA_DIR, "labels.csv")):
        print("Generating synthetic dataset …")
        generate_synthetic_dataset(output_dir=DATA_DIR, num_samples=3000)

    # 2. Train (or load existing checkpoint)
    if os.path.exists(CNN_MODEL_PATH):
        print(f"Found existing checkpoint: {CNN_MODEL_PATH}")
        ans = input("Re-train? [y/N] : ").strip().lower()
        if ans == "y":
            model, history = train_cnn_model(
                data_dir=DATA_DIR, save_path=CNN_MODEL_PATH,
                epochs=30, batch_size=64, lr=1e-3,
            )
        else:
            model = CNNRegressionModel()
            ckpt  = torch.load(CNN_MODEL_PATH, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded  Val MSE={ckpt['val_mse']:.6f}  R²={ckpt['val_r2']:.4f}")
    else:
        model, history = train_cnn_model(
            data_dir=DATA_DIR, save_path=CNN_MODEL_PATH,
            epochs=30, batch_size=64, lr=1e-3,
        )

    # 3. Metamorphic tests
    run_cnn_metamorphic_tests(model, data_dir=DATA_DIR, n_samples=10)

    # 4. Model comparison
    compare_models(model, data_dir=DATA_DIR)