import os
from dataset.generate_data import generate_synthetic_dataset
from training.train_regression import train_regression_model
from models.multiple_regression_model import MultipleRegressionModel
from metamorphic.metamorphic_tests import (
    brightness_invariance_test,
    horizontal_flip_regression_test,
    translation_invariance_test,
    vertical_crop_test,
)

ALL_TESTS = [
    brightness_invariance_test,
    horizontal_flip_regression_test,
    translation_invariance_test,
    vertical_crop_test,
]

def run_metamorphic_tests(model, data_dir="dataset/data", n_samples=10):
    """Run all metamorphic tests on n_samples images and print a summary."""
    img_dir = os.path.join(data_dir, "images")
    images  = sorted(os.listdir(img_dir))[:n_samples]

    print("\n" + "=" * 60)
    print("       METAMORPHIC TESTING RESULTS")
    print("=" * 60)

    total_runs   = 0
    total_passed = 0

    for img_name in images:
        img_path = os.path.join(img_dir, img_name)
        print(f"\nImage: {img_name}")

        for test_fn in ALL_TESTS:
            result = test_fn(model, img_path)
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            total_runs   += 1
            total_passed += int(result["passed"])
            print(
                f"  [{status}] {result['test']:<35} "
                f"diff={result['difference']:.4f}  "
                f"(tol={result['tolerance']})"
            )

    print("\n" + "-" * 60)
    print(f"  Pass rate: {total_passed}/{total_runs} "
          f"({100 * total_passed / total_runs:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":

    DATA_DIR   = "dataset/data"
    MODEL_PATH = "multiple_regression_model.pkl"

    # 1. Generate dataset if needed
    if not os.path.exists(os.path.join(DATA_DIR, "labels.csv")):
        print("Generating synthetic dataset...")
        generate_synthetic_dataset(output_dir=DATA_DIR, num_samples=3000)

    # 2. Train
    model, metrics = train_regression_model(
        data_dir=DATA_DIR,
        save_path=MODEL_PATH,
    )
    print(f"\n>>> Final Test R²: {metrics['r2']:.4f}  |  MSE: {metrics['mse']:.6f}")

    # 3. Metamorphic tests
    run_metamorphic_tests(model, data_dir=DATA_DIR, n_samples=10)