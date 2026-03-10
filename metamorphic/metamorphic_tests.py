import random
import torch
import torchvision.transforms.functional as TF


# helper utility -------------------------------------------------------------
def _to_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x)


def _from_log(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


# metamorphic relations ------------------------------------------------------
def horizontal_flip_test(model, image: torch.Tensor) -> float:
    flipped_image = TF.hflip(image)
    original_log = model(image.unsqueeze(0))
    flipped_log = model(flipped_image.unsqueeze(0))
    original_angle = _from_log(original_log)
    expected_log = _to_log(-original_angle)
    difference = torch.abs(flipped_log - expected_log)
    return difference.item()


def vertical_flip_test(model, image: torch.Tensor) -> float:
    vflipped = TF.vflip(image)
    orig_log = model(image.unsqueeze(0))
    flipped_log = model(vflipped.unsqueeze(0))
    return torch.abs(orig_log - flipped_log).item()


def brightness_test(model, image: torch.Tensor, factor: float = 1.0) -> float:
    adj = TF.adjust_brightness(image, factor)
    orig_log = model(image.unsqueeze(0))
    adj_log = model(adj.unsqueeze(0))
    return torch.abs(orig_log - adj_log).item()


def add_noise_test(model, image: torch.Tensor, std: float = 0.01) -> float:
    noise = torch.randn_like(image) * std
    noisy = image + noise
    orig_log = model(image.unsqueeze(0))
    noisy_log = model(noisy.unsqueeze(0))
    return torch.abs(orig_log - noisy_log).item()


def run_tests_on_sample(model,
                        image: torch.Tensor,
                        steering_log: torch.Tensor) -> dict:
    return {
        "horizontal_flip": horizontal_flip_test(model, image),
        "vertical_flip": vertical_flip_test(model, image),
        "brightness_up": brightness_test(model, image, factor=1.5),
        "brightness_down": brightness_test(model, image, factor=0.5),
        "noise": add_noise_test(model, image, std=0.02),
    }


def run_tests_on_dataset(model,
                         dataset,
                         num_samples: int = 100) -> dict:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    results = {"horizontal_flip": 0.0,
               "vertical_flip": 0.0,
               "brightness_up": 0.0,
               "brightness_down": 0.0,
               "noise": 0.0}

    count = 0
    for img, log in loader:
        if count >= num_samples:
            break
        res = run_tests_on_sample(model, img[0], log[0])
        for k, v in res.items():
            results[k] += v
        count += 1

    return {k: v / count for k, v in results.items()}


if __name__ == "__main__":
    import argparse
    from models.steering_model import SteeringLogRegression
    from dataset.dataset_loader import DrivingDataset

    parser = argparse.ArgumentParser(
        description="Run metamorphic relations against one or more steering models"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="model_lr.pth,model_sr.pth",
        help="comma-separated paths to saved model state dict(s)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="root directory of the dataset (overrides default)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="number of random examples to test"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    dataset = DrivingDataset(root_dir=args.dataset) if args.dataset else DrivingDataset()

    for model_path in models:
        model = SteeringLogRegression.load(model_path)
        scores = run_tests_on_dataset(model, dataset, num_samples=args.samples)
        print(f"Results for {model_path}:")
        for name, diff in scores.items():
            print(f"  {name}: {diff:.6f}")
        print()
