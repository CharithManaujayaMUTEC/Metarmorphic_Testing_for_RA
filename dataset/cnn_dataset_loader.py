import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ImageNet-style normalisation values (work well for synthetic data too) 
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transform():
    """
    Training transforms:
      Resize → random brightness/contrast jitter → tensor → normalize

    Augmentation keeps the lane geometry intact (no flips here —
    flipping is a metamorphic test, not a training trick).
    """
    return transforms.Compose([
        transforms.Resize((66, 200)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def get_eval_transform():
    """Evaluation / inference transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse the per-channel normalisation for visualisation.
    tensor : (3, H, W)  normalised
    returns: (3, H, W)  in [0, 1]
    """
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std  = torch.tensor(_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


class CNNDrivingDataset(Dataset):
    """
    Dataset for the CNN regression model.

    CSV format  : image,steering
    Image dir   : <root_dir>/images/
    Targets     : steering angles (float, already in [-1, 1] from generator)
    """

    def __init__(
        self,
        root_dir: str = "dataset/data",
        transform=None,
        mode: str = "train",          # "train" | "eval"
    ):
        self.root_dir = root_dir
        self.img_dir  = os.path.join(root_dir, "images")
        csv_path      = os.path.join(root_dir, "labels.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"labels.csv not found at {csv_path}. "
                "Run dataset/generate_data.py first."
            )

        self.data = pd.read_csv(csv_path)

        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = get_train_transform()
        else:
            self.transform = get_eval_transform()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_name = self.data.iloc[idx, 0]
        steering = float(self.data.iloc[idx, 1])

        img_path = os.path.join(self.img_dir, img_name)
        image    = Image.open(img_path).convert("RGB")
        image    = self.transform(image)

        return image, torch.tensor([steering], dtype=torch.float32)

    def split(self, train_ratio: float = 0.8, seed: int = 42):
        """
        Return two Subset objects: (train_dataset, val_dataset).
        Uses a reproducible random split.
        """
        from torch.utils.data import random_split, Subset
        import numpy as np

        n       = len(self)
        n_train = int(n * train_ratio)
        n_val   = n - n_train

        rng     = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(self, [n_train, n_val],
                                         generator=rng)

        # Give the val split eval transforms 
        # We do this by wrapping in a thin class rather than mutating self
        class _EvalSubset(Subset):
            def __getitem__(self, idx):
                orig_transform = self.dataset.transform
                self.dataset.transform = get_eval_transform()
                item = super().__getitem__(idx)
                self.dataset.transform = orig_transform
                return item

        return train_ds, _EvalSubset(self, val_ds.indices)