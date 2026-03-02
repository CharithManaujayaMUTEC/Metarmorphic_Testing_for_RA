import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DrivingDataset(Dataset):
    def __init__(self, root_dir="/content/Metarmorphic_Testing_for_RA/dataset/data"):
        self.root_dir = root_dir
        self.csv_path = os.path.join(root_dir, "labels.csv")
        self.img_dir = os.path.join(root_dir, "images")

        self.data = pd.read_csv(self.csv_path)

        self.transform = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        steering = self.data.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        steering = torch.tensor([steering], dtype=torch.float32)

        return image, steering