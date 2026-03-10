import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

def generate_synthetic_dataset(
    output_dir="dataset/data",
    num_samples=3000,
    image_size=128
):

    images_path = os.path.join(output_dir, "images")
    os.makedirs(images_path, exist_ok=True)

    labels = []

    for i in tqdm(range(num_samples), desc="Generating Data"):

        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        # Random lane curvature
        curvature = np.random.uniform(-0.5, 0.5)

        for y in range(image_size):
            x = int(image_size // 2 + curvature * (y - image_size // 2))
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)

        steering_angle = curvature
        filename = f"img_{i}.png"

        cv2.imwrite(os.path.join(images_path, filename), img)

        labels.append([filename, steering_angle])

    df = pd.DataFrame(labels, columns=["image", "steering"])
    df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    print("✅ Synthetic dataset generated successfully!")

if __name__ == "__main__":
    generate_synthetic_dataset()