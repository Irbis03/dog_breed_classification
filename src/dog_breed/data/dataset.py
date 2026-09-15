import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class DogDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_id = row["id"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Изображение не найдено: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        image = image.transpose(2, 0, 1) / 255.0
        image_tensor = torch.tensor(image, dtype=torch.float32)

        if "target" in row:
            label = torch.tensor(row["target"], dtype=torch.long)
            return image_tensor, label

        return image_tensor, img_id