import albumentations as A
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from dog_breed.data.dataset import DogDataset


class DogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        batch_size: int = 32,
        num_workers: int = 2,
        val_size: float = 0.2,
        img_size: int = 224,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.img_size = img_size
        self.label_encoder = LabelEncoder()

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        df["target"] = self.label_encoder.fit_transform(df["breed"])

        train_df, val_df = train_test_split(
            df,
            test_size=self.val_size,
            stratify=df["target"],
            random_state=42,
        )

        train_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        val_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.train_dataset = DogDataset(
            train_df.reset_index(drop=True), self.img_dir, transform=train_transform
        )
        self.val_dataset = DogDataset(
            val_df.reset_index(drop=True), self.img_dir, transform=val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )