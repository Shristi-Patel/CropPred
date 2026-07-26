"""
Dataset Module
"""

from pathlib import Path

import cv2
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from src.config import *
from src.transforms import get_train_transform, get_valid_transform


class RiceDiseaseDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        image_path = self.dataframe.loc[index, "Path"]

        label = self.dataframe.loc[index, "Target"]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


def create_dataframe():

    records = []

    image_extensions = (".jpg", ".jpeg", ".png")

    classes = sorted(
        [
            folder.name
            for folder in DATASET_PATH.iterdir()
            if folder.is_dir()
        ]
    )

    class_to_idx = {
        cls: idx
        for idx, cls in enumerate(classes)
    }

    for cls in classes:

        folder = DATASET_PATH / cls

        for image in folder.glob("*"):

            if image.suffix.lower() in image_extensions:

                records.append({

                    "Path": str(image),

                    "Label": cls,

                    "Target": class_to_idx[cls]

                })

    return pd.DataFrame(records)


def split_dataset(df):

    train_df, temp_df = train_test_split(

        df,

        test_size=0.30,

        stratify=df["Target"],

        random_state=SEED

    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        stratify=temp_df["Target"],

        random_state=SEED

    )

    return train_df, val_df, test_df


def get_dataloaders():

    df = create_dataframe()

    train_df, val_df, test_df = split_dataset(df)

    train_dataset = RiceDiseaseDataset(

        train_df,

        transform=get_train_transform()

    )

    val_dataset = RiceDiseaseDataset(

        val_df,

        transform=get_valid_transform()

    )

    test_dataset = RiceDiseaseDataset(

        test_df,

        transform=get_valid_transform()

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS

    )

    return train_loader, val_loader, test_loader