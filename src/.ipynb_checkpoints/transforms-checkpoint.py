"""
==========================================
Image Transformations
==========================================
"""

import albumentations as A

from albumentations.pytorch import ToTensorV2

from src.config import IMAGE_SIZE, MEAN, STD


def get_train_transform():

    train_transform = A.Compose([

        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        A.VerticalFlip(
            p=0.3
        ),

        A.Rotate(
            limit=20,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.GaussNoise(
            std_range=(0.1, 0.2),
            p=0.2
        ),

        A.Normalize(
            mean=MEAN,
            std=STD
        ),

        ToTensorV2()

    ])

    return train_transform


def get_valid_transform():

    valid_transform = A.Compose([

        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        A.Normalize(
            mean=MEAN,
            std=STD
        ),

        ToTensorV2()

    ])

    return valid_transform


def get_train_transform():

    train_transform = A.Compose([

        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        A.HorizontalFlip(
            p=0.5
        ),

        # Rice leaves are rarely upside down
        A.VerticalFlip(
            p=0.1
        ),

        # Mild camera angle variation
        A.Rotate(
            limit=10,
            p=0.5
        ),

        # Natural lighting variation
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Mild sensor noise
        A.GaussNoise(
            std_range=(0.1, 0.2),
            p=0.2
        ),

        A.Normalize(
            mean=MEAN,
            std=STD
        ),

        ToTensorV2()

    ])

    return train_transform