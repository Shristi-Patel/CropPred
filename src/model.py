"""
==========================================
Model Definition
==========================================
"""

import torch.nn as nn
import timm

from src.config import MODEL_NAME, NUM_CLASSES


def build_model(pretrained=True):
    """
    Build EfficientNet-B0 model
    """

    model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(

        nn.Dropout(0.3),

        nn.Linear(
            in_features,
            NUM_CLASSES
        )

    )

    return model


def freeze_backbone(model):
    """
    Freeze feature extractor
    """

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def unfreeze_backbone(model):
    """
    Unfreeze all layers
    """

    for param in model.parameters():
        param.requires_grad = True

    return model


def count_trainable_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def count_total_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def print_model_summary(model):

    trainable = count_trainable_parameters(model)

    total = count_total_parameters(model)

    print("=" * 50)

    print(f"Model : {MODEL_NAME}")

    print(f"Total Parameters      : {total:,}")

    print(f"Trainable Parameters  : {trainable:,}")

    print("=" * 50)