"""
Utility Functions
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, path):

    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):

    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    return model


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def plot_training_curves(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()