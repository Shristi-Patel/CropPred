"""
==========================================
Training Engine
==========================================
"""

import torch

from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device
):

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)

        running_correct += (preds == labels).sum().item()

        total += labels.size(0)

        progress_bar.set_postfix(

            Loss=f"{loss.item():.4f}"

        )

    epoch_loss = running_loss / total

    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc


def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False
        )

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)

            running_correct += (preds == labels).sum().item()

            total += labels.size(0)

            progress_bar.set_postfix(

                Loss=f"{loss.item():.4f}"

            )

    epoch_loss = running_loss / total

    epoch_acc = running_correct / total

    return epoch_loss, epoch_acc