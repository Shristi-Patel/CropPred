"""
Trainer
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import *
from src.engine import *
from src.utils import *


class Trainer:

    def __init__(self, model):

        self.model = model.to(DEVICE)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(

            self.model.parameters(),

            lr=LEARNING_RATE,

            weight_decay=WEIGHT_DECAY

        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(

            self.optimizer,

            mode="max",

            factor=0.5,

            patience=2

        )

    def fit(self, train_loader, val_loader):

        history = {

            "train_loss": [],

            "val_loss": [],

            "train_acc": [],

            "val_acc": []

        }

        best_acc = 0

        best_weights = copy.deepcopy(

            self.model.state_dict()

        )

        patience_counter = 0

        for epoch in range(EPOCHS):

            print("="*60)

            print(f"Epoch {epoch+1}/{EPOCHS}")

            train_loss, train_acc = train_one_epoch(

                self.model,

                train_loader,

                self.optimizer,

                self.criterion,

                DEVICE

            )

            val_loss, val_acc = validate_one_epoch(

                self.model,

                val_loader,

                self.criterion,

                DEVICE

            )

            self.scheduler.step(val_acc)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Train Loss : {train_loss:.4f}")

            print(f"Val Loss   : {val_loss:.4f}")

            print(f"Train Acc  : {train_acc:.4f}")

            print(f"Val Acc    : {val_acc:.4f}")

            if val_acc > best_acc:

                best_acc = val_acc

                best_weights = copy.deepcopy(

                    self.model.state_dict()

                )

                save_checkpoint(

                    self.model,

                    MODEL_PATH / "best_model.pth"

                )

                patience_counter = 0

            else:

                patience_counter += 1

            if patience_counter >= PATIENCE:

                print("\nEarly Stopping")

                break

        self.model.load_state_dict(best_weights)

        return history