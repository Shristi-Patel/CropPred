"""
==========================================
Project Configuration
==========================================
"""

from pathlib import Path
import torch

# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "Rice Leaf Disease Images"

OUTPUT_PATH = PROJECT_ROOT / "outputs"

MODEL_PATH = PROJECT_ROOT / "models"

REPORT_PATH = PROJECT_ROOT / "reports"

# Create directories automatically
OUTPUT_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)

# ==========================================
# Dataset
# ==========================================

NUM_CLASSES = 4

CLASS_NAMES = [
    "Bacterialblight",
    "Blast",
    "Brownspot",
    "Tungro"
]

# ==========================================
# Image Configuration
# ==========================================

IMAGE_SIZE = 224

# ==========================================
# Training Configuration
# ==========================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 5

NUM_WORKERS = 0          # Windows

SEED = 42

# ==========================================
# Model Configuration
# ==========================================

MODEL_NAME = "efficientnet_b0"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# Normalization (ImageNet)
# ==========================================

MEAN = (0.485, 0.456, 0.406)

STD = (0.229, 0.224, 0.225)