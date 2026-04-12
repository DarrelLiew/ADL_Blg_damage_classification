# ============================================================
# 1. Imports and config
# ============================================================


import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from shapely import wkt
from skimage.draw import polygon as draw_polygon
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

cv2.setNumThreads(0)

# ── Point this to a config.json to load experiment settings ──
# e.g. "training_runs/siamese_resunet_xview2_3/config.json"
CONFIG_PATH = "training_runs/siamese_resunet_xview2_3/config.json"

DEFAULTS = {
    "data_root": "3/train/train",
    "seed": 42,
    "output_root": "training_runs",
    "experiment_name": "siamese_resunet_xview2",
    "resume_training": True,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "image_size": [512, 512],
    "num_classes": 5,
    "backbone": "resnet34",
    "pretrained": True,
    "decoder_channels": [256, 128, 64, 32],
    "batch_size": 2,
    "num_workers": 0,
    "epochs": 50,
    "lr": 3e-4,
    "encoder_lr": 1e-5,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    "accumulation_steps": 4,
    "use_amp": True,
    "scheduler": "cosine",
    "min_lr": 1e-6,
    "ce_weight": 1.0,
    "dice_weight": 1.0,
    "label_smoothing": 0.05,
    "class_weight_method": "log_inverse",
    "class_weight_samples": None,
    "freeze_epochs": 15,
    "early_stopping_patience": 10,
    "run_training": False,
    "run_smoke_test": True,
    "show_examples": 3,
    "aug_hflip": 0.5,
    "aug_vflip": 0.2,
    "aug_rotate90": 0.5,
    "aug_brightness": 0.12,
    "aug_contrast": 0.12,
    "aug_saturation": 0.08,
    "aug_blur_prob": 0.10,
    "aug_noise_prob": 0.10,
}

if CONFIG_PATH and Path(CONFIG_PATH).exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        file_config = json.load(f)
    CONFIG = {**DEFAULTS, **file_config}
    print(f"Loaded config from {CONFIG_PATH}")
else:
    CONFIG = dict(DEFAULTS)
    if CONFIG_PATH:
        print(f"WARNING: {CONFIG_PATH} not found, using defaults")
    else:
        print("Using default config (set CONFIG_PATH to load from file)")

DAMAGE_CLASSES = {
    0: "Background",
    1: "No Damage",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
}
DAMAGE_COLORS = ["black", "green", "yellow", "orange", "red"]
DAMAGE_CMAP = ListedColormap(DAMAGE_COLORS)
DAMAGE_LEGEND = [
    Patch(facecolor=color, edgecolor="white", label=DAMAGE_CLASSES[i])
    for i, color in enumerate(DAMAGE_COLORS)
]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RUN_DIR = Path(CONFIG["output_root"]) / CONFIG["experiment_name"]
RUN_DIR.mkdir(parents=True, exist_ok=True)
CONFIG["run_dir"] = str(RUN_DIR)
CONFIG["best_checkpoint_path"] = str(RUN_DIR / "best_model.pt")
CONFIG["last_checkpoint_path"] = str(RUN_DIR / "last_model.pt")
CONFIG["history_json_path"] = str(RUN_DIR / "history.json")
CONFIG["metrics_csv_path"] = str(RUN_DIR / "metrics.csv")
CONFIG["config_json_path"] = str(RUN_DIR / "config.json")

with open(CONFIG["config_json_path"], "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2)

print(json.dumps(CONFIG, indent=2))
