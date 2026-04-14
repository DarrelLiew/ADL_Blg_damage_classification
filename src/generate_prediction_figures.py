"""
Generate prediction comparison figures showing model outputs across experiments
and disaster types. Loads checkpoints from Exp 1, Exp 3, and Exp 5 and runs
inference on selected samples from diverse disaster types.

Outputs to images/:
  D1_prediction_grid.png       -- 4 disaster types x (pre, post, GT, Exp1, Exp3, Exp5)
  D2_improvement_closeup.png   -- zoomed crops showing progressive improvement
  D3_failure_cases.png         -- examples where the model still struggles

Usage:
    python generate_prediction_figures.py
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from shapely import wkt
from skimage.draw import polygon as draw_polygon
from torchvision import models

BASE = Path(__file__).parent
IMG_OUT = BASE / "images"
DATA = BASE / "data" / "train" / "train"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Constants ────────────────────────────────────────────────
DAMAGE_MAP = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4, "un-classified": 1}
CLASS_NAMES = ["Background", "No Damage", "Minor Damage", "Major Damage", "Destroyed"]
CLASS_COLORS_RGB = [
    (0, 0, 0),         # background - black
    (76, 175, 80),     # no damage - green
    (255, 235, 59),    # minor - yellow
    (255, 152, 0),     # major - orange
    (244, 67, 54),     # destroyed - red
]
DAMAGE_CMAP = ListedColormap([(r/255, g/255, b/255) for r, g, b in CLASS_COLORS_RGB])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RESNET_CHANNELS = {
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
}


# ── Model definition (copied from notebook) ──────────────────
def build_backbone(name, pretrained=False):
    constructor = getattr(models, name)
    return constructor(weights=None)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class ResNetEncoder(nn.Module):
    def __init__(self, backbone_name="resnet34"):
        super().__init__()
        backbone = build_backbone(backbone_name, pretrained=False)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.channels = RESNET_CHANNELS[backbone_name]
    def forward(self, x):
        x0 = self.stem(x); x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = ConvBlock(channels * 3, channels)
    def forward(self, pre_feat, post_feat):
        diff = torch.abs(post_feat - pre_feat)
        return self.block(torch.cat([pre_feat, post_feat, diff], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SiameseResUNet(nn.Module):
    def __init__(self, backbone="resnet34", num_classes=5, decoder_channels=None):
        super().__init__()
        self.encoder = ResNetEncoder(backbone)
        enc = self.encoder.channels
        dec = decoder_channels or [256, 128, 64, 32]
        self.fuse0 = FusionBlock(enc[0]); self.fuse1 = FusionBlock(enc[1])
        self.fuse2 = FusionBlock(enc[2]); self.fuse3 = FusionBlock(enc[3])
        self.fuse4 = FusionBlock(enc[4])
        self.bottleneck = ConvBlock(enc[4], dec[0])
        self.dec4 = DecoderBlock(dec[0], enc[3], dec[0])
        self.dec3 = DecoderBlock(dec[0], enc[2], dec[1])
        self.dec2 = DecoderBlock(dec[1], enc[1], dec[2])
        self.dec1 = DecoderBlock(dec[2], enc[0], dec[3])
        self.head = nn.Conv2d(dec[3], num_classes, kernel_size=1)

    def forward(self, pre_img, post_img):
        out_size = pre_img.shape[-2:]
        pre = self.encoder(pre_img); post = self.encoder(post_img)
        f0 = self.fuse0(pre[0], post[0]); f1 = self.fuse1(pre[1], post[1])
        f2 = self.fuse2(pre[2], post[2]); f3 = self.fuse3(pre[3], post[3])
        f4 = self.fuse4(pre[4], post[4])
        x = self.bottleneck(f4); x = self.dec4(x, f3); x = self.dec3(x, f2)
        x = self.dec2(x, f1); x = self.dec1(x, f0); x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


# ── Mask / image helpers ─────────────────────────────────────
def polygon_list(geometry):
    if geometry.is_empty: return []
    if geometry.geom_type == "Polygon": return [geometry]
    if geometry.geom_type == "MultiPolygon": return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        out = []
        for item in geometry.geoms: out.extend(polygon_list(item))
        return out
    return []


def rasterize_mask(label_path, image_shape, is_post=True):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.int64)
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for obj in data["features"]["xy"]:
        geometry = wkt.loads(obj["wkt"])
        for poly in polygon_list(geometry):
            if poly.is_empty or poly.area < 1: continue
            coords = np.asarray(poly.exterior.coords)
            if coords.shape[0] < 3: continue
            rr, cc = draw_polygon(coords[:, 1], coords[:, 0], shape=(h, w))
            if rr.size == 0: continue
            if is_post:
                subtype = obj["properties"].get("subtype", "un-classified")
                mask[rr, cc] = DAMAGE_MAP.get(subtype, 1)
            else:
                mask[rr, cc] = 1
    return mask


def load_image(path, size=512):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def load_mask(label_path, target_size=512):
    """Rasterize at original 1024x1024, then resize with INTER_NEAREST (matches training pipeline)."""
    orig_shape = (1024, 1024)
    mask = rasterize_mask(label_path, orig_shape, is_post=True)
    mask = cv2.resize(mask.astype(np.uint8), (target_size, target_size),
                      interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return mask


def preprocess(img):
    """Convert HWC uint8 to NCHW float32 tensor, normalised."""
    t = img.astype(np.float32) / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)
    return t.to(DEVICE)


def mask_overlay(img, mask, alpha=0.5):
    """Blend mask colours onto image."""
    overlay = img.copy().astype(np.float32)
    for cls_id in range(1, 5):
        pixels = mask == cls_id
        if pixels.any():
            color = np.array(CLASS_COLORS_RGB[cls_id], dtype=np.float32)
            overlay[pixels] = overlay[pixels] * (1 - alpha) + color * alpha
    return overlay.astype(np.uint8)


def load_model(checkpoint_path, backbone, decoder_channels):
    """Load a model from checkpoint."""
    model = SiameseResUNet(backbone=backbone, num_classes=5, decoder_channels=decoder_channels)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # Handle DataParallel prefix
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "")] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def predict(model, pre_img, post_img):
    """Run inference, return class mask."""
    pre_t = preprocess(pre_img)
    post_t = preprocess(post_img)
    logits = model(pre_t, post_t)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


# ── Select samples ───────────────────────────────────────────
def find_samples():
    """Find good samples with actual damage from diverse disaster types."""
    images_dir = DATA / "images"
    labels_dir = DATA / "labels"

    # Target: one sample per disaster type, with visible damage
    target_disasters = [
        "hurricane-harvey",
        "mexico-earthquake",
        "santa-rosa-wildfire",
        "palu-tsunami",
    ]
    disaster_labels = [
        "Hurricane (Harvey)",
        "Earthquake (Mexico)",
        "Wildfire (Santa Rosa)",
        "Tsunami (Palu)",
    ]

    samples = []
    for disaster, label in zip(target_disasters, disaster_labels):
        # Find post-disaster images for this disaster
        candidates = sorted(images_dir.glob(f"{disaster}_*_post_disaster.png"))
        best_sample = None
        best_damage_pixels = 0

        for post_path in candidates[:40]:  # check up to 40 per disaster
            stem = post_path.stem.replace("_post_disaster", "")
            pre_path = images_dir / f"{stem}_pre_disaster.png"
            label_path = labels_dir / f"{stem}_post_disaster.json"

            if not pre_path.exists() or not label_path.exists():
                continue

            # Quick check: count damage pixels
            try:
                img = cv2.imread(str(post_path))
                mask = rasterize_mask(str(label_path), (1024, 1024), is_post=True)
                damage_pixels = np.sum(mask >= 2)  # minor, major, destroyed
                total_building = np.sum(mask >= 1)
                if damage_pixels > best_damage_pixels and total_building > 500:
                    best_damage_pixels = damage_pixels
                    best_sample = {
                        "disaster_label": label,
                        "stem": stem,
                        "pre_path": pre_path,
                        "post_path": post_path,
                        "label_path": label_path,
                        "damage_pixels": damage_pixels,
                    }
            except Exception:
                continue

        if best_sample:
            samples.append(best_sample)
            print(f"  {label}: {best_sample['stem']} ({best_sample['damage_pixels']} damage pixels)")
        else:
            print(f"  {label}: no suitable sample found!")

    return samples


# ── Figure D1: Main prediction grid ─────────────────────────
def generate_prediction_grid(samples, models_dict):
    """
    Grid: rows = disaster types, cols = pre | post | ground truth | Exp 1 | Exp 3 | Exp 5
    """
    n_rows = len(samples)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, n_rows * 5.2))

    col_titles = ["Pre-Disaster", "Post-Disaster", "Ground Truth",
                   "Exp 1: Baseline", "Exp 3: Training\nOverhaul", "Exp 5: Focal +\nTuned Phase 2"]

    for row, sample in enumerate(samples):
        pre_img = load_image(sample["pre_path"])
        post_img = load_image(sample["post_path"])
        gt_mask = load_mask(str(sample["label_path"]))

        # Col 0: pre image
        axes[row, 0].imshow(pre_img)
        axes[row, 0].set_ylabel(sample["disaster_label"], fontsize=15, fontweight="bold", rotation=90, labelpad=12)

        # Col 1: post image
        axes[row, 1].imshow(post_img)

        # Col 2: ground truth overlay
        gt_overlay = mask_overlay(post_img, gt_mask, alpha=0.55)
        axes[row, 2].imshow(gt_overlay)

        # Cols 3-5: model predictions
        for col_offset, (exp_name, model) in enumerate(models_dict.items()):
            pred_mask = predict(model, pre_img, post_img)
            pred_overlay = mask_overlay(post_img, pred_mask, alpha=0.55)
            axes[row, 3 + col_offset].imshow(pred_overlay)

        # Titles on first row only
        if row == 0:
            for c, title in enumerate(col_titles):
                axes[0, c].set_title(title, fontsize=20, fontweight="bold", pad=16)

        for c in range(n_cols):
            axes[row, c].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=(r/255, g/255, b/255), edgecolor="white", label=name)
        for (r, g, b), name in zip(CLASS_COLORS_RGB[1:], CLASS_NAMES[1:])
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=18,
               frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.01),
               markerscale=2.0, handlelength=3.0)

    fig.suptitle("Prediction Comparison Across Disaster Types and Experiments",
                 fontsize=20, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08, wspace=0.05)
    path = IMG_OUT / "D1_prediction_grid.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")


# ── Figure D2: Improvement close-ups ────────────────────────
def generate_improvement_closeup(samples, models_dict):
    """
    Pick 2 samples, show a zoomed crop comparing Exp 1 vs Exp 3 vs Exp 5 with GT.
    """
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    col_titles = ["Post-Disaster\n(Crop)", "Ground Truth", "Exp 1\nBaseline",
                   "Exp 3\nTraining Overhaul", "Exp 5\nFocal + Tuned"]

    # Pick the two samples with the most damage pixels (best visual impact)
    sorted_samples = sorted(samples, key=lambda s: s["damage_pixels"], reverse=True)
    for row_idx, sample in enumerate(sorted_samples[:2]):
        pre_img = load_image(sample["pre_path"])
        post_img = load_image(sample["post_path"])
        gt_mask = load_mask(str(sample["label_path"]))

        # Find a crop region with lots of damage
        # Scan 128x128 windows
        best_crop, best_score = (128, 128, 384, 384), 0
        for y in range(0, 384, 32):
            for x in range(0, 384, 32):
                region = gt_mask[y:y+192, x:x+192]
                score = np.sum(region >= 2)
                if score > best_score:
                    best_score = score
                    best_crop = (y, x, y+192, x+192)

        y1, x1, y2, x2 = best_crop
        post_crop = post_img[y1:y2, x1:x2]
        gt_crop = gt_mask[y1:y2, x1:x2]

        axes[row_idx, 0].imshow(post_crop)
        axes[row_idx, 0].set_ylabel(sample["disaster_label"], fontsize=12, fontweight="bold")

        gt_ov = mask_overlay(post_crop, gt_crop, alpha=0.55)
        axes[row_idx, 1].imshow(gt_ov)

        for col_off, (exp_name, model) in enumerate(models_dict.items()):
            pred = predict(model, pre_img, post_img)
            pred_crop = pred[y1:y2, x1:x2]
            pred_ov = mask_overlay(post_crop, pred_crop, alpha=0.55)
            axes[row_idx, 2 + col_off].imshow(pred_ov)

        if row_idx == 0:
            for c, title in enumerate(col_titles):
                axes[0, c].set_title(title, fontsize=14, fontweight="bold", pad=10)
        for c in range(5):
            axes[row_idx, c].axis("off")

    legend_patches = [
        mpatches.Patch(facecolor=(r/255, g/255, b/255), edgecolor="white", label=name)
        for (r, g, b), name in zip(CLASS_COLORS_RGB[1:], CLASS_NAMES[1:])
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=13,
               frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Close-Up: Progressive Improvement in Damage Segmentation",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08, wspace=0.05)
    path = IMG_OUT / "D2_improvement_closeup.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")


# ── Figure D3: Failure cases ────────────────────────────────
def generate_failure_cases(models_dict):
    """
    Show cases where Exp 5 still struggles: minor/major confusion,
    small buildings, etc.
    """
    images_dir = DATA / "images"
    labels_dir = DATA / "labels"

    # Find samples with lots of minor damage (where model is weakest)
    failure_types = [
        ("Minor Damage\nConfusion", "hurricane-michael", lambda m: np.sum(m == 2) > 200),
        ("Small Buildings", "mexico-earthquake", lambda m: np.sum(m >= 1) > 200 and np.sum(m >= 2) > 50),
        ("Mixed Damage\nLevels", "hurricane-harvey", lambda m: len(np.unique(m[m > 0])) >= 3),
    ]

    fig, axes = plt.subplots(len(failure_types), 4, figsize=(20, len(failure_types) * 4.5))
    col_titles = ["Post-Disaster", "Ground Truth", "Exp 5 Prediction", "Error Map"]

    for row, (label, disaster, criteria) in enumerate(failure_types):
        candidates = sorted(images_dir.glob(f"{disaster}_*_post_disaster.png"))
        found = False

        for post_path in candidates[:60]:
            stem = post_path.stem.replace("_post_disaster", "")
            pre_path = images_dir / f"{stem}_pre_disaster.png"
            label_path = labels_dir / f"{stem}_post_disaster.json"
            if not pre_path.exists() or not label_path.exists():
                continue
            try:
                img = cv2.imread(str(post_path))
                mask = rasterize_mask(str(label_path), (1024, 1024), is_post=True)
                if criteria(mask):
                    pre_img = load_image(pre_path)
                    post_img = load_image(post_path)
                    gt_mask = load_mask(str(label_path))

                    # Get Exp 5 prediction
                    exp5_model = list(models_dict.values())[-1]
                    pred = predict(exp5_model, pre_img, post_img)

                    axes[row, 0].imshow(post_img)
                    axes[row, 0].set_ylabel(label, fontsize=12, fontweight="bold", rotation=90, labelpad=10)

                    axes[row, 1].imshow(mask_overlay(post_img, gt_mask, alpha=0.55))
                    axes[row, 2].imshow(mask_overlay(post_img, pred, alpha=0.55))

                    # Error map: correct = dark, wrong = bright red
                    error = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
                    building_mask = gt_mask >= 1
                    correct = (pred == gt_mask) & building_mask
                    wrong = (pred != gt_mask) & building_mask
                    missed = (gt_mask >= 1) & (pred == 0)  # false negative
                    error[correct] = [100, 200, 100]  # green = correct
                    error[wrong] = [255, 80, 80]       # red = wrong class
                    error[missed] = [255, 165, 0]      # orange = missed building

                    axes[row, 3].imshow(error)

                    found = True
                    break
            except Exception:
                continue

        if not found:
            for c in range(4):
                axes[row, c].text(0.5, 0.5, "No sample found", ha="center", va="center", transform=axes[row, c].transAxes)
                axes[row, 0].set_ylabel(label, fontsize=12, fontweight="bold")

        if row == 0:
            for c, title in enumerate(col_titles):
                axes[0, c].set_title(title, fontsize=14, fontweight="bold", pad=10)
        for c in range(4):
            axes[row, c].axis("off")

    # Error map legend
    err_patches = [
        mpatches.Patch(facecolor=(100/255, 200/255, 100/255), label="Correct class"),
        mpatches.Patch(facecolor=(255/255, 80/255, 80/255), label="Wrong class"),
        mpatches.Patch(facecolor=(255/255, 165/255, 0), label="Missed building"),
        mpatches.Patch(facecolor="black", label="Background (ignored)"),
    ]
    fig.legend(handles=err_patches, loc="lower center", ncol=4, fontsize=12,
               frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Failure Analysis: Where Exp 5 Still Struggles",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08, wspace=0.05)
    path = IMG_OUT / "D3_failure_cases.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")


# ── Main ─────────────────────────────────────────────────────
def main():
    print("Loading models...")
    models_dict = {}

    # Exp 1: ResNet34, decoder [256,128,64,32]
    print("  Exp 1 (ResNet34, baseline)...")
    models_dict["Exp 1"] = load_model(
        BASE / "training_runs" / "siamese_resunet_xview2_1" / "best_model.pt",
        backbone="resnet34", decoder_channels=[256, 128, 64, 32]
    )

    # Exp 3: ResNet34, decoder [256,128,64,32]
    print("  Exp 3 (ResNet34, training overhaul)...")
    models_dict["Exp 3"] = load_model(
        BASE / "training_runs" / "siamese_resunet_xview2" / "best_model.pt",
        backbone="resnet34", decoder_channels=[256, 128, 64, 32]
    )

    # Exp 5: ResNet50, decoder [512,256,128,64]
    print("  Exp 5 (ResNet50, focal + tuned)...")
    models_dict["Exp 5"] = load_model(
        BASE / "training_runs" / "siamese_resunet_xview2_4" / "best_model (2).pt",
        backbone="resnet50", decoder_channels=[512, 256, 128, 64]
    )

    print("\nSelecting samples...")
    samples = find_samples()

    if len(samples) < 2:
        print("ERROR: Need at least 2 samples. Check data directory.")
        return

    print(f"\nGenerating D1: prediction grid ({len(samples)} disaster types)...")
    generate_prediction_grid(samples, models_dict)

    print("Generating D2: improvement close-ups...")
    generate_improvement_closeup(samples, models_dict)

    print("Generating D3: failure cases...")
    generate_failure_cases(models_dict)

    print(f"\nDone! Figures saved to {IMG_OUT}/")


if __name__ == "__main__":
    main()
