"""
Generate additional figures for the report that don't come from metrics CSVs.
  C1: Architecture diagram
  C2: Class distribution
  C3: Two-phase training schematic
  C4: All-experiments loss overlay

Usage:
    python generate_extra_figures.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BASE = Path(__file__).parent
OUT = BASE / "images"
RUNS = BASE / "training_runs"

def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.name}")

def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
            rows.append(row)
    return rows

def col(rows, key):
    return np.array([r[key] for r in rows if key in r])


# ====================================================================
# C1: Architecture Diagram
# ====================================================================
print("C1. Architecture diagram...")
fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

def draw_box(ax, x, y, w, h, text, color, fontsize=10, text_color="white", alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

# Title
ax.text(9, 9.5, "Siamese ResNet U-Net Architecture", fontsize=18,
        fontweight="bold", ha="center", va="center")

# Input images
draw_box(ax, 0.3, 7.0, 2.5, 1.4, "Pre-Disaster\nImage\n(512x512x3)", "#42a5f5", fontsize=11)
draw_box(ax, 0.3, 4.5, 2.5, 1.4, "Post-Disaster\nImage\n(512x512x3)", "#ef5350", fontsize=11)

# Shared encoder
draw_box(ax, 4.0, 5.0, 3.0, 3.0, "Shared ResNet\nEncoder\n(ImageNet\npretrained)\n\nResNet34/50", "#7e57c2", fontsize=11)
ax.text(5.5, 4.6, "Weight-shared", fontsize=10, ha="center", color="#4a148c", fontweight="bold")

# Arrows: inputs -> encoder
draw_arrow(ax, 2.8, 7.7, 4.0, 7.0)
draw_arrow(ax, 2.8, 5.2, 4.0, 5.8)

# Feature labels — positioned between input boxes and encoder, not overlapping boxes
ax.text(3.4, 7.5, "Pre features", fontsize=10, color="#1565c0", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="white", alpha=0.8))
ax.text(3.4, 5.3, "Post features", fontsize=10, color="#c62828", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="white", alpha=0.8))

# Fusion blocks
draw_box(ax, 8.0, 5.2, 2.8, 2.6, "Fusion Blocks\n(x5 levels)\n\nconcat(pre, post,\n|post - pre|)\n-> ConvBlock", "#ff9800", fontsize=10)

# Arrow: encoder -> fusion
draw_arrow(ax, 7.0, 6.5, 8.0, 6.5)

# Fusion detail box
draw_box(ax, 7.5, 3.0, 3.5, 1.6, "3-Stream Fusion:\npre + post + |diff|\n3C channels -> C channels", "#fff3e0",
         fontsize=10, text_color="#bf360c", alpha=0.9)

# Decoder
draw_box(ax, 11.8, 5.2, 2.5, 2.6, "U-Net\nDecoder\n\nBilinear Up\n+ Skip\n+ ConvBlock", "#66bb6a", fontsize=10)

# Arrow: fusion -> decoder
draw_arrow(ax, 10.8, 6.5, 11.8, 6.5)

# Skip connections annotation
ax.annotate("Skip connections\n(fused features)", xy=(11.3, 7.4), xytext=(10.2, 8.6),
            fontsize=10, color="#1b5e20", fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.5, connectionstyle="arc3,rad=-0.2"))

# Output head
draw_box(ax, 15.0, 5.6, 2.5, 1.8, "1x1 Conv\nHead\n\n5-class\nDamage Map", "#e91e63", fontsize=10)

# Arrow: decoder -> head
draw_arrow(ax, 14.3, 6.5, 15.0, 6.5)

# Bottom note
ax.text(9, 0.7, "ResNet34: 21M params, 512 deepest channels  |  ResNet50: 25M params, 2048 deepest channels\n"
        "FusionBlock at each level captures spatial change at multiple scales",
        fontsize=11, ha="center", va="center", color="#333",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", ec="#bbb"))

savefig(fig, "C1_architecture_diagram.png")


# ====================================================================
# C2: Class Distribution
# ====================================================================
print("C2. Class distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})

classes_all = ["Background", "No Damage", "Minor\nDamage", "Major\nDamage", "Destroyed"]
proportions = [97.0, 1.5, 0.3, 0.5, 0.7]
colors = ["#9e9e9e", "#4caf50", "#fdd835", "#ff9800", "#f44336"]

# Left: full scale
bars = ax1.bar(classes_all, proportions, color=colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, proportions):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.8,
             f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax1.set_ylabel("Pixel Proportion (%)", fontsize=12)
ax1.set_title("Full Scale: Background Dominates", fontsize=13, fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylim(0, 110)

# Right: damage classes only (zoomed)
damage_classes = ["No Damage", "Minor\nDamage", "Major\nDamage", "Destroyed"]
damage_props = [1.5, 0.3, 0.5, 0.7]
damage_colors = ["#4caf50", "#fdd835", "#ff9800", "#f44336"]

bars2 = ax2.bar(damage_classes, damage_props, color=damage_colors, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars2, damage_props):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
             f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax2.set_ylabel("Pixel Proportion (%)", fontsize=12)
ax2.set_title("Zoomed: Damage Classes Share ~3% of Pixels", fontsize=13, fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, 2.0)

# Annotation
ax2.annotate("Minor Damage:\nonly 0.3% of pixels\n(hardest class)",
             xy=(1, 0.3), xytext=(1.8, 1.35),
             arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.5),
             fontsize=10, color="#e65100", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#e65100", alpha=0.9))

fig.suptitle("xView2 Tier 3: Extreme Class Imbalance (97% Background)",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
savefig(fig, "C2_class_distribution.png")


# ====================================================================
# C3: Two-Phase Training Schematic
# ====================================================================
print("C3. Two-phase training schematic...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

for idx, (ax, phase, title, frozen_color, train_color, details) in enumerate(zip(
    axes,
    [1, 2],
    ["Phase 1: Frozen Encoder\n(Decoder + Fusion Learn)", "Phase 2: Full Fine-Tuning\n(Differential LR)"],
    ["#e0e0e0", "#bbdefb"],
    ["#4caf50", "#4caf50"],
    [
        {"Encoder": ("FROZEN", "#bdbdbd", "#333"),
         "Fusion Blocks": ("TRAINING  |  lr = 3e-4", "#81c784", "#1b5e20"),
         "Decoder": ("TRAINING  |  lr = 3e-4", "#81c784", "#1b5e20"),
         "Head": ("TRAINING  |  lr = 3e-4", "#81c784", "#1b5e20")},
        {"Encoder (top 4 layers)": ("TRAINING  |  lr = 5e-5", "#64b5f6", "#0d47a1"),
         "Encoder (bottom layers)": ("FROZEN", "#bdbdbd", "#333"),
         "Fusion Blocks": ("TRAINING  |  lr = 3e-4", "#81c784", "#1b5e20"),
         "Decoder": ("TRAINING  |  lr = 3e-4", "#81c784", "#1b5e20"),
         "Head": ("TRAINING  |  lr = 3e-4", "#81c784", "#1b5e20")},
    ]
)):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)

    y_pos = 8.8
    for name, (status, color, tcolor) in details.items():
        h = 1.3 if "bottom" not in name else 0.9
        box = FancyBboxPatch((0.5, y_pos - h), 9, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(box)
        ax.text(3.2, y_pos - h/2, name, fontsize=12, fontweight="bold",
                color=tcolor, va="center")
        ax.text(7.8, y_pos - h/2, status, fontsize=11, color=tcolor,
                va="center", ha="center", fontweight="bold")
        y_pos -= h + 0.35

    # Duration annotation
    if idx == 0:
        ax.text(5, 0.8, "Duration: 10-15 epochs\nPurpose: Learn to interpret pretrained features\nwithout corrupting ImageNet weights",
                fontsize=11, ha="center", color="#333",
                bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#4caf50", alpha=0.85))
    else:
        ax.text(5, 0.8, "Duration: 20-50 epochs\nPurpose: Adapt encoder to satellite imagery\nEncoder LR 6-30x lower than decoder LR",
                fontsize=11, ha="center", color="#333",
                bbox=dict(boxstyle="round,pad=0.4", fc="#e3f2fd", ec="#42a5f5", alpha=0.85))

fig.suptitle("Two-Phase Training Strategy", fontsize=17, fontweight="bold", y=1.02)
fig.tight_layout()
savefig(fig, "C3_two_phase_training.png")


# ====================================================================
# C4: All-Experiments Loss Overlay
# ====================================================================
print("C4. All-experiments loss overlay...")

EXPERIMENTS = {
    "Exp 1": RUNS / "siamese_resunet_xview2_1" / "metrics.csv",
    "Exp 2": RUNS / "siamese_resunet_xview2_2" / "metrics.csv",
    "Exp 3": RUNS / "siamese_resunet_xview2" / "metrics.csv",
    "Exp 4": RUNS / "siamese_resunet_xview2_3" / "metrics (1).csv",
    "Exp 5": RUNS / "siamese_resunet_xview2_4" / "epoch 60" / "metrics (4).csv",
}

COLORS = {
    "Exp 1": "#9e9e9e",
    "Exp 2": "#ff9800",
    "Exp 3": "#2196f3",
    "Exp 4": "#4caf50",
    "Exp 5": "#e91e63",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for label, path in EXPERIMENTS.items():
    if not path.exists():
        continue
    rows = load_csv(path)
    ep = col(rows, "epoch")
    tl = col(rows, "train_loss")
    vl = col(rows, "val_loss")
    ax1.plot(ep, tl, "-", color=COLORS[label], linewidth=2, label=label, alpha=0.85)
    ax2.plot(ep, vl, "-", color=COLORS[label], linewidth=2, label=label, alpha=0.85)

ax1.set_title("Training Loss", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", alpha=0.3)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)

ax2.set_title("Validation Loss", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", alpha=0.3)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)

fig.suptitle("Loss Curves Across All 5 Experiments",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
savefig(fig, "C4_loss_landscape_all.png")


print(f"\nDone! Extra figures saved to {OUT}/")
