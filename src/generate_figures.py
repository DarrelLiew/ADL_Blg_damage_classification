"""
Generate report figures from experiment metrics.
Output: images/ folder with publication-quality PNGs.

Follows academic-plotting skill standards:
  - Serif fonts (Times New Roman / DejaVu Serif)
  - 300 DPI
  - Okabe-Ito / Ocean Dusk colorblind-safe palette
  - No titles inside figures (captions handle it)
  - Frameless legends, subtle grids, clean spines

Usage:
    python generate_figures.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Academic-Plotting Publication Defaults ─────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "patch.edgecolor": "white",
    "patch.linewidth": 0.5,
})

# ── Ocean Dusk Palette (colorblind-safe) ───────────────────────
PALETTE = {
    "teal":   "#264653",
    "cyan":   "#2A9D8F",
    "gold":   "#E9C46A",
    "orange": "#F4A261",
    "coral":  "#E76F51",
    "blue":   "#0072B2",
    "sky":    "#56B4E9",
    "gray":   "#8C8C8C",
}
OUR_COLOR = "#E76F51"
BASELINE_COLOR = "#B0BEC5"

# Experiment colours (Ocean Dusk mapped)
COLORS = {
    "Exp 1": BASELINE_COLOR,
    "Exp 2": PALETTE["orange"],
    "Exp 3": PALETTE["blue"],
    "Exp 4": PALETTE["cyan"],
    "Exp 5": OUR_COLOR,
}

# Per-class colours (Okabe-Ito safe)
CLASS_NAMES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]
CLASS_KEYS  = ["val_f1_nodmg", "val_f1_minor", "val_f1_major", "val_f1_destroyed"]
CLASS_COLORS = ["#009E73", "#E69F00", "#D55E00", "#CC79A7"]
CLASS_MARKERS = ["o", "s", "^", "D"]

# ── Paths ──────────────────────────────────────────────────────
BASE = Path(__file__).parent
RUNS = BASE / "training_runs"
OUT = BASE / "images"
OUT.mkdir(exist_ok=True)

EXPERIMENTS = {
    "Exp 1": RUNS / "siamese_resunet_xview2_1" / "metrics.csv",
    "Exp 2": RUNS / "siamese_resunet_xview2_2" / "metrics.csv",
    "Exp 3": RUNS / "siamese_resunet_xview2" / "metrics.csv",
    "Exp 4": RUNS / "siamese_resunet_xview2_3" / "metrics (1).csv",
    "Exp 5": RUNS / "siamese_resunet_xview2_4" / "epoch 60" / "metrics (4).csv",
}

# ── Helpers ────────────────────────────────────────────────────
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


def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"  -> {path.name}")


def val_label(ax, bars, fmt=".2f", offset=0.008, fontsize=7, color="#444"):
    """Small, subtle value labels above bars."""
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=fontsize, color=color)


# ── Load Data ──────────────────────────────────────────────────
data = {}
for label, path in EXPERIMENTS.items():
    if path.exists():
        data[label] = load_csv(path)
    else:
        print(f"WARNING: {path} not found, skipping {label}")
print(f"Loaded {len(data)} experiments.\n")


# ╔════════════════════════════════════════════════════════════════╗
# ║  A1. Exp 2 Overfitting — train vs val loss divergence         ║
# ╚════════════════════════════════════════════════════════════════╝
print("A1. Exp 2 overfitting...")
fig, ax = plt.subplots(figsize=(5.5, 3.5))
rows2 = data["Exp 2"]
ep2 = col(rows2, "epoch")
tl2 = col(rows2, "train_loss")
vl2 = col(rows2, "val_loss")

ax.plot(ep2, tl2, color=OUR_COLOR, label="Train loss", marker="o",
        markevery=max(1, len(ep2)//8), markersize=4)
ax.plot(ep2, vl2, color=PALETTE["blue"], label="Val loss", marker="s",
        markevery=max(1, len(ep2)//8), markersize=4)
ax.fill_between(ep2, tl2, vl2, alpha=0.10, color=OUR_COLOR)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(loc="upper left")
savefig(fig, "A1_exp2_overfitting.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  A2. Exp 2 vs 3 — overfitting before/after                   ║
# ╚════════════════════════════════════════════════════════════════╝
print("A2. Exp 2 vs 3 — before/after...")
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.75, 3.0))

panels = [
    ("Exp 2", ax_l, "(a) Before: weight decay 1e\u22124"),
    ("Exp 3", ax_r, "(b) After: weight decay 0.05"),
]
for exp, ax, subtitle in panels:
    rows = data[exp]
    ep = col(rows, "epoch")
    tl = col(rows, "train_loss")
    vl = col(rows, "val_loss")
    ax.plot(ep, tl, color=OUR_COLOR, label="Train", marker="o",
            markevery=max(1, len(ep)//8), markersize=3)
    ax.plot(ep, vl, color=PALETTE["blue"], label="Val", marker="s",
            markevery=max(1, len(ep)//8), markersize=3)
    ax.fill_between(ep, tl, vl, alpha=0.08, color=OUR_COLOR)
    ax.set_title(subtitle, fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")

fig.tight_layout()
savefig(fig, "A2_exp2v3_overfitting_fix.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  A3. Exp 3 vs 4 — per-class F1 grouped bars                  ║
# ╚════════════════════════════════════════════════════════════════╝
print("A3. Exp 3 vs 4 — per-class F1...")
fig, ax = plt.subplots(figsize=(5.5, 3.5))

x = np.arange(len(CLASS_NAMES))
width = 0.35

for i, (exp, color, label) in enumerate([
    ("Exp 3", PALETTE["blue"], "Exp 3 (ResNet34)"),
    ("Exp 4", PALETTE["cyan"], "Exp 4 (ResNet50)"),
]):
    rows = data[exp]
    best_idx = int(col(rows, "val_damage_f1").argmax())
    vals = [rows[best_idx][k] for k in CLASS_KEYS]
    bars = ax.bar(x + i * width, vals, width, label=label, color=color)
    val_label(ax, bars, offset=0.01)

# Highlight Minor Damage gain — well above the value labels
ax.annotate("+53%", xy=(1 + width/2, 0.48), fontsize=10,
            fontweight="bold", color=PALETTE["cyan"], ha="center")
# Highlight Major Damage regression — well above the value labels
ax.annotate("\u221217%", xy=(2 + width/2, 0.50), fontsize=10,
            fontweight="bold", color="#C44E52", ha="center")

ax.set_xticks(x + width / 2)
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 0.90)
ax.legend(loc="upper right")
savefig(fig, "A3_exp3v4_perclass.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  A4. Exp 4 vs 5 — Phase 2 damage F1 comparison               ║
# ╚════════════════════════════════════════════════════════════════╝
print("A4. Exp 4 vs 5 — Phase 2 adaptation...")
fig, ax = plt.subplots(figsize=(5.5, 3.5))

for exp, color, marker, label in [
    ("Exp 4", PALETTE["cyan"], "s", "Exp 4 (lr=1e\u22125, 2 layers)"),
    ("Exp 5", OUR_COLOR, "o", "Exp 5 (lr=5e\u22125, 4 layers + Focal)"),
]:
    rows = data[exp]
    p2 = [r for r in rows if r.get("phase", 2) == 2]
    if not p2:
        continue
    ep = col(p2, "epoch")
    f1 = col(p2, "val_damage_f1")
    ax.plot(ep, f1, color=color, marker=marker,
            markevery=max(1, len(ep)//10), markersize=4, label=label)
    # Best marker
    bi = f1.argmax()
    ax.plot(ep[bi], f1[bi], "*", color=color, markersize=10, zorder=5)
    # Offset label to the side of the star so it does not sit above other text
    xoff = -8 if exp == "Exp 5" else 4
    yoff = 10 if exp == "Exp 5" else 8
    ha = "right" if exp == "Exp 5" else "left"
    ax.annotate(f"{f1[bi]:.3f}", xy=(ep[bi], f1[bi]),
                xytext=(xoff, yoff), textcoords="offset points",
                fontsize=9.5, fontweight="bold", color=color, ha=ha)

ax.set_xlabel("Epoch")
ax.set_ylabel("Val Damage Macro F1")
ax.set_ylim(top=0.60)
ax.legend(loc="lower right")
savefig(fig, "A4_exp4v5_phase2_adaptation.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  A5. Exp 5 deep-dive — two-panel (damage F1 + per-class)     ║
# ╚════════════════════════════════════════════════════════════════╝
print("A5. Exp 5 deep-dive...")
rows5 = data["Exp 5"]
ep5 = col(rows5, "epoch")
phases5 = col(rows5, "phase")
dmg_f1_5 = col(rows5, "val_damage_f1")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.75, 6.2), sharex=True,
                                gridspec_kw={"height_ratios": [1, 1.15]})

# --- (a) Damage F1 coloured by phase ---
p1m = phases5 == 1
p2m = phases5 == 2
ax1.plot(ep5[p1m], dmg_f1_5[p1m], "o-", color=PALETTE["blue"],
         markersize=4, label="Phase 1 (frozen encoder)")
ax1.plot(ep5[p2m], dmg_f1_5[p2m], "s-", color=OUR_COLOR,
         markersize=4, label="Phase 2 (unfrozen, Focal)")

# Phase boundary — place label to the right of the line, not above peak
ax1.axvline(x=10.5, color="#888", linestyle="--", linewidth=1, alpha=0.6)
ax1.text(10.5, 0.08, "Encoder\nunfrozen",
         fontsize=9, ha="center", color="#444", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

# Best epoch — annotation to the right and above the star
bi = dmg_f1_5.argmax()
ax1.plot(ep5[bi], dmg_f1_5[bi], "*", color=OUR_COLOR, markersize=10, zorder=5)
ax1.annotate(f"Best: {dmg_f1_5[bi]:.3f} (ep {int(ep5[bi])})",
             xy=(ep5[bi], dmg_f1_5[bi]),
             xytext=(-80, 14), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="#333", lw=1.0),
             fontsize=9.5, fontweight="bold", color="#333")

ax1.set_title("(a)", loc="left", fontweight="bold", fontsize=11)
ax1.set_ylabel("Val Damage Macro F1")
ax1.set_ylim(0, 0.65)
ax1.legend(loc="lower right", fontsize=8.5)

# --- (b) Per-class F1 ---
for key, name, color, marker in zip(CLASS_KEYS, CLASS_NAMES, CLASS_COLORS, CLASS_MARKERS):
    ax2.plot(ep5, col(rows5, key), color=color, marker=marker,
             markevery=max(1, len(ep5)//10), markersize=3.5, label=name)

ax2.axvline(x=10.5, color="#888", linestyle="--", linewidth=1, alpha=0.4)
ax2.set_title("(b)", loc="left", fontweight="bold", fontsize=11)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("F1 Score")
ax2.set_ylim(0, 0.85)
ax2.legend(loc="upper center", ncol=4, fontsize=8, framealpha=0.9,
           frameon=True, facecolor="white", edgecolor="#ccc")

fig.tight_layout()
savefig(fig, "A5_exp5_deepdive.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  B1. mIoU waterfall — training strategy dominates             ║
# ╚════════════════════════════════════════════════════════════════╝
print("B1. mIoU waterfall...")
fig, ax = plt.subplots(figsize=(8.0, 4.0))

steps = [
    ("Exp 1\nBaseline",             0.359, None,           BASELINE_COLOR),
    ("+Epochs\n(Exp 2)",            None,  0.394 - 0.359,  PALETTE["orange"]),
    ("+Training\nOverhaul\n(Exp 3)", None,  0.475 - 0.394,  PALETTE["blue"]),
    ("+ResNet50\n(Exp 4)",          None,  0.488 - 0.475,  PALETTE["cyan"]),
    ("+Focal+Tuned\n(Exp 5)",       None,  0.503 - 0.488,  OUR_COLOR),
    ("Total",                       0.503, None,           PALETTE["teal"]),
]

x_pos = np.arange(len(steps))
bottoms, heights = [], []
running = 0.0
for _, base, delta, _ in steps:
    if base is not None and delta is None:
        if running == 0:
            bottoms.append(0); heights.append(base); running = base
        else:
            bottoms.append(0); heights.append(base)
    else:
        bottoms.append(running); heights.append(delta); running += delta

bars = ax.bar(x_pos, heights, bottom=bottoms, width=0.55,
              color=[s[3] for s in steps])

# Connectors
for i in range(len(steps) - 2):
    top = bottoms[i] + heights[i]
    ax.plot([x_pos[i] + 0.28, x_pos[i+1] - 0.28], [top, top],
            color="#bbb", linewidth=0.8, linestyle="--")

# Value labels
for i, (bar, (label, base, delta, _)) in enumerate(zip(bars, steps)):
    top = bottoms[i] + heights[i]
    if i == 0:
        ax.text(bar.get_x() + bar.get_width()/2, top + 0.004,
                f"{top:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    elif label == "Total":
        ax.text(bar.get_x() + bar.get_width()/2, top + 0.004,
                f"{top:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=PALETTE["teal"])
    else:
        ax.text(bar.get_x() + bar.get_width()/2, top + 0.004,
                f"+{delta:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")

# Highlight Exp 3 — position below connectors to avoid overlap
ax.annotate("60% of total gain",
            xy=(2, bottoms[2] + heights[2] * 0.4),
            xytext=(3.8, 0.35),
            arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=1.5),
            fontsize=10, fontweight="bold", color=PALETTE["blue"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE["blue"], alpha=0.85))

ax.set_xticks(x_pos)
ax.set_xticklabels([s[0] for s in steps], fontsize=9)
ax.set_ylim(0, 0.60)
ax.set_ylabel("Val mIoU")
savefig(fig, "B1_miou_waterfall.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  B2. Rare-class challenge — per-class F1 across Exp 3/4/5    ║
# ╚════════════════════════════════════════════════════════════════╝
print("B2. Rare-class challenge...")
fig, ax = plt.subplots(figsize=(7.5, 4.2))

exps = ["Exp 3", "Exp 4", "Exp 5"]
exp_labels = ["Exp 3 (ResNet34)", "Exp 4 (ResNet50)", "Exp 5 (Focal+Tuned)"]
exp_colors = [PALETTE["blue"], PALETTE["cyan"], OUR_COLOR]
x = np.arange(len(CLASS_NAMES))
width = 0.24

for i, (exp, elabel, ecolor) in enumerate(zip(exps, exp_labels, exp_colors)):
    rows = data[exp]
    best_idx = int(col(rows, "val_damage_f1").argmax())
    vals = [rows[best_idx][k] for k in CLASS_KEYS]
    bars = ax.bar(x + i * width, vals, width, label=elabel, color=ecolor)
    # Only show value label on top bar per group to reduce clutter
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#333")

# Competition winner range for Minor Damage — positioned to not overlap bars
ax.axhspan(0.30, 0.45, xmin=0.22, xmax=0.50, alpha=0.06, color="#888")
ax.text(0.55, 0.37, "Competition\nwinner range", fontsize=8,
        color="#444", ha="center", fontweight="bold")

# Minor Damage trajectory arrow — well above all bars and labels, offset right
ax.annotate("+65%", xy=(1 + 2*width, 0.49),
            xytext=(2.1, 0.68),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2,
                            connectionstyle="arc3,rad=-0.2"),
            fontsize=10, fontweight="bold", color="#333")

ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 0.90)
ax.legend(loc="upper right", fontsize=9)
savefig(fig, "B2_rare_class_challenge.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  B3. Encoder adaptation arc — regression then recovery        ║
# ╚════════════════════════════════════════════════════════════════╝
print("B3. Encoder adaptation arc...")
fig, ax = plt.subplots(figsize=(6.0, 3.8))

exp_x = [0, 1, 2]
exp_xlabels = [
    "Exp 3\nResNet34",
    "Exp 4\nResNet50\n(lr=1e\u22125, 2 layers)",
    "Exp 5\nResNet50\n(lr=5e\u22125, 4 layers)",
]

for key, name, color, marker in zip(CLASS_KEYS, CLASS_NAMES, CLASS_COLORS, CLASS_MARKERS):
    vals = []
    for exp in ["Exp 3", "Exp 4", "Exp 5"]:
        rows = data[exp]
        best_idx = int(col(rows, "val_damage_f1").argmax())
        vals.append(rows[best_idx][key])
    ax.plot(exp_x, vals, color=color, marker=marker, markersize=7, label=name)
    # Offset right labels to avoid overlap (0.47 and 0.43 are close)
    right_offset = 0
    if name == "Major Damage":
        right_offset = 0.02
    elif name == "Minor Damage":
        right_offset = -0.02
    ax.text(2.15, vals[-1] + right_offset, f"{vals[-1]:.2f}", fontsize=9,
            va="center", color=color, fontweight="bold")
    # Offset left labels similarly
    left_offset = 0
    if name == "Major Damage":
        left_offset = 0.02
    elif name == "Minor Damage":
        left_offset = -0.02
    ax.text(-0.15, vals[0] + left_offset, f"{vals[0]:.2f}", fontsize=9,
            va="center", ha="right", color=color, fontweight="bold")

# Annotate Major Damage regression/recovery — positioned to avoid end-value labels
ax.annotate("Regressed", xy=(1, 0.384),
            xytext=(0.55, 0.28),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2),
            fontsize=9, fontweight="bold", color="#C44E52")
ax.annotate("Recovered", xy=(1.95, 0.44),
            xytext=(1.2, 0.34),
            arrowprops=dict(arrowstyle="->", color="#009E73", lw=1.2,
                            connectionstyle="arc3,rad=-0.2"),
            fontsize=9, fontweight="bold", color="#009E73")

ax.set_xticks(exp_x)
ax.set_xticklabels(exp_xlabels, fontsize=9)
ax.set_xlim(-0.35, 2.65)
ax.set_ylim(0.2, 0.82)
ax.set_ylabel("F1 Score at Best Epoch")
ax.legend(loc="upper center", fontsize=8.5, ncol=2)
savefig(fig, "B3_encoder_adaptation_arc.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  B4. SOTA comparison — horizontal bar                         ║
# ╚════════════════════════════════════════════════════════════════╝
print("B4. SOTA comparison...")
fig, ax = plt.subplots(figsize=(6.5, 3.8))

sota = [
    ("1st Place (vdurnov)\n12-model ensemble, full xBD",   0.75, BASELINE_COLOR),
    ("2nd Place (selimsef)\n2-model ensemble, full xBD",    0.72, BASELINE_COLOR),
    ("Single-model baseline\nfull xBD",                     0.625, BASELINE_COLOR),
    ("Ours (Exp 5)\nTier 3, single model, 60 ep",          0.546, OUR_COLOR),
]

ylabels = [s[0] for s in sota]
vals = [s[1] for s in sota]
colors = [s[2] for s in sota]

bars = ax.barh(ylabels, vals, color=colors, height=0.55)
for bar, val in zip(bars, vals):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", ha="left", va="center", fontsize=10, fontweight="bold", color="#333")

# Gap annotation — position to the right of the longest bar, not overlapping
ax.text(0.78, 3.2,
        "Gap: 10\u00d7 less data, no localisation\n"
        "pretraining, single model",
        fontsize=9.5, color="#333", fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#999", alpha=0.92))

ax.set_xlim(0, 0.95)
ax.set_xlabel("Damage Macro F1")
ax.invert_yaxis()
savefig(fig, "B4_sota_comparison.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  B5. Combined progression — mIoU + Damage F1 side by side    ║
# ╚════════════════════════════════════════════════════════════════╝
print("B5. Combined progression...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

all_labels = list(data.keys())
all_colors = [COLORS[l] for l in all_labels]

# --- (a) mIoU ---
miou_vals = [col(data[l], "val_miou").max() for l in all_labels]
bars = ax1.bar(all_labels, miou_vals, color=all_colors, width=0.55)
val_label(ax1, bars, fmt=".3f", offset=0.005, fontsize=8)

ax1.annotate(f"+{(miou_vals[-1] - miou_vals[0]) / miou_vals[0] * 100:.0f}% total",
             xy=(4, miou_vals[-1]),
             xytext=(2.5, miou_vals[-1] * 1.15),
             arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
             fontsize=9, fontweight="bold", color="#333")

ax1.set_title("(a) mIoU Progression", loc="left", fontsize=10, fontweight="bold")
ax1.set_ylim(0, max(miou_vals) * 1.28)
ax1.set_ylabel("Best Val mIoU")
ax1.tick_params(axis="x", labelsize=9)

# --- (b) Damage F1 ---
f1_vals = []
for l in all_labels:
    if "val_damage_f1" in data[l][0]:
        f1_vals.append(col(data[l], "val_damage_f1").max())
    else:
        f1_vals.append(0)

bars = ax2.bar(all_labels, f1_vals, color=all_colors, width=0.55)
for bar, val in zip(bars, f1_vals):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    else:
        ax2.text(bar.get_x() + bar.get_width()/2, 0.03,
                 "n/a", ha="center", va="bottom", fontsize=7, color="#777")

ax2.set_title("(b) Damage Macro F1 Progression", loc="left", fontsize=10, fontweight="bold")
ax2.set_ylim(0, max(f1_vals) * 1.22)
ax2.set_ylabel("Best Val Damage F1")
ax2.tick_params(axis="x", labelsize=9)

fig.tight_layout()
savefig(fig, "B5_combined_progression.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  C2. Class distribution — bar + zoomed inset                  ║
# ╚════════════════════════════════════════════════════════════════╝
print("C2. Class distribution...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.0),
                                gridspec_kw={"width_ratios": [1, 1.3]})

classes_all = ["BG", "No Dmg", "Minor\nDmg", "Major\nDmg", "Destroyed"]
pcts_all = [97.0, 1.5, 0.3, 0.5, 0.7]
colors_all = [BASELINE_COLOR] + CLASS_COLORS

# (a) Full scale
bars = ax1.bar(classes_all, pcts_all, color=colors_all, width=0.6)
val_label(ax1, bars, fmt=".1f", offset=0.5, fontsize=7)
ax1.set_title("(a) Full scale", loc="left", fontsize=10, fontweight="bold")
ax1.set_ylabel("Pixel proportion (%)")
ax1.tick_params(axis="x", labelsize=8)

# (b) Zoomed: damage classes only
classes_dmg = ["No Damage", "Minor\nDamage", "Major\nDamage", "Destroyed"]
pcts_dmg = [1.5, 0.3, 0.5, 0.7]
bars = ax2.bar(classes_dmg, pcts_dmg, color=CLASS_COLORS, width=0.55)
val_label(ax2, bars, fmt=".1f", offset=0.02, fontsize=7)
ax2.set_title("(b) Damage classes only (\u22643% total)",
              loc="left", fontsize=10, fontweight="bold")
ax2.set_ylabel("Pixel proportion (%)")
ax2.tick_params(axis="x", labelsize=7.5)

# Highlight Minor Damage
ax2.annotate("Hardest class:\n0.3% of pixels",
             xy=(1, 0.3), xytext=(2.2, 1.2),
             arrowprops=dict(arrowstyle="->", color=CLASS_COLORS[1], lw=1),
             fontsize=7, color=CLASS_COLORS[1], fontstyle="italic")

fig.tight_layout()
savefig(fig, "C2_class_distribution.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  C4. Loss curves across all experiments                       ║
# ╚════════════════════════════════════════════════════════════════╝
print("C4. Loss curves all experiments...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.0))

for exp, color in COLORS.items():
    if exp not in data:
        continue
    rows = data[exp]
    ep = col(rows, "epoch")
    marker_style = dict(markevery=max(1, len(ep)//8), markersize=3)
    ax1.plot(ep, col(rows, "train_loss"), color=color, label=exp,
             marker="o", **marker_style)
    ax2.plot(ep, col(rows, "val_loss"), color=color, label=exp,
             marker="s", **marker_style)

ax1.set_title("(a) Training loss", loc="left", fontsize=10, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend(loc="best", fontsize=7)

ax2.set_title("(b) Validation loss", loc="left", fontsize=10, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend(loc="best", fontsize=7)

# Clip val loss y-axis to avoid spike dominating
val_losses_all = []
for exp in data:
    val_losses_all.extend(col(data[exp], "val_loss").tolist())
p95 = np.percentile(val_losses_all, 95)
ax2.set_ylim(top=min(p95 * 1.3, max(val_losses_all)))

fig.tight_layout()
savefig(fig, "C4_loss_landscape_all.png")


# ╔════════════════════════════════════════════════════════════════╗
# ║  Summary                                                      ║
# ╚════════════════════════════════════════════════════════════════╝
figures = sorted(OUT.glob("*.png"))
print(f"\n{'='*60}")
print(f"Done! Generated figures saved to {OUT}/")
print(f"{'='*60}")
for f in figures:
    print(f"  {f.name}")
