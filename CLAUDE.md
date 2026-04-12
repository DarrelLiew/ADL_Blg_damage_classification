# xView2 Building Damage Segmentation

Semantic segmentation of building damage from pre/post-disaster satellite imagery using a Siamese ResNet U-Net. Trained on the [xView2 Tier 3 dataset](https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-tier-3-data).

## Project Structure

```
.
├── CLAUDE.md                              # This file -- project context for Claude
├── PROJECT_INSTRUCTIONS.md                # Course requirements (60.001 ADL Y2026)
├── REPORT.md                              # Project report (convert to PDF for submission)
├── PRESENTATION.md                        # Slide deck outline for Week 13 presentation
├── experiment_log.md                      # Full experiment history with analysis
│
├── Kaggle-latest.ipynb                    # THE notebook -- upload to Kaggle and run
├── training_script.ipynb                  # Local version (same code, local paths)
├── build_process_img_notebook.py          # Generator script for training_script.ipynb
├── verify_pre_mask_overlay.py             # Mask verification utility
│
├── training_runs/                         # All experiment configs, checkpoints, metrics
│   ├── experiments.md                     # Experiment roadmap (7 proposed experiments)
│   ├── siamese_resunet_xview2_1/          # Exp 1: Baseline (ResNet34, 5 epochs)
│   │   ├── config.json
│   │   ├── history.json
│   │   ├── metrics.csv
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   ├── siamese_resunet_xview2_2/          # Exp 2: Extended training (35 epochs)
│   │   ├── config.json
│   │   ├── changes.md
│   │   ├── history.json
│   │   ├── metrics.csv
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   ├── siamese_resunet_xview2/            # Exp 3: Training strategy overhaul (42 epochs)
│   │   ├── config.json
│   │   ├── history.json
│   │   ├── metrics.csv
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   ├── siamese_resunet_xview2_3/          # Exp 4: ResNet50 backbone (32 epochs, ongoing)
│   │   ├── config.json
│   │   ├── config (1).json
│   │   ├── changes.md
│   │   ├── history (1).json
│   │   ├── metrics (1).csv
│   │   ├── best_model.pt
│   │   └── last_model.pt
│   └── siamese_resunet_xview2_4/          # Exp 5: Tuned phase 2 + Focal Loss
│       ├── config.json
│       └── changes.md
│
├── data/                                  # Dataset (NOT in git, download from Kaggle)
│   ├── train/train/
│   │   ├── images/                        # ~3,732 PNGs (pre + post pairs)
│   │   └── labels/                        # ~3,732 JSON polygon annotations
│   └── test/test/
│       └── images/
│
└── verification_outputs/                  # Mask overlay debug images
```

## How Experiments Work

### Running a new experiment on Kaggle

1. Create a new folder under `training_runs/` (e.g. `siamese_resunet_xview2_5`)
2. Write a `config.json` with all hyperparameters
3. Write a `changes.md` documenting what changed from the previous experiment and why
4. Update **Cell 1** in `Kaggle-latest.ipynb` to match the new config
5. Upload `Kaggle-latest.ipynb` to Kaggle and run

### After a run completes

Download from Kaggle: `config.json`, `history.json`, `metrics.csv`, `best_model.pt`, `last_model.pt` into the experiment folder.

### What to update after each experiment

When a new experiment is run or results change, update **all applicable** files:

| File | When to update |
|------|---------------|
| `experiment_log.md` | **Every experiment.** Add results, per-class F1, observations, and key takeaways. Update the Results Summary table. |
| `REPORT.md` | **Every experiment.** Add the experiment to Section 4, update the Results Summary table (4.5), update SOTA comparison (4.6), and update the Conclusion. |
| `PRESENTATION.md` | **When results change significantly.** Update Slides 5-8 (results, per-class, SOTA comparison). Update the demo if the best model changes. |
| `training_runs/<exp>/changes.md` | **Before each experiment.** Document what changed and why, with expected impact. This is written before the run starts. |
| `README.md` | **When project structure changes.** Update experiments table and usage instructions. |

### Key metrics to track

- **Primary:** Val Damage Macro F1 (macro average of F1 for classes 1-4, excludes background)
- **Secondary:** Val mIoU, Val Mean Dice
- **Per-class:** Val F1 for No Damage, Minor Damage, Major Damage, Destroyed
- **Diagnostic:** Train-val loss gap (overfitting indicator)

## Architecture

Siamese ResNet U-Net:
- Shared ResNet encoder (ResNet34 or ResNet50) processes pre + post images
- FusionBlocks at each level: concat(pre, post, |diff|) -> ConvBlock
- U-Net decoder with skip connections from fused features
- 1x1 conv head -> 5-class output

## Training Strategy

Two-phase training:
- **Phase 1** (frozen encoder): Decoder + fusion blocks learn from pretrained features
- **Phase 2** (unfrozen encoder): Full fine-tuning with differential LR (encoder gets lower LR)

Loss: Cross-Entropy (or Focal) + Dice, with log-inverse class weights.

## Current Best Results

| Experiment | Damage F1 | mIoU | Backbone |
|-----------|-----------|------|----------|
| Exp 3 (training overhaul) | 0.510 | 0.475 | ResNet34 |
| Exp 4 (ResNet50) | 0.521 | 0.488 | ResNet50 |
| Exp 5 (tuned phase 2 + Focal) | **0.546** | **0.503** | ResNet50 |

Exp 5 completed the full 60 epochs. Best epoch: 49.

Competition reference: top solutions achieved ~0.72-0.78 damage F1 with ensembles on the full xBD dataset.

## Dataset

- Source: [xView2 Tier 3 on Kaggle](https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-tier-3-data)
- ~2,200 image pairs, 1024x1024 resized to 512x512
- 5 classes: Background (~97%), No Damage, Minor Damage, Major Damage, Destroyed
- Split: 80/10/10 train/val/test (seed=42)

## Dependencies

```
Python 3.11+
torch, torchvision (with CUDA)
opencv-python, scikit-image, scikit-learn
shapely, matplotlib, numpy, tqdm
```
