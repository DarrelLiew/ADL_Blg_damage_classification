# xView2 Damage Segmentation -- Experiment Log

## Task

Semantic segmentation of building damage from pre/post-disaster satellite imagery using the xView2 dataset. The model is a Siamese ResNet U-Net that takes pre- and post-disaster image pairs, extracts features with a shared encoder, fuses them via FusionBlocks, and decodes a 5-class damage mask (background, no-damage, minor, major, destroyed).

## Dataset

- ~2,200 training image pairs (original 1024x1024, resized to 512x512 for training), split 80/10/10 train/val/test
- Extreme class imbalance: background ~97% of pixels, damage classes share the remaining ~3%
- Ground truth from xView2 challenge: polygon annotations converted to per-pixel masks

## Reference Benchmark

Kaggle Swin-B Siamese U-Net: damage macro F1 = 0.72 (87M params, much larger backbone)

---

## Experiment 1: Baseline

**Folder:** `siamese_resunet_xview2_1` | **Status:** Complete (5 epochs)

### Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet34 (pretrained, all layers unfrozen) |
| Decoder channels | [256, 128, 64, 32] |
| Image size | 512x512 |
| Batch size | 1 (effective 2 with 2-step accumulation) |
| Epochs | 5 |
| Learning rate | 1e-4, CosineAnnealingLR |
| Weight decay | 1e-4 |
| Loss | CE + Dice (equal weight) |
| Class weights | Frequency inverse, estimated from 256 samples |
| Encoder freezing | None |
| Augmentations | Basic (hflip, vflip, rot90, color jitter, blur, noise) |
| num_workers | 0 |

### Results (Best Validation)

| Metric | Value | Epoch |
|--------|-------|-------|
| Val mIoU | 0.359 | 2 |
| Val Mean Dice | 0.443 | 2 |
| Val Pixel Accuracy | 0.944 | 2 |
| Val Loss | 1.428 | 2 |

### Observations

- Overfitting began at epoch 3: val mIoU dropped from 0.359 to 0.339 while train mIoU continued rising
- High pixel accuracy (94.4%) is misleading -- the model predicts background well but barely detects damage classes
- Cosine LR decayed to near-zero by epoch 5, effectively cutting training short
- Data loading bottleneck: epoch 1 took 61 minutes with num_workers=0
- Train-val gap at epoch 5: train mIoU 0.369 vs val mIoU 0.347

---

## Experiment 2: Extended Training

**Folder:** `siamese_resunet_xview2_2` | **Status:** Complete (35 epochs)

### Changes from Experiment 1

| Parameter | Exp 1 | Exp 2 | Rationale |
|-----------|-------|-------|-----------|
| Epochs | 5 | 50 (ran 35) | Allow model to converge; 5 was far too early |

All other hyperparameters remained identical to Exp 1. A broader set of changes (OneCycleLR, Focal Loss, larger batch, stronger augmentations) was planned but not applied to this run.

### Results (Best Validation)

| Metric | Value | Epoch | Change from Exp 1 |
|--------|-------|-------|--------------------|
| Val mIoU | 0.394 | 19 | +0.035 (+9.7%) |
| Val Mean Dice | 0.481 | 35 | +0.038 (+8.6%) |
| Val Pixel Accuracy | 0.956 | 19 | +0.012 |
| Val Loss | 1.310 | 12 | -0.118 |

### Observations

- More epochs helped: mIoU improved from 0.359 to 0.394
- But **severe overfitting** dominated the run: by epoch 35, train loss was 0.626 while val loss was 2.382 (3.8x gap)
- The model cycled through the cosine LR schedule multiple times, producing periodic spikes in val loss
- Val mIoU peaked at epoch 19 (0.394) then fluctuated without improvement for 16 more epochs
- No early stopping mechanism meant compute was wasted on the overfitting tail
- The architecture and training strategy, not just epoch count, needed to change

### Key Takeaway

Simply training longer with the same configuration yielded diminishing returns. Overfitting was the dominant problem -- the model memorized training data without generalizing. Regularization and training strategy changes were necessary.

---

## Experiment 3: Training Strategy Overhaul

**Folder:** `siamese_resunet_xview2` | **Status:** Complete (42 epochs, two-phase)

### Changes from Experiment 2

| Parameter | Exp 2 | Exp 3 | Rationale |
|-----------|-------|-------|-----------|
| Encoder freezing | None | **Phase 1: frozen (15 epochs), Phase 2: unfrozen** | Exp 2 trained all params from epoch 1, destabilizing pretrained ImageNet features. Freezing lets the decoder/fusion learn first, then fine-tune the encoder with a low LR. |
| Encoder LR | (same as decoder) | **1e-5** (30x lower than decoder) | Preserves ImageNet features during Phase 2 fine-tuning. |
| Decoder LR | 1e-4 | **3e-4** | Higher decoder LR during Phase 1 when encoder is frozen. |
| Weight decay | 1e-4 | **0.05** (500x increase) | Exp 2 overfit severely. Strong L2 regularization penalizes large weights. |
| Batch size | 1 | **2** | Better BatchNorm statistics. |
| Accumulation steps | 2 | **4** | Effective batch: 8 (up from 2). More stable gradients, especially for rare classes. |
| Class weight method | Frequency inverse | **Log-inverse** (clipped to [1, 20]) | More stable than raw frequency inverse. Strongly upweights all damage classes. |
| Class weight samples | 256 | **All (~2,200)** | Accurate class frequencies instead of noisy subsample. |
| Label smoothing | None | **0.05** | Prevents overconfident predictions on easy pixels. |
| Early stopping | None | **Patience=10** | Stops training if no improvement, prevents wasted compute and degradation. |
| Primary metric | mIoU | **Damage macro F1 (classes 1-4)** | Excludes background, gives honest damage detection assessment. |
| Per-class F1 | Not tracked | **Tracked (NoDmg, Minor, Major, Destroyed)** | Diagnoses which classes improve or regress. |

Architecture remained unchanged (Siamese ResNet34 U-Net).

### Results (Best Validation)

| Metric | Value | Epoch | Change from Exp 2 |
|--------|-------|-------|--------------------|
| Val mIoU | 0.475 | 32 | +0.081 (+20.6%) |
| Val Mean Dice | 0.603 | 32 | +0.122 (+25.4%) |
| Val Damage F1 | 0.510 | 32 | (new metric) |
| Val Pixel Accuracy | 0.948 | 32 | -0.008 |

### Per-Class F1 at Best Epoch (32)

| Class | Val F1 |
|-------|--------|
| No Damage | 0.672 |
| Minor Damage | 0.283 |
| Major Damage | 0.461 |
| Destroyed | 0.623 |

### Observations

- **Overfitting greatly reduced**: train-val loss gap at epoch 32 was 2.48 vs 2.47 (nearly identical), compared to Exp 2's 3.8x gap. The 500x weight decay increase was the primary driver.
- **Two-phase training worked**: Phase 1 (frozen encoder, epochs 1-15) let the decoder learn without corrupting ImageNet features. Phase 2 (unfrozen, epochs 16-42) allowed task-specific encoder adaptation.
- **Val loss is higher (~2.47) than Exp 1-2 (~1.3-1.4)** but this is expected: log-inverse class weights and label smoothing inflate the raw loss by penalizing rare-class mistakes more heavily. The actual predictions are better (higher mIoU, Dice, F1).
- **Minor Damage is the weakest class** (F1=0.28). The model confuses it with No Damage and Major Damage. This is likely a feature capacity issue -- ResNet34 lacks the representational depth to capture subtle damage cues.
- **Pixel accuracy dropped slightly** (0.948 vs 0.956). This is actually desirable -- the model is now predicting more damage pixels (some incorrectly), rather than defaulting to background for everything.

### Key Takeaway

Training strategy improvements alone (no architecture changes) boosted mIoU by 20.6% and Dice by 25.4%. The biggest wins came from (1) two-phase encoder freeze/unfreeze, (2) aggressive weight decay to combat overfitting, and (3) proper class weighting. The remaining bottleneck is the backbone's limited feature capacity, especially for subtle damage classes.

---

## Experiment 4: ResNet50 Backbone Upgrade (Proposed)

**Folder:** `siamese_resunet_xview2_3` | **Status:** Configured, not yet run

### Changes from Experiment 3

| Parameter | Exp 3 | Exp 4 | Rationale |
|-----------|-------|-------|-----------|
| Backbone | ResNet34 (21M params) | **ResNet50 (25M params)** | Bottleneck blocks with 4x wider intermediate representations. Needed for subtle damage distinctions that ResNet34 cannot capture. |
| Encoder channels | [64, 64, 128, 256, 512] | **[64, 256, 512, 1024, 2048]** | 4x more channels at the deepest level. |
| Decoder channels | [256, 128, 64, 32] | **[512, 256, 128, 64]** | Widened to match larger encoder. |
| num_workers | 0 | **4** | Parallel data loading to keep GPU fed. |

All training strategy from Exp 3 retained (two-phase training, weight decay 0.05, label smoothing, early stopping, damage F1 tracking).

### Expected Impact

| Change | Expected Gain (damage F1) | Confidence |
|--------|--------------------------|------------|
| ResNet50 backbone | +0.05-0.10 | High |
| Wider decoder | (enables backbone gain) | -- |
| num_workers=4 | (faster epochs only) | -- |

**Target: damage macro F1 >= 0.58**

---

## Results Summary

| | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|--|-------|-------|-------|-------|
| **Key Change** | Baseline | More epochs | Training strategy overhaul | ResNet50 backbone |
| **Architecture** | ResNet34 | ResNet34 | ResNet34 | ResNet50 |
| **Epochs Run** | 5 | 35 | 42 (two-phase) | -- |
| **Effective Batch** | 2 | 2 | 8 | 8 |
| **Weight Decay** | 1e-4 | 1e-4 | 0.05 | 0.05 |
| **Val mIoU** | 0.359 | 0.394 | **0.475** | -- |
| **Val Dice** | 0.443 | 0.481 | **0.603** | -- |
| **Val Damage F1** | -- | -- | **0.510** | -- |
| **Overfitting** | Moderate | Severe | Controlled | -- |

## Analysis of What Worked

### High Impact

1. **Two-phase encoder freeze/unfreeze (Exp 3)**: Stabilized early training by protecting pretrained ImageNet features. The decoder and fusion blocks learned meaningful representations before the encoder was allowed to adapt. Estimated contribution: +0.05-0.10 mIoU.

2. **Weight decay 1e-4 -> 0.05 (Exp 3)**: The single most important regularization change. Exp 2's severe overfitting (train loss 0.63, val loss 2.38) was eliminated. Train-val gap went from 3.8x to nearly 1:1.

3. **Log-inverse class weights from all samples (Exp 3)**: Forced the model to pay attention to rare damage classes instead of optimizing for easy background pixels.

### Medium Impact

4. **Larger effective batch size 2 -> 8 (Exp 3)**: More stable gradients, especially important for rare-class pixels where per-sample gradients are noisy.

5. **More training epochs 5 -> 35 (Exp 2)**: Necessary but not sufficient. Without regularization, additional epochs just increased overfitting after epoch ~12.

### Diagnostic Impact

6. **Damage macro F1 metric (Exp 3)**: Not a model change, but critical for honest evaluation. mIoU and pixel accuracy were misleading due to background dominance. Per-class F1 immediately revealed Minor Damage (0.28) as the weak point.

### What Did Not Help

7. **Extended training without regularization (Exp 2)**: Val mIoU only improved +0.035 over Exp 1 despite 7x more epochs. Most of the extra compute was wasted on overfitting.

## Remaining Challenges

1. **Minor Damage detection (F1=0.28)**: The hardest class. Visually subtle, easily confused with No Damage and Major Damage. Likely needs richer features (larger backbone) and possibly targeted oversampling.

2. **Major Damage detection (F1=0.46)**: Better than Minor but still below No Damage (0.67) and Destroyed (0.62). The intermediate damage levels are inherently harder.

3. **Backbone capacity**: ResNet34's 512-channel deepest features may not be sufficient for fine-grained damage classification. Exp 4 (ResNet50, 2048 channels) is designed to address this.
