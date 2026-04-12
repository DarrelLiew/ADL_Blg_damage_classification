# Experiment 5 -- Changes and Justification

Baseline: Experiment 4 (`siamese_resunet_xview2_3`, 32 epochs, best val damage F1 0.521, best val mIoU 0.488)

---

## Diagnosis: Why Exp 4 Underperformed

Exp 4 upgraded ResNet34 -> ResNet50 but only gained +0.011 damage F1 (0.510 -> 0.521). The ResNet50 upgrade was expected to yield +0.05-0.10. Analysis of the training curves reveals:

1. **Phase 2 transition shock**: At epoch 16, val damage F1 dropped from 0.488 to 0.389 (-0.099). Only 2 encoder layers were unfrozen, but even this caused instability.
2. **Slow recovery**: It took 12 epochs to recover to phase 1 levels (epoch 27 reached 0.521). The encoder_lr of 1e-5 was too conservative for the unfrozen layers to adapt meaningfully.
3. **Late phase 2 start**: With freeze_epochs=15 out of 50 total, only 35 epochs remained for phase 2. Combined with slow recovery, the model had limited time to benefit from encoder fine-tuning.
4. **Minor Damage F1 improved (+0.150)**: Confirming ResNet50 helps with subtle features, but Major (-0.077) and Destroyed (-0.066) regressed, likely due to insufficient encoder adaptation.

---

## Changes

### 1. Encoder Learning Rate: 1e-5 -> 5e-5 (CHANGED)

| Parameter | Exp 4 | Exp 5 | Why |
|-----------|-------|-------|-----|
| `encoder_lr` | 1e-5 | **5e-5** | 5x increase allows the unfrozen encoder to adapt faster in phase 2. The 1e-5 rate was too conservative -- by the time cosine decayed it to near min_lr, the encoder had barely moved from its ImageNet initialization. At 5e-5, the encoder gets meaningful gradient updates for ~30 epochs before decay takes over. |

### 2. Freeze Epochs: 15 -> 10 (CHANGED)

| Parameter | Exp 4 | Exp 5 | Why |
|-----------|-------|-------|-----|
| `freeze_epochs` | 15 | **10** | Gives phase 2 a full 50 epochs instead of 35. Exp 4 showed the decoder converges within 10 epochs (val damage F1 plateaued around epoch 10-12 in phase 1). The extra 5 frozen epochs added minimal value but stole time from phase 2 fine-tuning. |

### 3. Unfreeze Layers: 2 -> 4 (CHANGED)

| Parameter | Exp 4 | Exp 5 | Why |
|-----------|-------|-------|-----|
| `unfreeze_layers` | 2 (hardcoded) | **4** (config param) | Exp 4 only unfroze layer3+layer4 of ResNet50. This left layers 1-2 (which contain low/mid-level features) frozen at ImageNet values. Satellite imagery differs significantly from ImageNet -- unfreezing 4 layers (layer1-4) allows the encoder to adapt its feature hierarchy to satellite textures and damage patterns. The differential LR (5e-5 vs 3e-4) still protects the encoder from catastrophic forgetting. |

### 4. Focal Loss: CE -> Focal (CHANGED)

| Parameter | Exp 4 | Exp 5 | Why |
|-----------|-------|-------|-----|
| Loss | CE + Dice | **Focal + Dice** | Every xView2 competition winner used Focal Loss. CE treats all pixels equally regardless of prediction confidence. Focal loss (gamma=2) downweights easy/confident predictions (background, clear no-damage) and focuses training on hard, ambiguous pixels (damage boundaries, minor vs major). This directly targets the model's weakest area. |
| `focal_gamma` | -- | **2.0** | Standard value from Lin et al. (2017). gamma=2 reduces the loss contribution of well-classified examples by ~100x compared to CE. |

### 5. Total Epochs: 50 -> 60 (CHANGED)

| Parameter | Exp 4 | Exp 5 | Why |
|-----------|-------|-------|-----|
| `epochs` | 50 | **60** | With freeze_epochs=10, phase 2 gets 50 full epochs. Combined with higher encoder_lr and more unfrozen layers, the model needs time to converge. Early stopping (patience=15) will cut training short if it plateaus. |

### 6. Early Stopping Patience: 10 -> 15 (CHANGED)

| Parameter | Exp 4 | Exp 5 | Why |
|-----------|-------|-------|-----|
| `early_stopping_patience` | 10 | **15** | Phase 2 transition causes a temporary performance drop (seen in Exp 4: ~12 epochs to recover). With patience=10, the model could be stopped during recovery. Patience=15 gives it time to recover and then plateau before stopping. |

### 7. Stronger Augmentations (CHANGED)

| Parameter | Exp 4 | Exp 5 | Why |
|-----------|-------|-------|-----|
| `aug_brightness` | 0.12 | **0.15** | With more unfrozen layers and higher encoder LR, overfitting risk increases. Slightly stronger augmentations act as additional regularization. |
| `aug_contrast` | 0.12 | **0.15** | Same rationale. |
| `aug_saturation` | 0.08 | **0.10** | Same rationale. |
| `aug_blur_prob` | 0.10 | **0.15** | Same rationale. |
| `aug_noise_prob` | 0.10 | **0.15** | Same rationale. |

### 8. Everything Else: Unchanged

All other settings retained from Exp 4:
- ResNet50 backbone with decoder [512, 256, 128, 64]
- Batch size 16, accumulation 4 (effective batch 64)
- Weight decay 0.05
- Cosine scheduler with min_lr=1e-6 (reset at phase 2)
- Log-inverse class weights from all samples
- Label smoothing 0.05
- Damage macro F1 as primary metric

---

## Code Changes Required

Two code changes in the notebook (see Kaggle-latest.ipynb):

### 1. Read `unfreeze_layers` from config (training loop)

Replace all hardcoded `n_layers=2` with `CONFIG.get("unfreeze_layers", 2)`:
```python
_raw_model.unfreeze_encoder(n_layers=CONFIG.get("unfreeze_layers", 2))
```

### 2. Add FocalLoss class and update CombinedLoss (loss cell)

Add `FocalLoss` class and modify `CombinedLoss` to use it when `focal_gamma` is set in config.

---

## Expected Impact

| Change | Expected Gain (damage F1) | Confidence |
|--------|--------------------------|------------|
| encoder_lr 5e-5 + unfreeze 4 layers | +0.03-0.06 | High |
| Focal Loss | +0.02-0.04 | Medium-High |
| freeze_epochs 10 + epochs 60 | (enables above gains) | -- |
| Stronger augmentations | (regularization, no direct gain) | -- |

**Target: damage macro F1 >= 0.58-0.62**

The main bottleneck in Exp 4 was insufficient encoder adaptation in phase 2. With 5x higher encoder LR, 2x more unfrozen layers, and 43% more phase 2 epochs, the ResNet50 backbone should finally deliver the feature richness improvement that was expected but unrealized in Exp 4.

---

## How to Run

Set `CONFIG_PATH` at the top of the notebook:
```python
CONFIG_PATH = "training_runs/siamese_resunet_xview2_4/config.json"
```

### VRAM Fix (applied after OOM on resume)

Resuming at epoch 17 (phase 2) with 4 unfrozen encoder layers caused OOM on Kaggle P100 (16GB). Phase 1 fit at batch_size=16 because the encoder was frozen (~30M trainable params, no activations stored for encoder backprop). Phase 2 with 4 unfrozen layers has ~249M trainable params, requiring activation storage for all encoder layers — exceeding VRAM at batch_size=16.

**Fix:** `batch_size`: 16 → 8, `accumulation_steps`: 4 → 8. Effective batch size unchanged (8×8 = 64). Training ~20-30% slower per epoch but mathematically equivalent.

Also: optimizer state could not be restored on resume (param group mismatch due to different unfrozen layers), so phase 2 optimizer starts fresh. Only momentum/velocity history is lost; model weights are correctly restored. Negligible impact on convergence (optimizer warms up within 1-2 epochs).
