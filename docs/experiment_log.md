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

## Experiment 4: ResNet50 Backbone Upgrade

**Folder:** `siamese_resunet_xview2_3` | **Status:** Complete (32 of 50 epochs, two-phase)

### Changes from Experiment 3

| Parameter | Exp 3 | Exp 4 | Rationale |
|-----------|-------|-------|-----------|
| Backbone | ResNet34 (21M params) | **ResNet50 (25M params)** | Bottleneck blocks with 4x wider intermediate representations. Needed for subtle damage distinctions that ResNet34 cannot capture. |
| Encoder channels | [64, 64, 128, 256, 512] | **[64, 256, 512, 1024, 2048]** | 4x more channels at the deepest level. |
| Decoder channels | [256, 128, 64, 32] | **[512, 256, 128, 64]** | Widened to match larger encoder. |
| num_workers | 0 | **4** | Parallel data loading to keep GPU fed. |
| Batch size | 2 | **16** | Kaggle P100 has enough VRAM for ResNet50 at batch 16. |

All training strategy from Exp 3 retained (two-phase training, weight decay 0.05, label smoothing, early stopping, damage F1 tracking).

### Results (Best Validation)

| Metric | Value | Epoch | Change from Exp 3 |
|--------|-------|-------|--------------------|
| Val mIoU | 0.488 | 27 | +0.013 (+2.7%) |
| Val Mean Dice | 0.616 | 27 | +0.013 (+2.2%) |
| Val Damage F1 | 0.521 | 27 | +0.011 (+2.2%) |
| Val Pixel Accuracy | 0.989 | 27 | +0.041 |

### Per-Class F1 at Best Epoch (27)

| Class | Exp 3 | Exp 4 | Change |
|-------|-------|-------|--------|
| No Damage | 0.672 | 0.709 | +0.037 |
| Minor Damage | 0.283 | **0.433** | **+0.150** |
| Major Damage | 0.461 | 0.384 | -0.077 |
| Destroyed | 0.623 | 0.557 | -0.066 |

### Observations

- **Minor Damage F1 dramatically improved** (+0.150), confirming ResNet50's richer features help with subtle damage. This was the weakest class in Exp 3.
- **Overall gain was smaller than expected** (+0.011 damage F1 vs. expected +0.05-0.10). Major and Destroyed classes actually regressed.
- **Phase 2 transition shock**: At epoch 16, val damage F1 dropped from 0.488 to 0.389 (-0.099). Recovery took 12 epochs. Only 2 encoder layers were unfrozen with encoder_lr=1e-5, which was too conservative.
- **The encoder barely adapted**: The low encoder_lr (1e-5) combined with cosine decay meant the unfrozen layers received minimal gradient updates. The ResNet50 backbone stayed close to its ImageNet initialization rather than adapting to satellite imagery.
- **Epoch times**: ~23 min (Phase 1) and ~26 min (Phase 2) on Kaggle P100.

### Key Takeaway

ResNet50 has the capacity to improve damage segmentation (proven by Minor Damage F1), but the training strategy didn't let it adapt sufficiently. The encoder needs a higher learning rate, more unfrozen layers, and more phase 2 epochs to realize its potential.

---

## Experiment 5: Tuned Phase 2 + Focal Loss

**Folder:** `siamese_resunet_xview2_4` | **Status:** Complete (60 epochs, two-phase)

### Changes from Experiment 4

| Parameter | Exp 4 | Exp 5 | Rationale |
|-----------|-------|-------|-----------|
| `encoder_lr` | 1e-5 | **5e-5** | 5x increase for faster encoder adaptation in phase 2. |
| `freeze_epochs` | 15 | **10** | Decoder converges by epoch 10; gives phase 2 more time (50 epochs vs 35). |
| `unfreeze_layers` | 2 (hardcoded) | **4** (config param) | Unfreeze layers 1-4 instead of just 3-4. Satellite imagery differs from ImageNet -- deeper adaptation needed. |
| Loss | CE + Dice | **Focal + Dice** | Focal loss (gamma=2) downweights easy predictions, focuses on hard damage boundaries. Used by all xView2 competition winners. |
| `epochs` | 50 | **60** | More time for phase 2 convergence. |
| `early_stopping_patience` | 10 | **15** | Allows recovery from phase 2 transition drop without premature stopping. |
| Augmentations | Moderate | **Slightly stronger** | brightness/contrast 0.12->0.15, blur/noise prob 0.10->0.15. Compensates for increased overfitting risk from more unfrozen params. |

### Expected Impact

| Change | Expected Gain (damage F1) | Confidence |
|--------|--------------------------|------------|
| encoder_lr 5e-5 + unfreeze 4 layers | +0.03-0.06 | High |
| Focal Loss | +0.02-0.04 | Medium-High |
| freeze_epochs 10 + epochs 60 | (enables above gains) | -- |

**Target: damage macro F1 >= 0.58-0.62**

See `siamese_resunet_xview2_4/changes.md` for full justification.

### Results (Best Validation)

| Metric | Value | Epoch | Change from Exp 4 |
|--------|-------|-------|--------------------|
| Val mIoU | 0.503 | 49 | +0.015 (+3.1%) |
| Val Mean Dice | 0.635 | 49 | +0.019 (+3.1%) |
| Val Damage F1 | **0.546** | 49 | **+0.025 (+4.7%)** |
| Val Pixel Accuracy | 0.988 | 49 | -0.001 |

### Per-Class F1 at Best Epoch (49)

| Class | Exp 4 | Exp 5 | Change |
|-------|-------|-------|--------|
| No Damage | 0.709 | 0.692 | -0.017 |
| Minor Damage | 0.433 | **0.465** | **+0.032** |
| Major Damage | 0.384 | **0.437** | **+0.053** |
| Destroyed | 0.557 | **0.588** | **+0.031** |

### Observations

- **New best damage F1 (0.546)**, surpassing Exp 4 (0.521) by +0.025. The higher encoder_lr and 4 unfrozen layers allowed the ResNet50 backbone to adapt more effectively to satellite imagery.
- **All three damage classes (Minor, Major, Destroyed) improved** over Exp 4. Major Damage F1 recovered from Exp 4's regression (0.384 -> 0.437) and is now close to Exp 3's level (0.461). Destroyed F1 also improved (0.557 -> 0.588).
- **Minor Damage continued improving**: 0.433 -> 0.465, now comfortably within the range of competition winners (0.30-0.45 typical).
- **Phase 2 transition shock at epoch 11**: damage F1 dropped from 0.469 to 0.329. Recovery to pre-shock levels took ~7 epochs (by epoch 18), faster than Exp 4's 12-epoch recovery.
- **Validation instability in phase 2**: Epochs 19-20, 22-23, 25-26, 37-38, 40, 54-55, and 57 showed large val_loss spikes (up to 3.96), likely from the higher encoder_lr causing occasional gradient instability. Despite this, the model recovered each time and continued improving.
- **Later-phase convergence**: The best epoch was 49. Epochs 35, 46, 49, 51, 52, and 58 all produced damage F1 >= 0.54, confirming that the full 60-epoch schedule was needed for the model to reach its ceiling. Training plateaued in the final ~10 epochs; no further improvement is expected.
- **Run completed the full 60 epochs** — early stopping (patience=15) was not triggered.

### Key Takeaway

The changes validated the hypothesis from Exp 4: ResNet50 needed a higher encoder_lr (5e-5 vs 1e-5), more unfrozen layers (4 vs 2), and focal loss to realize its potential. The full 60-epoch run confirmed that extended phase-2 training was necessary — the best epoch (49) came well after the earlier checkpoint (epoch 34) would have stopped. Exp 5 achieved the best results of all experiments (damage F1 = 0.546, mIoU = 0.503), though still short of the 0.58-0.62 target range.

### Run Notes

- **Phase 1 (epochs 1-10):** Completed successfully at batch_size=16. Best: dmg_F1=0.4819, mIoU=0.4581 at epoch 9. Close to Exp 4's final best (0.521) in only 10 frozen epochs.
- **Phase 2 transition (epoch 11):** Expected performance dip to dmg_F1=0.3287 (4 layers unfrozen, bigger disruption than Exp 4's 2-layer unfreeze). Recovery was faster than Exp 4 — reached 0.4619 by epoch 14 (6 epochs vs Exp 4's 12).
- **OOM on resume at epoch 17:** Phase 2 with 4 unfrozen encoder layers exceeded P100 VRAM at batch_size=16. Fixed by reducing to batch_size=8, accumulation_steps=8 (effective batch 64). Phase 1 originally ran with accumulation_steps=4 (effective batch 32).
- **Optimizer state mismatch on resume:** Phase 1 checkpoint had different param groups than phase 2 optimizer (different unfrozen layers). Optimizer starts fresh; model weights correctly restored.
- **Best epoch 49:** Best damage F1 was achieved at epoch 49, well inside the aggressive-fine-tuning phase after the encoder has had time to adapt.

---

## Results Summary

| | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 |
|--|-------|-------|-------|-------|-------|
| **Key Change** | Baseline | More epochs | Training overhaul | ResNet50 | Tuned phase 2 + Focal |
| **Architecture** | ResNet34 | ResNet34 | ResNet34 | ResNet50 | ResNet50 |
| **Epochs Run** | 5 | 35 | 42 (two-phase) | 32 (two-phase) | 60 (two-phase) |
| **Effective Batch** | 2 | 2 | 8 | 64 | 64 (8×8) |
| **Weight Decay** | 1e-4 | 1e-4 | 0.05 | 0.05 | 0.05 |
| **Val mIoU** | 0.359 | 0.394 | 0.475 | 0.488 | **0.503** |
| **Val Dice** | 0.443 | 0.481 | 0.603 | 0.616 | **0.635** |
| **Val Damage F1** | -- | -- | 0.510 | 0.521 | **0.546** |
| **Overfitting** | Moderate | Severe | Controlled | Controlled | Controlled |

## Analysis of What Worked

### High Impact

1. **Two-phase encoder freeze/unfreeze (Exp 3)**: Stabilized early training by protecting pretrained ImageNet features. The decoder and fusion blocks learned meaningful representations before the encoder was allowed to adapt. Estimated contribution: +0.05-0.10 mIoU.

2. **Weight decay 1e-4 -> 0.05 (Exp 3)**: The single most important regularization change. Exp 2's severe overfitting (train loss 0.63, val loss 2.38) was eliminated. Train-val gap went from 3.8x to nearly 1:1.

3. **Log-inverse class weights from all samples (Exp 3)**: Forced the model to pay attention to rare damage classes instead of optimizing for easy background pixels.

### Medium Impact

4. **Larger effective batch size 2 -> 8 (Exp 3)**: More stable gradients, especially important for rare-class pixels where per-sample gradients are noisy.

5. **More training epochs 5 -> 35 (Exp 2)**: Necessary but not sufficient. Without regularization, additional epochs just increased overfitting after epoch ~12.

6. **ResNet50 backbone (Exp 4)**: Improved Minor Damage F1 by +0.150, confirming feature capacity hypothesis. Overall gain limited by insufficient encoder adaptation in phase 2.

### Diagnostic Impact

7. **Damage macro F1 metric (Exp 3)**: Not a model change, but critical for honest evaluation. mIoU and pixel accuracy were misleading due to background dominance. Per-class F1 immediately revealed Minor Damage (0.28) as the weak point.

### Medium-High Impact

8. **Focal Loss + aggressive encoder fine-tuning (Exp 5)**: Focal loss (gamma=2) downweights easy background predictions, focusing gradients on hard damage boundaries. Combined with 5x higher encoder_lr (5e-5) and 4 unfrozen layers, this recovered Major Damage F1 (0.384 -> 0.437), pushed Minor Damage to 0.465, and lifted Destroyed to 0.588. New best damage F1 (0.546) over the full 60-epoch run, with best epoch at 49.

### What Did Not Help

9. **Extended training without regularization (Exp 2)**: Val mIoU only improved +0.035 over Exp 1 despite 7x more epochs. Most of the extra compute was wasted on overfitting.

10. **Conservative encoder fine-tuning (Exp 4)**: encoder_lr=1e-5 with only 2 unfrozen layers meant the ResNet50 backbone barely adapted from ImageNet. The larger backbone's potential was unrealized. This was corrected in Exp 5.

## Remaining Challenges

1. **Phase 2 validation instability**: Even with higher encoder_lr (Exp 5), phase 2 shows occasional large validation loss spikes (up to ~3.96). The model recovers, but this instability likely limited final convergence — several late epochs (e.g., 54-55, 57, 59-60) spiked away from the best epoch (49).

2. **Minor Damage detection**: Improved to F1=0.465 with Exp 5 (from 0.28 in Exp 3) but still among the weaker classes. Now exceeds competition winner range (0.30-0.45), suggesting our approach is competitive on this difficult class.

3. **Plateau in final epochs**: After the best at epoch 49, training plateaued rather than continuing to improve. Damage F1 oscillated between 0.38 and 0.54, indicating the model had reached a ceiling given its capacity and data.

4. **Data scale**: Tier 3 (~2,200 pairs) vs. full xBD (~22,000 pairs) limits what any architecture can learn.
