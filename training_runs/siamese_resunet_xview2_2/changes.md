# Experiment 2 -- Hyperparameter Changes and Justification

Baseline: `siamese_resunet_xview2_1` (5 epochs, best val mIoU 0.359, best val Dice 0.443)

Architecture is **unchanged** (Siamese ResUNet, shared ResNet34 encoder, FusionBlocks, UNet decoder).
All changes below are hyperparameter / training-strategy only.

---

## Training Schedule

| Parameter | v1 | v2 | Why |
|-----------|----|----|-----|
| `epochs` | 5 | **50** | v1 stopped far too early. Train loss was still dropping at epoch 5 (1.18) with no plateau. The model never had a chance to converge. 50 epochs with early stopping lets the model reach its potential without wasting time if it plateaus. |
| `patience` | (none) | **10** | Early stopping after 10 epochs without val mIoU improvement prevents overfitting and wasted compute. Without this, v1 would have kept training past the epoch-2 peak. |
| `lr` | 1e-4 | **3e-4** | Higher peak LR paired with OneCycleLR warmup. The warmup phase prevents instability, and the higher peak lets the model explore the loss landscape more aggressively before annealing down. |
| `scheduler` | cosine | **onecycle** | OneCycleLR includes built-in linear warmup (first 10% of training) then cosine decay. CosineAnnealingLR in v1 started decaying from step 1, reaching 1e-6 by epoch 5 -- the model spent most of training at a near-zero LR. OneCycle is better suited for training-from-scratch schedules. |
| `pct_start` | (n/a) | **0.1** | 10% of training (5 epochs) spent warming up. Prevents early gradient explosions from the randomly-initialized decoder while the pretrained encoder is still adapting. |
| `weight_decay` | 1e-4 | **5e-4** | Stronger L2 regularization. v1 showed overfitting by epoch 3 (val mIoU peaked at epoch 2 while train mIoU kept climbing). Higher weight decay penalises large weights and should delay overfitting. |

## Batch Size and Data Loading

| Parameter | v1 | v2 | Why |
|-----------|----|----|-----|
| `batch_size` | 1 | **2** | Doubling real batch size improves BatchNorm statistics (computed over 2 images instead of 1, less noisy). Fits in 8GB VRAM with AMP at 512x512. |
| `accumulation_steps` | 2 | **8** | Effective batch size goes from 2 (1x2) to **16** (2x8). Larger effective batches produce more stable gradient estimates, smoother training curves, and better generalisation. 16 is a well-tested sweet spot for segmentation tasks. |
| `num_workers` | 0 | **4** | v1 loaded data on the main process, creating a CPU bottleneck. 4 parallel workers keep the GPU fed. Epoch 1 took 61 min in v1 -- much of that was likely data loading stalls. |

## Loss Function

| Parameter | v1 | v2 | Why |
|-----------|----|----|-----|
| `loss_type` | CE + Dice | **Focal + Dice** | Standard CE gives equal per-pixel attention. Focal Loss down-weights easy/well-classified pixels (background, no-damage) and focuses on hard examples (damage boundaries, rare classes). With gamma=2.0, a pixel classified at 0.9 confidence gets 100x less loss weight than one at 0.1 confidence. This is critical for imbalanced satellite imagery. |
| `focal_gamma` | (n/a) | **2.0** | gamma=2.0 is the original paper default (Lin et al., 2017). Provides strong focus on hard examples without being so aggressive that easy-class learning collapses entirely. |
| `focal_weight` | (ce_weight=1.0) | **1.0** | Kept at 1.0 -- the focal mechanism itself handles the rebalancing. |
| `dice_weight` | 1.0 | **1.5** | Increasing Dice weight emphasises region-overlap optimisation. Dice directly optimises for IoU-like behaviour, which is the evaluation metric (mIoU). Giving it more weight aligns training objective with evaluation. |
| `class_weight_samples` | 256 | **null (all)** | v1 computed class frequency weights from only 256 of 2,239 training samples -- a noisy 11% subsample. Using all samples gives accurate class weights. Slower to compute once at startup but much more representative. |
| `weight_dice_by_class` | (no) | **true** | v1's DiceLoss averaged over all 5 classes equally, meaning background (class 0, ~85% of pixels) contributed as much as destroyed (class 4, <1% of pixels). Weighting Dice by inverse class frequency forces the model to care about minority damage classes. |

## Data Augmentation

| Parameter | v1 | v2 | Why |
|-----------|----|----|-----|
| `aug_vflip` | 0.2 | **0.5** | Satellite imagery has no canonical "up". Vertical flip should be as likely as horizontal flip. No reason for the asymmetry in v1. |
| `aug_brightness` | 0.12 | **0.20** | Satellite images vary significantly in illumination across captures (time of day, atmospheric conditions). Stronger brightness jitter improves robustness. |
| `aug_contrast` | 0.12 | **0.20** | Same reasoning. Post-disaster images often have different contrast from pre-disaster (smoke, dust, different atmospheric conditions). |
| `aug_saturation` | 0.08 | **0.15** | Conservative increase. Too much saturation jitter can destroy color-based damage cues, but 0.08 was too timid. |
| `aug_hue` | (none) | **0.05** | Small hue shift was missing entirely. Helps with sensor variation between pre/post captures. Kept small (0.05) since damage classes may have color correlations. |
| `aug_blur_prob` | 0.10 | **0.15** | Slightly more aggressive blur simulation. Satellite imagery can have atmospheric blur, motion blur from sensor. |
| `aug_noise_prob` | 0.10 | **0.15** | Satellite sensors produce noise especially in shadow regions. More noise augmentation improves robustness. |
| `aug_coarse_dropout_prob` | (none) | **0.10** | CutOut/CoarseDropout randomly erases rectangular patches. Forces the model to not rely on any single spatial region and improves generalisation. Well-established regulariser for segmentation. |
| `aug_elastic_prob` | (none) | **0.15** | Elastic deformation simulates slight geometric distortions from orthorectification errors and building lean in oblique views. Common augmentation for building segmentation. |
| `aug_scale_range` | (none) | **[0.8, 1.2]** | Random scale simulates GSD variation. Buildings appear at different scales in different tiles. Forces multi-scale feature learning without changing the architecture. |

---

## Expected Impact

Conservative estimate based on each change category:

| Change Category | Expected mIoU Gain | Confidence |
|-----------------|--------------------| -----------|
| Training schedule (50 epochs + OneCycleLR + warmup) | +5-10% | High |
| Batch size (effective 16) + num_workers | +2-4% | High |
| Focal Loss + weighted Dice | +3-6% | Medium-High |
| Stronger augmentations | +2-4% | Medium |
| Weight decay + early stopping | +1-2% (via reduced overfitting) | Medium |

**Target: val mIoU 0.50-0.60** (up from 0.359 baseline)

These are not strictly additive -- gains overlap. But collectively, moving from 0.36 to 0.50+ mIoU is realistic with hyperparameter tuning alone.
