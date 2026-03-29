# Experiment Tracker -- xView2 Siamese ResUNet

## Experiment 1: Baseline (`siamese_resunet_xview2_1`)

**Status:** Complete (5 epochs)

| Config | Value |
|--------|-------|
| Backbone | ResNet34 (pretrained, unfrozen) |
| Image size | 512x512 |
| Batch size | 1 (effective 2 with accumulation) |
| Epochs | 5 |
| LR | 1e-4, CosineAnnealingLR |
| Loss | CE + Dice (equal weight) |
| Augmentations | Basic (hflip, vflip, rot90, color jitter, blur, noise) |

**Results:**

| Metric | Best Val (Epoch 2) |
|--------|-------------------|
| mIoU | 0.359 |
| Mean Dice | 0.443 |
| Pixel Accuracy | 0.944 |
| Val Loss | 1.428 |

**Observations:**
- Overfitting starts at epoch 3 (val mIoU drops while train mIoU rises)
- High pixel accuracy but low mIoU = model handles background well, struggles with damage classes
- Cosine LR decayed to near-zero by epoch 5, cutting training short
- Data loading bottleneck (num_workers=0, epoch 1 took 61 min)

---

## Experiment 2: Hyperparameter Tuning (`siamese_resunet_xview2_2`)

**Status:** Configured, not yet run

**Changes from Experiment 1** (architecture unchanged):
- Epochs: 5 -> 50 with early stopping (patience=10)
- Scheduler: CosineAnnealing -> OneCycleLR with 10% warmup
- LR: 1e-4 -> 3e-4 peak
- Batch size: 1 -> 2, accumulation: 2 -> 8 (effective 16)
- num_workers: 0 -> 4
- Loss: CE+Dice -> Focal(gamma=2)+weighted Dice, dice_weight 1.0->1.5
- Class weights: computed from all samples instead of 256
- Weight decay: 1e-4 -> 5e-4
- Augmentation: stronger jitter, added hue/elastic/coarse dropout/scale

**Target:** val mIoU 0.50-0.60

Full justification: see `siamese_resunet_xview2_2/changes.md`

---

## Experiment 3 (Proposed): Architecture -- Attention Fusion

**Requires:** Same GPU (8GB RTX 4060)

**Changes from Experiment 2:**
- Add Squeeze-and-Excitation (SE) channel attention to each FusionBlock
- SE reduces fused features to a channel descriptor via global avg pool, then learns per-channel importance weights through a small bottleneck FC layer
- Minimal VRAM overhead (~50MB), meaningful feature selection improvement
- The current fusion (concat + conv) treats all channels equally. SE lets the model learn that e.g. texture-change channels matter more than brightness-change channels

```python
class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = ConvBlock(channels * 3, channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, pre_feat, post_feat):
        diff = torch.abs(post_feat - pre_feat)
        x = torch.cat([pre_feat, post_feat, diff], dim=1)
        x = self.block(x)
        return x * self.se(x)
```

**Target:** +1-3% mIoU over Experiment 2

---

## Experiment 4 (Proposed): Architecture -- Auxiliary Building Localization

**Requires:** Same GPU (8GB RTX 4060)

**Changes from Experiment 3:**
- Add a secondary segmentation head that predicts building footprints from pre-image features
- The pre_mask (binary building map) is already loaded in the dataset but never used for supervision
- Acts as a multi-task regularizer: encoder must learn "where are buildings" (localization) in addition to "how damaged are they" (classification)
- Auxiliary loss weighted at 0.3x the main loss, decayed linearly to 0 over training

```python
# In SiameseResUNet.__init__:
self.loc_head = nn.Sequential(
    nn.Conv2d(dec[3], 16, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 2, 1),  # binary: building / not-building
)

# In forward: branch off before the damage head
loc_logits = self.loc_head(x)  # predict pre_mask
damage_logits = self.head(x)   # predict post_mask (existing)

# In loss:
aux_weight = 0.3 * (1 - epoch / total_epochs)  # linear decay
total_loss = damage_loss + aux_weight * F.cross_entropy(loc_logits, pre_mask)
```

**Target:** +2-4% mIoU over Experiment 3

---

## Experiment 5 (Proposed): Backbone Upgrade -- ResNet50

**Requires:** 12-16GB VRAM GPU (RTX 4070 Ti, RTX 3090, or similar)

**Changes from Experiment 4:**
- Backbone: ResNet34 -> ResNet50
- Encoder channels: [64,64,128,256,512] -> [64,256,512,1024,2048]
- Bottleneck blocks with 4x wider features at each stage
- Decoder channels: [256,128,64,32] -> [512,256,128,64] to match wider encoder
- batch_size may need to drop to 1-2 depending on available VRAM

**Why not on 8GB:**
FusionBlock at deepest level takes 2048x3 = 6144 input channels. Combined with decoder upsampling at 512x512, activation memory alone exceeds 8GB even at batch_size=1.

**Why it matters:**
ResNet50's Bottleneck blocks learn richer intermediate representations than ResNet34's BasicBlocks. For subtle damage classification (minor vs major damage), the additional feature capacity is critical. xView2 competition winners universally use ResNet50+ or EfficientNet-B4+ backbones.

**Target:** +3-5% mIoU over Experiment 4

---

## Experiment 6 (Proposed): Backbone Upgrade -- EfficientNet-B4

**Requires:** 12-16GB VRAM GPU

**Changes:**
- Replace ResNet encoder entirely with EfficientNet-B4 (pretrained on ImageNet)
- Compound scaling (depth + width + resolution) gives better features per FLOP
- Encoder channels: [24, 32, 56, 160, 448] -- narrower than ResNet50 but more efficient
- Add a new EfficientNetEncoder class using timm library
- MBConv blocks with built-in SE attention -- no need for separate SE in fusion

```python
import timm

class EfficientNetEncoder(nn.Module):
    def __init__(self, name="efficientnet_b4", pretrained=True):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, features_only=True)
        self.channels = self.model.feature_info.channels()
```

**Target:** +2-4% mIoU over ResNet50 variant, with similar or less VRAM

---

## Experiment 7 (Proposed): Advanced Training Techniques

**Requires:** 16GB+ VRAM recommended, works on 8GB with reduced batch

**Changes (can be applied to any of the above):**
- **Test-Time Augmentation (TTA):** Average predictions over original + hflip + vflip + rot90 at inference. Free +1-3% mIoU at the cost of 4-8x inference time.
- **Exponential Moving Average (EMA):** Maintain a shadow copy of model weights averaged over training. Usually gives a more stable, slightly better model. torch EMA with decay=0.999.
- **MixUp / CutMix:** Blend training samples and their labels. Strong regularizer that reduces overfitting and improves calibration.
- **Label Smoothing:** Smooth one-hot targets by epsilon=0.1. Prevents overconfident predictions and improves generalisation.
- **Stochastic Weight Averaging (SWA):** Average checkpoints from last N epochs. PyTorch has built-in `torch.optim.swa_utils`.

---

## Summary Roadmap

| Experiment | Key Change | GPU | Target mIoU |
|------------|-----------|-----|-------------|
| 1 (done) | Baseline | 8GB | 0.36 |
| 2 | Hyperparameter tuning | 8GB | 0.50-0.60 |
| 3 | + SE attention fusion | 8GB | 0.53-0.63 |
| 4 | + Auxiliary localization head | 8GB | 0.55-0.65 |
| 5 | + ResNet50 backbone | 12-16GB | 0.60-0.70 |
| 6 | + EfficientNet-B4 backbone | 12-16GB | 0.62-0.72 |
| 7 | + TTA, EMA, CutMix, SWA | any | +1-3% on top |

Note: xView2 competition top solutions achieved ~0.75-0.80 mIoU using ensemble models, larger backbones (ResNet101/EfficientNet-B7), multi-scale inference, and heavily tuned post-processing. The roadmap above targets single-model performance without ensembling.
