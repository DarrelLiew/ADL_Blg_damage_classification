# Experiment 4 -- Changes and Justification

Baseline: Experiment 3 (`siamese_resunet_xview2`, 42 epochs, best val mIoU 0.475, best val damage F1 0.510)

---

## 1. Backbone Upgrade: ResNet34 -> ResNet50 (CHANGED)

| Parameter | Exp 3 | Exp 4 | Why |
|-----------|-------|-------|-----|
| `backbone` | resnet34 | **resnet50** | ResNet34 (21M params) lacks feature capacity for subtle damage distinctions. Minor Damage F1 was only 0.28 in Exp 3 -- the encoder can't extract rich enough features to separate minor from major damage. ResNet50 (25M params) uses Bottleneck blocks with 4x wider intermediate representations, giving much richer features at each stage. Every xView2 competition winner used ResNet50+ or larger backbones. |
| Encoder channels | [64, 64, 128, 256, 512] | **[64, 256, 512, 1024, 2048]** | Bottleneck blocks output 4x more channels per stage. The deepest features go from 512 to 2048 channels, giving the fusion blocks far more information about texture and structural changes. |
| `decoder_channels` | [256, 128, 64, 32] | **[512, 256, 128, 64]** | Decoder must be widened to match the larger encoder. With the old [256,128,64,32] decoder, information from the 2048-channel bottleneck would be severely compressed. Wider decoder preserves spatial detail through upsampling. |

## 2. Data Loading: num_workers (CHANGED)

| Parameter | Exp 3 | Exp 4 | Why |
|-----------|-------|-------|-----|
| `num_workers` | 0 | **4** | Exp 1 showed a data loading bottleneck (epoch 1 took 61 min with num_workers=0). With a proper GPU setup, parallel data loading removes this bottleneck and keeps the GPU fed. |

## 3. Everything Else: Unchanged

All training strategy improvements from Exp 3 are retained:
- Two-phase training (15 epochs frozen, then unfreeze with encoder_lr=1e-5)
- Log-inverse class weights computed from all samples
- Label smoothing (0.05)
- Weight decay (0.05)
- Effective batch size 8 (2 x 4 accumulation)
- Early stopping (patience=10)
- Damage macro F1 as primary metric with per-class F1 tracking
- Cosine scheduler with min_lr=1e-6

---

## Expected Impact

| Change | Expected Gain (damage F1) | Confidence |
|--------|--------------------------|------------|
| ResNet50 backbone | +0.05-0.10 | High |
| Wider decoder | (enables backbone gain) | -- |
| num_workers=4 | (faster epochs, no accuracy change) | -- |

**Target: damage macro F1 >= 0.58** (up from 0.51 in Exp 3)

The main bottleneck in Exp 3 was Minor Damage F1 (0.28). ResNet50's richer feature representations should help the most with distinguishing these visually subtle damage levels.

---

## How to Run

Set `CONFIG_PATH` at the top of the notebook:
```python
CONFIG_PATH = "training_runs/siamese_resunet_xview2_3/config.json"
```

Note: If VRAM is tight, reduce `batch_size` to 1 and increase `accumulation_steps` to 8 to keep the same effective batch size.
