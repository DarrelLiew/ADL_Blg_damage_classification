# xView2 Building Damage Segmentation

Semantic segmentation of building damage from pre- and post-disaster satellite imagery using a Siamese ResNet U-Net architecture, trained on the [xView2](https://xview2.org/) dataset.

## Architecture

The model uses a **Siamese encoder** -- a shared ResNet backbone processes both the pre-disaster and post-disaster images. Features from both branches are combined at each encoder level via **FusionBlocks** (concatenation of pre, post, and absolute difference features, followed by convolution). The fused features are decoded through a U-Net-style decoder to produce a 5-class segmentation mask:

| Class | Label        |
| ----- | ------------ |
| 0     | Background   |
| 1     | No Damage    |
| 2     | Minor Damage |
| 3     | Major Damage |
| 4     | Destroyed    |

## Dataset

The xView2 dataset provides ~2,200 pairs of 1024x1024 satellite images (pre/post disaster) with polygon annotations for building damage. Images are resized to 512x512 for training. The dataset is split 80/10/10 for train/val/test.

The dataset is not included in this repository. To set up:

1. Download the xView2 dataset from [Kaggle](https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-tier-3-data)
2. Place it in the following structure:

```
data/
  train/
    train/
      images/       # pre and post disaster PNGs
      labels/       # JSON annotation files
  test/
    test/
      images/
      labels/
```

3. Update `data_root` in your config JSON if using a different path.

## Project Structure

```
.
├── build_process_img_notebook.py   # Script that generates the training notebook
├── process_img.ipynb               # Generated training notebook (all-in-one)
├── experiment_log.md               # Detailed experiment comparison and analysis
├── training_runs/                  # Experiment configs, checkpoints, and metrics
│   ├── experiments.md              # Experiment roadmap
│   ├── siamese_resunet_xview2_1/   # Exp 1: Baseline
│   ├── siamese_resunet_xview2_2/   # Exp 2: Extended training
│   ├── siamese_resunet_xview2/     # Exp 3: Training strategy overhaul
│   └── siamese_resunet_xview2_3/   # Exp 4: ResNet50 backbone (configured)
├── verification_outputs/           # Mask overlay visualizations
└── verify_pre_mask_overlay.py      # Script to verify mask generation
```

## Experiments

| Experiment              | Key Change                                         | Val mIoU | Val Dice | Val Damage F1 |
| ----------------------- | -------------------------------------------------- | -------- | -------- | ------------- |
| 1 - Baseline            | ResNet34, 5 epochs                                 | 0.359    | 0.443    | --            |
| 2 - Extended training   | 35 epochs (same config)                            | 0.394    | 0.481    | --            |
| 3 - Training overhaul   | Two-phase freeze, weight decay 0.05, class weights | 0.475    | 0.603    | 0.510         |
| 4 - ResNet50 (proposed) | Backbone upgrade, wider decoder                    | --       | --       | --            |

See [experiment_log.md](experiment_log.md) for full analysis of what changed and why.

## Usage

### Running an experiment

1. Open `process_img.ipynb`
2. Set `CONFIG_PATH` in the first code cell to point to an experiment config:
   ```python
   CONFIG_PATH = "training_runs/siamese_resunet_xview2_3/config.json"
   ```
3. Run all cells

### Regenerating the notebook

The notebook is generated from `build_process_img_notebook.py`:

```bash
python build_process_img_notebook.py
```

## Requirements

- Python 3.11+
- PyTorch (with CUDA for GPU training)
- torchvision
- OpenCV (`cv2`)
- scikit-image
- scikit-learn
- shapely
- matplotlib
- numpy
