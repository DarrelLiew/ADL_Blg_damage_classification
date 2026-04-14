# Presentation: xView2 Building Damage Segmentation

**Duration:** ~12-15 minutes + demo
**Format:** Slides + live demo (or pre-recorded video backup)
**Narrative arc:** Problem -> Architecture -> "It didn't work" -> "We fixed it" -> "We pushed further" -> Results -> Impact

The presentation tells the story of iterative experimentation -- not just final results, but the journey of failures, insights, and systematic fixes that led to improvement. Each slide maps to a rubric criterion.

> **Note:** We ran an initial 5-epoch trial run (the original Experiment 1 in the codebase, folder `siamese_resunet_xview2_1/`) to verify the pipeline worked. Since it was too short to produce meaningful results, we omitted it from the report and presentation. Experiments are renumbered starting from our first full training run.

---

## Slide 1: Title

**xView2 Building Damage Segmentation: AI for Disaster Response**

- Team members and student IDs
- 60.001 Applied Deep Learning, Y2026
- GitHub link | Dataset: Kaggle xView2 Tier 3 | Model weights: Google Drive

---

## Slide 2: The Problem (AI for Social Good)

> *Rubric: Concept and Relevance [5%], Demonstration of impact*

**"When a disaster strikes, every hour matters."**

- Traditional ground surveys take days to weeks -- people die waiting
- Satellite imagery is available within hours, but manually inspecting thousands of buildings doesn't scale
- UN, Red Cross, FEMA have all identified automated satellite damage assessment as a priority capability
- **Our goal:** Given a satellite image pair (before and after disaster), produce a pixel-level damage map classifying every building into 4 damage levels

**Speaker notes:** Open with a compelling disaster image. Frame this as a real problem that organizations are actively trying to solve. This establishes "AI for Social Good" immediately.

*Visual: Full-bleed before/after satellite image pair from the dataset. One side shows an intact neighbourhood, the other shows destruction.*

---

## Slide 3: The Dataset -- xView2

> *Rubric: Experiments - clarity of specifications*

- **Source:** xView2 Challenge (Defense Innovation Unit), WorldView-3 satellite imagery
- ~2,200 image pairs (pre + post disaster), resized to 512x512
- 5 classes: Background, No Damage, Minor Damage, Major Damage, Destroyed
- **The core challenge:** Extreme class imbalance

| Class | Pixel % | Reality |
|-------|---------|---------|
| Background | ~97% | Non-building pixels |
| No Damage | ~1.5% | Intact buildings |
| Minor Damage | ~0.3% | Cracked walls, missing tiles |
| Major Damage | ~0.5% | Structural damage |
| Destroyed | ~0.7% | Collapsed/razed |

**"A model predicting 'background' everywhere gets 97% accuracy -- and is completely useless."**

**Speaker notes:** This sets up why naive metrics fail and why our later switch to damage F1 was critical. Spend ~30 seconds here.

*Visual: Sample image pair with coloured overlay mask (green/yellow/orange/red). Show the class distribution pie chart.*

---

## Slide 4: Architecture -- Siamese ResNet U-Net

> *Rubric: Method - clear presentation of architecture*

```
Pre-disaster image  --> [Shared ResNet Encoder] --+
                                                  |--> FusionBlocks --> U-Net Decoder --> 5-class Damage Map
Post-disaster image --> [Shared ResNet Encoder] --+
```

Four key design decisions:

1. **Shared encoder:** Weight-tied ResNet processes both images in the same feature space -- ensures consistent comparison
2. **Three-stream fusion:** FusionBlock = concat(pre, post, |pre - post|) -- the absolute difference explicitly encodes *what changed*
3. **U-Net decoder with skip connections:** Preserves spatial detail from encoder to output
4. **Pretrained ImageNet backbone:** Transfer learning bootstraps feature extraction; we don't train from scratch

**Speaker notes:** Keep this to ~90 seconds. The audience needs to understand the architecture enough to follow the experiment story. Don't get bogged down in layer details.

*Visual: Architecture diagram (draw.io/Figma). Colour-code the three streams in the fusion block.*

---

## Slide 5: The Journey Begins -- and Fails

> *Rubric: Hyper-parameter tuning and comparison of results*

**Experiment 1 (Baseline):** ResNet34, 5 epochs, basic setup
- Val mIoU: 0.359. Model learned to predict background well (94% pixel accuracy) but barely detected damage.

**Experiment 2 (More Epochs):** Same model, 35 epochs
- Val mIoU: 0.394 (+9.7%). Marginal gain, but...

**"The model started memorizing the training data instead of learning."**

*Visual:* `A1_exp2_overfitting.png` -- *Exp 2 train vs val loss showing dramatic divergence. The pink shaded gap between the lines is the visual punchline.*

**Speaker notes:** This is the "failure" beat. The audience sees the model failing. Train loss drops to 0.63 while val loss balloons to 2.38 -- a 3.8x gap. "Simply training longer wasn't the answer."

---

## Slide 6: The Breakthrough -- Training Strategy Overhaul

> *Rubric: Hyper-parameter tuning, comparison of results*

**Experiment 3: Zero architecture changes. Only training strategy.**

| What we changed | Before (Exp 2) | After (Exp 3) |
|-----------------|----------------|----------------|
| Weight decay | 1e-4 | **0.05** (500x increase) |
| Encoder training | All layers from epoch 1 | **Phase 1: frozen, Phase 2: unfrozen** |
| Class weights | Frequency inverse (256 samples) | **Log-inverse (all samples, clipped)** |
| Effective batch | 2 | **8** |
| Primary metric | mIoU | **Damage macro F1** |

**Result: +20.6% mIoU improvement. The single largest gain across all experiments.**

*Visual:* `A2_exp2v3_overfitting_fix.png` -- *Side-by-side: Exp 2 (diverging train/val loss) vs Exp 3 (parallel lines). The contrast is dramatic.*

**Speaker notes:** This is the "aha moment." Spend 2 minutes here. The message: "Training strategy matters more than architecture. A 500x weight decay increase did more than upgrading the entire backbone later."

---

## Slide 7: Scaling Up -- ResNet50 (A Mixed Result)

> *Rubric: Comparison of results, demonstrating difficulty*

**Experiment 4:** Upgraded ResNet34 (21M params) to ResNet50 (25M params, 4x wider bottleneck layers)

**Overall:** damage F1 0.510 -> 0.521. Modest.

**But the per-class story is dramatic:**

*Visual:* `A3_exp3v4_perclass.png` -- *Grouped bars showing Minor Damage jumping +53% while Major Damage drops -17%.*

- **Minor Damage F1: 0.28 -> 0.43 (+53%)** -- richer features helped with subtle cracks/tiles
- **Major Damage F1: 0.46 -> 0.38 (-17%)** -- the backbone wasn't adapting; encoder_lr was too conservative (1e-5) with only 2 layers unfrozen

**"A bigger model isn't automatically a better model -- you have to let it learn."**

**Speaker notes:** This sets up the next experiment. The audience understands why we need Exp 5.

---

## Slide 8: Fixing the Backbone -- Focal Loss + Aggressive Fine-Tuning

> *Rubric: Hyper-parameter tuning, demonstrating difficulty*

**Experiment 5:** Let ResNet50 actually adapt to satellite imagery

| Parameter | Exp 4 | Exp 5 | Why |
|-----------|-------|-------|-----|
| Encoder LR | 1e-5 | **5e-5** (5x) | Faster adaptation |
| Unfrozen layers | 2 | **4** | Deeper adaptation |
| Loss | CE + Dice | **Focal + Dice** | Focus on hard boundary pixels |

*Visual:* `A4_exp4v5_phase2_adaptation.png` -- *Exp 4 vs Exp 5 damage F1 over Phase 2 epochs, showing Exp 5 reaching higher despite spikes.*

**New best: Damage F1 = 0.546** (at epoch 49 of 60)

**Speaker notes:** Point out that the best epoch (49) is well inside the Phase 2 fine-tuning window. The validation curve is noisy because aggressive encoder fine-tuning plus Focal Loss concentrate the gradient on rare-class pixels; the underlying trend is upward.

---

## Slide 9: The Rare-Class Challenge

> *Rubric: Performance & Evaluation, appropriate metrics*

**"Pixel accuracy was 97%. It was a lie."**

Switching to Damage Macro F1 revealed Minor Damage (F1=0.28) as the real bottleneck. Across 3 experiments, we improved it by 65%:

*Visual:* `B2_rare_class_challenge.png` -- *Grouped bars for Exp 3/4/5 with competition winner range shaded for Minor Damage. Arrow showing 0.28 -> 0.47 trajectory.*

- Minor Damage: 0.28 -> 0.43 -> **0.47** (now exceeds competition winner range of 0.30-0.45)
- Major Damage: 0.46 -> 0.38 (regressed with ResNet50) -> **0.44** (recovered with Focal Loss)

*Visual:* `B3_encoder_adaptation_arc.png` -- *Line chart showing the regression-then-recovery arc across experiments. Annotated: "Regressed (too conservative)" and "Recovered (5x encoder_lr + Focal)".*

**Speaker notes:** This slide demonstrates we understand *why* our metrics behave the way they do, not just what the numbers are.

---

## Slide 10: What Made the Biggest Difference

> *Rubric: Hyper-parameter tuning, comparison of results to baseline*

*Visual:* `B1_miou_waterfall.png` -- *Waterfall chart showing cumulative mIoU gain. Exp 3 bar dominates (~56% of total gain). TOTAL: 0.503.*

**Key insight: Training strategy accounted for the majority of total improvement. Architecture upgrades were secondary.**

| Rank | Change | mIoU gain |
|------|--------|-----------|
| 1 | Weight decay 0.05 + two-phase training (Exp 3) | **+0.081** |
| 2 | More epochs (Exp 2) | +0.035 |
| 3 | Focal Loss + tuned fine-tuning (Exp 5) | +0.015 |
| 4 | ResNet50 backbone (Exp 4) | +0.013 |

**Speaker notes:** This is a strong slide for the rubric's "comparison of results" requirement. The waterfall makes the contribution of each experiment visually unambiguous.

---

## Slide 11: Comparison to State-of-the-Art

> *Rubric: Comparison of results to some baseline/reference*

*Visual:* `B4_sota_comparison.png` -- *Horizontal bars with gap explanation box.*

| Model | Damage F1 | Data | Models |
|-------|-----------|------|--------|
| 1st Place (vdurnov) | ~0.75 | Full xBD (22K pairs) | 12-model ensemble |
| 2nd Place (selimsef) | ~0.72 | Full xBD | 2-model ensemble |
| Single-model baseline | ~0.625 | Full xBD | 1 model |
| **Ours (Exp 5)** | **0.546** | Tier 3 (2.2K pairs) | 1 model |

**The gap is explained by resource constraints, not architectural limitations:**
1. 10x less training data (Tier 3 vs full xBD)
2. No localization pretraining (winners used 2-stage training)
3. Single model, no ensemble or test-time augmentation

**Speaker notes:** Frame this honestly. The rubric says "you are not expected to beat SOTA." Show you understand *why* the gap exists.

---

## Slide 12: Failure Cases and Limitations

> *Rubric: "Show examples of your model malfunctioning if any and discuss what might be the reason"*

Show 3-4 side-by-side predictions (ground truth vs model output):

1. **Minor vs Major confusion** -- damage boundaries are subjective, even human annotators disagree
2. **Small buildings** -- buildings occupying very few pixels at 512x512 are near the classification limit
3. **Underrepresented disaster types** -- performance varies by event; some disaster types have fewer training samples
4. **Phase 2 instability** -- validation spikes during aggressive fine-tuning (epochs 20, 23, 25 in Exp 5)

**Honest assessment:** The model is useful for rapid coarse assessment (minutes vs days), but would need higher resolution imagery and more training data for operational deployment.

*Visual: 2x2 grid of failure cases. Each shows post-disaster image, ground truth mask, predicted mask.*

---

## Slide 13: Demo

> *Rubric: "A small demo, along with some slides or a small video would be appreciated"*

**Live demonstration** (pre-record a backup video):

1. Load trained model from `best_model.pt` checkpoint
2. Feed in a test image pair the model has never seen
3. Show predicted damage map overlaid on the post-disaster image
4. Compare side-by-side with ground truth

```python
model = SiameseResUNet(backbone="resnet50", num_classes=5, decoder_channels=[512, 256, 128, 64])
checkpoint = torch.load("best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
pred = model(pre_image, post_image).argmax(dim=1)
visualize(post_image, pred, ground_truth)
```

**Speaker notes:** Show 2-3 examples: one good prediction, one mediocre, one failure. This demonstrates honesty and understanding. The coloured damage maps on satellite imagery are visually striking.

---

## Slide 14: Key Takeaways

1. **Training strategy > architecture:** 500x weight decay increase contributed more than upgrading ResNet34 -> ResNet50
2. **Honest metrics reveal real problems:** Pixel accuracy (97%) was meaningless; Damage macro F1 (0.53) told the true story
3. **Systematic experimentation:** 5 experiments, each isolating variables, with clear ablation analysis and per-class tracking
4. **Bigger models need tuning, not just plugging in:** ResNet50 regressed on Major Damage until we fixed the fine-tuning strategy
5. **AI for Social Good:** Automated damage assessment can reduce response time from days to minutes

---

## Slide 15: Future Work

- Complete Exp 5 training (26 remaining epochs) when GPU compute becomes available -- model was still improving
- Test-time augmentation (free +1-3% with no retraining)
- Auxiliary building localization head (multi-task learning, as used by competition winners)
- Scale to full xBD dataset (~22K pairs vs our 2.2K)

---

## Slide 16: Thank You / Q&A

- **GitHub:** [repository link]
- **Dataset:** [Kaggle xView2 Tier 3](https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-tier-3-data)
- **Model weights:** [Google Drive link]
- **Reproducibility:** All configs, metrics CSVs, and figure generation scripts included in repo

---

## Figure-to-Slide Mapping

| Slide | Figure file | What it shows |
|-------|-------------|---------------|
| 5 | `A1_exp2_overfitting.png` | Exp 2 train vs val loss diverging |
| 6 | `A2_exp2v3_overfitting_fix.png` | Side-by-side before/after overfitting fix |
| 7 | `A3_exp3v4_perclass.png` | Per-class F1: ResNet50 helped Minor, hurt Major |
| 8 | `A4_exp4v5_phase2_adaptation.png` | Exp 4 vs 5 Phase 2 damage F1 |
| 9 | `B2_rare_class_challenge.png` | Per-class F1 trajectory + competition range |
| 9 | `B3_encoder_adaptation_arc.png` | Regression-then-recovery arc |
| 10 | `B1_miou_waterfall.png` | Cumulative contribution waterfall |
| 11 | `B4_sota_comparison.png` | SOTA horizontal bar comparison |

---

## Presentation Strategy

### Narrative Arc
The presentation follows a **failure -> insight -> fix -> result** loop for each experiment transition. This is more compelling than just presenting final numbers.

1. **"It didn't work"** (Slides 5) -- baseline failed, more training just overfitted
2. **"We figured out why"** (Slide 6) -- overfitting diagnosis, weight decay as the fix
3. **"Bigger model, mixed results"** (Slide 7) -- ResNet50 helped some classes, hurt others
4. **"We fixed that too"** (Slide 8) -- aggressive fine-tuning + Focal Loss, but GPU ran out
5. **"Here's the big picture"** (Slides 9-11) -- trends, SOTA comparison, honest assessment

### Time Allocation (~13 minutes)
| Section | Slides | Time |
|---------|--------|------|
| Problem + dataset | 2-3 | 2 min |
| Architecture | 4 | 1.5 min |
| Experiments story | 5-8 | 5 min |
| Analysis + SOTA | 9-11 | 2.5 min |
| Failures + demo | 12-13 | 3 min |
| Takeaways + Q&A | 14-16 | 2 min |

### Rubric Coverage Checklist
- [x] **Concept & Relevance [5%]:** Slide 2 (AI for Social Good framing)
- [x] **Coding [25%]:** GitHub link, documented notebook, reproducibility in Slide 16
- [x] **Performance & Evaluation [20%]:** Slides 5-11 (F1, mIoU, per-class, SOTA comparison, waterfall)
- [x] **Presentation & Communication [20%]:** Narrative arc, visual-heavy slides, figure-per-slide
- [x] **Creativity bonus [5%]:** Three-stream fusion, two-phase training, systematic ablation approach

### Anticipate Questions
- **"Why not use the full dataset?"** -- Tier 3 is what's publicly available on Kaggle; full xBD requires original competition registration
- **"Why not ensemble?"** -- Single-model focus for practical deployment and cleaner ablation. Also compute-limited.
- **"Why is pixel accuracy so high but F1 so low?"** -- 97% of pixels are background; predicting all-background gives 97% accuracy. This is why we switched to damage macro F1.
- **"What would you do with more compute?"** -- Complete Exp 5 (26 epochs left), then try TTA and localization pretraining
- **"Why did ResNet50 make Major Damage worse?"** -- encoder_lr was too conservative (1e-5), only 2 layers unfrozen. The backbone stayed near ImageNet features instead of adapting to satellite imagery. Fixed in Exp 5.
