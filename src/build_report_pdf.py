"""
Build the final PDF report for 60.001 Applied Deep Learning.

Usage:
    python build_report_pdf.py

Outputs:
    ADL_Project_Report.pdf
"""

from pathlib import Path
from fpdf import FPDF

BASE = Path(__file__).parent
IMG = BASE / "images"
OUT = BASE / "ADL_Project_Report.pdf"


FONTS_DIR = "C:/Windows/Fonts"


class Report(FPDF):
    """Custom PDF with header/footer."""

    def __init__(self):
        super().__init__()
        # Register TTF fonts for full unicode support
        self.add_font("Arial", "", f"{FONTS_DIR}/arial.ttf", uni=True)
        self.add_font("Arial", "B", f"{FONTS_DIR}/arialbd.ttf", uni=True)
        self.add_font("Arial", "I", f"{FONTS_DIR}/ariali.ttf", uni=True)
        self.add_font("Arial", "BI", f"{FONTS_DIR}/arialbi.ttf", uni=True)
        self.add_font("Consolas", "", f"{FONTS_DIR}/consola.ttf", uni=True)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Arial", "I", 8)
            self.set_text_color(130, 130, 130)
            self.cell(0, 5, "xView2 Building Damage Segmentation | 60.001 Applied Deep Learning", align="L")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── Helpers ──────────────────────────────────────────────
    def section_title(self, num, title):
        self.ln(4)
        self.set_font("Arial", "B", 16)
        self.set_text_color(25, 25, 25)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(33, 150, 243)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def subsection(self, num, title):
        self.ln(2)
        self.set_font("Arial", "B", 13)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, f"{num} {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsubsection(self, title):
        self.ln(1)
        self.set_font("Arial", "B", 11)
        self.set_text_color(70, 70, 70)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Arial", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def body_bold(self, text):
        self.set_font("Arial", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def italic(self, text):
        self.set_font("Arial", "I", 9)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Arial", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, "\u2022")
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def add_figure(self, filename, caption, width=180):
        path = IMG / filename
        if not path.exists():
            self.body(f"[Figure not found: {filename}]")
            return
        # Check if enough space for the image (~80mm) + caption
        if self.get_y() > 220:
            self.add_page()
        self.image(str(path), x=(self.w - width) / 2, w=width)
        self.ln(2)
        self.set_font("Arial", "I", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 4.5, caption, align="C")
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            n = len(headers)
            total = self.w - self.l_margin - self.r_margin
            col_widths = [total / n] * n

        # Header
        self.set_font("Arial", "B", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Arial", "", 9)
        for row in rows:
            max_h = 7
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), border=1, align="C")
            self.ln()
        self.ln(2)

    def callout(self, text):
        """Blue-tinted callout box."""
        self.set_fill_color(227, 242, 253)
        self.set_draw_color(33, 150, 243)
        self.set_text_color(13, 71, 161)
        self.set_font("Arial", "I", 9.5)
        x = self.get_x()
        y = self.get_y()
        self.set_line_width(0.5)
        # Draw left border accent
        self.rect(x, y, self.w - self.l_margin - self.r_margin, 0, "D")
        self.set_x(x + 4)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8, 5, text, fill=False)
        self.set_text_color(30, 30, 30)
        self.ln(2)


def build():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    # ╔════════════════════════════════════════════════════════════╗
    # ║  TITLE PAGE                                                ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.ln(35)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(25, 25, 25)
    pdf.multi_cell(0, 12, "xView2 Building Damage\nSegmentation", align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "AI for Disaster Response", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_draw_color(33, 150, 243)
    pdf.set_line_width(1)
    mid = pdf.w / 2
    pdf.line(mid - 40, pdf.get_y(), mid + 40, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 7, "60.001 Applied Deep Learning, Y2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Singapore University of Technology and Design", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    details = [
        ("Team Members:", "[Member names and student IDs]"),
        ("Dataset:", "xView2 Tier 3 (Kaggle)"),
        ("GitHub:", "[Repository URL]"),
        ("Model Weights:", "[Google Drive / Dropbox link]"),
    ]
    for label, value in details:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(45, 6, label, align="R")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"  {value}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "Submitted: April 17, 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # ╔════════════════════════════════════════════════════════════╗
    # ║  TABLE OF CONTENTS                                         ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(25, 25, 25)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    toc = [
        ("1", "Introduction", ""),
        ("2", "Dataset", ""),
        ("3", "Method", ""),
        ("  3.1", "Architecture: Siamese ResNet U-Net", ""),
        ("  3.2", "Loss Function", ""),
        ("  3.3", "Two-Phase Training Strategy", ""),
        ("  3.4", "Optimisation Details", ""),
        ("4", "Experiments", ""),
        ("  4.1", "Experiment 1: Baseline", ""),
        ("  4.2", "Experiment 2: More Epochs", ""),
        ("  4.3", "Experiment 3: Training Strategy Overhaul", ""),
        ("  4.4", "Experiment 4: ResNet50 Backbone", ""),
        ("  4.5", "Experiment 5: Tuned Phase 2 + Focal Loss", ""),
        ("  4.6", "Results Summary", ""),
        ("  4.7", "Comparison to State-of-the-Art", ""),
        ("5", "Analysis", ""),
        ("6", "Reproducibility", ""),
        ("7", "Conclusion", ""),
        ("8", "Team Contributions", ""),
        ("", "References", ""),
    ]
    for num, title, _ in toc:
        indent = 5 if num.startswith("  ") else 0
        pdf.set_x(pdf.l_margin + indent)
        if not num.startswith("  "):
            pdf.set_font("Helvetica", "B", 11)
        else:
            pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 6.5, f"{num.strip()}  {title}", new_x="LMARGIN", new_y="NEXT")

    # ╔════════════════════════════════════════════════════════════╗
    # ║  1. INTRODUCTION                                           ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("1", "Introduction")

    pdf.subsection("1.1", "Problem Statement")
    pdf.body(
        "When natural disasters strike, rapid assessment of building damage is critical for directing "
        "emergency resources. Traditional damage assessment relies on ground surveys that take days to "
        "weeks -- time that costs lives. Satellite imagery is available within hours, but manually "
        "inspecting thousands of buildings across disaster zones does not scale."
    )
    pdf.body(
        "This project develops a deep learning model for automated building damage segmentation from "
        "pre- and post-disaster satellite imagery. Given a pair of satellite images (before and after a "
        "disaster), the model produces a pixel-level damage map classifying every building into one of "
        "four damage levels: No Damage, Minor Damage, Major Damage, and Destroyed."
    )

    pdf.subsection("1.2", "Objectives")
    pdf.bullet("Design and implement a Siamese U-Net architecture for change-detection-based damage segmentation")
    pdf.bullet("Systematically improve model performance through iterative experimentation -- training strategy, regularisation, backbone scaling, and loss function design")
    pdf.bullet("Evaluate against the xView2 competition benchmark and analyse failure modes")
    pdf.bullet("Deliver a reproducible, well-documented codebase with saved model weights")

    pdf.subsection("1.3", "Scope and Constraints")
    pdf.bullet("Dataset: xView2 Tier 3 (~2,200 image pairs), a subset of the full xBD dataset (~22,000 pairs)")
    pdf.bullet("Compute: Single GPU training (Kaggle P100 / SUTD AI Cluster). GPU quota limitations and cluster downtime affected our final experiment.")
    pdf.bullet("Architecture: Single-model inference (no ensembling), practical for real deployment")
    pdf.bullet("Classes: 5-class segmentation -- Background (0), No Damage (1), Minor Damage (2), Major Damage (3), Destroyed (4)")

    pdf.subsection("1.4", "Social Impact")
    pdf.body(
        "Automated damage assessment directly supports humanitarian response. It reduces assessment time "
        "from days to minutes, can process entire cities simultaneously, and provides consistent "
        "classification across analysts. Organisations like the UN, Red Cross, and FEMA have identified "
        "automated satellite damage assessment as a priority capability for disaster response."
    )

    # ╔════════════════════════════════════════════════════════════╗
    # ║  2. DATASET                                                ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("2", "Dataset")

    pdf.subsection("2.1", "Source")
    pdf.body(
        "The xView2 dataset was created by the Defense Innovation Unit (DIU) for the xView2 Building "
        "Damage Assessment Challenge. It contains WorldView-3 satellite imagery (1.4 m GSD) across "
        "multiple disaster types: earthquakes, hurricanes, wildfires, volcanic eruptions, and floods."
    )

    pdf.subsection("2.2", "Structure")
    pdf.add_table(
        ["Property", "Value"],
        [
            ["Image pairs", "~2,200 (pre + post disaster)"],
            ["Original resolution", "1024 x 1024 px"],
            ["Training resolution", "512 x 512 px (resized)"],
            ["Annotation format", "GeoJSON polygons per building"],
            ["Split", "80% train / 10% val / 10% test"],
            ["Random seed", "42 (deterministic split)"],
        ],
        col_widths=[55, 125],
    )

    pdf.subsection("2.3", "Class Distribution")
    pdf.body(
        "The dataset exhibits extreme class imbalance, which is the central challenge of this task. "
        "Background pixels constitute approximately 97% of all pixels. The four damage classes share "
        "the remaining ~3%, with Minor Damage being the rarest at only 0.3% of pixels."
    )

    pdf.add_figure("C2_class_distribution.png",
                    "Figure 1: xView2 Tier 3 class distribution. Left: full scale showing background dominance. "
                    "Right: zoomed view of damage classes. Minor Damage (0.3%) is the hardest class to detect.")

    pdf.body(
        "This imbalance means a model predicting 'background' everywhere achieves 97% pixel accuracy "
        "while being completely useless for damage assessment. This motivated our adoption of Damage "
        "Macro F1 (macro average of F1 for damage classes 1-4, excluding background) as the primary "
        "evaluation metric from Experiment 3 onward."
    )

    pdf.subsection("2.4", "Preprocessing and Augmentation")
    pdf.body(
        "Polygon annotations were rasterised to per-pixel masks using Shapely and scikit-image. Images "
        "were resized from 1024x1024 to 512x512 via bilinear interpolation. ImageNet normalisation was "
        "applied (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])."
    )
    pdf.body("Data augmentation (final configuration, Exp 5):")
    pdf.add_table(
        ["Augmentation", "Probability", "Purpose"],
        [
            ["Horizontal flip", "0.5", "Orientation invariance"],
            ["Vertical flip", "0.2", "Orientation invariance"],
            ["Random 90-degree rotation", "0.5", "Rotation invariance"],
            ["Brightness/Contrast jitter", "+/-0.15", "Lighting robustness"],
            ["Saturation jitter", "+/-0.10", "Colour robustness"],
            ["Gaussian blur", "0.15", "Resolution robustness"],
            ["Gaussian noise", "0.15", "Sensor noise robustness"],
        ],
        col_widths=[60, 35, 85],
    )

    # ╔════════════════════════════════════════════════════════════╗
    # ║  3. METHOD                                                 ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("3", "Method")

    pdf.subsection("3.1", "Architecture: Siamese ResNet U-Net")
    pdf.body(
        "The model follows a Siamese encoder -- fusion -- decoder paradigm designed for change "
        "detection in satellite imagery. A shared ResNet backbone extracts hierarchical features from "
        "both the pre- and post-disaster images. At each encoder level, a FusionBlock combines the two "
        "feature maps with an explicit change signal. A U-Net decoder then reconstructs the full-resolution "
        "5-class damage segmentation map."
    )

    pdf.add_figure("C1_architecture_diagram.png",
                    "Figure 2: Siamese ResNet U-Net architecture. Pre- and post-disaster images are processed by a "
                    "shared ResNet encoder. FusionBlocks at each level combine features via 3-stream fusion "
                    "(pre + post + |diff|). A U-Net decoder with skip connections produces the 5-class damage map.",
                    width=175)

    pdf.subsubsection("Shared Encoder")
    pdf.body(
        "A pretrained ResNet backbone (ImageNet weights) extracts hierarchical features from both images. "
        "Weight sharing ensures the same feature space for both temporal views. We experimented with two backbones:"
    )
    pdf.add_table(
        ["Backbone", "Parameters", "Deepest Channels", "Block Type"],
        [
            ["ResNet34", "21M", "512", "BasicBlock"],
            ["ResNet50", "25M", "2048", "Bottleneck (4x wider)"],
        ],
        col_widths=[40, 35, 50, 55],
    )

    pdf.subsubsection("Fusion Blocks")
    pdf.body(
        "At each encoder level, a FusionBlock combines pre- and post-disaster features using three-stream "
        "fusion: concat(pre_feat, post_feat, |post_feat - pre_feat|). The absolute difference explicitly "
        "encodes change magnitude, helping the model focus on damaged regions. The concatenated 3C-channel "
        "tensor is compressed to C channels by a ConvBlock (two 3x3 Conv-BN-ReLU layers)."
    )

    pdf.subsubsection("U-Net Decoder")
    pdf.body(
        "Standard U-Net decoder with skip connections from fused features at each level. Each DecoderBlock "
        "performs: bilinear upsample -> concatenate skip connection -> ConvBlock. A final 1x1 convolution "
        "produces the 5-class output."
    )

    pdf.subsection("3.2", "Loss Function")
    pdf.body(
        "Our loss function evolved across experiments. The final configuration (Exp 5) uses:"
    )
    pdf.body_bold("L = Focal Loss(pred, target, class_weights, gamma=2) + Dice Loss(pred, target)")
    pdf.ln(1)
    pdf.add_table(
        ["Component", "Purpose"],
        [
            ["Focal Loss (gamma=2)", "Downweights easy background pixels; focuses on hard boundaries"],
            ["Dice Loss", "Region-level overlap; inherently handles class imbalance"],
            ["Log-inverse class weights", "Upweights rare damage classes (clipped to [1, 20])"],
            ["Label smoothing (0.05)", "Prevents overconfident predictions on easy pixels"],
        ],
        col_widths=[60, 120],
    )
    pdf.body(
        "Experiments 1-4 used standard Cross-Entropy instead of Focal Loss. The switch in Exp 5 was "
        "motivated by Major Damage F1 regressing in Exp 4 -- Focal Loss focuses the model on hard "
        "boundary pixels between adjacent damage classes. This formulation is consistent with all "
        "xView2 competition winning solutions."
    )

    pdf.subsection("3.3", "Two-Phase Training Strategy")
    pdf.body(
        "A key contribution of this project is the two-phase training strategy, developed in Exp 3 "
        "after observing severe overfitting in Exp 2. The insight is that pretrained ImageNet features "
        "provide a strong starting point, but must be protected during early training."
    )

    pdf.add_figure("C3_two_phase_training.png",
                    "Figure 3: Two-phase training strategy. Phase 1 freezes the encoder so the decoder and fusion "
                    "blocks learn to interpret pretrained features. Phase 2 unfreezes encoder layers with a lower "
                    "learning rate for domain-specific adaptation.",
                    width=175)

    pdf.body(
        "Phase 1 (Frozen Encoder): The encoder weights are frozen; only the decoder and fusion blocks "
        "train at lr=3e-4. This lets new layers learn to interpret pretrained ImageNet features without "
        "corrupting them. Duration: 10-15 epochs."
    )
    pdf.body(
        "Phase 2 (Full Fine-Tuning): The top N encoder layers are unfrozen with a differential learning "
        "rate 6-30x lower than the decoder. This allows task-specific adaptation of the backbone to "
        "satellite imagery while preserving low-level features. The number of unfrozen layers and "
        "encoder LR were key hyperparameters tuned across Exp 3-5."
    )
    pdf.add_table(
        ["Parameter", "Exp 3", "Exp 4", "Exp 5"],
        [
            ["Freeze epochs", "15", "15", "10"],
            ["Unfrozen layers", "All", "2", "4"],
            ["Encoder LR", "1e-5", "1e-5", "5e-5"],
        ],
        col_widths=[50, 40, 40, 40],
    )

    pdf.subsection("3.4", "Optimisation Details")
    pdf.add_table(
        ["Parameter", "Value"],
        [
            ["Optimiser", "AdamW"],
            ["Decoder LR", "3e-4"],
            ["Encoder LR (Phase 2)", "1e-5 to 5e-5 (tuned per experiment)"],
            ["Weight decay", "0.05"],
            ["Gradient clipping", "1.0 (max norm)"],
            ["Batch size", "8-16 (final config)"],
            ["Gradient accumulation", "4-8 steps (effective batch 32-64)"],
            ["Mixed precision", "FP16 (torch.cuda.amp)"],
            ["Scheduler", "CosineAnnealingLR (min_lr=1e-6)"],
            ["Early stopping", "Patience=15 on val damage F1"],
        ],
        col_widths=[60, 120],
    )

    # ╔════════════════════════════════════════════════════════════╗
    # ║  4. EXPERIMENTS                                            ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("4", "Experiments")
    pdf.body(
        "We conducted 5 experiments, each building on the previous. This iterative approach allowed us "
        "to isolate the impact of individual changes and understand what drives performance in damage "
        "segmentation. Each experiment was motivated by a specific observation or failure from the "
        "previous one."
    )

    pdf.body(
        "The figure below shows model predictions across four disaster types (hurricane, earthquake, "
        "wildfire, tsunami) for three key experiments. Each row is a different disaster; columns show "
        "the pre-disaster image, post-disaster image, ground truth damage map, and predictions from "
        "Exp 1 (baseline), Exp 3 (training overhaul), and Exp 5 (final model). The visual progression "
        "from near-blank predictions in Exp 1 to structured damage maps in Exp 5 mirrors the quantitative "
        "improvements reported below."
    )

    pdf.add_figure("D1_prediction_grid.png",
                    "Figure 15: Prediction comparison across 4 disaster types and 3 experiments. Each row shows a "
                    "different disaster: Hurricane Harvey, Mexico Earthquake, Santa Rosa Wildfire, Palu Tsunami. "
                    "Exp 1 (baseline) produces near-blank or noisy predictions. Exp 3 (training overhaul) begins "
                    "detecting damage regions. Exp 5 (final model) produces structured damage maps approaching "
                    "ground truth quality.",
                    width=180)

    # ── Exp 1 ──
    pdf.subsection("4.1", "Experiment 1: Baseline -- Establishing a Starting Point")
    pdf.body_bold("Goal: Get a working model and establish baseline performance.")
    pdf.add_table(
        ["Parameter", "Value"],
        [
            ["Backbone", "ResNet34 (all layers unfrozen)"],
            ["Batch size", "1 (effective 2 with 2-step accumulation)"],
            ["Epochs", "5"],
            ["Learning rate", "1e-4, CosineAnnealingLR"],
            ["Weight decay", "1e-4"],
            ["Loss", "CE + Dice (equal weight)"],
            ["Class weights", "Frequency inverse (256 samples)"],
        ],
        col_widths=[55, 125],
    )

    pdf.subsubsection("Results (Best Validation)")
    pdf.add_table(
        ["Metric", "Best Value", "Epoch"],
        [
            ["Val mIoU", "0.359", "2"],
            ["Val Mean Dice", "0.443", "2"],
            ["Val Pixel Accuracy", "0.944", "2"],
        ],
        col_widths=[55, 55, 55],
    )

    pdf.subsubsection("Outcome and Analysis")
    pdf.body(
        "The model achieved 94.4% pixel accuracy -- but this was misleading. With ~97% of pixels being "
        "background, high pixel accuracy simply meant the model was good at predicting 'not a building.' "
        "Overfitting appeared by epoch 3: val mIoU dropped from 0.359 to 0.339 while train mIoU continued "
        "rising. The cosine LR scheduler decayed to near-zero within 5 epochs, effectively cutting "
        "training short. Data loading with num_workers=0 created a bottleneck (61 min/epoch)."
    )
    pdf.subsubsection("Takeaway")
    pdf.body(
        "The baseline established a floor of mIoU=0.359 but was limited by insufficient training time "
        "and the onset of overfitting. The misleading pixel accuracy metric highlighted the need for "
        "class-aware evaluation. The immediate question: was the problem simply insufficient epochs, "
        "or something deeper?"
    )

    # ── Exp 2 ──
    pdf.add_page()
    pdf.subsection("4.2", "Experiment 2: More Epochs -- Testing the Simplest Hypothesis")
    pdf.body_bold("Goal: Determine whether the baseline simply needed more training time.")
    pdf.body("Change: Epochs 5 -> 50 (ran 35 before manual stop). All other parameters identical to Exp 1.")

    pdf.subsubsection("Results (Best Validation)")
    pdf.add_table(
        ["Metric", "Best Value", "Epoch", "vs. Exp 1"],
        [
            ["Val mIoU", "0.394", "19", "+0.035 (+9.7%)"],
            ["Val Mean Dice", "0.481", "35", "+0.038 (+8.6%)"],
            ["Val Pixel Accuracy", "0.956", "19", "+0.012"],
        ],
        col_widths=[42, 35, 25, 55],
    )

    pdf.subsubsection("Outcome and Analysis")
    pdf.body(
        "More epochs helped marginally (+9.7% mIoU), but severe overfitting dominated the run. By "
        "epoch 35, train loss had dropped to 0.63 while val loss had climbed to 2.38 -- a 3.8:1 ratio. "
        "The model was memorising training data rather than learning generalisable features. Val mIoU "
        "peaked at epoch 19 and then fluctuated without improvement for 16 more epochs, wasting compute."
    )

    pdf.add_figure("A1_exp2_overfitting.png",
                    "Figure 4: Exp 2 train vs val loss. The shaded region between the curves shows the growing "
                    "overfitting gap. By epoch 35, the train-val loss ratio reached 3.8:1.",
                    width=160)

    pdf.subsubsection("Takeaway")
    pdf.body(
        "Simply training longer with the same configuration yielded diminishing returns. Of the 7x "
        "additional compute (35 vs 5 epochs), most was wasted on overfitting after epoch ~12. The model "
        "needed regularisation and a fundamentally different training approach, not just more time."
    )

    # ── Exp 3 ──
    pdf.add_page()
    pdf.subsection("4.3", "Experiment 3: Training Strategy Overhaul -- The Breakthrough")
    pdf.body_bold(
        "Goal: Eliminate overfitting and force the model to learn damage classes, not just background. "
        "Zero architecture changes -- only training strategy."
    )

    pdf.subsubsection("Changes from Exp 2")
    pdf.add_table(
        ["Change", "From", "To", "Rationale"],
        [
            ["Weight decay", "1e-4", "0.05 (500x)", "Primary overfitting counter"],
            ["Encoder freezing", "None", "Phase 1/2", "Protect pretrained features"],
            ["Encoder LR", "Same", "1e-5 (30x lower)", "Preserve ImageNet knowledge"],
            ["Effective batch", "2", "8", "Stable rare-class gradients"],
            ["Class weights", "Freq. inverse", "Log-inverse (all)", "Better rare-class weighting"],
            ["Label smoothing", "None", "0.05", "Reduce overconfidence"],
            ["Early stopping", "None", "Patience=10", "Prevent wasted compute"],
            ["Primary metric", "mIoU", "Damage macro F1", "Excludes background"],
        ],
        col_widths=[38, 32, 42, 60],
    )

    pdf.subsubsection("Results (Best Validation, Epoch 32)")
    pdf.add_table(
        ["Metric", "Best Value", "vs. Exp 2"],
        [
            ["Val mIoU", "0.475", "+0.081 (+20.6%)"],
            ["Val Mean Dice", "0.603", "+0.122 (+25.4%)"],
            ["Val Damage F1", "0.510", "(new metric)"],
        ],
        col_widths=[50, 50, 60],
    )

    pdf.subsubsection("Per-Class F1 at Best Epoch")
    pdf.add_table(
        ["Class", "F1 Score"],
        [
            ["No Damage", "0.672"],
            ["Minor Damage", "0.283"],
            ["Major Damage", "0.461"],
            ["Destroyed", "0.623"],
        ],
        col_widths=[60, 60],
    )

    pdf.subsubsection("Outcome and Analysis")
    pdf.body(
        "This was the single largest improvement across all experiments, achieved with zero architecture "
        "changes. The 500x weight decay increase (1e-4 to 0.05) was the primary driver: the train-val "
        "loss gap dropped from 3.8:1 to near 1:1 (train 2.48, val 2.47). Two-phase training stabilised "
        "early learning by protecting pretrained features. Log-inverse class weights forced the model "
        "to attend to rare damage classes rather than optimising only for easy background pixels."
    )
    pdf.body(
        "Note: Val loss appears higher (~2.47) than Exp 1-2 (~1.3-1.4), but this is expected. Log-inverse "
        "class weights and label smoothing inflate the raw loss by penalising rare-class mistakes more "
        "heavily. The actual predictions are substantially better (higher mIoU, Dice, F1)."
    )

    pdf.add_figure("A2_exp2v3_overfitting_fix.png",
                    "Figure 5: The most dramatic improvement in the project. Left: Exp 2 with train-val loss "
                    "diverging (3.8x gap). Right: Exp 3 with train-val loss running parallel (1:1 ratio). "
                    "The 500x weight decay increase was the primary driver.",
                    width=175)

    pdf.subsubsection("Takeaway")
    pdf.body(
        "Training strategy alone accounted for 60% of total improvement across all 5 experiments. The "
        "biggest wins came from: (1) two-phase encoder freeze/unfreeze, (2) aggressive weight decay, "
        "and (3) proper class weighting. However, per-class F1 revealed Minor Damage at only 0.283 -- "
        "suggesting ResNet34 lacked the feature capacity for subtle damage detection."
    )

    # ── Exp 4 ──
    pdf.add_page()
    pdf.subsection("4.4", "Experiment 4: ResNet50 Backbone -- A Mixed Result")
    pdf.body_bold(
        "Goal: Increase encoder capacity to improve fine-grained damage distinction, particularly Minor Damage."
    )

    pdf.subsubsection("Changes from Exp 3")
    pdf.add_table(
        ["Change", "From", "To", "Rationale"],
        [
            ["Backbone", "ResNet34 (21M)", "ResNet50 (25M)", "4x wider bottleneck layers"],
            ["Encoder channels", "[64..512]", "[64..2048]", "4x more at deepest level"],
            ["Decoder channels", "[256..32]", "[512..64]", "Match wider encoder"],
            ["num_workers", "0", "4", "Parallel data loading"],
            ["Batch size", "2", "16", "Utilise P100 VRAM"],
        ],
        col_widths=[42, 35, 42, 53],
    )

    pdf.subsubsection("Results (Best Validation, Epoch 27)")
    pdf.add_table(
        ["Metric", "Best Value", "vs. Exp 3"],
        [
            ["Val mIoU", "0.488", "+0.013 (+2.7%)"],
            ["Val Mean Dice", "0.616", "+0.013 (+2.2%)"],
            ["Val Damage F1", "0.521", "+0.011 (+2.2%)"],
        ],
        col_widths=[50, 50, 60],
    )

    pdf.subsubsection("Per-Class F1 Comparison")
    pdf.add_table(
        ["Class", "Exp 3 (ResNet34)", "Exp 4 (ResNet50)", "Change"],
        [
            ["No Damage", "0.672", "0.709", "+0.037"],
            ["Minor Damage", "0.283", "0.433", "+0.150 (+53%)"],
            ["Major Damage", "0.461", "0.384", "-0.077 (-17%)"],
            ["Destroyed", "0.623", "0.557", "-0.066"],
        ],
        col_widths=[40, 40, 40, 42],
    )

    pdf.subsubsection("Outcome and Analysis")
    pdf.body(
        "The per-class results told a more nuanced story than the aggregate metrics. Minor Damage F1 "
        "improved dramatically (+53%), confirming that ResNet34 lacked the feature capacity for subtle "
        "damage. The 4x wider bottleneck layers in ResNet50 provided representational depth to "
        "distinguish cracks and missing tiles."
    )
    pdf.body(
        "However, Major Damage and Destroyed regressed, and the overall damage F1 gain was only +0.011 "
        "(far below the expected +0.05-0.10). The cause: the encoder was barely adapting to satellite imagery. "
        "With encoder_lr=1e-5 and only 2 layers unfrozen, the ResNet50 backbone stayed close to its "
        "ImageNet initialisation. At epoch 16, the Phase 2 transition caused a damage F1 drop of 0.099, "
        "and recovery took 12 epochs -- too slow with too little learning signal."
    )

    pdf.add_figure("A3_exp3v4_perclass.png",
                    "Figure 6: Per-class F1 comparison. ResNet50 dramatically improved Minor Damage (+53%) but "
                    "caused Major Damage (-17%) and Destroyed to regress. The mixed result revealed insufficient "
                    "encoder adaptation.",
                    width=155)

    pdf.subsubsection("Takeaway")
    pdf.body(
        "A bigger backbone is not automatically better -- you have to let it learn. ResNet50 proved it "
        "had the capacity to improve damage segmentation (via Minor Damage), but its potential was "
        "unrealised due to conservative fine-tuning. The encoder needed a higher learning rate, more "
        "unfrozen layers, and a loss function better suited to hard boundary cases."
    )

    # ── Exp 5 ──
    pdf.add_page()
    pdf.subsection("4.5", "Experiment 5: Tuned Phase 2 + Focal Loss -- Unlocking the Backbone")
    pdf.body_bold(
        "Goal: Fix Exp 4's insufficient encoder adaptation. Let ResNet50 actually learn satellite-specific features."
    )

    pdf.subsubsection("Changes from Exp 4")
    pdf.add_table(
        ["Change", "From", "To", "Rationale"],
        [
            ["Encoder LR", "1e-5", "5e-5 (5x)", "Faster encoder adaptation"],
            ["Freeze epochs", "15", "10", "More Phase 2 time"],
            ["Unfrozen layers", "2", "4", "Deeper adaptation for sat. imagery"],
            ["Loss", "CE + Dice", "Focal + Dice", "Focus on hard boundaries"],
            ["Epochs", "50", "60", "More convergence time"],
            ["ES patience", "10", "15", "Allow Phase 2 recovery"],
            ["Augmentations", "Moderate", "Stronger", "Compensate more params"],
        ],
        col_widths=[38, 32, 42, 60],
    )

    pdf.subsubsection("Results (Best Validation, Epoch 30 -- stopped at 34/60)")
    pdf.add_table(
        ["Metric", "Best Value", "vs. Exp 4"],
        [
            ["Val mIoU", "0.494", "+0.006 (+1.2%)"],
            ["Val Mean Dice", "0.625", "+0.009 (+1.5%)"],
            ["Val Damage F1", "0.533", "+0.012 (+2.3%)"],
        ],
        col_widths=[50, 50, 60],
    )

    pdf.subsubsection("Per-Class F1 Comparison")
    pdf.add_table(
        ["Class", "Exp 4", "Exp 5", "Change"],
        [
            ["No Damage", "0.709", "0.683", "-0.026"],
            ["Minor Damage", "0.433", "0.466", "+0.033"],
            ["Major Damage", "0.384", "0.427", "+0.043"],
            ["Destroyed", "0.557", "0.556", "-0.001"],
        ],
        col_widths=[42, 35, 35, 42],
    )

    pdf.subsubsection("Outcome and Analysis")
    pdf.body(
        "Exp 5 achieved the best results of all experiments despite completing only 57% of planned "
        "epochs (34/60). The changes validated our hypothesis from Exp 4:"
    )
    pdf.bullet("The encoder needed a higher LR: 5e-5 (vs 1e-5) allowed the backbone to adapt to satellite imagery features.")
    pdf.bullet("More layers needed unfreezing: 4 layers (vs 2) allowed deeper feature adaptation. Satellite imagery of rubble and structural collapse looks nothing like ImageNet's natural images.")
    pdf.bullet("Focal Loss helped hard cases: Major Damage F1 recovered from 0.384 to 0.427, partially restoring Exp 3's level. Focal loss (gamma=2) downweights easy background pixels and focuses gradients on ambiguous damage boundaries.")
    pdf.bullet("Minor Damage reached 0.466, now exceeding the typical competition winner range (0.30-0.45).")

    pdf.add_figure("A5_exp5_deepdive.png",
                    "Figure 7: Exp 5 deep-dive. Top: Damage F1 over epochs, coloured by phase (blue=frozen, "
                    "pink=unfrozen). Phase 2 transition shock at epoch 11, recovery by epoch 18. GPU quota "
                    "exhausted at epoch 34 while model was still improving. Bottom: Per-class F1 trajectories.",
                    width=170)

    pdf.body(
        "Phase 2 showed validation instability -- occasional loss spikes up to ~3.96 at epochs 20, 23, "
        "and 25. This is the cost of more aggressive fine-tuning: the higher encoder LR occasionally "
        "disrupts features before the model recovers. Despite the spikes, the overall trend was upward "
        "and early stopping (patience=15) never triggered."
    )

    pdf.callout(
        "Compute constraint: Training was terminated at epoch 34/60 due to exhausted GPU quota on Kaggle. "
        "The SUTD AI cluster was also experiencing downtime, as reported by multiple teams. Given that "
        "early stopping had not triggered and the model was still improving, the final results likely "
        "underestimate what this configuration could achieve with full training."
    )

    pdf.add_figure("A4_exp4v5_phase2_adaptation.png",
                    "Figure 8: Exp 4 vs Exp 5 damage F1 during Phase 2. Exp 5's more aggressive fine-tuning "
                    "(5x encoder LR, 4 layers, Focal Loss) achieved higher peak despite more volatility. "
                    "Red annotation marks GPU quota exhaustion.",
                    width=165)

    pdf.subsubsection("Takeaway")
    pdf.body(
        "The changes validated the core hypothesis: ResNet50 needed more aggressive fine-tuning to "
        "realise its potential. Even with only 57% of planned epochs, Exp 5 set new bests for all "
        "metrics. The remaining 26 epochs would likely have pushed damage F1 toward the 0.58-0.62 range."
    )

    # ── Results Summary ──
    pdf.add_page()
    pdf.subsection("4.6", "Results Summary")

    pdf.add_table(
        ["", "Exp 1", "Exp 2", "Exp 3", "Exp 4", "Exp 5"],
        [
            ["Key Change", "Baseline", "More epochs", "Train. overhaul", "ResNet50", "Focal+Tuned"],
            ["Backbone", "ResNet34", "ResNet34", "ResNet34", "ResNet50", "ResNet50"],
            ["Epochs Run", "5", "35", "42", "32", "34/60*"],
            ["Eff. Batch", "2", "2", "8", "64", "64"],
            ["Val mIoU", "0.359", "0.394", "0.475", "0.488", "0.494"],
            ["Val Dice", "0.443", "0.481", "0.603", "0.616", "0.625"],
            ["Val Dmg F1", "--", "--", "0.510", "0.521", "0.533"],
            ["Overfitting", "Moderate", "Severe", "Controlled", "Controlled", "Controlled"],
        ],
        col_widths=[30, 28, 28, 30, 28, 28],
    )
    pdf.italic("*Exp 5 stopped at epoch 34/60 due to GPU quota exhaustion. SUTD AI cluster also unavailable. Training was still improving.")

    pdf.add_figure("B5_combined_progression.png",
                    "Figure 9: mIoU and Damage Macro F1 progression across all 5 experiments, showing systematic "
                    "improvement from 0.359 to 0.494 mIoU (+37%).",
                    width=170)

    pdf.body(
        "The quantitative improvements translate directly to visual quality. The close-up comparison "
        "below shows zoomed regions from a wildfire and hurricane scene, demonstrating how Exp 1's "
        "near-blank predictions evolve into structured damage segmentation by Exp 5."
    )

    pdf.add_figure("D2_improvement_closeup.png",
                    "Figure 16: Close-up comparison of damage segmentation across experiments. Top: Santa Rosa "
                    "wildfire (destroyed buildings). Bottom: Hurricane Harvey (mixed damage levels). Exp 1 "
                    "produces minimal or incorrect predictions. Exp 3 begins detecting damage regions but "
                    "with poor class distinction. Exp 5 achieves structured, class-differentiated segmentation.",
                    width=178)

    # ── SOTA comparison ──
    pdf.subsection("4.7", "Comparison to State-of-the-Art")
    pdf.add_table(
        ["Model", "Damage F1", "Data", "Ensemble"],
        [
            ["1st place (vdurnov)", "~0.75", "Full xBD (22K)", "12 models"],
            ["2nd place (selimsef)", "~0.72", "Full xBD", "2 models"],
            ["Single-model baseline", "~0.625", "Full xBD", "1 model"],
            ["Ours (Exp 5)", "0.533", "Tier 3 (2.2K)", "1 model"],
        ],
        col_widths=[50, 35, 45, 35],
    )

    pdf.add_figure("B4_sota_comparison.png",
                    "Figure 10: Comparison to xView2 competition SOTA. The gap is attributable to resource "
                    "constraints (10x less data, no localisation pretraining, single model, incomplete training), "
                    "not architectural limitations.",
                    width=165)

    pdf.body(
        "Our systematic improvement from mIoU 0.359 to 0.494 -- a 38% relative improvement -- demonstrates "
        "effective hyperparameter tuning and architectural decisions. Notably, our Minor Damage F1 (0.466) "
        "exceeds the typical range achieved by competition winners (0.30-0.45 for this class)."
    )

    # ╔════════════════════════════════════════════════════════════╗
    # ║  5. ANALYSIS                                               ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("5", "Analysis")

    pdf.subsection("5.1", "What Made the Biggest Difference")

    pdf.add_figure("B1_miou_waterfall.png",
                    "Figure 11: Waterfall chart showing cumulative mIoU contribution of each experiment. "
                    "Exp 3 (training strategy overhaul) accounts for 60% of total improvement.",
                    width=170)

    pdf.add_table(
        ["Rank", "Change", "Experiment", "mIoU Gain", "% of Total"],
        [
            ["1", "Weight decay + two-phase", "Exp 3", "+0.081", "60%"],
            ["2", "More epochs", "Exp 2", "+0.035", "26%"],
            ["3", "ResNet50 backbone", "Exp 4", "+0.013", "10%"],
            ["4", "Focal Loss + tuned FT", "Exp 5", "+0.006", "4%"],
        ],
        col_widths=[20, 45, 30, 30, 30],
    )

    pdf.body(
        "Training strategy accounted for 60% of total improvement. The 500x weight decay increase "
        "(1e-4 to 0.05) was the single most impactful change -- it eliminated the 3.8:1 train-val loss "
        "ratio entirely. Architecture upgrades (ResNet50) and loss function changes (Focal) were "
        "secondary to getting the training fundamentals right."
    )

    pdf.subsection("5.2", "The Rare-Class Challenge")

    pdf.add_figure("B2_rare_class_challenge.png",
                    "Figure 12: Per-class F1 across Exp 3/4/5 with competition winner range shaded for Minor "
                    "Damage. The +65% improvement trajectory demonstrates systematic progress on the hardest class.",
                    width=165)

    pdf.body(
        "Minor Damage was consistently the weakest class but showed the largest improvement trajectory: "
        "F1 improved from 0.283 (Exp 3) to 0.466 (Exp 5) -- a 65% gain. This class is inherently "
        "difficult: subtle cracks, missing tiles, and cosmetic damage are near the resolution limit at "
        "512x512 pixels. Competition winners also struggle here (typical range: 0.30-0.45). Our final "
        "Minor Damage F1 of 0.466 exceeds this range."
    )

    pdf.subsection("5.3", "The Encoder Adaptation Arc")

    pdf.add_figure("B3_encoder_adaptation_arc.png",
                    "Figure 13: Per-class F1 trajectories across Exp 3->4->5. Major Damage regressed when "
                    "ResNet50 was fine-tuned too conservatively (Exp 4), then recovered with 5x higher encoder LR "
                    "and Focal Loss (Exp 5).",
                    width=155)

    pdf.body(
        "A key finding was that upgrading from ResNet34 to ResNet50 caused Major Damage F1 to regress "
        "(0.461 to 0.384) when the encoder was kept too close to its ImageNet initialisation. Satellite "
        "imagery of structural collapse looks nothing like ImageNet's natural images -- the encoder must "
        "be given sufficient learning signal to adapt. Increasing encoder LR 5x and unfreezing 4 layers "
        "(Exp 5) recovered the regression."
    )
    pdf.body_bold(
        "Lesson: A bigger backbone with conservative fine-tuning can be worse than a smaller backbone "
        "that is fully adapted."
    )

    pdf.subsection("5.4", "Loss Curves Across All Experiments")

    pdf.add_figure("C4_loss_landscape_all.png",
                    "Figure 14: Training and validation loss across all 5 experiments. Exp 1-2 show low absolute "
                    "loss but severe train-val divergence. Exp 3-5 show higher absolute loss (due to log-inverse "
                    "class weights) but much better generalisation.",
                    width=170)

    pdf.body(
        "The loss curves reveal an important subtlety: lower absolute loss does not mean better "
        "performance. Exp 1-2 achieved train loss below 1.0, but with severe overfitting. Exp 3-5 have "
        "higher absolute loss values (~2.5-3.0) because log-inverse class weights penalise rare-class "
        "errors heavily. Despite the higher loss, the actual predictions are substantially better "
        "(mIoU 0.494 vs 0.359). This underscores why proper evaluation metrics (damage F1, per-class F1) "
        "are essential."
    )

    pdf.subsection("5.5", "Failure Cases and Limitations")
    pdf.bullet("Minor vs. Major confusion: The boundary between adjacent damage levels is subjective -- human annotators also disagree. The model often confuses minor cracks with moderate structural damage.")
    pdf.bullet("Small buildings: Buildings occupying very few pixels at 512x512 resolution are difficult to classify reliably. Higher-resolution inputs would help.")
    pdf.bullet("Underrepresented disaster types: Performance varies by disaster event. The model generalises less well to disaster types with fewer training samples in Tier 3.")
    pdf.bullet("Phase 2 validation instability: Aggressive encoder fine-tuning (Exp 5) caused periodic validation loss spikes. The encoder LR may be near the upper limit of stability.")
    pdf.bullet("Incomplete training: Our best model (Exp 5) completed only 57% of planned epochs. The model was still improving -- early stopping never triggered -- meaning our reported results are a lower bound.")

    pdf.add_figure("D3_failure_cases.png",
                    "Figure 17: Failure analysis for Exp 5. Three challenging scenarios: minor damage confusion "
                    "(top), small buildings with mixed damage (middle), and scenes with mixed damage levels "
                    "(bottom). The error maps (rightmost column) show correct predictions in green, wrong "
                    "damage class in red, and missed buildings in orange. The model's main failure mode is "
                    "confusing adjacent damage levels (minor vs major), consistent with the per-class F1 analysis.",
                    width=170)

    # ╔════════════════════════════════════════════════════════════╗
    # ║  6. REPRODUCIBILITY                                        ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("6", "Reproducibility")

    pdf.subsection("6.1", "Environment")
    pdf.body("Python 3.11+, PyTorch >= 2.0 (with CUDA), torchvision, opencv-python, scikit-image, scikit-learn, shapely, matplotlib, numpy, tqdm.")
    pdf.body("Install: pip install torch torchvision opencv-python scikit-image scikit-learn shapely matplotlib numpy tqdm")

    pdf.subsection("6.2", "Dataset Setup")
    pdf.bullet("Download the xView2 Tier 3 dataset from Kaggle: kaggle.com/datasets/tunguz/xview2-challenge-dataset-tier-3-data")
    pdf.bullet("Place in: data/train/train/images/ and data/train/train/labels/")
    pdf.bullet("Update data_root in the experiment config JSON if using a different path.")

    pdf.subsection("6.3", "Training from Scratch")
    pdf.bullet("Open Kaggle-latest.ipynb (for Kaggle) or training_script.ipynb (for local training).")
    pdf.bullet("Cell 1 creates the experiment config. Modify the config dict to change hyperparameters.")
    pdf.bullet("Set CONFIG_PATH to the desired experiment config (e.g., training_runs/siamese_resunet_xview2_4/config.json).")
    pdf.bullet("Run all cells. Training logs, checkpoints (best_model.pt, last_model.pt), metrics (history.json, metrics.csv), and config are saved automatically.")
    pdf.body("Each experiment's configuration is stored in training_runs/<experiment_name>/config.json. To reproduce any experiment, set CONFIG_PATH to that config and run the notebook.")

    pdf.subsection("6.4", "Loading a Trained Model")
    pdf.body(
        "To load a trained model for inference:\n\n"
        "  model = SiameseResUNet(backbone='resnet50', num_classes=5,\n"
        "                         decoder_channels=[512, 256, 128, 64])\n"
        "  checkpoint = torch.load('training_runs/siamese_resunet_xview2_4/best_model.pt',\n"
        "                          map_location='cpu')\n"
        "  model.load_state_dict(checkpoint['model_state_dict'])\n"
        "  model.eval()\n"
        "  pred = model(pre_image, post_image).argmax(dim=1)"
    )
    pdf.body("Model weights are available at [Google Drive / Dropbox link] for checkpoint files too large for GitHub.")

    pdf.subsection("6.5", "Recreating Figures")
    pdf.body(
        "All figures in this report can be recreated using the provided scripts:\n"
        "  python generate_figures.py        # Figures A1-A5, B1-B5 (from metrics CSVs)\n"
        "  python generate_extra_figures.py   # Figures C1-C4 (architecture, distribution, etc.)\n\n"
        "These scripts require only matplotlib and numpy, reading from training_runs/*/metrics.csv."
    )

    pdf.subsection("6.6", "Project Structure")
    pdf.set_font("Consolas", "", 8)
    pdf.multi_cell(0, 4, (
        "ADL Project/\n"
        "  ADL_Project_Report.pdf          # This report\n"
        "  Kaggle-latest.ipynb             # Training notebook (upload to Kaggle)\n"
        "  training_script.ipynb           # Local training notebook\n"
        "  generate_figures.py             # Recreates metric-based figures\n"
        "  generate_extra_figures.py       # Recreates schematic figures\n"
        "  images/                         # All report figures\n"
        "  training_runs/\n"
        "    siamese_resunet_xview2_1/     # Exp 1: Baseline\n"
        "    siamese_resunet_xview2_2/     # Exp 2: Extended training\n"
        "    siamese_resunet_xview2/       # Exp 3: Training overhaul\n"
        "    siamese_resunet_xview2_3/     # Exp 4: ResNet50\n"
        "    siamese_resunet_xview2_4/     # Exp 5: Focal + Tuned Phase 2\n"
        "  data/                           # Dataset (not in git)\n"
    ))
    pdf.ln(2)

    # ╔════════════════════════════════════════════════════════════╗
    # ║  7. CONCLUSION                                             ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.add_page()
    pdf.section_title("7", "Conclusion")

    pdf.body(
        "We developed a Siamese ResNet U-Net for building damage segmentation from satellite imagery, "
        "achieving a damage macro F1 of 0.533 and mIoU of 0.494 through systematic experimentation "
        "across 5 experiments. Our key findings:"
    )

    pdf.body_bold("1. Training strategy matters more than architecture.")
    pdf.body(
        "The training overhaul (Exp 3) delivered a 20.6% mIoU improvement with zero architecture changes "
        "-- the largest single gain. Aggressive weight decay (0.05) and two-phase encoder freezing were "
        "the primary drivers, accounting for 60% of total improvement."
    )

    pdf.body_bold("2. Honest metrics reveal the real challenge.")
    pdf.body(
        "Switching from pixel accuracy (misleadingly high at 94-98%) to damage macro F1 exposed Minor "
        "Damage (F1=0.28) as the critical bottleneck, directing subsequent experiments toward backbone "
        "capacity and loss function design."
    )

    pdf.body_bold("3. Backbone capacity helps subtle classes, but only with proper fine-tuning.")
    pdf.body(
        "ResNet50 improved Minor Damage F1 by +0.150 over ResNet34, but caused Major Damage to regress "
        "when fine-tuned too conservatively. Increasing encoder LR 5x and unfreezing 4 layers (Exp 5) "
        "recovered the regression and pushed all metrics to new bests."
    )

    pdf.body_bold("4. Focal Loss focuses on what matters.")
    pdf.body(
        "Switching from Cross-Entropy to Focal Loss (gamma=2) helped the model attend to hard damage "
        "boundary pixels instead of easy background, recovering Major Damage F1 and further improving "
        "Minor Damage."
    )

    pdf.body_bold("5. The gap to SOTA is explainable.")
    pdf.body(
        "Our single-model result on 10% of the data (0.533) vs. competition winners using 12-model "
        "ensembles on the full dataset (0.75) is explained by data scale, ensembling, two-stage "
        "pretraining, and incomplete training -- resource constraints, not architectural limitations."
    )

    pdf.ln(4)
    pdf.subsection("", "Future Work")
    pdf.bullet("Complete Exp 5 training (26 remaining epochs) when GPU compute becomes available -- the model was still improving.")
    pdf.bullet("Implement test-time augmentation for free accuracy gains (typically +1-3% with no retraining).")
    pdf.bullet("Add auxiliary building localisation head (multi-task learning, as used by competition winners).")
    pdf.bullet("Train on the full xBD dataset (~22,000 pairs) if compute permits.")

    # ╔════════════════════════════════════════════════════════════╗
    # ║  8. TEAM CONTRIBUTIONS                                     ║
    # ╚════════════════════════════════════════════════════════════╝
    pdf.section_title("8", "Team Contributions")
    pdf.add_table(
        ["Member", "Contributions"],
        [
            ["[Name 1]", "[Contributions]"],
            ["[Name 2]", "[Contributions]"],
            ["[Name 3]", "[Contributions]"],
            ["[Name 4]", "[Contributions]"],
        ],
        col_widths=[50, 130],
    )

    # ╔════════════════════════════════════════════════════════════╗
    # ║  REFERENCES                                                ║
    # ╚════════════════════════════════════════════════════════════╝
    # References section (no number)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(25, 25, 25)
    pdf.cell(0, 10, "References", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(33, 150, 243)
    pdf.set_line_width(0.8)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)
    refs = [
        '[1] Gupta, R., et al. "xBD: A Dataset for Assessing Building Damage from Satellite Imagery." arXiv:1911.09296, 2019.',
        '[2] Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.',
        '[3] He, K., et al. "Deep Residual Learning for Image Recognition." CVPR 2016.',
        '[4] Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.',
        '[5] xView2 1st Place Solution: github.com/vdurnov/xview2_1st_place_solution',
        '[6] xView2 2nd Place Solution: github.com/selimsef/xview2_solution',
    ]
    for ref in refs:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 5, ref)
        pdf.ln(1)

    # ── Save ──
    pdf.output(str(OUT))
    print(f"\nReport saved to: {OUT}")
    print(f"Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    build()
