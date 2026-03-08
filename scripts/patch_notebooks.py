"""
Patch all 4 scenario notebooks to add:
1. Sample image preview (3x3 grid, with original vs enhanced for LLIE scenarios)
2. Training curves (loss + metrics line charts for overfitting/underfitting analysis)
3. Detection visualization (predictions vs ground truth with bounding boxes)
4. Confusion matrix (normalized + counts)
5. Fix per_class TypeError (defensive handling)
"""
import json
import copy
import uuid
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")

SCENARIOS = {
    "scenario_s1_raw.ipynb": {
        "key": "s1_raw",
        "name": "S1_Raw",
        "has_enhancement": False,
    },
    "scenario_s2_hvi_cidnet.ipynb": {
        "key": "s2_hvi_cidnet",
        "name": "S2_HVI_CIDNet",
        "has_enhancement": True,
    },
    "scenario_s3_retinexformer.ipynb": {
        "key": "s3_retinexformer",
        "name": "S3_RetinexFormer",
        "has_enhancement": True,
    },
    "scenario_s4_lyt_net.ipynb": {
        "key": "s4_lyt_net",
        "name": "S4_LYT_Net",
        "has_enhancement": True,
    },
}


def make_md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def find_cell_index(cells, marker_text):
    """Find cell index by searching for a marker in the source text."""
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if marker_text in src:
            return i
    return None


# ═══════════════════════════════════════════════════════════════════
#  NEW CELL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

# ── Fase 2.5: Sample Image Preview (NO enhancement) ───────────
PREVIEW_MD_NO_ENH = make_md_cell([
    "---\n",
    "## Fase 2.5: Sample Test Images Preview"
])

PREVIEW_CODE_NO_ENH = make_code_cell([
    "#@title Fase 2.5 · Sample Test Images (3×3 Grid)\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import glob as _glob\n",
    "\n",
    "test_img_dir = os.path.join(OUTPUT_ROOT, \"ExDark_yolo\", \"images\", \"test\")\n",
    "sample_images = sorted(_glob.glob(os.path.join(test_img_dir, \"*.*\")))[:9]\n",
    "\n",
    "fig, axes = plt.subplots(3, 3, figsize=(14, 14))\n",
    "fig.suptitle(f\"Sample Test Images — {SCENARIO_NAME}\\n(No Enhancement — Raw Low-Light)\",\n",
    "             fontsize=16, fontweight='bold', y=1.02)\n",
    "\n",
    "for idx, ax in enumerate(axes.flat):\n",
    "    if idx < len(sample_images):\n",
    "        img = mpimg.imread(sample_images[idx])\n",
    "        ax.imshow(img)\n",
    "        ax.set_title(os.path.basename(sample_images[idx]), fontsize=9)\n",
    "    ax.axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "save_dir = os.path.join(OUTPUT_ROOT, \"evaluation\", SCENARIO_NAME)\n",
    "os.makedirs(save_dir, exist_ok=True)\n",
    "plt.savefig(os.path.join(save_dir, \"sample_test_images.png\"),\n",
    "            dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f\"✓ Saved → evaluation/{SCENARIO_NAME}/sample_test_images.png\")\n",
])

# ── Fase 2.5: Sample Image Preview (WITH enhancement) ─────────
PREVIEW_MD_ENH = make_md_cell([
    "---\n",
    "## Fase 2.5: Sample Test Images — Original vs Enhanced"
])

PREVIEW_CODE_ENH = make_code_cell([
    "#@title Fase 2.5 · Original vs Enhanced (3×3 Paired Grid)\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import glob as _glob\n",
    "\n",
    "raw_test_dir = os.path.join(OUTPUT_ROOT, \"ExDark_yolo\", \"images\", \"test\")\n",
    "enh_test_dir = os.path.join(OUTPUT_ROOT, f\"ExDark_enhanced_{enhancer_name}\", \"images\", \"test\")\n",
    "\n",
    "raw_images = sorted(_glob.glob(os.path.join(raw_test_dir, \"*.*\")))[:9]\n",
    "\n",
    "fig, axes = plt.subplots(6, 3, figsize=(15, 28))\n",
    "fig.suptitle(f\"Original (Low-Light) vs Enhanced ({enhancer_name}) — {SCENARIO_NAME}\",\n",
    "             fontsize=16, fontweight='bold', y=1.01)\n",
    "\n",
    "for i in range(9):\n",
    "    row_orig = (i // 3) * 2\n",
    "    row_enh  = row_orig + 1\n",
    "    col      = i % 3\n",
    "\n",
    "    if i < len(raw_images):\n",
    "        fname = os.path.basename(raw_images[i])\n",
    "\n",
    "        # ── Original (top of pair) ──\n",
    "        raw_img = mpimg.imread(raw_images[i])\n",
    "        axes[row_orig, col].imshow(raw_img)\n",
    "        axes[row_orig, col].set_title(f\"Original: {fname}\", fontsize=8)\n",
    "        axes[row_orig, col].axis('off')\n",
    "\n",
    "        # ── Enhanced (bottom of pair) ──\n",
    "        enh_path = os.path.join(enh_test_dir, fname)\n",
    "        if os.path.exists(enh_path):\n",
    "            enh_img = mpimg.imread(enh_path)\n",
    "            axes[row_enh, col].imshow(enh_img)\n",
    "            axes[row_enh, col].set_title(f\"Enhanced: {fname}\", fontsize=8)\n",
    "        else:\n",
    "            axes[row_enh, col].text(0.5, 0.5, \"Enhanced not found\",\n",
    "                                     ha='center', va='center',\n",
    "                                     transform=axes[row_enh, col].transAxes, fontsize=10)\n",
    "        axes[row_enh, col].axis('off')\n",
    "    else:\n",
    "        axes[row_orig, col].axis('off')\n",
    "        axes[row_enh, col].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "save_dir = os.path.join(OUTPUT_ROOT, \"evaluation\", SCENARIO_NAME)\n",
    "os.makedirs(save_dir, exist_ok=True)\n",
    "plt.savefig(os.path.join(save_dir, \"sample_original_vs_enhanced.png\"),\n",
    "            dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f\"✓ Saved → evaluation/{SCENARIO_NAME}/sample_original_vs_enhanced.png\")\n",
])

# ── Fase 3.5: Training Curves ─────────────────────────────────
TRAIN_CURVES_MD = make_md_cell([
    "---\n",
    "## Fase 3.5: Training Curves (Loss & Metrics)\n",
    "Visualisasi kurva training untuk analisis **overfitting / underfitting**.\n",
    "- **Row 1**: Train vs Val loss (box, cls, dfl) — gap besar = overfitting\n",
    "- **Row 2**: Precision, Recall, mAP progression per epoch"
])

TRAIN_CURVES_CODE = make_code_cell([
    "#@title Fase 3.5 · Training Curves\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import glob as _glob\n",
    "\n",
    "run_dir_train = os.path.join(OUTPUT_ROOT, \"runs\", SCENARIO_NAME)\n",
    "results_csv = os.path.join(run_dir_train, \"results.csv\")\n",
    "if not os.path.exists(results_csv):\n",
    "    candidates = _glob.glob(os.path.join(run_dir_train, \"**\", \"results.csv\"), recursive=True)\n",
    "    if candidates:\n",
    "        results_csv = candidates[0]\n",
    "\n",
    "if not os.path.exists(results_csv):\n",
    "    print(f\"⚠ results.csv not found in {run_dir_train}. Skipping training curves.\")\n",
    "else:\n",
    "    df = pd.read_csv(results_csv)\n",
    "    df.columns = df.columns.str.strip()\n",
    "    epochs = df.index + 1\n",
    "\n",
    "    # ── Figure 1: Train vs Val Loss + Metrics ──\n",
    "    fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
    "    fig.suptitle(f\"Training Curves — {SCENARIO_NAME}\", fontsize=16, fontweight='bold')\n",
    "\n",
    "    # Row 1: Loss (train vs val → detect overfitting)\n",
    "    loss_pairs = [\n",
    "        (\"train/box_loss\", \"val/box_loss\", \"Box Loss\"),\n",
    "        (\"train/cls_loss\", \"val/cls_loss\", \"Classification Loss\"),\n",
    "        (\"train/dfl_loss\", \"val/dfl_loss\", \"DFL Loss\"),\n",
    "    ]\n",
    "    for i, (tr, vl, title) in enumerate(loss_pairs):\n",
    "        if tr in df.columns and vl in df.columns:\n",
    "            axes[0, i].plot(epochs, df[tr], label=\"Train\", linewidth=2, color='#2196F3')\n",
    "            axes[0, i].plot(epochs, df[vl], label=\"Val\", linewidth=2,\n",
    "                            linestyle='--', color='#F44336')\n",
    "            axes[0, i].set_title(title, fontsize=12, fontweight='bold')\n",
    "            axes[0, i].set_xlabel(\"Epoch\")\n",
    "            axes[0, i].set_ylabel(\"Loss\")\n",
    "            axes[0, i].legend(fontsize=10)\n",
    "            axes[0, i].grid(True, alpha=0.3)\n",
    "\n",
    "    # Row 2: Metrics progression\n",
    "    metric_items = [\n",
    "        (\"metrics/precision(B)\", \"Precision\", '#4CAF50'),\n",
    "        (\"metrics/recall(B)\",    \"Recall\",    '#FF9800'),\n",
    "        (\"metrics/mAP50(B)\",     \"mAP@0.5\",   '#9C27B0'),\n",
    "    ]\n",
    "    for i, (col, title, color) in enumerate(metric_items):\n",
    "        if col in df.columns:\n",
    "            axes[1, i].plot(epochs, df[col], linewidth=2, color=color)\n",
    "            axes[1, i].set_title(title, fontsize=12, fontweight='bold')\n",
    "            axes[1, i].set_xlabel(\"Epoch\")\n",
    "            axes[1, i].set_ylabel(title)\n",
    "            axes[1, i].grid(True, alpha=0.3)\n",
    "            axes[1, i].set_ylim(0, 1.05)\n",
    "\n",
    "    plt.tight_layout()\n",
    "    save_dir = os.path.join(OUTPUT_ROOT, \"evaluation\", SCENARIO_NAME)\n",
    "    os.makedirs(save_dir, exist_ok=True)\n",
    "    plt.savefig(os.path.join(save_dir, \"training_curves.png\"),\n",
    "                dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "\n",
    "    # ── Figure 2: mAP@0.5 vs mAP@0.5:0.95 ──\n",
    "    if \"metrics/mAP50(B)\" in df.columns and \"metrics/mAP50-95(B)\" in df.columns:\n",
    "        fig2, ax2 = plt.subplots(figsize=(10, 5))\n",
    "        ax2.plot(epochs, df[\"metrics/mAP50(B)\"],\n",
    "                 label=\"mAP@0.5\", linewidth=2, color='#9C27B0')\n",
    "        ax2.plot(epochs, df[\"metrics/mAP50-95(B)\"],\n",
    "                 label=\"mAP@0.5:0.95\", linewidth=2, color='#E91E63')\n",
    "        ax2.set_title(f\"mAP Progression — {SCENARIO_NAME}\",\n",
    "                      fontsize=14, fontweight='bold')\n",
    "        ax2.set_xlabel(\"Epoch\")\n",
    "        ax2.set_ylabel(\"mAP\")\n",
    "        ax2.legend(fontsize=11)\n",
    "        ax2.grid(True, alpha=0.3)\n",
    "        ax2.set_ylim(0, 1.05)\n",
    "        plt.tight_layout()\n",
    "        plt.savefig(os.path.join(save_dir, \"mAP_progression.png\"),\n",
    "                    dpi=150, bbox_inches='tight')\n",
    "        plt.show()\n",
    "\n",
    "    # ── Figure 3: Learning Rate Schedule ──\n",
    "    lr_cols = [c for c in df.columns if c.startswith(\"lr/\")]\n",
    "    if lr_cols:\n",
    "        fig3, ax3 = plt.subplots(figsize=(10, 4))\n",
    "        for c in lr_cols:\n",
    "            ax3.plot(epochs, df[c], label=c, linewidth=1.5)\n",
    "        ax3.set_title(f\"Learning Rate Schedule — {SCENARIO_NAME}\",\n",
    "                      fontsize=14, fontweight='bold')\n",
    "        ax3.set_xlabel(\"Epoch\")\n",
    "        ax3.set_ylabel(\"Learning Rate\")\n",
    "        ax3.legend(fontsize=9)\n",
    "        ax3.grid(True, alpha=0.3)\n",
    "        plt.tight_layout()\n",
    "        plt.savefig(os.path.join(save_dir, \"lr_schedule.png\"),\n",
    "                    dpi=150, bbox_inches='tight')\n",
    "        plt.show()\n",
    "\n",
    "    print(f\"✓ Saved training curves → evaluation/{SCENARIO_NAME}/\")\n",
])

# ── Fase 4.5: Detection Visualization ─────────────────────────
DET_VIZ_MD = make_md_cell([
    "---\n",
    "## Fase 4.5: Detection Samples — Predictions vs Ground Truth\n",
    "Visualisasi 9 sample test images:\n",
    "- **Kolom kiri**: Ground truth bounding boxes\n",
    "- **Kolom kanan**: Prediksi model (bounding box + class label + confidence score)"
])

DET_VIZ_CODE = make_code_cell([
    "#@title Fase 4.5 · Detection Visualization (GT vs Prediction)\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as mpatches\n",
    "import matplotlib.image as mpimg\n",
    "from ultralytics import YOLO\n",
    "import numpy as np\n",
    "import glob as _glob\n",
    "\n",
    "CLASS_NAMES = {\n",
    "    0: \"Bicycle\", 1: \"Boat\", 2: \"Bottle\", 3: \"Bus\",\n",
    "    4: \"Car\", 5: \"Cat\", 6: \"Chair\", 7: \"Cup\",\n",
    "    8: \"Dog\", 9: \"Motorbike\", 10: \"People\", 11: \"Table\",\n",
    "}\n",
    "BOX_COLORS = {i: plt.cm.Set3(i / 12) for i in range(12)}\n",
    "\n",
    "# ── Paths ──\n",
    "if enhancer_name and str(enhancer_name).lower() != \"none\":\n",
    "    test_img_dir = os.path.join(\n",
    "        OUTPUT_ROOT, f\"ExDark_enhanced_{enhancer_name}\", \"images\", \"test\")\n",
    "else:\n",
    "    test_img_dir = os.path.join(OUTPUT_ROOT, \"ExDark_yolo\", \"images\", \"test\")\n",
    "test_lbl_dir = os.path.join(OUTPUT_ROOT, \"ExDark_yolo\", \"labels\", \"test\")\n",
    "\n",
    "_w = get_best_weights(os.path.join(OUTPUT_ROOT, \"runs\", SCENARIO_NAME))\n",
    "_model = YOLO(_w)\n",
    "\n",
    "sample_imgs = sorted(_glob.glob(os.path.join(test_img_dir, \"*.*\")))[:9]\n",
    "\n",
    "def _draw_gt(ax, label_path, h, w):\n",
    "    \"\"\"Draw YOLO-format ground-truth boxes.\"\"\"\n",
    "    if not os.path.exists(label_path):\n",
    "        return\n",
    "    with open(label_path) as f:\n",
    "        for line in f:\n",
    "            parts = line.strip().split()\n",
    "            if len(parts) < 5:\n",
    "                continue\n",
    "            cid = int(parts[0])\n",
    "            xc, yc, bw, bh = (float(x) for x in parts[1:5])\n",
    "            x1 = (xc - bw / 2) * w\n",
    "            y1 = (yc - bh / 2) * h\n",
    "            rect = mpatches.FancyBboxPatch(\n",
    "                (x1, y1), bw * w, bh * h,\n",
    "                linewidth=2, edgecolor=BOX_COLORS[cid],\n",
    "                facecolor='none', boxstyle='round,pad=0')\n",
    "            ax.add_patch(rect)\n",
    "            ax.text(x1, max(y1 - 4, 0),\n",
    "                    CLASS_NAMES.get(cid, str(cid)),\n",
    "                    fontsize=7, color='white',\n",
    "                    bbox=dict(boxstyle='round,pad=0.2',\n",
    "                              facecolor=BOX_COLORS[cid], alpha=0.85))\n",
    "\n",
    "def _draw_pred(ax, pred_results):\n",
    "    \"\"\"Draw prediction boxes with confidence.\"\"\"\n",
    "    for box in pred_results[0].boxes:\n",
    "        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()\n",
    "        cid  = int(box.cls[0])\n",
    "        conf = float(box.conf[0])\n",
    "        rect = mpatches.FancyBboxPatch(\n",
    "            (x1, y1), x2 - x1, y2 - y1,\n",
    "            linewidth=2, edgecolor=BOX_COLORS.get(cid, 'red'),\n",
    "            facecolor='none', boxstyle='round,pad=0')\n",
    "        ax.add_patch(rect)\n",
    "        ax.text(x1, max(y1 - 4, 0),\n",
    "                f\"{CLASS_NAMES.get(cid, str(cid))} {conf:.2f}\",\n",
    "                fontsize=7, color='white',\n",
    "                bbox=dict(boxstyle='round,pad=0.2',\n",
    "                          facecolor=BOX_COLORS.get(cid, 'red'), alpha=0.85))\n",
    "\n",
    "fig, axes = plt.subplots(3, 6, figsize=(30, 15))\n",
    "fig.suptitle(\n",
    "    f\"Detection Results — {SCENARIO_NAME}\\n\"\n",
    "    \"Left: Ground Truth  |  Right: Prediction\",\n",
    "    fontsize=16, fontweight='bold')\n",
    "\n",
    "for idx, img_path in enumerate(sample_imgs):\n",
    "    row      = idx // 3\n",
    "    col_gt   = (idx % 3) * 2\n",
    "    col_pred = col_gt + 1\n",
    "    fname    = os.path.basename(img_path)\n",
    "\n",
    "    img = mpimg.imread(img_path)\n",
    "    h, w = img.shape[:2]\n",
    "\n",
    "    # Ground Truth\n",
    "    axes[row, col_gt].imshow(img)\n",
    "    lbl_path = os.path.join(test_lbl_dir,\n",
    "                            os.path.splitext(fname)[0] + \".txt\")\n",
    "    _draw_gt(axes[row, col_gt], lbl_path, h, w)\n",
    "    axes[row, col_gt].set_title(f\"GT: {fname}\", fontsize=8)\n",
    "    axes[row, col_gt].axis('off')\n",
    "\n",
    "    # Prediction\n",
    "    axes[row, col_pred].imshow(img)\n",
    "    pred = _model.predict(img_path, conf=0.25, verbose=False)\n",
    "    _draw_pred(axes[row, col_pred], pred)\n",
    "    axes[row, col_pred].set_title(f\"Pred: {fname}\", fontsize=8)\n",
    "    axes[row, col_pred].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "save_path = os.path.join(OUTPUT_ROOT, \"evaluation\", SCENARIO_NAME,\n",
    "                         \"detection_samples_gt_vs_pred.png\")\n",
    "plt.savefig(save_path, dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f\"✓ Saved → evaluation/{SCENARIO_NAME}/detection_samples_gt_vs_pred.png\")\n",
    "\n",
    "del _model\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.empty_cache()\n",
])

# ── Fase 4.6: Confusion Matrix ────────────────────────────────
CM_MD = make_md_cell([
    "---\n",
    "## Fase 4.6: Confusion Matrix\n",
    "Per-class confusion matrix untuk evaluasi kesalahan klasifikasi objek."
])

CM_CODE = make_code_cell([
    "#@title Fase 4.6 · Confusion Matrix\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "from ultralytics import YOLO\n",
    "\n",
    "CLASS_NAMES_LIST = [\"Bicycle\", \"Boat\", \"Bottle\", \"Bus\", \"Car\", \"Cat\",\n",
    "                    \"Chair\", \"Cup\", \"Dog\", \"Motorbike\", \"People\", \"Table\"]\n",
    "nc = len(CLASS_NAMES_LIST)\n",
    "\n",
    "_w = get_best_weights(os.path.join(OUTPUT_ROOT, \"runs\", SCENARIO_NAME))\n",
    "_model = YOLO(_w)\n",
    "\n",
    "# Run validation with plots to compute confusion matrix\n",
    "val_results = _model.val(data=data_yaml, split=\"test\", plots=False,\n",
    "                         conf=0.25, verbose=False)\n",
    "\n",
    "try:\n",
    "    cm_full = val_results.confusion_matrix.matrix\n",
    "except AttributeError:\n",
    "    print(\"⚠ Confusion matrix not available from val(). Skipping.\")\n",
    "    cm_full = None\n",
    "\n",
    "if cm_full is not None:\n",
    "    # Extract class-only matrix (exclude background row/col)\n",
    "    cm = cm_full[:nc, :nc].copy()\n",
    "    row_sums = cm.sum(axis=1, keepdims=True)\n",
    "    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)\n",
    "\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(22, 9))\n",
    "\n",
    "    # ── Counts ──\n",
    "    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',\n",
    "                xticklabels=CLASS_NAMES_LIST,\n",
    "                yticklabels=CLASS_NAMES_LIST,\n",
    "                ax=axes[0], linewidths=0.5)\n",
    "    axes[0].set_title(f\"Confusion Matrix (Counts) — {SCENARIO_NAME}\",\n",
    "                      fontsize=13, fontweight='bold')\n",
    "    axes[0].set_ylabel(\"True Class\")\n",
    "    axes[0].set_xlabel(\"Predicted Class\")\n",
    "    axes[0].tick_params(axis='x', rotation=45)\n",
    "    axes[0].tick_params(axis='y', rotation=0)\n",
    "\n",
    "    # ── Normalized ──\n",
    "    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',\n",
    "                xticklabels=CLASS_NAMES_LIST,\n",
    "                yticklabels=CLASS_NAMES_LIST,\n",
    "                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)\n",
    "    axes[1].set_title(f\"Confusion Matrix (Normalized) — {SCENARIO_NAME}\",\n",
    "                      fontsize=13, fontweight='bold')\n",
    "    axes[1].set_ylabel(\"True Class\")\n",
    "    axes[1].set_xlabel(\"Predicted Class\")\n",
    "    axes[1].tick_params(axis='x', rotation=45)\n",
    "    axes[1].tick_params(axis='y', rotation=0)\n",
    "\n",
    "    plt.tight_layout()\n",
    "    save_path = os.path.join(OUTPUT_ROOT, \"evaluation\", SCENARIO_NAME,\n",
    "                             \"confusion_matrix.png\")\n",
    "    plt.savefig(save_path, dpi=150, bbox_inches='tight')\n",
    "    plt.show()\n",
    "    print(f\"✓ Saved → evaluation/{SCENARIO_NAME}/confusion_matrix.png\")\n",
    "\n",
    "del _model\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.empty_cache()\n",
])


# ── Updated Fase 4 code (fixed per_class + richer table) ──────
FASE4_CODE_TEMPLATE = [
    "#@title Fase 4 · Detection Evaluation  (auto-skip if metrics.json exists)\n",
    "from src.evaluation.eval_yolo import evaluate_yolo\n",
    "import pandas as pd\n",
    "\n",
    "run_dir      = os.path.join(OUTPUT_ROOT, \"runs\", SCENARIO_NAME)\n",
    "weights_path = get_best_weights(run_dir)\n",
    "eval_dir     = os.path.join(OUTPUT_ROOT, \"evaluation\", SCENARIO_NAME)\n",
    "\n",
    "results  = evaluate_yolo(weights_path=weights_path, dataset_yaml=data_yaml,\n",
    "                         output_dir=eval_dir, scenario_name=SCENARIO_NAME)\n",
    "overall  = results.get(\"overall\", {})\n",
    "\n",
    "print(f\"\\n{'='*50}\")\n",
    "print(f\"Detection Results — {SCENARIO_NAME}\")\n",
    "print(f\"{'='*50}\")\n",
    "print(f\"  mAP@0.5      : {overall.get('mAP_50',0):.4f}\")\n",
    "print(f\"  mAP@0.5:0.95 : {overall.get('mAP_50_95',0):.4f}\")\n",
    "print(f\"  Precision    : {overall.get('precision',0):.4f}\")\n",
    "print(f\"  Recall       : {overall.get('recall',0):.4f}\")\n",
    "\n",
    "# ── Per-class table (defensive: handles both dict and float values) ──\n",
    "per_cls = results.get(\"per_class\", {})\n",
    "if per_cls:\n",
    "    def _extract(v, key, default=0):\n",
    "        if isinstance(v, dict):\n",
    "            return v.get(key, default)\n",
    "        return float(v) if key == \"mAP_50\" else default\n",
    "\n",
    "    cls_rows = [{\n",
    "        \"Class\": k,\n",
    "        \"AP@0.5\": f\"{_extract(v, 'mAP_50'):.4f}\",\n",
    "        \"AP@0.5:0.95\": f\"{_extract(v, 'mAP_50_95'):.4f}\",\n",
    "    } for k, v in per_cls.items()]\n",
    "\n",
    "    df_cls = pd.DataFrame(cls_rows).set_index(\"Class\")\n",
    "    display(df_cls)\n",
]


def patch_notebook(nb_path, scenario_info):
    """Patch a single scenario notebook with all new visualization cells."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    has_enh = scenario_info["has_enhancement"]

    # ── 1. Find insertion points ──
    # We search for unique marker text inside existing cells

    # Fase 2 code cell marker
    if has_enh:
        fase2_marker = "Fase 2" # Enhancement code
        fase2_idx = None
        for i, c in enumerate(cells):
            src = "".join(c.get("source", []))
            if "enhance_dataset" in src or "Enhancement" in src:
                if c["cell_type"] == "code":
                    fase2_idx = i
                    break
        if fase2_idx is None:
            # fallback: find by title
            fase2_idx = find_cell_index(cells, "Fase 2")
    else:
        fase2_idx = find_cell_index(cells, "S1_Raw uses raw images")
        if fase2_idx is None:
            fase2_idx = find_cell_index(cells, "SKIP")

    # Fase 3 code cell (Training)
    fase3_idx = find_cell_index(cells, "train_yolo(dataset_yaml")
    if fase3_idx is None:
        fase3_idx = find_cell_index(cells, "Fase 3")

    # Fase 4 code cell (Detection Eval)
    fase4_idx = find_cell_index(cells, "evaluate_yolo(")
    if fase4_idx is None:
        fase4_idx = find_cell_index(cells, "Fase 4")

    if any(x is None for x in [fase2_idx, fase3_idx, fase4_idx]):
        print(f"  ⚠ Could not find all insertion points! "
              f"fase2={fase2_idx}, fase3={fase3_idx}, fase4={fase4_idx}")
        return False

    # ── 2. Update existing Fase 4 code to fix TypeError ──
    cells[fase4_idx]["source"] = FASE4_CODE_TEMPLATE

    # ── 3. Insert new cells (insert from bottom to top to preserve indices) ──
    # After Fase 4: Confusion Matrix (insert last so it ends up at bottom)
    cells.insert(fase4_idx + 1, copy.deepcopy(CM_MD))
    cells.insert(fase4_idx + 2, copy.deepcopy(CM_CODE))

    # After Fase 4: Detection Visualization (before CM)
    cells.insert(fase4_idx + 1, copy.deepcopy(DET_VIZ_MD))
    cells.insert(fase4_idx + 2, copy.deepcopy(DET_VIZ_CODE))

    # Adjust indices since we added 4 cells after fase4
    # fase3_idx might be before fase4, so no change needed for it
    # But we need to recalculate from scratch for safety

    # Re-find Fase 3 (it shifted if fase3 < fase4, which it always is)
    fase3_idx = find_cell_index(cells, "train_yolo(dataset_yaml")

    # After Fase 3: Training Curves
    cells.insert(fase3_idx + 1, copy.deepcopy(TRAIN_CURVES_MD))
    cells.insert(fase3_idx + 2, copy.deepcopy(TRAIN_CURVES_CODE))

    # Re-find Fase 2
    if has_enh:
        fase2_idx = None
        for i, c in enumerate(cells):
            src = "".join(c.get("source", []))
            if "enhance_dataset" in src:
                if c["cell_type"] == "code":
                    fase2_idx = i
                    break
        if fase2_idx is None:
            fase2_idx = find_cell_index(cells, "Enhancement")
    else:
        fase2_idx = find_cell_index(cells, "S1_Raw uses raw images")
        if fase2_idx is None:
            fase2_idx = find_cell_index(cells, "SKIP")

    # After Fase 2: Sample Image Preview
    if has_enh:
        cells.insert(fase2_idx + 1, copy.deepcopy(PREVIEW_MD_ENH))
        cells.insert(fase2_idx + 2, copy.deepcopy(PREVIEW_CODE_ENH))
    else:
        cells.insert(fase2_idx + 1, copy.deepcopy(PREVIEW_MD_NO_ENH))
        cells.insert(fase2_idx + 2, copy.deepcopy(PREVIEW_CODE_NO_ENH))

    # ── 4. Save ──
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return True


def main():
    for filename, info in SCENARIOS.items():
        nb_path = os.path.join(NOTEBOOKS_DIR, filename)
        if not os.path.exists(nb_path):
            print(f"⚠ Not found: {nb_path}")
            continue

        print(f"\nPatching {filename} ...")
        ok = patch_notebook(nb_path, info)
        if ok:
            print(f"  ✓ Done — {filename}")
        else:
            print(f"  ✗ Failed — {filename}")

    print("\n✓ All notebooks patched.")


if __name__ == "__main__":
    main()
