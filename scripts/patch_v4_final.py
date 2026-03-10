"""
Patch v4: Final notebook rebuild.

Changes vs v3:
- Flat runs/ (no redundant scenario subfolder) via run_name="runs"
- Fase 3.5: Show Ultralytics results.png + train batch images
- NEW Fase 4.55: Validation batch grid (pred vs labels)
- Fase 4.6: Use Ultralytics CM images instead of recomputing
- get_best_weights(SCENARIO_RUNS) everywhere
- Fase 2.5: GridSpec layout with width_ratios for uniform row height
- Fase 4.5 & 4.55: Tighter spacing between title/content/columns
- cellView: "form" for auto-collapse from Fase 2.5 onwards

Rebuilds ALL 4 scenario notebooks from scratch.
"""
import json
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")

SCENARIOS = {
    "scenario_s1_raw.ipynb": {
        "key": "s1_raw",
        "name": "S1_Raw",
        "has_enhancement": False,
        "enhancer_name": None,
    },
    "scenario_s2_hvi_cidnet.ipynb": {
        "key": "s2_hvi_cidnet",
        "name": "S2_HVI_CIDNet",
        "has_enhancement": True,
        "enhancer_name": "hvi_cidnet",
    },
    "scenario_s3_retinexformer.ipynb": {
        "key": "s3_retinexformer",
        "name": "S3_RetinexFormer",
        "has_enhancement": True,
        "enhancer_name": "retinexformer",
    },
    "scenario_s4_lyt_net.ipynb": {
        "key": "s4_lyt_net",
        "name": "S4_LYT_Net",
        "has_enhancement": True,
        "enhancer_name": "lyt_net",
    },
}

def _lines(text):
    return [line + "\n" for line in text.strip().split("\n")]


# ═══════════════════════════════════════════════════════════════════
#  CELL TEMPLATES
# ═══════════════════════════════════════════════════════════════════

SETUP_CONFIG_CODE = r'''#@title 0.3 · Load Configuration & Define Paths
from src.config import load_config, save_environment_info
from src.seed import set_global_seed

cfg = load_config(SCENARIO_KEY, quick_test=QUICK_TEST)
set_global_seed(cfg["seed"])

paths      = cfg.get("paths", {})
data_paths = paths.get("data", {})

OUTPUT_ROOT = paths.get("output_root") or paths.get("drive_root") or paths.get("project_root")
EXDARK_ROOT = paths.get("exdark_root") or data_paths.get("exdark_original")

if OUTPUT_ROOT is None: raise KeyError("Cannot resolve OUTPUT_ROOT from config")
if EXDARK_ROOT is None: raise KeyError("Cannot resolve EXDARK_ROOT from config")

cfg["paths"]["output_root"]  = OUTPUT_ROOT
cfg["paths"]["exdark_root"]  = EXDARK_ROOT
if "exdark_structure" not in cfg["paths"]:
    m = cfg.get("paths_meta", {}).get("exdark", {})
    cfg["paths"]["exdark_structure"] = {
        "images":      m.get("images_dir",      "Dataset"),
        "groundtruth": m.get("groundtruth_dir",  "Groundtruth"),
        "classlist":   m.get("classlist_file",   "Groundtruth/imageclasslist.txt"),
    }

# ── Per-scenario directory layout ───────────────────────────────
PREPARED_DIR   = os.path.join(OUTPUT_ROOT, "prepared")
SCENARIO_DIR   = os.path.join(OUTPUT_ROOT, "scenarios", SCENARIO_NAME)
SCENARIO_RUNS  = os.path.join(SCENARIO_DIR, "runs")
SCENARIO_EVAL  = os.path.join(SCENARIO_DIR, "evaluation")

os.makedirs(PREPARED_DIR, exist_ok=True)
os.makedirs(SCENARIO_RUNS, exist_ok=True)
os.makedirs(SCENARIO_EVAL, exist_ok=True)

print(f"Output root  : {OUTPUT_ROOT}")
print(f"ExDark root  : {EXDARK_ROOT}")
print(f"Prepared dir : {PREPARED_DIR}")
print(f"Scenario dir : {SCENARIO_DIR}")
assert os.path.exists(EXDARK_ROOT), f"ExDark not found: {EXDARK_ROOT}"
print("\n✓ ExDark dataset found")
save_environment_info(SCENARIO_DIR)
'''

FASE1_CODE = r'''#@title Fase 1 · Data Preparation  (auto-skip if already done)
from src.data.split_dataset     import parse_split_file
from src.data.convert_exdark    import convert_exdark_to_yolo
from src.data.build_yolo_dataset import build_yolo_dataset

classlist_path = os.path.join(EXDARK_ROOT,
    cfg["paths"]["exdark_structure"]["groundtruth"], "imageclasslist.txt")
img_dir        = os.path.join(EXDARK_ROOT, cfg["paths"]["exdark_structure"]["images"])
gt_dir         = os.path.join(EXDARK_ROOT, cfg["paths"]["exdark_structure"]["groundtruth"])
split_output   = os.path.join(PREPARED_DIR, "splits")
labels_dir     = os.path.join(PREPARED_DIR, "ExDark_yolo_labels")
yolo_dir       = os.path.join(PREPARED_DIR, "ExDark_yolo")

# 1.1 Splits
splits = parse_split_file(classlist_path, split_output)
print(f"Splits  -> Train:{splits['train']} Val:{splits['val']} Test:{splits['test']}")

# 1.2 Convert annotations
stats = convert_exdark_to_yolo(img_dir, gt_dir, labels_dir)
print(f"Labels  -> {stats['total_labels']} files, {stats['total_objects']} objects")

# 1.3 Build YOLO dir + dataset.yaml
build_stats = build_yolo_dataset(img_dir, labels_dir, split_output, yolo_dir,
                                  target_size=cfg["yolo"]["imgsz"])
total = sum(s["processed"] for s in build_stats["splits"].values())
print(f"YOLO dir-> {total} images built  ({yolo_dir})")
'''

FASE2_SKIP_CODE = r'''#@title Fase 2 · Enhancement  (S1_Raw — SKIPPED, no enhancement)
print("[SKIP] S1_Raw uses raw images. No LLIE enhancement needed.")
enhancer_name = None
enhanced_dir  = None
'''

FASE2_ENH_CODE = r'''#@title Fase 2 · Enhancement  (auto-skip if already done)
import torch
from src.enhancement.run_enhancement import enhance_dataset, get_enhancer

enhancer_name = cfg.get("scenario", {}).get("enhancer", None)
assert enhancer_name and enhancer_name.lower() != "none", \
    "No enhancer configured for this scenario!"

enhanced_dir = os.path.join(SCENARIO_DIR, "enhanced")
cache_dir    = os.path.join(OUTPUT_ROOT, "model_cache")

print(f"Enhancer  : {enhancer_name}")
print(f"Output    : {enhanced_dir}")

enhancer = get_enhancer(enhancer_name, cache_dir)
enhancer.load_model()

stats = enhance_dataset(
    enhancer=enhancer,
    source_dataset_dir=yolo_dir,
    output_dir=enhanced_dir,
    yolo_labels_dir=yolo_dir,
)

print(f"\nDone -> {stats['total_processed']} processed, "
      f"{stats['total_skipped']} skipped, {stats['total_failed']} failed")

del enhancer
if torch.cuda.is_available(): torch.cuda.empty_cache()
'''

# ╔═══════════════════════════════════════════════════════════════════╗
# ║ Fase 2.5 — Preview grids with GridSpec uniform-height rows      ║
# ╚═══════════════════════════════════════════════════════════════════╝

PREVIEW_NO_ENH = r'''#@title Fase 2.5 · Sample Test Images (3x3 Grid)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob as _glob

test_img_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")
sample_paths = sorted(_glob.glob(os.path.join(test_img_dir, "*.*")))[:9]

# Load images and compute aspect ratios (width / height)
images, aspects, fnames = [], [], []
for p in sample_paths:
    img = mpimg.imread(p)
    images.append(img)
    aspects.append(img.shape[1] / img.shape[0])
    fnames.append(os.path.basename(p))

n_cols     = 3
n_rows     = (len(images) + n_cols - 1) // n_cols
row_height = 4.5                                         # inches per row
fig_h      = row_height * n_rows + 1.2                   # + title margin

fig = plt.figure(figsize=(18, fig_h))
fig.suptitle(
    f"Sample Test Images — {SCENARIO_NAME}\n"
    "(No Enhancement — Raw Low-Light)",
    fontsize=16, fontweight='bold')

# Outer grid: one slot per row
outer = gridspec.GridSpec(n_rows, 1, figure=fig,
                          hspace=0.08, top=0.92, bottom=0.02,
                          left=0.02, right=0.98)

for row in range(n_rows):
    s, e = row * n_cols, min((row + 1) * n_cols, len(images))
    row_imgs    = images[s:e]
    row_aspects = aspects[s:e]
    row_fnames  = fnames[s:e]

    # Pad if last row has fewer images
    padded = list(row_aspects)
    while len(padded) < n_cols:
        padded.append(1.0)

    # Inner grid: width_ratios = aspect ratios → uniform height
    inner = gridspec.GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=outer[row],
        width_ratios=padded, wspace=0.03)

    for col in range(n_cols):
        ax = fig.add_subplot(inner[col])
        if col < len(row_imgs):
            ax.imshow(row_imgs[col])
            ax.set_title(row_fnames[col], fontsize=9)
        ax.axis('off')

os.makedirs(SCENARIO_EVAL, exist_ok=True)
plt.savefig(os.path.join(SCENARIO_EVAL, "sample_test_images.png"),
            dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved -> {SCENARIO_EVAL}/sample_test_images.png")
'''

PREVIEW_ENH = r'''#@title Fase 2.5 · Original vs Enhanced (Paired Grid)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import glob as _glob

raw_test_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")
enh_test_dir = os.path.join(SCENARIO_DIR, "enhanced", "images", "test")

raw_paths = sorted(_glob.glob(os.path.join(raw_test_dir, "*.*")))[:9]

# Load raw images and compute aspect ratios
images_raw, aspects, fnames = [], [], []
for p in raw_paths:
    img = mpimg.imread(p)
    images_raw.append(img)
    aspects.append(img.shape[1] / img.shape[0])
    fnames.append(os.path.basename(p))

n_cols     = 3
n_groups   = (len(images_raw) + n_cols - 1) // n_cols    # pair-groups
row_height = 3.0                                          # per sub-row
fig_h      = row_height * n_groups * 2 + 1.5              # 2 sub-rows/group

fig = plt.figure(figsize=(18, fig_h))
fig.suptitle(
    f"Original (Low-Light) vs Enhanced ({enhancer_name}) — {SCENARIO_NAME}",
    fontsize=16, fontweight='bold')

outer = gridspec.GridSpec(n_groups, 1, figure=fig,
                          hspace=0.12, top=0.95, bottom=0.01,
                          left=0.02, right=0.98)

for g in range(n_groups):
    s, e = g * n_cols, min((g + 1) * n_cols, len(images_raw))
    grp_imgs    = images_raw[s:e]
    grp_aspects = aspects[s:e]
    grp_fnames  = fnames[s:e]

    padded = list(grp_aspects)
    while len(padded) < n_cols:
        padded.append(1.0)

    # Inner: 2 rows (orig, enh), n_cols columns
    inner = gridspec.GridSpecFromSubplotSpec(
        2, n_cols, subplot_spec=outer[g],
        width_ratios=padded, hspace=0.04, wspace=0.03)

    for col in range(n_cols):
        # Top: original
        ax_o = fig.add_subplot(inner[0, col])
        if col < len(grp_imgs):
            ax_o.imshow(grp_imgs[col])
            ax_o.set_title(f"Original: {grp_fnames[col]}", fontsize=8)
        ax_o.axis('off')

        # Bottom: enhanced
        ax_e = fig.add_subplot(inner[1, col])
        if col < len(grp_imgs):
            enh_path = os.path.join(enh_test_dir, grp_fnames[col])
            if os.path.exists(enh_path):
                ax_e.imshow(mpimg.imread(enh_path))
                ax_e.set_title(f"Enhanced: {grp_fnames[col]}", fontsize=8)
            else:
                ax_e.text(0.5, 0.5, "Enhanced not found",
                          ha='center', va='center',
                          transform=ax_e.transAxes, fontsize=10)
        ax_e.axis('off')

os.makedirs(SCENARIO_EVAL, exist_ok=True)
plt.savefig(os.path.join(SCENARIO_EVAL, "sample_original_vs_enhanced.png"),
            dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved -> {SCENARIO_EVAL}/sample_original_vs_enhanced.png")
'''

FASE3_CODE = r'''#@title Fase 3 · Training
# Set FORCE_RETRAIN = True to retrain even if best.pt already exists.
# (This will DELETE the previous run folder and retrain from scratch.)
FORCE_RETRAIN = False  # @param {type:"boolean"}

from src.training.train_yolo import train_yolo, get_best_weights
import shutil

data_yaml = (
    os.path.join(SCENARIO_DIR, "enhanced", "dataset.yaml")
    if enhancer_name and enhancer_name.lower() != "none"
    else os.path.join(PREPARED_DIR, "ExDark_yolo", "dataset.yaml")
)
assert os.path.exists(data_yaml), f"dataset.yaml not found: {data_yaml}"

if FORCE_RETRAIN and os.path.exists(SCENARIO_RUNS):
    print(f"FORCE_RETRAIN=True -- removing previous run: {SCENARIO_RUNS}")
    shutil.rmtree(SCENARIO_RUNS)
    os.makedirs(SCENARIO_RUNS, exist_ok=True)

# NOTE: output_dir=SCENARIO_DIR + run_name="runs" → files land directly in
#       scenarios/<name>/runs/  (no redundant subfolder inside runs)
result = train_yolo(dataset_yaml=data_yaml, scenario_name=SCENARIO_NAME,
                    output_dir=SCENARIO_DIR, run_name="runs",
                    config=cfg, force=FORCE_RETRAIN)

best = get_best_weights(SCENARIO_RUNS)
print(f"\nbest.pt  : {best}")
'''

TRAIN_CURVES_CODE = r'''#@title Fase 3.5 · Training Curves & Figures
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob as _glob

run_dir_train = SCENARIO_RUNS
results_csv = os.path.join(run_dir_train, "results.csv")
if not os.path.exists(results_csv):
    candidates = _glob.glob(os.path.join(run_dir_train, "**", "results.csv"), recursive=True)
    if candidates:
        results_csv = candidates[0]

if not os.path.exists(results_csv):
    print(f"results.csv not found in {run_dir_train}. Skipping training curves.")
else:
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    epochs = df.index + 1

    # ── Figure 1: Train vs Val Loss + Metrics ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Curves — {SCENARIO_NAME}", fontsize=16, fontweight='bold')

    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Classification Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]
    for i, (tr, vl, title) in enumerate(loss_pairs):
        if tr in df.columns and vl in df.columns:
            axes[0, i].plot(epochs, df[tr], label="Train", linewidth=2, color='#2196F3')
            axes[0, i].plot(epochs, df[vl], label="Val", linewidth=2,
                            linestyle='--', color='#F44336')
            axes[0, i].set_title(title, fontsize=12, fontweight='bold')
            axes[0, i].set_xlabel("Epoch"); axes[0, i].set_ylabel("Loss")
            axes[0, i].legend(fontsize=10); axes[0, i].grid(True, alpha=0.3)

    metric_items = [
        ("metrics/precision(B)", "Precision", '#4CAF50'),
        ("metrics/recall(B)",    "Recall",    '#FF9800'),
        ("metrics/mAP50(B)",     "mAP@0.5",   '#9C27B0'),
    ]
    for i, (col, title, color) in enumerate(metric_items):
        if col in df.columns:
            axes[1, i].plot(epochs, df[col], linewidth=2, color=color)
            axes[1, i].set_title(title, fontsize=12, fontweight='bold')
            axes[1, i].set_xlabel("Epoch"); axes[1, i].set_ylabel(title)
            axes[1, i].grid(True, alpha=0.3); axes[1, i].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(SCENARIO_EVAL, "training_curves.png"),
                dpi=150, bbox_inches='tight')
    plt.show()

    # ── Figure 2: mAP@0.5 vs mAP@0.5:0.95 ──
    if "metrics/mAP50(B)" in df.columns and "metrics/mAP50-95(B)" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(epochs, df["metrics/mAP50(B)"],
                 label="mAP@0.5", linewidth=2, color='#9C27B0')
        ax2.plot(epochs, df["metrics/mAP50-95(B)"],
                 label="mAP@0.5:0.95", linewidth=2, color='#E91E63')
        ax2.set_title(f"mAP Progression — {SCENARIO_NAME}",
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP")
        ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(SCENARIO_EVAL, "mAP_progression.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()

    # ── Figure 3: Learning Rate Schedule ──
    lr_cols = [c for c in df.columns if c.startswith("lr/")]
    if lr_cols:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        for c in lr_cols:
            ax3.plot(epochs, df[c], label=c, linewidth=1.5)
        ax3.set_title(f"Learning Rate Schedule — {SCENARIO_NAME}",
                      fontsize=14, fontweight='bold')
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("Learning Rate")
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SCENARIO_EVAL, "lr_schedule.png"),
                    dpi=150, bbox_inches='tight')
        plt.show()

    print(f"Saved custom training curves -> {SCENARIO_EVAL}/")

# ═══════════════════════════════════════════════════════════════
#  Ultralytics-generated results.png  (comprehensive overview)
# ═══════════════════════════════════════════════════════════════
results_png = os.path.join(run_dir_train, "results.png")
if os.path.exists(results_png):
    fig_r, ax_r = plt.subplots(figsize=(20, 9))
    ax_r.imshow(mpimg.imread(results_png))
    ax_r.set_title(f"Ultralytics Training Summary — {SCENARIO_NAME}",
                   fontsize=15, fontweight='bold')
    ax_r.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print(f"results.png not found in {run_dir_train}")

# ═══════════════════════════════════════════════════════════════
#  Train batch samples  (shows augmented training images)
# ═══════════════════════════════════════════════════════════════
train_batches = sorted(_glob.glob(os.path.join(run_dir_train, "train_batch*.jpg")))
if train_batches:
    n_tb = len(train_batches)
    fig_tb, axes_tb = plt.subplots(1, n_tb, figsize=(7 * n_tb, 7))
    if n_tb == 1: axes_tb = [axes_tb]
    fig_tb.suptitle(f"Training Batch Samples — {SCENARIO_NAME}",
                    fontsize=14, fontweight='bold')
    for ax, path in zip(axes_tb, train_batches):
        ax.imshow(mpimg.imread(path))
        ax.set_title(os.path.basename(path), fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
'''

FASE4_CODE = r'''#@title Fase 4 · Detection Evaluation
# Set FORCE_EVALUATION = True to re-evaluate even if metrics.json exists.
FORCE_EVALUATION = False  # @param {type:"boolean"}

from src.evaluation.eval_yolo import evaluate_yolo
import pandas as pd

weights_path = get_best_weights(SCENARIO_RUNS)

results  = evaluate_yolo(weights_path=weights_path, dataset_yaml=data_yaml,
                         output_dir=SCENARIO_EVAL, scenario_name=SCENARIO_NAME,
                         force=FORCE_EVALUATION)
overall  = results.get("overall", {})

print(f"\n{'='*50}")
print(f"Detection Results — {SCENARIO_NAME}")
print(f"{'='*50}")
print(f"  mAP@0.5      : {overall.get('mAP_50',0):.4f}")
print(f"  mAP@0.5:0.95 : {overall.get('mAP_50_95',0):.4f}")
print(f"  Precision    : {overall.get('precision',0):.4f}")
print(f"  Recall       : {overall.get('recall',0):.4f}")

# ── Per-class table (defensive: handles both dict and float values) ──
per_cls = results.get("per_class", {})
if per_cls:
    def _extract(v, key, default=0):
        if isinstance(v, dict):
            return v.get(key, default)
        return float(v) if key == "mAP_50" else default

    cls_rows = [{
        "Class": k,
        "AP@0.5": f"{_extract(v, 'mAP_50'):.4f}",
        "AP@0.5:0.95": f"{_extract(v, 'mAP_50_95'):.4f}",
    } for k, v in per_cls.items()]

    df_cls = pd.DataFrame(cls_rows).set_index("Class")
    display(df_cls)
'''

# ╔═══════════════════════════════════════════════════════════════════╗
# ║ Fase 4.5 — Detection GT vs Pred  (tighter spacing)              ║
# ╚═══════════════════════════════════════════════════════════════════╝

DET_VIZ_CODE = r'''#@title Fase 4.5 · Detection Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from ultralytics import YOLO
import numpy as np
import glob as _glob
import torch

CLASS_NAMES = {
    0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus",
    4: "Car", 5: "Cat", 6: "Chair", 7: "Cup",
    8: "Dog", 9: "Motorbike", 10: "People", 11: "Table",
}
BOX_COLORS = {i: plt.cm.tab20(i / 12) for i in range(12)}

# ── Determine layout ──
has_enh = enhancer_name and str(enhancer_name).lower() != "none"

if has_enh:
    raw_test_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")
    enh_test_dir = os.path.join(SCENARIO_DIR, "enhanced", "images", "test")
    test_img_dir = enh_test_dir
    n_samples, n_cols = 6, 4   # Original | Enhanced | Pred | GT
else:
    test_img_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")
    n_samples, n_cols = 9, 2   # Pred | GT

test_lbl_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "labels", "test")

_w = get_best_weights(SCENARIO_RUNS)
_model = YOLO(_w)

sample_imgs = sorted(_glob.glob(os.path.join(test_img_dir, "*.*")))[:n_samples]

def _draw_boxes(ax, boxes_data, mode="gt"):
    for b in boxes_data:
        cid = b["cid"]
        color = BOX_COLORS.get(cid, (1, 0, 0, 1))
        rect = plt.Rectangle(
            (b["x1"], b["y1"]), b["w"], b["h"],
            linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        label = CLASS_NAMES.get(cid, str(cid))
        if mode == "pred":
            label = f"{label} {b['conf']:.2f}"
        ax.text(b["x1"], max(b["y1"] - 3, 0), label,
                fontsize=6, color="white",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor=color, alpha=0.85))

def _parse_gt(label_path, h, w):
    boxes = []
    if not os.path.exists(label_path): return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cid = int(parts[0])
            xc, yc, bw, bh = (float(x) for x in parts[1:5])
            boxes.append({"cid": cid,
                          "x1": (xc - bw/2)*w, "y1": (yc - bh/2)*h,
                          "w": bw*w, "h": bh*h})
    return boxes

def _parse_pred(results):
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        boxes.append({"cid": int(box.cls[0]),
                      "x1": x1, "y1": y1,
                      "w": x2 - x1, "h": y2 - y1,
                      "conf": float(box.conf[0])})
    return boxes

# ─────────────────────────────────────────────────────────────────
# Layout (left to right):
#   Enhanced: Original | Enhanced | Prediction | Ground Truth
#   Baseline: Prediction | Ground Truth
# ─────────────────────────────────────────────────────────────────
n = len(sample_imgs)
fig_w = 20 if has_enh else 16
fig, axes = plt.subplots(n, n_cols, figsize=(fig_w, 4.0 * n),
                         gridspec_kw={"wspace": 0.02, "hspace": 0.06})

if has_enh:
    suptitle = (f"Detection Pipeline — {SCENARIO_NAME}\n"
                "Original  |  Enhanced  |  Prediction (conf >= 0.25)  |  Ground Truth")
else:
    suptitle = (f"Detection Results — {SCENARIO_NAME}\n"
                "Prediction (conf >= 0.25)  |  Ground Truth")

fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.998)

if n == 1:
    axes = axes.reshape(1, n_cols)

for idx, img_path in enumerate(sample_imgs):
    fname = os.path.basename(img_path)
    img = mpimg.imread(img_path)
    h, w = img.shape[:2]
    lbl_path = os.path.join(test_lbl_dir, os.path.splitext(fname)[0] + ".txt")

    if has_enh:
        # Col 0: Original (raw low-light, no boxes)
        raw_path = os.path.join(raw_test_dir, fname)
        if os.path.exists(raw_path):
            axes[idx, 0].imshow(mpimg.imread(raw_path))
        axes[idx, 0].set_title(f"Original: {fname}", fontsize=8, loc="left")
        axes[idx, 0].axis("off")

        # Col 1: Enhanced (no boxes)
        axes[idx, 1].imshow(img, aspect="equal")
        axes[idx, 1].set_title("Enhanced", fontsize=8, loc="left")
        axes[idx, 1].axis("off")

        pred_col, gt_col = 2, 3
    else:
        pred_col, gt_col = 0, 1

    # Prediction column
    axes[idx, pred_col].imshow(img, aspect="equal")
    pred_results = _model.predict(img_path, conf=0.25, verbose=False)
    _draw_boxes(axes[idx, pred_col], _parse_pred(pred_results), mode="pred")
    lbl_pred = f"Pred: {fname}" if not has_enh else "Prediction"
    axes[idx, pred_col].set_title(lbl_pred, fontsize=8, loc="left")
    axes[idx, pred_col].axis("off")

    # Ground Truth column
    axes[idx, gt_col].imshow(img, aspect="equal")
    _draw_boxes(axes[idx, gt_col], _parse_gt(lbl_path, h, w), mode="gt")
    lbl_gt = f"GT: {fname}" if not has_enh else "Ground Truth"
    axes[idx, gt_col].set_title(lbl_gt, fontsize=8, loc="left")
    axes[idx, gt_col].axis("off")

plt.subplots_adjust(top=0.97)
save_path = os.path.join(SCENARIO_EVAL, "detection_samples.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")

del _model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''

# ╔═══════════════════════════════════════════════════════════════════╗
# ║ Fase 4.55 — Validation batch grid  (tighter spacing)            ║
# ╚═══════════════════════════════════════════════════════════════════╝

VAL_BATCH_CODE = r'''#@title Fase 4.55 · Validation Batch Grid (Pred vs Labels)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob as _glob

val_pred   = sorted(_glob.glob(os.path.join(SCENARIO_RUNS, "val_batch*_pred.jpg")))
val_labels = sorted(_glob.glob(os.path.join(SCENARIO_RUNS, "val_batch*_labels.jpg")))

if val_pred and val_labels:
    n_cols = max(len(val_pred), len(val_labels))

    # Config:  figsize=(8*n_cols, 14)  top=0.94  hspace=0.04  wspace=0.02
    fig, axes = plt.subplots(2, n_cols, figsize=(8 * n_cols, 14))
    fig.suptitle(
        f"Validation Batches — {SCENARIO_NAME}\n"
        "Top row: Predictions  |  Bottom row: Ground Truth Labels",
        fontsize=15, fontweight='bold', y=0.998)

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_cols):
        # Top row: predictions
        if i < len(val_pred):
            axes[0, i].imshow(mpimg.imread(val_pred[i]))
            axes[0, i].set_title(os.path.basename(val_pred[i]), fontsize=10)
        axes[0, i].axis('off')

        # Bottom row: labels (ground truth)
        if i < len(val_labels):
            axes[1, i].imshow(mpimg.imread(val_labels[i]))
            axes[1, i].set_title(os.path.basename(val_labels[i]), fontsize=10)
        axes[1, i].axis('off')

    plt.subplots_adjust(top=0.94, hspace=0.04, wspace=0.02)
    save_path = os.path.join(SCENARIO_EVAL, "val_batch_pred_vs_labels.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved -> {save_path}")
else:
    print(f"No val_batch images found in {SCENARIO_RUNS}")
    print("These are generated during Ultralytics training (Fase 3).")
'''

CM_CODE = r'''#@title Fase 4.6 · Confusion Matrix  (from Ultralytics training output)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil

# Ultralytics generates these during training in the runs directory
cm_path      = os.path.join(SCENARIO_RUNS, "confusion_matrix.png")
cm_norm_path = os.path.join(SCENARIO_RUNS, "confusion_matrix_normalized.png")

found = False

if os.path.exists(cm_path) and os.path.exists(cm_norm_path):
    found = True
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle(f"Confusion Matrix — {SCENARIO_NAME}",
                 fontsize=16, fontweight='bold')

    axes[0].imshow(mpimg.imread(cm_path))
    axes[0].set_title("Counts", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(mpimg.imread(cm_norm_path))
    axes[1].set_title("Normalized", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Copy to evaluation folder for archival
    shutil.copy2(cm_path, os.path.join(SCENARIO_EVAL, "confusion_matrix.png"))
    shutil.copy2(cm_norm_path, os.path.join(SCENARIO_EVAL, "confusion_matrix_normalized.png"))
    print(f"Copied confusion matrices -> {SCENARIO_EVAL}/")

elif os.path.exists(cm_path):
    found = True
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(mpimg.imread(cm_path))
    ax.set_title(f"Confusion Matrix — {SCENARIO_NAME}",
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    shutil.copy2(cm_path, os.path.join(SCENARIO_EVAL, "confusion_matrix.png"))

if not found:
    print(f"No confusion matrix images found in {SCENARIO_RUNS}")
    print("These are generated by Ultralytics during training (Fase 3).")
    print("If QUICK_TEST=True (1 epoch), CM may not be generated.")
'''

FASE5_CODE = r'''#@title Fase 5 · Image Quality Metrics  (auto-skip if summary.json exists)
from src.evaluation.nr_metrics import compute_nr_metrics

raw_test_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")
test_dir = (
    os.path.join(SCENARIO_DIR, "enhanced", "images", "test")
    if enhancer_name and enhancer_name.lower() != "none"
    else raw_test_dir
)
raw_dir_for_loe = raw_test_dir if (enhancer_name and enhancer_name.lower() != "none") else None

nr = compute_nr_metrics(images_dir=test_dir, output_dir=SCENARIO_EVAL,
                         scenario_name=SCENARIO_NAME, raw_images_dir=raw_dir_for_loe,
                         force=FORCE_EVALUATION)
print(f"\nNR-IQA — {SCENARIO_NAME}")
print(f"  NIQE (lower=better)    : {nr.get('niqe_mean','N/A')}")
print(f"  BRISQUE (lower=better) : {nr.get('brisque_mean','N/A')}")
print(f"  LOE (lower=better)     : {nr.get('loe_mean','N/A')}")
'''

FASE6_CODE = r'''#@title Fase 6 · Latency & FLOPs  (auto-skip if cached)
import torch
from src.evaluation.latency import measure_latency
from src.evaluation.flops   import compute_all_flops

raw_test_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")
weights_path = get_best_weights(SCENARIO_RUNS)

enhancer_obj = None
if enhancer_name and enhancer_name.lower() != "none":
    from src.enhancement.run_enhancement import get_enhancer
    cache_dir    = os.path.join(OUTPUT_ROOT, "model_cache")
    enhancer_obj = get_enhancer(enhancer_name, cache_dir)
    enhancer_obj.load_model()

lat  = measure_latency(yolo_weights=weights_path, output_dir=SCENARIO_EVAL,
                        scenario_name=SCENARIO_NAME, test_images_dir=raw_test_dir,
                        enhancer=enhancer_obj,
                        num_images=cfg.get("latency",{}).get("iterations",200),
                        warmup=cfg.get("latency",{}).get("warmup",50),
                        force=FORCE_EVALUATION)

flops = compute_all_flops(yolo_weights=weights_path, output_dir=SCENARIO_EVAL,
                           scenario_name=SCENARIO_NAME,
                           enhancer_model=enhancer_obj.model if enhancer_obj else None,
                           enhancer_name=enhancer_name if enhancer_obj else None,
                           force=FORCE_EVALUATION)

print(f"\nLatency — {SCENARIO_NAME}")
print(f"  T_enhance : {lat.get('T_enhance_ms_mean',0):.2f} ms")
print(f"  T_detect  : {lat.get('T_detect_ms_mean',0):.2f} ms")
print(f"  T_total   : {lat.get('T_total_ms_mean',0):.2f} ms")
print(f"\nFLOPs — {SCENARIO_NAME}")
print(f"  Enhancer  : {flops.get('enhancer',{}).get('gflops',0) or 0:.2f} GFLOPs")
print(f"  YOLO      : {flops.get('yolo',{}).get('gflops',0) or 0:.2f} GFLOPs")
print(f"  Total     : {flops.get('total_gflops',0) or 0:.2f} GFLOPs")

if enhancer_obj: del enhancer_obj
if torch.cuda.is_available(): torch.cuda.empty_cache()
'''

# ═══════════════════════════════════════════════════════════════════
#  MARKDOWN CELLS
# ═══════════════════════════════════════════════════════════════════

PREVIEW_MD_NO_ENH = ["---\n", "## Fase 2.5: Sample Test Images Preview"]
PREVIEW_MD_ENH = ["---\n", "## Fase 2.5: Sample Test Images — Original vs Enhanced"]
TRAIN_CURVES_MD = [
    "---\n",
    "## Fase 3.5: Training Curves & Figures\n",
    "Visualisasi kurva training untuk analisis **overfitting / underfitting**.\n",
    "- **Row 1**: Train vs Val loss (box, cls, dfl) — gap besar = overfitting\n",
    "- **Row 2**: Precision, Recall, mAP progression per epoch\n",
    "- **results.png**: Grafik lengkap dari Ultralytics\n",
    "- **train_batch**: Sample gambar augmentasi training"
]
DET_VIZ_MD_NO_ENH = [
    "---\n",
    "## Fase 4.5: Detection Samples — Predictions vs Ground Truth\n",
    "Visualisasi 9 sample test images:\n",
    "- **Kolom kiri**: Prediction (bounding box + confidence score)\n",
    "- **Kolom kanan**: Ground Truth (bounding box anotasi)\n",
    "\n",
    "Urutan: **Prediction \u2192 Ground Truth**"
]
DET_VIZ_MD_ENH = [
    "---\n",
    "## Fase 4.5: Detection Pipeline Visualization\n",
    "Visualisasi 6 sample test images — pipeline lengkap:\n",
    "- **Kolom 1**: Original (low-light, tanpa bounding box)\n",
    "- **Kolom 2**: Enhanced (setelah LLIE, tanpa bounding box)\n",
    "- **Kolom 3**: Prediction (bounding box + confidence score)\n",
    "- **Kolom 4**: Ground Truth (bounding box anotasi)\n",
    "\n",
    "Urutan konsisten: **Original \u2192 Enhanced \u2192 Prediction \u2192 Ground Truth**"
]
VAL_BATCH_MD = [
    "---\n",
    "## Fase 4.55: Validation Batch — Predictions vs Labels\n",
    "Visualisasi output batch validasi dari Ultralytics training:\n",
    "- **Baris atas**: Prediksi model pada validation batch\n",
    "- **Baris bawah**: Ground truth labels pada batch yang sama"
]
CM_MD = [
    "---\n",
    "## Fase 4.6: Confusion Matrix\n",
    "Confusion matrix dari output training Ultralytics (bukan recompute).\n",
    "Menampilkan versi **counts** dan **normalized**."
]


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def mk_md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def mk_code(text, hide=False):
    """Create a code cell.  If hide=True, add cellView:'form' for Colab auto-collapse."""
    meta = {}
    if hide:
        meta["cellView"] = "form"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": _lines(text),
    }


# ═══════════════════════════════════════════════════════════════════
#  NOTEBOOK BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_notebook(scenario_info):
    name = scenario_info["name"]
    key = scenario_info["key"]
    has_enh = scenario_info["has_enhancement"]

    cells = []

    # ── Header ──
    enh_desc = "SKIPPED — baseline" if not has_enh else f"menggunakan {name.split('_',1)[-1] if '_' in name else name}"
    cells.append(mk_md([
        f"# Scenario Notebook: {name}\n",
        "\n",
        f"Pipeline Fase 1–6 untuk skenario **{name}**.\n",
        "\n",
        "| Fase | Deskripsi | Auto-skip? |\n",
        "|------|-----------|------------|\n",
        "| 1    | Data Preparation (parse, convert, build YOLO) | jika output sudah ada |\n",
        f"| 2    | LLIE Enhancement ({enh_desc}) | jika enhanced dir sudah ada |\n",
        "| 3    | YOLOv11n Training | jika best.pt sudah ada |\n",
        "| 4    | Detection Evaluation (mAP, P, R) | jika metrics.json sudah ada |\n",
        "| 5    | NR-IQA (NIQE, BRISQUE, LOE) | jika summary.json sudah ada |\n",
        "| 6    | Latency & FLOPs | jika cache sudah ada |\n",
        "\n",
        "Setelah **semua 4 skenario** selesai buka **comparison.ipynb**.\n",
    ]))

    # ── 0. Setup (visible — early cells stay open) ──
    cells.append(mk_md(["## 0. Setup"]))

    cells.append(mk_code(f'''#@title 0.1 · Environment Setup & Clone Repo
import os, subprocess, sys, shutil

# ── Config ───────────────────────────────────────────────────────
QUICK_TEST  = True   # @param {{type:"boolean"}}
REPO_URL    = "https://github.com/Otachiking/Object-Detection-ExDARK-with-LLIE.git"
SCENARIO_KEY  = "{key}"   # DO NOT CHANGE
SCENARIO_NAME = "{name}"  # DO NOT CHANGE

# ── Detect Environment ───────────────────────────────────────────
_IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
_IS_COLAB  = not _IS_KAGGLE and os.path.exists("/content")

if _IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_DIR = "/content/TA-IQBAL-ObjectDetectionExDARKwithLLIE"
    print("[ENV] Google Colab")
elif _IS_KAGGLE:
    REPO_DIR = "/kaggle/working/TA-IQBAL-ObjectDetectionExDARKwithLLIE"
    print("[ENV] Kaggle Notebook")
    print("  -> Pastikan ExDark dataset ditambahkan sebagai Input Dataset")
    print("     dengan nama 'exdark-dataset'")
else:
    raise RuntimeError(
        "Notebook ini dirancang untuk Google Colab atau Kaggle.\\n"
        "Jalankan di salah satu platform tersebut."
    )

# ── Clone / Pull ─────────────────────────────────────────────────
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("Repo exists — resetting to origin/main ...")
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"], check=True)
else:
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    print("Cloning repo ...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"\\nScenario  : {{SCENARIO_NAME}}")
print(f"Quick test: {{QUICK_TEST}}")'''))

    cells.append(mk_code(r'''#@title 0.2 · Install Dependencies
!pip install -q ultralytics pyiqa thop fvcore scipy pandas pyyaml seaborn tqdm gdown huggingface_hub

import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram  = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"VRAM     : {vram/1e9:.1f} GB")
else:
    print("No GPU — Runtime > Change runtime type > T4 GPU")'''))

    cells.append(mk_code(SETUP_CONFIG_CODE))

    # ── Fase 1 (visible) ──
    cells.append(mk_md(["---\n", "## Fase 1: Data Preparation"]))
    cells.append(mk_code(FASE1_CODE))

    # ── Fase 2 (visible) ──
    cells.append(mk_md(["---\n", "## Fase 2: Image Enhancement"]))
    if has_enh:
        cells.append(mk_code(FASE2_ENH_CODE))
    else:
        cells.append(mk_code(FASE2_SKIP_CODE))

    # ── Fase 2.5+ → hide=True (auto-collapse in Colab) ──
    if has_enh:
        cells.append(mk_md(PREVIEW_MD_ENH))
        cells.append(mk_code(PREVIEW_ENH, hide=True))
    else:
        cells.append(mk_md(PREVIEW_MD_NO_ENH))
        cells.append(mk_code(PREVIEW_NO_ENH, hide=True))

    # ── Fase 3 ──
    cells.append(mk_md(["---\n", "## Fase 3: Training"]))
    cells.append(mk_code(FASE3_CODE, hide=True))

    # ── Fase 3.5 ──
    cells.append(mk_md(TRAIN_CURVES_MD))
    cells.append(mk_code(TRAIN_CURVES_CODE, hide=True))

    # ── Fase 4 ──
    cells.append(mk_md(["---\n", "## Fase 4: Detection Evaluation"]))
    cells.append(mk_code(FASE4_CODE, hide=True))

    # ── Fase 4.5 ──
    if has_enh:
        cells.append(mk_md(DET_VIZ_MD_ENH))
    else:
        cells.append(mk_md(DET_VIZ_MD_NO_ENH))
    cells.append(mk_code(DET_VIZ_CODE, hide=True))

    # ── Fase 4.55 ──
    cells.append(mk_md(VAL_BATCH_MD))
    cells.append(mk_code(VAL_BATCH_CODE, hide=True))

    # ── Fase 4.6 ──
    cells.append(mk_md(CM_MD))
    cells.append(mk_code(CM_CODE, hide=True))

    # ── Fase 5 ──
    cells.append(mk_md(["---\n", "## Fase 5: Image Quality Metrics"]))
    cells.append(mk_code(FASE5_CODE, hide=True))

    # ── Fase 6 ──
    cells.append(mk_md(["---\n", "## Fase 6: Latency & FLOPs"]))
    cells.append(mk_code(FASE6_CODE, hide=True))

    # ── Done ──
    cells.append(mk_code(f'''#@title Done — {name} complete
print("=" * 60)
print(f"  {{SCENARIO_NAME}} — all Fase 1-6 complete")
print(f"  Results saved under:")
print(f"    {{SCENARIO_DIR}}/")
print(f"    ├── runs/       (training weights & figures)")
print(f"    └── evaluation/ (metrics, charts)")
print()
print("  Next steps:")
print("    -> Run the other scenario notebooks")
print("    -> After all 4 scenarios done, open comparison.ipynb")
print("=" * 60)'''))

    return cells


def main():
    for filename, info in SCENARIOS.items():
        nb_path = os.path.join(NOTEBOOKS_DIR, filename)

        if os.path.exists(nb_path):
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
        else:
            nb = {
                "nbformat": 4,
                "nbformat_minor": 2,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.10.0"},
                },
                "cells": [],
            }

        nb["cells"] = build_notebook(info)

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  -> Rebuilt {filename} ({len(nb['cells'])} cells)")

    print("\nDone — all 4 notebooks rebuilt.")


if __name__ == "__main__":
    main()
