"""
Patch v2: Fix detection grid, suppress verbose validation, add FORCE_RETRAIN.
"""
import json
import glob
import copy
import os

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")

# ═══════════════════════════════════════════════════════════════
# FIXED Detection Viz: vertical layout (Nx2: GT left, Pred right)
# ═══════════════════════════════════════════════════════════════
DET_VIZ_CODE = r'''#@title Fase 4.5 · Detection Visualization (GT vs Prediction)
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

# ── Paths ──
if enhancer_name and str(enhancer_name).lower() != "none":
    test_img_dir = os.path.join(
        OUTPUT_ROOT, f"ExDark_enhanced_{enhancer_name}", "images", "test")
else:
    test_img_dir = os.path.join(OUTPUT_ROOT, "ExDark_yolo", "images", "test")
test_lbl_dir = os.path.join(OUTPUT_ROOT, "ExDark_yolo", "labels", "test")

_w = get_best_weights(os.path.join(OUTPUT_ROOT, "runs", SCENARIO_NAME))
_model = YOLO(_w)

sample_imgs = sorted(_glob.glob(os.path.join(test_img_dir, "*.*")))[:9]

def _draw_boxes(ax, boxes_data, mode="gt"):
    """Draw bounding boxes on axes. mode=gt|pred"""
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
                fontsize=7, color="white",
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

# ── Plot: 9 rows x 2 cols (GT left, Pred right) ──
n = len(sample_imgs)
fig, axes = plt.subplots(n, 2, figsize=(16, 4.5 * n),
                         gridspec_kw={"wspace": 0.03, "hspace": 0.12})
fig.suptitle(
    f"Detection Results — {SCENARIO_NAME}\n"
    "Left: Ground Truth  |  Right: Prediction (conf >= 0.25)",
    fontsize=16, fontweight="bold", y=1.0)

if n == 1: axes = axes.reshape(1, 2)

for idx, img_path in enumerate(sample_imgs):
    fname = os.path.basename(img_path)
    img = mpimg.imread(img_path)
    h, w = img.shape[:2]

    # GT
    axes[idx, 0].imshow(img, aspect="equal")
    lbl = os.path.join(test_lbl_dir, os.path.splitext(fname)[0] + ".txt")
    _draw_boxes(axes[idx, 0], _parse_gt(lbl, h, w), mode="gt")
    axes[idx, 0].set_title(f"GT: {fname}", fontsize=9, loc="left")
    axes[idx, 0].axis("off")

    # Pred
    axes[idx, 1].imshow(img, aspect="equal")
    pred = _model.predict(img_path, conf=0.25, verbose=False)
    _draw_boxes(axes[idx, 1], _parse_pred(pred), mode="pred")
    axes[idx, 1].set_title(f"Pred: {fname}", fontsize=9, loc="left")
    axes[idx, 1].axis("off")

plt.tight_layout()
save_path = os.path.join(OUTPUT_ROOT, "evaluation", SCENARIO_NAME,
                         "detection_samples_gt_vs_pred.png")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> evaluation/{SCENARIO_NAME}/detection_samples_gt_vs_pred.png")

del _model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''.strip().split("\n")
DET_VIZ_CODE = [line + "\n" for line in DET_VIZ_CODE]

# ═══════════════════════════════════════════════════════════════
# FIXED Fase 1: suppress verbose validate
# ═══════════════════════════════════════════════════════════════
FASE1_CODE = r'''#@title Fase 1 · Data Preparation  (auto-skip if already done)
from src.data.split_dataset     import parse_split_file
from src.data.convert_exdark    import convert_exdark_to_yolo
from src.data.build_yolo_dataset import build_yolo_dataset
from src.data.validate_dataset  import validate_yolo_dataset

classlist_path = os.path.join(EXDARK_ROOT,
    cfg["paths"]["exdark_structure"]["groundtruth"], "imageclasslist.txt")
img_dir        = os.path.join(EXDARK_ROOT, cfg["paths"]["exdark_structure"]["images"])
gt_dir         = os.path.join(EXDARK_ROOT, cfg["paths"]["exdark_structure"]["groundtruth"])
split_output   = os.path.join(OUTPUT_ROOT, "splits")
labels_dir     = os.path.join(OUTPUT_ROOT, "ExDark_yolo_labels")
yolo_dir       = os.path.join(OUTPUT_ROOT, "ExDark_yolo")

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
print(f"YOLO dir-> {total} images built")

# 1.4 Quick validate (suppress verbose per-class output)
import io as _io, contextlib as _ctx
with _ctx.redirect_stdout(_io.StringIO()):
    vr = validate_yolo_dataset(yolo_dir)
_summary = vr.get("summary", {})
print(f"Validate-> {'PASSED' if vr['valid'] else 'FAILED'} "
      f"({_summary.get('total_images',0)} imgs, {_summary.get('total_objects',0)} objs)")
if not vr["valid"]:
    for sp, sd in vr.get("splits",{}).items():
        if sd.get("error"): print(f"  {sp}: {sd['error']}")
'''.strip().split("\n")
FASE1_CODE = [line + "\n" for line in FASE1_CODE]

# ═══════════════════════════════════════════════════════════════
# FIXED Fase 3: FORCE_RETRAIN toggle
# ═══════════════════════════════════════════════════════════════
FASE3_CODE = r'''#@title Fase 3 · Training
# Set FORCE_RETRAIN = True to retrain even if best.pt already exists.
# (This will DELETE the previous run folder and retrain from scratch.)
FORCE_RETRAIN = False  # @param {type:"boolean"}

from src.training.train_yolo import train_yolo, get_best_weights
import shutil

data_yaml = (
    os.path.join(OUTPUT_ROOT, f"ExDark_enhanced_{enhancer_name}", "dataset.yaml")
    if enhancer_name and enhancer_name.lower() != "none"
    else os.path.join(OUTPUT_ROOT, "ExDark_yolo", "dataset.yaml")
)
assert os.path.exists(data_yaml), f"dataset.yaml not found: {data_yaml}"

runs_dir = os.path.join(OUTPUT_ROOT, "runs")
run_dir  = os.path.join(runs_dir, SCENARIO_NAME)

if FORCE_RETRAIN and os.path.exists(run_dir):
    print(f"FORCE_RETRAIN=True -- removing previous run: {run_dir}")
    shutil.rmtree(run_dir)

result = train_yolo(dataset_yaml=data_yaml, scenario_name=SCENARIO_NAME,
                    output_dir=runs_dir, config=cfg, force=FORCE_RETRAIN)

best = get_best_weights(os.path.join(runs_dir, SCENARIO_NAME))
print(f"\nbest.pt  : {best}")
'''.strip().split("\n")
FASE3_CODE = [line + "\n" for line in FASE3_CODE]


def main():
    for nb_path in sorted(glob.glob(os.path.join(NOTEBOOKS_DIR, "scenario_s*.ipynb"))):
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb["cells"]
        changed = False

        for i, c in enumerate(cells):
            if c["cell_type"] != "code":
                continue
            src = "".join(c.get("source", []))

            # 1. Replace detection viz cell
            if "Fase 4.5" in src and "Detection Visualization" in src:
                c["source"] = copy.deepcopy(DET_VIZ_CODE)
                changed = True
                print(f"  [det-viz] Updated in {os.path.basename(nb_path)}")

            # 2. Replace Fase 1 (suppress verbose validate)
            if "Fase 1" in src and "parse_split_file" in src:
                c["source"] = copy.deepcopy(FASE1_CODE)
                changed = True
                print(f"  [fase1]   Updated in {os.path.basename(nb_path)}")

            # 3. Replace Fase 3 (add FORCE_RETRAIN)
            if "train_yolo(" in src and ("Fase 3" in src or "Training" in src) and "Fase 3.5" not in src:
                c["source"] = copy.deepcopy(FASE3_CODE)
                changed = True
                print(f"  [fase3]   Updated in {os.path.basename(nb_path)}")

        if changed:
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"  -> Saved {os.path.basename(nb_path)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
