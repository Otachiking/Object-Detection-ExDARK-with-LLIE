import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from ultralytics import YOLO

def plot_sample_images(raw_test_dir, enh_test_dir, output_dir, scenario_name, enhancer_name=None):
    os.makedirs(output_dir, exist_ok=True)
    raw_paths = sorted(glob.glob(os.path.join(raw_test_dir, "*.*")))[:9]
    if not raw_paths:
        print(f"[WARN] No test images found in {raw_test_dir}")
        return

    images_raw, aspects, fnames = [], [], []
    for p in raw_paths:
        img = mpimg.imread(p)
        images_raw.append(img)
        aspects.append(img.shape[1] / img.shape[0])
        fnames.append(os.path.basename(p))

    n_cols = 3
    if not enh_test_dir or str(enhancer_name).lower() == "none":
        # S1: Raw only (3x3 grid)
        n_rows = (len(images_raw) + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(18, 4.0 * n_rows + 1.8))
        fig.suptitle(f"Sample Test Images — {scenario_name}\n(No Enhancement — Raw Low-Light)", fontsize=16, fontweight='bold', y=0.99)
        gs = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.20, top=0.93, bottom=0.02, left=0.02, right=0.98)
        
        for r in range(n_rows):
            s = r * n_cols
            e = min(s + n_cols, len(images_raw))
            wr = list(aspects[s:e]) + [1.0] * (n_cols - (e - s))
            inner = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs[r], width_ratios=wr, wspace=0.04)
            for col in range(n_cols):
                ax = fig.add_subplot(inner[0, col])
                if col < (e - s):
                    ax.imshow(images_raw[s + col])
                    ax.set_title(fnames[s + col], fontsize=9, pad=6)
                ax.axis('off')
        
        save_path = os.path.join(output_dir, "sample_test_images.png")
    else:
        # S2/3/4: Raw vs Enhanced paired
        n_groups = (len(images_raw) + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(18, 3.5 * n_groups * 2 + 2.0))
        fig.suptitle(f"Original (Low-Light) vs Enhanced ({enhancer_name}) — {scenario_name}", fontsize=16, fontweight='bold', y=0.99)
        outer = gridspec.GridSpec(n_groups, 1, figure=fig, hspace=0.28, top=0.95, bottom=0.02, left=0.02, right=0.98)
        
        for g in range(n_groups):
            s = g * n_cols
            e = min(s + n_cols, len(images_raw))
            wr = list(aspects[s:e]) + [1.0] * (n_cols - (e - s))
            inner = gridspec.GridSpecFromSubplotSpec(2, n_cols, subplot_spec=outer[g], width_ratios=wr, hspace=0.18, wspace=0.04)
            for col in range(n_cols):
                ax_o = fig.add_subplot(inner[0, col])
                ax_e = fig.add_subplot(inner[1, col])
                if col < (e - s):
                    ax_o.imshow(images_raw[s + col])
                    ax_o.set_title(f"Original: {fnames[s + col]}", fontsize=8, pad=6)
                    enh_path = os.path.join(enh_test_dir, fnames[s + col])
                    if os.path.exists(enh_path):
                        ax_e.imshow(mpimg.imread(enh_path))
                        ax_e.set_title(f"Enhanced: {fnames[s + col]}", fontsize=8, pad=6)
                    else:
                        ax_e.text(0.5, 0.5, "Enhanced not found", ha='center', va='center', transform=ax_e.transAxes, fontsize=10)
                ax_o.axis('off')
                ax_e.axis('off')
        save_path = os.path.join(output_dir, "sample_original_vs_enhanced.png")

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {save_path}")
    return save_path

def plot_training_curves(run_dir_train, output_dir, scenario_name):
    os.makedirs(output_dir, exist_ok=True)
    out_paths = []
    results_csv = os.path.join(run_dir_train, "results.csv")
    if not os.path.exists(results_csv):
        candidates = glob.glob(os.path.join(run_dir_train, "**", "results.csv"), recursive=True)
        if candidates:
            results_csv = candidates[0]
            
    if not os.path.exists(results_csv):
        print(f"results.csv not found in {run_dir_train}. Skipping custom curves.")
    else:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        epochs = df.index + 1
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Training Curves — {scenario_name}", fontsize=16, fontweight='bold')
        loss_pairs = [
            ("train/box_loss", "val/box_loss", "Box Loss"),
            ("train/cls_loss", "val/cls_loss", "Classification Loss"),
            ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
        ]
        for i, (tr, vl, title) in enumerate(loss_pairs):
            if tr in df.columns and vl in df.columns:
                axes[0, i].plot(epochs, df[tr], label="Train", linewidth=2, color='#2196F3')
                axes[0, i].plot(epochs, df[vl], label="Val", linewidth=2, linestyle='--', color='#F44336')
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
        p1 = os.path.join(output_dir, "training_curves.png")
        plt.savefig(p1, dpi=150, bbox_inches='tight')
        plt.close()
        out_paths.append(p1)
        
        if "metrics/mAP50(B)" in df.columns and "metrics/mAP50-95(B)" in df.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", linewidth=2, color='#9C27B0')
            ax2.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", linewidth=2, color='#E91E63')
            ax2.set_title(f"mAP Progression — {scenario_name}", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP")
            ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 1.05)
            plt.tight_layout()
            p2 = os.path.join(output_dir, "mAP_progression.png")
            plt.savefig(p2, dpi=150, bbox_inches='tight')
            plt.close()
            out_paths.append(p2)
            
    # Copy generated summaries
    ultralytics_results = os.path.join(run_dir_train, "results.png")
    if os.path.exists(ultralytics_results):
        fig_r, ax_r = plt.subplots(figsize=(20, 9))
        ax_r.imshow(mpimg.imread(ultralytics_results))
        ax_r.set_title(f"Ultralytics Training Summary — {scenario_name}", fontsize=15, fontweight='bold')
        ax_r.axis('off')
        plt.tight_layout()
        save_path = os.path.join(output_dir, "ultralytics_results.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved -> {save_path}")
        out_paths.append(save_path)
        
    return out_paths

def plot_detection_results(weights_path, test_img_dir, test_lbl_dir, output_dir, scenario_name):
    os.makedirs(output_dir, exist_ok=True)
    CLASS_NAMES = {
        0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus",
        4: "Car", 5: "Cat", 6: "Chair", 7: "Cup",
        8: "Dog", 9: "Motorbike", 10: "People", 11: "Table",
    }
    BOX_COLORS = {i: plt.cm.tab20(i / 12) for i in range(12)}
    
    if not os.path.exists(weights_path):
        print(f"[WARN] Weights not found: {weights_path}")
        return
        
    model = YOLO(weights_path)
    sample_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))[:9]
    if not sample_imgs:
        print("[WARN] No images to predict!")
        return

    def _draw_boxes(ax, boxes_data, mode="gt"):
        for b in boxes_data:
            cid = b["cid"]
            color = BOX_COLORS.get(cid, (1, 0, 0, 1))
            rect = plt.Rectangle((b["x1"], b["y1"]), b["w"], b["h"], linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            label = CLASS_NAMES.get(cid, str(cid))
            if mode == "pred":
                label = f"{label} {b['conf']:.2f}"
            ax.text(b["x1"], max(b["y1"] - 3, 0), label, fontsize=7, color="white", bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.85))

    def _parse_gt(label_path, h, w):
        boxes = []
        if not os.path.exists(label_path): return boxes
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cid = int(parts[0])
                xc, yc, bw, bh = (float(x) for x in parts[1:5])
                boxes.append({"cid": cid, "x1": (xc - bw/2)*w, "y1": (yc - bh/2)*h, "w": bw*w, "h": bh*h})
        return boxes

    def _parse_pred(results):
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append({"cid": int(box.cls[0]), "x1": x1, "y1": y1, "w": x2 - x1, "h": y2 - y1, "conf": float(box.conf[0])})
        return boxes

    n = len(sample_imgs)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4.5 * n), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
    fig.suptitle(f"Detection Results — {scenario_name}\nLeft: Ground Truth  |  Right: Prediction (conf >= 0.25)", fontsize=16, fontweight="bold", y=1.0)
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
        pred = model.predict(img_path, conf=0.25, verbose=False)
        _draw_boxes(axes[idx, 1], _parse_pred(pred), mode="pred")
        axes[idx, 1].set_title(f"Pred: {fname}", fontsize=9, loc="left")
        axes[idx, 1].axis("off")
        
    save_path = os.path.join(output_dir, "detection_samples_gt_vs_pred.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")
    return save_path
