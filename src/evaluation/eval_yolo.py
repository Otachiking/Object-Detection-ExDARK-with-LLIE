"""
YOLOv11n evaluation on test set.

Extracts:
- Overall: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- Per-class: mAP@0.5 for each of 12 ExDark classes

Uses best.pt from training run.
Confidence threshold locked at 0.001 (Ultralytics default) for fair comparison.
"""

import os
import json
import torch
import pandas as pd
from typing import Dict, Optional
from ultralytics import YOLO

from src.utils.io import patch_dataset_yaml_path


CLASS_NAMES = {
    0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus",
    4: "Car", 5: "Cat", 6: "Chair", 7: "Cup",
    8: "Dog", 9: "Motorbike", 10: "People", 11: "Table",
}


def evaluate_yolo(
    weights_path: str,
    dataset_yaml: str,
    output_dir: str,
    scenario_name: str,
    split: str = "test",
    conf: float = 0.001,
    iou: float = 0.7,
    device=None,
    imgsz: int = 640,
    force: bool = False,
) -> dict:
    """Evaluate YOLO model on test set.

    Skips evaluation if metrics.json already exists (unless force=True).

    Args:
        weights_path: Path to best.pt
        dataset_yaml: Path to dataset.yaml
        output_dir: Where to save evaluation results
        scenario_name: Scenario identifier
        split: Dataset split to evaluate on
        conf: Confidence threshold (MUST be same for all scenarios)
        iou: NMS IoU threshold
        device: GPU device
        imgsz: Input image size
        force: If True, re-evaluate even if results exist

    Returns:
        Dict with overall and per-class metrics
    """
    # --- Skip logic ---
    json_path = os.path.join(output_dir, "metrics.json")
    if not force and os.path.exists(json_path):
        print(f"\n[SKIP] Evaluation results already exist for {scenario_name}")
        print(f"  Loaded from: {json_path}")
        print(f"  → To re-evaluate, pass force=True")
        with open(json_path, "r") as f:
            cached = json.load(f)
        overall = cached.get("overall", {})
        print(f"  mAP@0.5: {overall.get('mAP_50', 0):.4f} | "
              f"mAP@0.5:0.95: {overall.get('mAP_50_95', 0):.4f}")
        return cached

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    # --- Defensive: ensure dataset.yaml uses absolute path (fixes Windows path issue) ---
    patch_dataset_yaml_path(dataset_yaml)

    print(f"\n{'='*60}")
    print(f"[EVAL] Scenario: {scenario_name}")
    print(f"[EVAL] Weights: {weights_path}")
    print(f"[EVAL] Dataset: {dataset_yaml}")
    print(f"[EVAL] Split: {split}, Conf: {conf}, IoU: {iou}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(weights_path)

    # Run validation on specified split
    results = model.val(
        data=dataset_yaml,
        split=split,
        conf=conf,
        iou=iou,
        device=device,
        imgsz=imgsz,
        project=output_dir,
        name="val_plots",
        exist_ok=True,
        verbose=True,
    )

    # Extract overall metrics
    overall = {
        "mAP_50": float(results.box.map50),           # mAP@0.5
        "mAP_50_95": float(results.box.map),           # mAP@0.5:0.95
        "precision": float(results.box.mp),             # Mean precision
        "recall": float(results.box.mr),                # Mean recall
    }

    # Extract per-class metrics
    per_class = {}
    if hasattr(results.box, "ap50") and results.box.ap50 is not None:
        ap50_per_class = results.box.ap50
        for i, ap in enumerate(ap50_per_class):
            class_name = CLASS_NAMES.get(i, f"class_{i}")
            per_class[class_name] = {
                "mAP_50": float(ap),
            }

    # Also get per-class AP@0.5:0.95 if available
    if hasattr(results.box, "ap") and results.box.ap is not None:
        ap_per_class = results.box.ap
        for i, ap in enumerate(ap_per_class):
            class_name = CLASS_NAMES.get(i, f"class_{i}")
            if class_name in per_class:
                per_class[class_name]["mAP_50_95"] = float(ap)

    # Build complete results dict
    eval_results = {
        "scenario": scenario_name,
        "weights": weights_path,
        "dataset": dataset_yaml,
        "split": split,
        "conf_threshold": conf,
        "iou_threshold": iou,
        "overall": overall,
        "per_class": per_class,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[EVAL] Metrics saved: {json_path}")

    # Per-class CSV
    if per_class:
        csv_path = os.path.join(output_dir, "metrics_per_class.csv")
        df = pd.DataFrame.from_dict(per_class, orient="index")
        df.index.name = "class"
        df.to_csv(csv_path)
        print(f"[EVAL] Per-class CSV: {csv_path}")

    # Print summary
    print(f"\n[EVAL] === {scenario_name} Results ===")
    print(f"  mAP@0.5:      {overall['mAP_50']:.4f}")
    print(f"  mAP@0.5:0.95: {overall['mAP_50_95']:.4f}")
    print(f"  Precision:     {overall['precision']:.4f}")
    print(f"  Recall:        {overall['recall']:.4f}")

    if per_class:
        print(f"\n  Per-class mAP@0.5:")
        for cls_name, metrics in sorted(per_class.items()):
            print(f"    {cls_name:12s}: {metrics.get('mAP_50', 0):.4f}")

    return eval_results


def aggregate_detection_results(
    results_dirs: Dict[str, str],
    output_path: str,
) -> pd.DataFrame:
    """Aggregate detection results from multiple scenarios into a single table.

    Args:
        results_dirs: Dict mapping scenario_name → eval output directory
        output_path: Path to save aggregated CSV

    Returns:
        DataFrame with all scenarios
    """
    rows = []

    for scenario_name, eval_dir in results_dirs.items():
        json_path = os.path.join(eval_dir, "metrics.json")
        if not os.path.exists(json_path):
            print(f"[WARN] Metrics not found for {scenario_name}: {json_path}")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        row = {"scenario": scenario_name}
        row.update(data.get("overall", {}))
        rows.append(row)

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[EVAL] Aggregated results saved: {output_path}")
    print(df.to_string(index=False))

    return df
