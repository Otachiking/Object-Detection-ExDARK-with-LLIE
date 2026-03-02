"""
YOLOv11n training wrapper with config injection.

Ensures:
- ALL scenarios use identical YOLO hyperparameters (fairness)
- Reproducibility via seed + config snapshot + env snapshot
- Clean output organization per scenario
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from src.seed import set_global_seed
from src.config import save_config_snapshot, save_environment_info


def train_yolo(
    dataset_yaml: str,
    scenario_name: str,
    output_dir: str,
    config: dict,
    resume: bool = False,
) -> dict:
    """Train YOLOv11n for a specific scenario.

    Args:
        dataset_yaml: Path to dataset.yaml (Ultralytics format)
        scenario_name: Scenario identifier (e.g., "s1_raw")
        output_dir: Root output directory for runs
        config: Merged configuration dict (base + scenario)
        resume: Resume from last.pt if available

    Returns:
        Training results dict
    """
    yolo_cfg = config.get("yolo", {})
    seed = config.get("seed", 42)

    # Set global seed
    set_global_seed(seed)

    # Determine epochs and batch (respect quick_test mode)
    epochs = yolo_cfg.get("epochs", 100)
    batch = yolo_cfg.get("batch", 16)

    if config.get("quick_test", False):
        epochs = config.get("quick_test_epochs", 1)
        batch = config.get("quick_test_batch", 8)
        print(f"[TRAIN] ⚡ Quick test mode: epochs={epochs}, batch={batch}")

    # Project and name for Ultralytics directory structure
    project_dir = output_dir
    run_name = scenario_name

    print(f"\n{'='*60}")
    print(f"[TRAIN] Scenario: {scenario_name}")
    print(f"[TRAIN] Dataset: {dataset_yaml}")
    print(f"[TRAIN] Output: {project_dir}/{run_name}")
    print(f"[TRAIN] Epochs: {epochs}, Batch: {batch}, Seed: {seed}")
    print(f"[TRAIN] Model: {yolo_cfg.get('model', 'yolo11n.pt')}")
    print(f"{'='*60}\n")

    # Initialize YOLO model
    model_name = yolo_cfg.get("model", "yolo11n.pt")
    model = YOLO(model_name)

    # Train with unified hyperparameters
    results = model.train(
        data=dataset_yaml,
        imgsz=yolo_cfg.get("imgsz", 640),
        epochs=epochs,
        batch=batch,
        seed=seed,
        patience=yolo_cfg.get("patience", 20),
        project=project_dir,
        name=run_name,
        exist_ok=yolo_cfg.get("exist_ok", True),
        device=yolo_cfg.get("device", 0),
        workers=yolo_cfg.get("workers", 2),
        pretrained=yolo_cfg.get("pretrained", True),
        resume=resume,

        # Augmentation — identical for all scenarios
        hsv_h=yolo_cfg.get("hsv_h", 0.015),
        hsv_s=yolo_cfg.get("hsv_s", 0.7),
        hsv_v=yolo_cfg.get("hsv_v", 0.4),
        degrees=yolo_cfg.get("degrees", 0.0),
        translate=yolo_cfg.get("translate", 0.1),
        scale=yolo_cfg.get("scale", 0.5),
        fliplr=yolo_cfg.get("fliplr", 0.5),
        mosaic=yolo_cfg.get("mosaic", 1.0),
        mixup=yolo_cfg.get("mixup", 0.0),
        copy_paste=yolo_cfg.get("copy_paste", 0.0),

        # Validation
        val=True,
    )

    # Save config snapshot and environment info
    run_dir = os.path.join(project_dir, run_name)
    save_config_snapshot(config, run_dir)
    save_environment_info(run_dir)

    # Check if best.pt exists
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    last_pt = os.path.join(run_dir, "weights", "last.pt")

    print(f"\n[TRAIN] ✓ Training complete: {scenario_name}")
    print(f"  best.pt: {'✓' if os.path.exists(best_pt) else '✗'} {best_pt}")
    print(f"  last.pt: {'✓' if os.path.exists(last_pt) else '✗'} {last_pt}")

    return {
        "scenario": scenario_name,
        "run_dir": run_dir,
        "best_pt": best_pt if os.path.exists(best_pt) else None,
        "last_pt": last_pt if os.path.exists(last_pt) else None,
        "epochs_requested": epochs,
    }


def get_best_weights(run_dir: str) -> str:
    """Get path to best.pt for a completed training run.

    Args:
        run_dir: Path to Ultralytics run directory

    Returns:
        Path to best.pt

    Raises:
        FileNotFoundError if best.pt doesn't exist
    """
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        # Fallback to last.pt
        last_pt = os.path.join(run_dir, "weights", "last.pt")
        if os.path.exists(last_pt):
            print(f"[WARN] best.pt not found, using last.pt: {last_pt}")
            return last_pt
        raise FileNotFoundError(f"No weights found in {run_dir}/weights/")
    return best_pt
