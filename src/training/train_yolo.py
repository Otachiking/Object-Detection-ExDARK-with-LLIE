"""
YOLOv11n training wrapper with config injection.

Ensures:
- ALL scenarios use identical YOLO hyperparameters (fairness)
- Reproducibility via seed + config snapshot + env snapshot
- Clean output organization per scenario
"""

import os
import json
import time
import yaml
import torch
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from src.seed import set_global_seed
from src.config import save_config_snapshot, save_environment_info
from src.utils.io import patch_dataset_yaml_path


def train_yolo(
    dataset_yaml: str,
    scenario_name: str,
    output_dir: str,
    config: dict,
    run_name: str = None,
    resume: bool = False,
    force: bool = False,
) -> dict:
    """Train YOLOv11n for a specific scenario.

    Skips training if best.pt already exists (unless force=True).

    Args:
        dataset_yaml: Path to dataset.yaml (Ultralytics format)
        scenario_name: Scenario identifier (e.g., "S1_Raw") — used for logging
        output_dir: Root output directory (Ultralytics 'project' dir)
        config: Merged configuration dict (base + scenario)
        run_name: Ultralytics 'name' dir inside output_dir.
                  Defaults to scenario_name if None.
        resume: Resume from last.pt if available
        force: If True, retrain even if best.pt exists

    Returns:
        Training results dict
    """
    yolo_cfg = config.get("yolo", {})
    seed = config.get("seed", 42)
    _run_name = run_name if run_name else scenario_name

    # --- Skip logic: check if training already completed ---
    run_dir = os.path.join(output_dir, _run_name)
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    last_pt = os.path.join(run_dir, "weights", "last.pt")

    if not force and os.path.exists(best_pt):
        print(f"\n[SKIP] Training already complete for {scenario_name}")
        print(f"  best.pt: {best_pt}")
        try:
            ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            if "epoch" in ckpt and isinstance(ckpt["epoch"], int) and ckpt["epoch"] >= 0:
                print(f"  [INFO] 🌟 BEST WEIGHTS TERPILIH: Model 'best.pt' ini diambil dari Epoch ke-{(ckpt['epoch'] + 1)} (berdasarkan mAP tertinggi).")
        except Exception as e:
            print(f"  [WARN] Tidak bisa membaca file weights untuk mendapatkan informasi epoch: {e}")
            
        print(f"  → To retrain, pass force=True or delete {run_dir}")
        return {
            "scenario": scenario_name,
            "run_dir": run_dir,
            "best_pt": best_pt,
            "last_pt": last_pt if os.path.exists(last_pt) else None,
            "epochs_requested": yolo_cfg.get("epochs", 100),
            "skipped": True,
        }

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
    run_name_dir = _run_name

    print(f"\n{'='*60}")
    print(f"[TRAIN] Scenario: {scenario_name}")
    print(f"[TRAIN] Dataset: {dataset_yaml}")
    print(f"[TRAIN] Output: {project_dir}/{run_name_dir}")
    print(f"[TRAIN] Epochs: {epochs}, Batch: {batch}, Seed: {seed}")
    print(f"[TRAIN] Model: {yolo_cfg.get('model', 'yolov7n.pt')}")
    print(f"{'='*60}\n")

    # --- Defensive: ensure dataset.yaml uses absolute path (fixes Windows path issue) ---
    patch_dataset_yaml_path(dataset_yaml)

    # Initialize YOLO model
    model_name = yolo_cfg.get("model", "yolov7n.pt")
    model = YOLO(model_name)

    # Train with unified hyperparameters
    _t_start = time.time()
    results = model.train(
        data=dataset_yaml,
        imgsz=yolo_cfg.get("imgsz", 640),
        epochs=epochs,
        batch=batch,
        seed=seed,
        patience=yolo_cfg.get("patience", 20),
        project=project_dir,
        name=run_name_dir,
        exist_ok=yolo_cfg.get("exist_ok", True),
        device=[i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() > 1 else (0 if torch.cuda.is_available() else "cpu"),
        workers=yolo_cfg.get("workers", 4),
        pretrained=yolo_cfg.get("pretrained", True),
        resume=resume,

        # Optimizer & Learning Rate
        optimizer=yolo_cfg.get("optimizer", "auto"),
        lr0=yolo_cfg.get("lr0", 0.01),
        lrf=yolo_cfg.get("lrf", 0.01),
        weight_decay=yolo_cfg.get("weight_decay", 0.0005),
        warmup_epochs=yolo_cfg.get("warmup_epochs", 3.0),

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

    _elapsed = time.time() - _t_start

    # Save config snapshot and environment info
    run_dir = os.path.join(project_dir, run_name_dir)
    save_config_snapshot(config, run_dir)
    save_environment_info(run_dir)

    # Check if best.pt exists
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    last_pt = os.path.join(run_dir, "weights", "last.pt")

    _hrs, _rem = divmod(_elapsed, 3600)
    _mins, _secs = divmod(_rem, 60)
    print(f"\n{'='*60}")
    print(f"[TRAIN] ✓ Training complete: {scenario_name}")
    print(f"  Duration : {int(_hrs)}h {int(_mins)}m {_secs:.1f}s ({_elapsed/60:.1f} min total)")
    print(f"  Epochs   : {epochs}")
    print(f"  best.pt  : {'✓' if os.path.exists(best_pt) else '✗'} {best_pt}")
    if os.path.exists(best_pt):
        try:
            ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            if "epoch" in ckpt and isinstance(ckpt["epoch"], int) and ckpt["epoch"] >= 0:
                print(f"  [INFO] 🌟 BEST WEIGHTS TERPILIH: Model 'best.pt' ini diambil dari Epoch ke-{(ckpt['epoch'] + 1)} (berdasarkan mAP tertinggi).")
        except Exception as e:
            print(f"  [WARN] Tidak bisa membaca file weights untuk mendapatkan informasi epoch: {e}")
            
    print(f"  last.pt  : {'✓' if os.path.exists(last_pt) else '✗'} {last_pt}")
    print(f"{'='*60}")

    return {
        "scenario": scenario_name,
        "run_dir": run_dir,
        "best_pt": best_pt if os.path.exists(best_pt) else None,
        "last_pt": last_pt if os.path.exists(last_pt) else None,
        "epochs_requested": epochs,
        "elapsed_seconds": round(_elapsed, 1),
        "elapsed_formatted": f"{int(_hrs)}h {int(_mins)}m {_secs:.1f}s",
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
    else:
        try:
            import torch
            ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            if "epoch" in ckpt and isinstance(ckpt["epoch"], int) and ckpt["epoch"] >= 0:
                print(f"[INFO] 🌟 BEST WEIGHTS TERPILIH: Model 'best.pt' ini diambil dari Epoch ke-{(ckpt['epoch'] + 1)} (berdasarkan mAP tertinggi).")
        except Exception:
            pass
    return best_pt
