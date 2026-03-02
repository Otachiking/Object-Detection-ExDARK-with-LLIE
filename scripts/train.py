#!/usr/bin/env python3
"""
Fase 3: YOLO Training
=====================
Trains YOLOv11n on the specified scenario's dataset.

Usage:
    python scripts/train.py --scenario s1_raw [--quick-test]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, save_config_snapshot, save_environment_info
from src.seed import set_global_seed
from src.training.train_yolo import train_yolo, get_best_weights


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11n on ExDark")
    parser.add_argument("--scenario", required=True, help="Scenario config name")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode (1 epoch)")
    parser.add_argument("--config-dir", default=None, help="Config directory override")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.scenario, config_dir=args.config_dir, quick_test=args.quick_test)
    set_global_seed(cfg["seed"])

    scenario_name = cfg.get("scenario_name", args.scenario)
    paths = cfg["paths"]
    output_root = paths["output_root"]

    print("=" * 60)
    print(f"FASE 3: YOLO TRAINING ({scenario_name})")
    print("=" * 60)

    # Determine dataset YAML
    enhancer_name = cfg.get("enhancer", {}).get("name", None)
    if enhancer_name and enhancer_name.lower() != "none":
        data_yaml = os.path.join(output_root, f"ExDark_enhanced_{enhancer_name}", "dataset.yaml")
    else:
        data_yaml = os.path.join(output_root, "ExDark_yolo", "dataset.yaml")

    if not os.path.exists(data_yaml):
        print(f"ERROR: dataset.yaml not found at {data_yaml}")
        print("Run prepare_data.py (and enhance_dataset.py if needed) first.")
        sys.exit(1)

    print(f"Dataset: {data_yaml}")

    # Training
    run_dir = os.path.join(output_root, "runs", scenario_name)
    results = train_yolo(
        data_yaml=data_yaml,
        run_dir=run_dir,
        cfg=cfg,
        resume=args.resume,
    )

    # Save config snapshot
    save_config_snapshot(cfg, run_dir)
    save_environment_info(run_dir)

    # Report
    best_weights = get_best_weights(run_dir)
    print("\n" + "=" * 60)
    print(f"FASE 3 COMPLETE ({scenario_name})")
    print(f"Run directory: {run_dir}")
    print(f"Best weights: {best_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
