#!/usr/bin/env python3
"""
Fase 4: Detection Evaluation
=============================
Evaluates trained YOLOv11n on the test split.

Usage:
    python scripts/evaluate.py --scenario s1_raw [--quick-test]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.seed import set_global_seed
from src.evaluation.eval_yolo import evaluate_yolo
from src.training.train_yolo import get_best_weights


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11n on ExDark test set")
    parser.add_argument("--scenario", required=True, help="Scenario config name")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode")
    parser.add_argument("--config-dir", default=None, help="Config directory override")
    parser.add_argument("--weights", default=None, help="Override weights path (default: auto-detect best.pt)")
    args = parser.parse_args()

    cfg = load_config(args.scenario, config_dir=args.config_dir, quick_test=args.quick_test)
    set_global_seed(cfg["seed"])

    scenario_name = cfg.get("scenario_name", args.scenario)
    paths = cfg["paths"]
    output_root = paths["output_root"]

    print("=" * 60)
    print(f"FASE 4: DETECTION EVALUATION ({scenario_name})")
    print("=" * 60)

    # Find weights
    run_dir = os.path.join(output_root, "runs", scenario_name)
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = get_best_weights(run_dir)

    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        print("Run train.py first.")
        sys.exit(1)

    print(f"Weights: {weights_path}")

    # Dataset YAML
    enhancer_name = cfg.get("scenario", {}).get("enhancer", None)
    if enhancer_name and enhancer_name.lower() != "none":
        data_yaml = os.path.join(output_root, f"ExDark_enhanced_{enhancer_name}", "dataset.yaml")
    else:
        data_yaml = os.path.join(output_root, "ExDark_yolo", "dataset.yaml")

    # Evaluate
    eval_dir = os.path.join(output_root, "evaluation", scenario_name)
    results = evaluate_yolo(
        weights_path=weights_path,
        data_yaml=data_yaml,
        eval_dir=eval_dir,
        cfg=cfg,
    )

    print("\n" + "=" * 60)
    print(f"FASE 4 COMPLETE ({scenario_name})")
    print(f"mAP@0.5: {results.get('mAP_50', 'N/A'):.4f}")
    print(f"mAP@0.5:0.95: {results.get('mAP_50_95', 'N/A'):.4f}")
    print(f"Results saved: {eval_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
