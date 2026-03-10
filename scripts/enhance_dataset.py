#!/usr/bin/env python3
"""
Fase 2: Image Enhancement
=========================
Applies the specified LLIE method to all images in the YOLO dataset.
Skipped for S1_Raw (no enhancement needed).

Usage:
    python scripts/enhance_dataset.py --scenario s2_hvi_cidnet [--quick-test]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.seed import set_global_seed
from src.enhancement.run_enhancement import run_enhancement_pipeline


def main():
    parser = argparse.ArgumentParser(description="Enhance ExDark images with LLIE method")
    parser.add_argument("--scenario", required=True, help="Scenario config (s2_hvi_cidnet, s3_retinexformer, s4_lyt_net)")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode (limited images)")
    parser.add_argument("--config-dir", default=None, help="Config directory override")
    args = parser.parse_args()

    cfg = load_config(args.scenario, config_dir=args.config_dir, quick_test=args.quick_test)
    set_global_seed(cfg["seed"])

    scenario_name = cfg.get("scenario_name", args.scenario)
    enhancer_name = cfg.get("scenario", {}).get("enhancer", None)

    if enhancer_name is None or enhancer_name.lower() == "none":
        print(f"[ENHANCE] Scenario {scenario_name} has no enhancer. Nothing to do.")
        return

    print("=" * 60)
    print(f"FASE 2: IMAGE ENHANCEMENT ({scenario_name})")
    print(f"Enhancer: {enhancer_name}")
    print("=" * 60)

    paths = cfg["paths"]
    yolo_dir = os.path.join(paths["output_root"], "ExDark_yolo")
    enhanced_dir = os.path.join(paths["output_root"], f"ExDark_enhanced_{enhancer_name}")
    cache_dir = os.path.join(paths["output_root"], "model_cache")

    stats = run_enhancement_pipeline(
        yolo_dir=yolo_dir,
        output_dir=enhanced_dir,
        enhancer_name=enhancer_name,
        cache_dir=cache_dir,
        cfg=cfg,
    )

    print("\n" + "=" * 60)
    print(f"FASE 2 COMPLETE ({scenario_name})")
    print(f"Enhanced dataset at: {enhanced_dir}")
    print(f"Processed: {stats['processed']} | Skipped: {stats['skipped']} | Failed: {stats['failed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
