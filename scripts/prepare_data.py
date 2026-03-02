#!/usr/bin/env python3
"""
Fase 1: Data Preparation
========================
Parses ExDark official split, converts annotations to YOLO format,
builds the YOLO directory structure, and validates everything.

Usage:
    python scripts/prepare_data.py --scenario s1_raw [--quick-test]
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.seed import set_global_seed
from src.data.split_dataset import parse_split_file
from src.data.convert_exdark import convert_all_annotations
from src.data.build_yolo_dataset import build_yolo_dataset
from src.data.validate_dataset import validate_yolo_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare ExDark dataset for YOLO training")
    parser.add_argument("--scenario", default="s1_raw", help="Scenario config name (default: s1_raw)")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode")
    parser.add_argument("--config-dir", default=None, help="Config directory override")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.scenario, config_dir=args.config_dir, quick_test=args.quick_test)
    set_global_seed(cfg["seed"])

    paths = cfg["paths"]
    exdark_root = paths["exdark_root"]
    output_root = paths["output_root"]

    print("=" * 60)
    print("FASE 1: DATA PREPARATION")
    print("=" * 60)

    # Step 1: Parse official split
    print("\n[Step 1/4] Parsing official split...")
    classlist_path = os.path.join(
        exdark_root, paths["exdark_structure"]["groundtruth"], "imageclasslist.txt"
    )
    split_output = os.path.join(output_root, "splits")
    splits = parse_split_file(classlist_path, split_output)
    print(f"  Train: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}")

    # Step 2: Convert annotations
    print("\n[Step 2/4] Converting ExDark annotations to YOLO format...")
    gt_dir = os.path.join(exdark_root, paths["exdark_structure"]["groundtruth"])
    img_dir = os.path.join(exdark_root, paths["exdark_structure"]["images"])
    yolo_labels_dir = os.path.join(output_root, "ExDark_yolo_labels")
    convert_stats = convert_all_annotations(gt_dir, img_dir, yolo_labels_dir, cfg)
    print(f"  Converted: {convert_stats['converted']} | Skipped: {convert_stats['skipped']}")

    # Step 3: Build YOLO directory structure
    print("\n[Step 3/4] Building YOLO dataset structure...")
    yolo_dir = os.path.join(output_root, "ExDark_yolo")
    build_stats = build_yolo_dataset(
        split_dir=split_output,
        img_dir=img_dir,
        label_dir=yolo_labels_dir,
        output_dir=yolo_dir,
        imgsz=cfg["yolo"]["imgsz"],
        cfg=cfg,
    )
    print(f"  Images: {build_stats['total_images']} | Labels: {build_stats['total_labels']}")

    # Step 4: Validate
    print("\n[Step 4/4] Validating dataset...")
    val_results = validate_yolo_dataset(yolo_dir, cfg)
    if val_results["valid"]:
        print("  ✓ Dataset validation PASSED")
    else:
        print("  ✗ Dataset validation FAILED:")
        for issue in val_results.get("issues", []):
            print(f"    - {issue}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("FASE 1 COMPLETE")
    print(f"YOLO dataset ready at: {yolo_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
