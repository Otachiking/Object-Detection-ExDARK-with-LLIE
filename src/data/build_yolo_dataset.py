"""
Build YOLO-structured dataset from ExDark images + converted labels + split files.

Produces:
  ExDark_yolo/
    images/{train,val,test}/   — resized images (640px longest side)
    labels/{train,val,test}/   — YOLO format .txt labels
    dataset.yaml               — Ultralytics dataset config

Key design decisions:
- Images are resized to 640px (longest side, preserve aspect ratio) BEFORE saving
- Labels are in normalized YOLO format (immune to resize)
- Flat structure (no subfolder per class) — YOLO matches labels by filename stem
"""

import os
import cv2
import shutil
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.data.split_dataset import load_split_file


def resize_longest_side(
    img: np.ndarray,
    target_size: int = 640,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize image so longest side = target_size, preserving aspect ratio.

    Args:
        img: BGR image (numpy array)
        target_size: Target size for longest side
        interpolation: OpenCV interpolation method

    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    if max(h, w) <= target_size:
        return img  # Don't upscale

    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)

    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def build_yolo_dataset(
    exdark_images_dir: str,
    labels_dir: str,
    splits_dir: str,
    output_dir: str,
    target_size: int = 640,
    class_names: Optional[Dict[int, str]] = None,
    force: bool = False,
    splits_to_build: Optional[List[str]] = None,
) -> dict:
    """Build complete YOLO dataset structure from ExDark.

    Resume-safe: skips images that already exist (unless force=True).

    Args:
        exdark_images_dir: ExDark images root (contains Bicycle/, Boat/, etc.)
        labels_dir: Directory with converted YOLO labels (flat .txt files)
        splits_dir: Directory with train.txt, val.txt, test.txt
        output_dir: Output root directory (will contain images/, labels/, dataset.yaml)
        target_size: Resize longest side to this value
        class_names: Dict mapping class_id → class_name
        force: If True, overwrite existing files
        splits_to_build: List of splits to process (e.g. ["train", "val", "test"]). Defaults to all 3.

    Returns:
        Summary statistics
    """
    if class_names is None:
        class_names = {
            0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus",
            4: "Car", 5: "Cat", 6: "Chair", 7: "Cup",
            8: "Dog", 9: "Motorbike", 10: "People", 11: "Table",
        }

    summary = {"splits": {}, "errors": []}

    if splits_to_build is None:
        splits_to_build = ["train", "val", "test"]

    # --- Overall skip check (all requested splits) ---
    if not force:
        print("[CHECK] Checking if requested YOLO dataset splits are already built...")
        yaml_path = os.path.join(output_dir, "dataset.yaml")
        
        is_built = os.path.exists(yaml_path)
        if is_built:
            for split_name in splits_to_build:
                images_split = os.path.join(output_dir, "images", split_name)
                # Check if dir exists and has files (at least some images)
                if not os.path.isdir(images_split) or len(os.listdir(images_split)) == 0:
                    is_built = False
                    break
        
        if is_built:
            print(f"[SKIP] YOLO dataset splits already built in {output_dir}")
            for split_name in splits_to_build:
                images_out = os.path.join(output_dir, "images", split_name)
                n = len([f for f in os.listdir(images_out)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]) if os.path.isdir(images_out) else 0
                summary["splits"][split_name] = {
                    "total": n, "processed": n, "skipped": 0,
                }
                print(f"  {split_name}: {n} images ✓")
            print(f"  → To rebuild, pass force=True")
            return summary

    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(splits_dir, f"{split_name}.txt")
        if not os.path.exists(split_file):
            print(f"[WARN] Split file not found: {split_file}")
            continue

        entries = load_split_file(split_file)
        
        # Decide if we need to build images for this split
        build_images = split_name in splits_to_build

        print(f"\n[BUILD] Processing {split_name}: {len(entries)} items (build_images={build_images})")

        # Create output directories
        images_out = os.path.join(output_dir, "images", split_name)
        labels_out = os.path.join(output_dir, "labels", split_name)
        if build_images:
            os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        count_img_ok = 0
        count_img_skip = 0
        count_lbl_ok = 0

        for filename, class_folder in tqdm(entries, desc=f"  {split_name}"):
            # Source paths
            src_image = os.path.join(exdark_images_dir, class_folder, filename)
            label_stem = os.path.splitext(filename)[0]
            src_label = os.path.join(labels_dir, label_stem + ".txt")

            # Output paths
            out_image_name = label_stem + ".jpg"
            dst_image = os.path.join(images_out, out_image_name)
            dst_label = os.path.join(labels_out, label_stem + ".txt")

            # 1. Copy label file (ALWAYS DO THIS)
            if not os.path.exists(dst_label) or force:
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    with open(dst_label, "w") as f: pass
                    summary["errors"].append(f"Label not found (created empty): {src_label}")
            count_lbl_ok += 1

            # 2. Resize and save image (ONLY if build_images)
            if build_images:
                if not os.path.exists(src_image):
                    summary["errors"].append(f"Image not found: {src_image}")
                    count_img_skip += 1
                    continue

                if not force and os.path.exists(dst_image):
                    count_img_ok += 1
                    continue

                try:
                    img = cv2.imread(src_image)
                    if img is None:
                        summary["errors"].append(f"Cannot read: {src_image}")
                        count_img_skip += 1
                        continue

                    img_resized = resize_longest_side(img, target_size)
                    cv2.imwrite(dst_image, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    count_img_ok += 1
                except Exception as e:
                    summary["errors"].append(f"Error processing {src_image}: {e}")
                    count_img_skip += 1
                    continue

        if build_images:
            summary["splits"][split_name] = {
                "total": len(entries),
                "processed": count_img_ok,
                "skipped": count_img_skip,
            }
            print(f"  {split_name}: {count_img_ok} imgs OK, {count_img_skip} imgs skipped, {count_lbl_ok} labels")
        else:
            print(f"  {split_name}: {count_lbl_ok} labels synced (images skipped)")

    # Generate dataset.yaml (use relative path so it works on any machine)
    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"\n[BUILD] dataset.yaml written: {yaml_path}")

    # Print errors summary
    if summary["errors"]:
        print(f"\n[BUILD] {len(summary['errors'])} errors/warnings:")
        for e in summary["errors"][:20]:
            print(f"  - {e}")
        if len(summary["errors"]) > 20:
            print(f"  ... and {len(summary['errors']) - 20} more")

    return summary


def generate_enhanced_dataset_yaml(
    enhanced_images_dir: str,
    yolo_labels_dir: str,
    output_yaml_path: str,
    class_names: Optional[Dict[int, str]] = None,
) -> str:
    """Generate dataset.yaml for an enhanced dataset that shares labels with raw.

    The enhanced dataset has its own images but uses the same labels as ExDark_yolo.
    Ultralytics expects labels in a 'labels' folder parallel to 'images'.

    Args:
        enhanced_images_dir: Root of enhanced images (contains images/{train,val,test}/)
        yolo_labels_dir: Root of YOLO labels (ExDark_yolo path that has labels/{train,val,test}/)
        output_yaml_path: Where to save dataset.yaml
        class_names: Class name mapping

    Returns:
        Path to generated yaml
    """
    if class_names is None:
        class_names = {
            0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus",
            4: "Car", 5: "Cat", 6: "Chair", 7: "Cup",
            8: "Dog", 9: "Motorbike", 10: "People", 11: "Table",
        }

    # Use relative path so yaml is portable across machines
    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }

    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    # Create symlinks for labels (or copy if symlink fails)
    enhanced_root = enhanced_images_dir
    for split in ["train", "val", "test"]:
        src_labels = os.path.join(yolo_labels_dir, "labels", split)
        dst_labels = os.path.join(enhanced_root, "labels", split)

        if os.path.exists(dst_labels):
            continue

        os.makedirs(os.path.dirname(dst_labels), exist_ok=True)
        try:
            os.symlink(src_labels, dst_labels)
            print(f"[YAML] Symlinked labels: {dst_labels} → {src_labels}")
        except (OSError, NotImplementedError):
            # Fallback: copy labels with progress
            print(f"[YAML] Copying labels: {src_labels} → {dst_labels} ...")
            os.makedirs(dst_labels, exist_ok=True)
            label_files = [f for f in os.listdir(src_labels) if f.endswith(".txt")]
            for lf in tqdm(label_files, desc=f"  Copying {split} labels", unit="file"):
                shutil.copy2(os.path.join(src_labels, lf), os.path.join(dst_labels, lf))
            print(f"[YAML] Copied {len(label_files)} label files")

    print(f"[YAML] Enhanced dataset.yaml: {output_yaml_path}")
    return output_yaml_path
