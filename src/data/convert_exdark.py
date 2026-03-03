"""
Convert ExDark annotations (bbGt format) to YOLO format.

ExDark annotation format (per line after header):
  ClassName left top width height 0 0 0 0 0 0 0

YOLO format (per line):
  class_id x_center y_center width height   (all normalized 0-1)

Critical notes:
- Annotation files named: <image_filename>.txt (e.g., 2015_00001.png.txt)
- Images can contain objects from MULTIPLE classes
- Bounding boxes may slightly exceed image bounds → clipped to [0,1]
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# ExDark class name → YOLO class ID (0-indexed)
CLASS_NAME_TO_ID = {
    "Bicycle": 0, "Boat": 1, "Bottle": 2, "Bus": 3,
    "Car": 4, "Cat": 5, "Chair": 6, "Cup": 7,
    "Dog": 8, "Motorbike": 9, "People": 10, "Table": 11,
}


def parse_exdark_annotation(
    annot_path: str,
) -> List[Tuple[str, int, int, int, int]]:
    """Parse a single ExDark annotation file.

    Args:
        annot_path: Path to annotation .txt file

    Returns:
        List of (class_name, left, top, width, height) tuples
    """
    objects = []

    with open(annot_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip header: "% bbGt version=3"
        if line.startswith("%"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        class_name = parts[0]
        if class_name not in CLASS_NAME_TO_ID:
            print(f"[WARN] Unknown class '{class_name}' in {annot_path}")
            continue

        try:
            left = int(parts[1])
            top = int(parts[2])
            width = int(parts[3])
            height = int(parts[4])
        except ValueError:
            # Handle float values (some annotations may have decimals)
            left = int(float(parts[1]))
            top = int(float(parts[2]))
            width = int(float(parts[3]))
            height = int(float(parts[4]))

        objects.append((class_name, left, top, width, height))

    return objects


def convert_bbox_to_yolo(
    left: int, top: int, w: int, h: int,
    img_width: int, img_height: int,
) -> Tuple[float, float, float, float]:
    """Convert ExDark bbox [l,t,w,h] to YOLO [x_center, y_center, w, h] normalized.

    Args:
        left, top, w, h: Absolute pixel coordinates (ExDark format)
        img_width, img_height: Image dimensions

    Returns:
        (x_center, y_center, w_norm, h_norm) all in [0, 1]
    """
    x_center = (left + w / 2.0) / img_width
    y_center = (top + h / 2.0) / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    # Clip to [0, 1] — some ExDark bboxes slightly exceed image bounds
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    # Additional safety: ensure bbox doesn't exceed image when reconstructed
    # If x_center - w/2 < 0 or x_center + w/2 > 1, adjust width
    if x_center - w_norm / 2 < 0:
        w_norm = x_center * 2
    if x_center + w_norm / 2 > 1:
        w_norm = (1 - x_center) * 2
    if y_center - h_norm / 2 < 0:
        h_norm = y_center * 2
    if y_center + h_norm / 2 > 1:
        h_norm = (1 - y_center) * 2

    return x_center, y_center, w_norm, h_norm


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height without loading full image into memory.

    Returns:
        (width, height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def convert_single_image(
    image_path: str,
    annot_path: str,
    output_label_path: str,
) -> dict:
    """Convert annotations for a single image to YOLO format.

    Args:
        image_path: Path to the image file
        annot_path: Path to ExDark annotation file
        output_label_path: Where to write YOLO label file

    Returns:
        Dict with conversion stats
    """
    stats = {"success": False, "num_objects": 0, "classes": set(), "warnings": []}

    # Get image dimensions
    try:
        img_w, img_h = get_image_dimensions(image_path)
    except ValueError as e:
        stats["warnings"].append(str(e))
        return stats

    # Parse ExDark annotations
    if not os.path.exists(annot_path):
        stats["warnings"].append(f"Annotation not found: {annot_path}")
        return stats

    objects = parse_exdark_annotation(annot_path)

    if not objects:
        stats["warnings"].append(f"No valid objects in: {annot_path}")
        # Still create empty label file (YOLO expects it)
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        with open(output_label_path, "w") as f:
            pass
        stats["success"] = True
        return stats

    # Convert each object
    yolo_lines = []
    for class_name, left, top, w, h in objects:
        class_id = CLASS_NAME_TO_ID[class_name]
        xc, yc, wn, hn = convert_bbox_to_yolo(left, top, w, h, img_w, img_h)

        # Validate: skip zero-area boxes
        if wn <= 0 or hn <= 0:
            stats["warnings"].append(f"Zero-area bbox skipped: {class_name} [{left},{top},{w},{h}]")
            continue

        yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        stats["classes"].add(class_id)

    # Write YOLO label file
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    with open(output_label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

    stats["success"] = True
    stats["num_objects"] = len(yolo_lines)

    return stats


def convert_exdark_to_yolo(
    exdark_images_dir: str,
    exdark_gt_dir: str,
    output_labels_dir: str,
    class_folders: Optional[List[str]] = None,
) -> dict:
    """Convert all ExDark annotations to YOLO format.

    Args:
        exdark_images_dir: Root dir containing class folders with images
                           (e.g., ExDark_original/Dataset/ExDark/)
        exdark_gt_dir: Root dir containing class folders with annotations
                       (e.g., ExDark_original/Groundtruth/)
        output_labels_dir: Where to write YOLO label files (flat structure)
        class_folders: List of class folder names. Default: all 12 ExDark classes.

    Returns:
        Summary statistics dict
    """
    if class_folders is None:
        class_folders = list(CLASS_NAME_TO_ID.keys())

    summary = {
        "total_images": 0,
        "total_labels": 0,
        "total_objects": 0,
        "failed": 0,
        "warnings": [],
        "per_class_objects": {name: 0 for name in class_folders},
    }

    os.makedirs(output_labels_dir, exist_ok=True)
    all_files = []

    # Collect all image files across all class folders
    for class_folder in class_folders:
        img_dir = os.path.join(exdark_images_dir, class_folder)
        gt_dir = os.path.join(exdark_gt_dir, class_folder)

        if not os.path.isdir(img_dir):
            print(f"[WARN] Image directory not found: {img_dir}")
            continue
        if not os.path.isdir(gt_dir):
            print(f"[WARN] Groundtruth directory not found: {gt_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            # Match image extensions (case-insensitive)
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            image_path = os.path.join(img_dir, fname)
            # ExDark annotation filename = image_filename + ".txt"
            annot_path = os.path.join(gt_dir, fname + ".txt")
            # YOLO label: same stem as image, .txt extension (flat)
            label_stem = os.path.splitext(fname)[0]
            label_path = os.path.join(output_labels_dir, label_stem + ".txt")

            all_files.append((image_path, annot_path, label_path, class_folder))

    summary["total_images"] = len(all_files)
    print(f"[CONVERT] Found {len(all_files)} images across {len(class_folders)} classes")

    # Convert all
    for image_path, annot_path, label_path, class_folder in tqdm(
        all_files, desc="Converting ExDark → YOLO"
    ):
        stats = convert_single_image(image_path, annot_path, label_path)

        if stats["success"]:
            summary["total_labels"] += 1
            summary["total_objects"] += stats["num_objects"]
        else:
            summary["failed"] += 1

        summary["warnings"].extend(stats.get("warnings", []))

    print(f"\n[CONVERT] Summary:")
    print(f"  Images: {summary['total_images']}")
    print(f"  Labels created: {summary['total_labels']}")
    print(f"  Total objects: {summary['total_objects']}")
    print(f"  Failed: {summary['failed']}")
    if summary["warnings"]:
        print(f"  Warnings: {len(summary['warnings'])}")
        for w in summary["warnings"][:10]:
            print(f"    - {w}")
        if len(summary["warnings"]) > 10:
            print(f"    ... and {len(summary['warnings']) - 10} more")

    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python convert_exdark.py <images_dir> <gt_dir> <output_labels_dir>")
        sys.exit(1)
    convert_exdark_to_yolo(sys.argv[1], sys.argv[2], sys.argv[3])
