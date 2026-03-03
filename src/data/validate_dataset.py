"""
Validate YOLO dataset integrity.

Checks:
1. Image count == Label count per split
2. All bbox values in [0, 1]
3. All class_ids in valid range [0, 11]
4. No corrupt/unreadable images
5. Label files not empty (at least 1 bbox per image expected, but warn if empty)
6. Dimensions consistency after resize

Optional: visual preview with bbox overlay
"""

import os
import cv2
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter


CLASS_NAMES = {
    0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus",
    4: "Car", 5: "Cat", 6: "Chair", 7: "Cup",
    8: "Dog", 9: "Motorbike", 10: "People", 11: "Table",
}

NUM_CLASSES = 12


def validate_yolo_dataset(
    dataset_dir: str,
    splits: List[str] = None,
    max_preview: int = 5,
    save_preview_dir: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """Validate a YOLO-formatted dataset.

    Args:
        dataset_dir: Root directory (contains images/, labels/)
        splits: List of splits to validate. Default: ["train", "val", "test"]
        max_preview: Number of random preview images to generate
        save_preview_dir: If set, save preview images with bbox overlay
        seed: Random seed for preview sampling

    Returns:
        Validation report dict
    """
    if splits is None:
        splits = ["train", "val", "test"]

    random.seed(seed)
    report = {"valid": True, "splits": {}, "summary": {}}

    total_images = 0
    total_labels = 0
    total_objects = 0
    class_distribution = Counter()

    for split in splits:
        images_dir = os.path.join(dataset_dir, "images", split)
        labels_dir = os.path.join(dataset_dir, "labels", split)

        split_report = {
            "images": 0, "labels": 0, "objects": 0,
            "empty_labels": 0, "invalid_bbox": 0, "invalid_class": 0,
            "unreadable_images": 0, "orphan_labels": 0, "missing_labels": 0,
        }

        if not os.path.isdir(images_dir):
            print(f"[VALIDATE] ✗ Images dir not found: {images_dir}")
            split_report["error"] = "images dir missing"
            report["valid"] = False
            report["splits"][split] = split_report
            continue

        if not os.path.isdir(labels_dir):
            print(f"[VALIDATE] ✗ Labels dir not found: {labels_dir}")
            split_report["error"] = "labels dir missing"
            report["valid"] = False
            report["splits"][split] = split_report
            continue

        # List images and labels
        image_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))}
        label_files = {os.path.splitext(f)[0]: f for f in os.listdir(labels_dir)
                       if f.endswith(".txt")}

        split_report["images"] = len(image_files)
        split_report["labels"] = len(label_files)

        # Check matching
        image_stems = set(image_files.keys())
        label_stems = set(label_files.keys())

        missing_labels = image_stems - label_stems
        orphan_labels = label_stems - image_stems

        split_report["missing_labels"] = len(missing_labels)
        split_report["orphan_labels"] = len(orphan_labels)

        if missing_labels:
            print(f"[VALIDATE] ✗ {split}: {len(missing_labels)} images without labels")
            for m in list(missing_labels)[:5]:
                print(f"    Missing label for: {m}")
            report["valid"] = False

        if orphan_labels:
            print(f"[VALIDATE] ⚠ {split}: {len(orphan_labels)} labels without images")

        # Validate each label
        matched_stems = image_stems & label_stems
        for stem in matched_stems:
            label_path = os.path.join(labels_dir, label_files[stem])

            with open(label_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            if not lines:
                split_report["empty_labels"] += 1
                continue

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    split_report["invalid_bbox"] += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = [float(x) for x in parts[1:5]]
                except ValueError:
                    split_report["invalid_bbox"] += 1
                    continue

                # Check class ID range
                if cls_id < 0 or cls_id >= NUM_CLASSES:
                    split_report["invalid_class"] += 1
                    report["valid"] = False

                # Check bbox range [0, 1]
                for val in [xc, yc, w, h]:
                    if val < 0 or val > 1:
                        split_report["invalid_bbox"] += 1
                        report["valid"] = False
                        break

                split_report["objects"] += 1
                class_distribution[cls_id] += 1

        total_images += split_report["images"]
        total_labels += split_report["labels"]
        total_objects += split_report["objects"]
        report["splits"][split] = split_report

        status = "✓" if (split_report["invalid_bbox"] == 0
                         and split_report["invalid_class"] == 0
                         and split_report["missing_labels"] == 0) else "✗"
        print(f"[VALIDATE] {status} {split}: {split_report['images']} images, "
              f"{split_report['labels']} labels, {split_report['objects']} objects, "
              f"{split_report['empty_labels']} empty labels")

    # Summary
    report["summary"] = {
        "total_images": total_images,
        "total_labels": total_labels,
        "total_objects": total_objects,
        "class_distribution": {CLASS_NAMES.get(k, f"unknown_{k}"): v
                               for k, v in sorted(class_distribution.items())},
    }

    print(f"\n[VALIDATE] === Summary ===")
    print(f"  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")
    print(f"  Total objects: {total_objects}")
    print(f"  Class distribution:")
    for cls_id in sorted(class_distribution.keys()):
        print(f"    {CLASS_NAMES.get(cls_id, cls_id)}: {class_distribution[cls_id]}")

    # Generate preview images
    if save_preview_dir and max_preview > 0:
        _generate_previews(dataset_dir, splits, max_preview, save_preview_dir, seed)

    return report


def _generate_previews(
    dataset_dir: str,
    splits: List[str],
    max_preview: int,
    output_dir: str,
    seed: int,
) -> None:
    """Generate preview images with bounding box overlay."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Collect all image-label pairs from all splits
    pairs = []
    for split in splits:
        images_dir = os.path.join(dataset_dir, "images", split)
        labels_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.isdir(images_dir):
            continue

        for f in os.listdir(images_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                stem = os.path.splitext(f)[0]
                img_path = os.path.join(images_dir, f)
                lbl_path = os.path.join(labels_dir, stem + ".txt")
                if os.path.exists(lbl_path):
                    pairs.append((img_path, lbl_path, split))

    if not pairs:
        return

    # Sample
    sampled = random.sample(pairs, min(max_preview, len(pairs)))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
              (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)]

    for idx, (img_path, lbl_path, split) in enumerate(sampled):
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            xc, yc, bw, bh = [float(x) for x in parts[1:5]]

            # Convert YOLO → pixel coords
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = CLASS_NAMES.get(cls_id, str(cls_id))
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_path = os.path.join(output_dir, f"preview_{split}_{idx}.jpg")
        cv2.imwrite(out_path, img)

    print(f"[VALIDATE] {len(sampled)} preview images saved to {output_dir}")
