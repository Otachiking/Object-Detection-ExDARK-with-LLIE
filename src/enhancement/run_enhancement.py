"""
Batch enhancement pipeline for entire YOLO-formatted dataset.

Takes a YOLO dataset (ExDark_yolo/) and produces an enhanced copy (ExDark_enhanced/<method>/).
Labels are shared via symlink (not duplicated).

Features:
- Resume-safe: skips already-processed images
- Progress bar with ETA
- Manifest CSV for debugging
- Dimension validation (output must match input)
"""

import os
import cv2
import csv
import time
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from src.enhancers.base import BaseEnhancer
from src.data.build_yolo_dataset import generate_enhanced_dataset_yaml


def enhance_dataset(
    enhancer: BaseEnhancer,
    source_dataset_dir: str,
    output_dir: str,
    yolo_labels_dir: str,
    splits: list = None,
    save_manifest: bool = True,
    force: bool = False,
) -> dict:
    """Enhance all images in a YOLO-formatted dataset.

    Resume-safe: skips images that already exist (unless force=True).

    Args:
        enhancer: Loaded BaseEnhancer instance
        source_dataset_dir: Source YOLO dataset root (ExDark_yolo/)
        output_dir: Output root for enhanced images
        yolo_labels_dir: Path to share labels from (ExDark_yolo/ root)
        splits: Which splits to process. Default: ["train", "val", "test"]
        save_manifest: Whether to save per-image manifest CSV
        force: If True, re-enhance even if output exists

    Returns:
        Summary dict with counts and timing
    """
    if splits is None:
        splits = ["train", "val", "test"]

    assert enhancer.is_loaded, f"Enhancer {enhancer.name} not loaded. Call load_model() first."

    summary = {
        "enhancer": enhancer.name,
        "splits": {},
        "total_processed": 0,
        "total_skipped": 0,
        "total_failed": 0,
        "total_time_s": 0,
    }

    # --- Overall skip check ---
    if not force:
        all_complete = True
        for split in splits:
            src_dir = os.path.join(source_dataset_dir, "images", split)
            out_dir = os.path.join(output_dir, "images", split)
            if not os.path.isdir(src_dir):
                all_complete = False
                break
            expected = len([f for f in os.listdir(src_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            actual = len([f for f in os.listdir(out_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]) \
                if os.path.isdir(out_dir) else 0
            if actual < expected:
                all_complete = False
                break

        if all_complete:
            print(f"\n[SKIP] All enhanced images already exist for {enhancer.name}")
            for split in splits:
                out_dir = os.path.join(output_dir, "images", split)
                n = len([f for f in os.listdir(out_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]) \
                    if os.path.isdir(out_dir) else 0
                summary["splits"][split] = {
                    "total": n, "processed": 0, "skipped": n, "failed": 0, "time_s": 0,
                }
                summary["total_skipped"] += n
                print(f"  {split}: {n} images ✓")
            print(f"  → To re-enhance, pass force=True")
            return summary

    manifest_rows = []

    for split in splits:
        source_images_dir = os.path.join(source_dataset_dir, "images", split)
        output_images_dir = os.path.join(output_dir, "images", split)

        if not os.path.isdir(source_images_dir):
            print(f"[ENHANCE] ⚠ Source dir not found: {source_images_dir}")
            continue

        os.makedirs(output_images_dir, exist_ok=True)

        # List all images
        image_files = sorted([
            f for f in os.listdir(source_images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        processed = 0
        skipped = 0
        failed = 0
        split_time = 0

        print(f"\n[ENHANCE] {enhancer.name} — {split}: {len(image_files)} images")

        for fname in tqdm(image_files, desc=f"  {split}"):
            src_path = os.path.join(source_images_dir, fname)
            dst_path = os.path.join(output_images_dir, fname)

            # Resume-safe: skip if output exists
            if not force and os.path.exists(dst_path):
                skipped += 1
                continue

            try:
                # Read source image
                img = cv2.imread(src_path)
                if img is None:
                    raise ValueError(f"Cannot read: {src_path}")

                # Enhance with safety checks
                t0 = time.perf_counter()
                enhanced = enhancer.enhance_safe(img)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000

                # Save enhanced image (same format as source)
                cv2.imwrite(dst_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

                processed += 1
                split_time += elapsed_ms

                manifest_rows.append({
                    "filename": fname,
                    "split": split,
                    "status": "ok",
                    "time_ms": f"{elapsed_ms:.1f}",
                    "input_shape": f"{img.shape}",
                    "output_shape": f"{enhanced.shape}",
                })

            except Exception as e:
                failed += 1
                print(f"\n  [ERROR] {fname}: {e}")
                manifest_rows.append({
                    "filename": fname,
                    "split": split,
                    "status": f"failed: {e}",
                    "time_ms": "0",
                    "input_shape": "",
                    "output_shape": "",
                })

        summary["splits"][split] = {
            "total": len(image_files),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "time_s": split_time / 1000,
        }
        summary["total_processed"] += processed
        summary["total_skipped"] += skipped
        summary["total_failed"] += failed
        summary["total_time_s"] += split_time / 1000

        print(f"  Done: {processed} processed, {skipped} skipped, {failed} failed "
              f"({split_time/1000:.1f}s)")

    # Save manifest CSV
    if save_manifest and manifest_rows:
        manifest_path = os.path.join(output_dir, "enhance_manifest.csv")
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"\n[ENHANCE] Manifest saved: {manifest_path}")

    # Generate dataset.yaml with shared labels
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    generate_enhanced_dataset_yaml(
        enhanced_images_dir=output_dir,
        yolo_labels_dir=yolo_labels_dir,
        output_yaml_path=yaml_path,
    )

    # Validate: count output images per split
    print(f"\n[ENHANCE] === {enhancer.name} Summary ===")
    for split, stats in summary["splits"].items():
        out_dir = os.path.join(output_dir, "images", split)
        actual = len([f for f in os.listdir(out_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        expected = stats["total"]
        match = "✓" if actual == expected else f"✗ ({actual}/{expected})"
        print(f"  {split}: {match}")

    print(f"  Total time: {summary['total_time_s']:.1f}s")
    if summary["total_processed"] > 0:
        avg_ms = (summary["total_time_s"] * 1000) / summary["total_processed"]
        print(f"  Avg per image: {avg_ms:.1f} ms")

    return summary


def get_enhancer(name: str, cache_dir: str) -> BaseEnhancer:
    """Factory function to create an enhancer by name.

    Args:
        name: "hvi_cidnet", "retinexformer", or "lyt_net"
        cache_dir: Root cache directory

    Returns:
        Unloaded BaseEnhancer instance
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower in ("hvi_cidnet", "hvicidnet", "hvi"):
        from src.enhancers.hvi_cidnet import HVICIDNetEnhancer
        return HVICIDNetEnhancer(cache_dir=os.path.join(cache_dir, "HVI-CIDNet"))

    elif name_lower in ("retinexformer", "retinex"):
        from src.enhancers.retinexformer import RetinexFormerEnhancer
        return RetinexFormerEnhancer(cache_dir=os.path.join(cache_dir, "Retinexformer"))

    elif name_lower in ("lyt_net", "lytnet", "lyt"):
        from src.enhancers.lyt_net import LYTNetEnhancer
        return LYTNetEnhancer(cache_dir=os.path.join(cache_dir, "LYT-Net"))

    else:
        raise ValueError(f"Unknown enhancer: {name}. Choose from: hvi_cidnet, retinexformer, lyt_net")
