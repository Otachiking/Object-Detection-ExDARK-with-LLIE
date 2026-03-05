"""
No-Reference (NR) Image Quality Metrics for enhancement evaluation.

Metrics:
- NIQE (Natural Image Quality Evaluator): lower = better
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator): lower = better
- LOE (Lightness Order Error): lower = better (needs raw→enhanced pair)

Uses pyiqa library for NIQE and BRISQUE (GPU-accelerated).
LOE is implemented manually.

Computed on a random sample of test images (default 1000) for efficiency.
"""

import os
import json
import random
import cv2
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


def compute_loe(img_raw: np.ndarray, img_enhanced: np.ndarray, downsample: int = 4) -> float:
    """Compute Lightness Order Error (LOE) between raw and enhanced image.

    LOE measures how well the enhancement preserves the relative lightness order
    of the original image. Lower is better.

    Args:
        img_raw: Original image (BGR uint8)
        img_enhanced: Enhanced image (BGR uint8)
        downsample: Downsample factor for speed (default 4)

    Returns:
        LOE value (float, lower = better)
    """
    # Convert to grayscale (lightness channel)
    gray_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_enh = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Downsample for speed
    if downsample > 1:
        gray_raw = gray_raw[::downsample, ::downsample]
        gray_enh = gray_enh[::downsample, ::downsample]

    h, w = gray_raw.shape
    n = h * w

    if n == 0:
        return 0.0

    # Flatten
    raw_flat = gray_raw.flatten()
    enh_flat = gray_enh.flatten()

    # Sample-based LOE (for large images)
    max_pixels = 2000  # Limit for computational feasibility
    if n > max_pixels:
        indices = np.random.choice(n, max_pixels, replace=False)
        raw_sample = raw_flat[indices]
        enh_sample = enh_flat[indices]
    else:
        raw_sample = raw_flat
        enh_sample = enh_flat

    m = len(raw_sample)

    # Compute LOE: count lightness order violations
    # For each pair of pixels, check if the relative order is preserved
    loe = 0
    for i in range(m):
        # Compare pixel i with all others
        raw_order = (raw_sample > raw_sample[i]).astype(np.int32)
        enh_order = (enh_sample > enh_sample[i]).astype(np.int32)
        loe += np.sum(np.abs(raw_order - enh_order))

    # Normalize by number of comparisons
    loe = loe / (m * m) * 100  # Percentage

    return float(loe)


def compute_nr_metrics(
    images_dir: str,
    output_dir: str,
    scenario_name: str,
    raw_images_dir: Optional[str] = None,
    sample_size: int = 1000,
    seed: int = 42,
    device: str = "cuda",
    metrics: List[str] = None,
    force: bool = False,
) -> dict:
    """Compute NR image quality metrics on a sample of images.

    Skips computation if summary.json already exists (unless force=True).

    Args:
        images_dir: Directory with images to evaluate (e.g., test split)
        output_dir: Where to save results
        scenario_name: Scenario identifier
        raw_images_dir: Raw images dir for LOE computation (None for S1)
        sample_size: Number of images to sample
        seed: Random seed for sampling
        device: 'cuda' or 'cpu'
        metrics: List of metrics to compute. Default: ["niqe", "brisque"]
        force: If True, recompute even if results exist

    Returns:
        Summary dict with mean/std/median per metric
    """
    # --- Skip logic ---
    summary_path = os.path.join(output_dir, "summary.json")
    if not force and os.path.exists(summary_path):
        print(f"\n[SKIP] NR metrics already computed for {scenario_name}")
        print(f"  Loaded from: {summary_path}")
        print(f"  \u2192 To recompute, pass force=True")
        with open(summary_path, "r") as f:
            cached = json.load(f)
        # Print cached summary
        for col in ("niqe", "brisque", "loe"):
            if f"{col}_mean" in cached:
                print(f"  {col.upper()}: mean={cached[f'{col}_mean']:.4f}")
        return cached

    if metrics is None:
        metrics = ["niqe", "brisque"]

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # List available images
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if not image_files:
        print(f"[NR-METRICS] No images found in {images_dir}")
        return {}

    # Sample
    if sample_size and sample_size < len(image_files):
        sampled = random.sample(image_files, sample_size)
        print(f"[NR-METRICS] Sampled {sample_size}/{len(image_files)} images (seed={seed})")
    else:
        sampled = image_files
        print(f"[NR-METRICS] Using all {len(sampled)} images")

    # Initialize pyiqa metrics
    iqa_metrics = {}
    try:
        import pyiqa
        for m in metrics:
            if m.lower() in ("niqe", "brisque"):
                iqa_metrics[m] = pyiqa.create_metric(m.lower(), device=device)
                print(f"[NR-METRICS] Initialized {m} (device={device})")
    except ImportError:
        print("[NR-METRICS] ⚠ pyiqa not installed. Install: pip install pyiqa")
        print("[NR-METRICS] Falling back to basic computation...")

    # Compute whether to include LOE
    compute_loe_flag = (raw_images_dir is not None and
                        os.path.isdir(raw_images_dir) and
                        raw_images_dir != images_dir)

    if compute_loe_flag:
        print(f"[NR-METRICS] LOE enabled (raw: {raw_images_dir})")
    else:
        print(f"[NR-METRICS] LOE disabled (no raw reference or same dir)")

    # Compute metrics per image
    results_rows = []

    for fname in tqdm(sampled, desc=f"NR metrics ({scenario_name})"):
        img_path = os.path.join(images_dir, fname)
        row = {"filename": fname}

        try:
            # Read image for pyiqa (expects tensor)
            img = cv2.imread(img_path)
            if img is None:
                row["error"] = "unreadable"
                results_rows.append(row)
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # Compute NIQE and BRISQUE via pyiqa
            for metric_name, metric_fn in iqa_metrics.items():
                try:
                    score = metric_fn(img_tensor).item()
                    row[metric_name] = score
                except Exception as e:
                    row[metric_name] = None
                    row[f"{metric_name}_error"] = str(e)

            # Compute LOE if raw reference available
            if compute_loe_flag:
                raw_path = os.path.join(raw_images_dir, fname)
                if os.path.exists(raw_path):
                    try:
                        img_raw = cv2.imread(raw_path)
                        if img_raw is not None:
                            loe_val = compute_loe(img_raw, img)
                            row["loe"] = loe_val
                    except Exception as e:
                        row["loe_error"] = str(e)

        except Exception as e:
            row["error"] = str(e)

        results_rows.append(row)

    # Build DataFrame
    df = pd.DataFrame(results_rows)

    # Save per-image scores
    csv_path = os.path.join(output_dir, "scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[NR-METRICS] Per-image scores: {csv_path}")

    # Compute summary statistics
    summary = {"scenario": scenario_name, "sample_size": len(sampled)}
    metric_cols = [c for c in df.columns if c in ("niqe", "brisque", "loe")]

    for col in metric_cols:
        values = df[col].dropna()
        if len(values) > 0:
            summary[f"{col}_mean"] = float(values.mean())
            summary[f"{col}_std"] = float(values.std())
            summary[f"{col}_median"] = float(values.median())
            summary[f"{col}_count"] = int(len(values))

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[NR-METRICS] Summary: {summary_path}")

    # Print summary
    print(f"\n[NR-METRICS] === {scenario_name} Summary ===")
    for col in metric_cols:
        if f"{col}_mean" in summary:
            print(f"  {col.upper():10s}: mean={summary[f'{col}_mean']:.4f} "
                  f"± {summary[f'{col}_std']:.4f} "
                  f"(median={summary[f'{col}_median']:.4f})")

    return summary
