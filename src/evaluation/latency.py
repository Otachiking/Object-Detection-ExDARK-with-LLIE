"""
Inference latency measurement with proper GPU timing.

Measures:
- T_enhance: LLIE inference time per image (0 for S1)
- T_detect: YOLO inference time per image
- T_total: end-to-end (T_enhance + T_detect)

Reports: mean, std, median, p95

Critical: Uses torch.cuda.synchronize() for accurate GPU timing.
"""

import os
import json
import time
import cv2
import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from src.enhancers.base import BaseEnhancer


def measure_latency(
    yolo_weights: str,
    output_dir: str,
    scenario_name: str,
    test_images_dir: str,
    enhancer: Optional[BaseEnhancer] = None,
    num_images: int = 200,
    warmup: int = 50,
    device: int = 0,
    imgsz: int = 640,
    seed: int = 42,
) -> dict:
    """Measure inference latency for a scenario.

    Args:
        yolo_weights: Path to YOLO best.pt
        output_dir: Where to save results
        scenario_name: Scenario identifier
        test_images_dir: Directory with test images
        enhancer: Loaded enhancer (None for S1 baseline)
        num_images: Number of images to measure
        warmup: Warmup iterations before timing
        device: GPU device
        imgsz: YOLO input size
        seed: Random seed for image selection

    Returns:
        Latency statistics dict
    """
    import random
    from ultralytics import YOLO

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[LATENCY] Scenario: {scenario_name}")
    print(f"[LATENCY] Warmup: {warmup}, Measurements: {num_images}")
    print(f"[LATENCY] Enhancer: {enhancer.name if enhancer else 'None (S1)'}")
    print(f"{'='*60}\n")

    # Load YOLO model
    model = YOLO(yolo_weights)

    # Select test images
    image_files = sorted([
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if len(image_files) < num_images:
        print(f"[LATENCY] Using all {len(image_files)} images (requested {num_images})")
        selected = image_files
    else:
        selected = random.sample(image_files, num_images)

    # Load images into memory first (exclude I/O from timing)
    print("[LATENCY] Loading images into memory...")
    images = []
    for fname in selected:
        img = cv2.imread(os.path.join(test_images_dir, fname))
        if img is not None:
            images.append(img)

    print(f"[LATENCY] Loaded {len(images)} images")

    # === Warmup Phase ===
    print(f"[LATENCY] Warming up ({warmup} iterations)...")
    dummy = images[0] if images else np.zeros((640, 640, 3), dtype=np.uint8)

    for _ in range(warmup):
        if enhancer:
            _ = enhancer.enhance(dummy)
        _ = model(dummy, imgsz=imgsz, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # === Measurement Phase ===
    enhance_times = []
    detect_times = []
    total_times = []

    print(f"[LATENCY] Measuring ({len(images)} images)...")
    for img in tqdm(images, desc="Latency"):
        # --- Enhancement timing ---
        t_enhance = 0.0
        enhanced_img = img

        if enhancer:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            enhanced_img = enhancer.enhance(img)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_enhance = (t1 - t0) * 1000  # ms

        # --- Detection timing ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(enhanced_img, imgsz=imgsz, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_detect = (t1 - t0) * 1000  # ms

        t_total = t_enhance + t_detect

        enhance_times.append(t_enhance)
        detect_times.append(t_detect)
        total_times.append(t_total)

    # === Compute Statistics ===
    def stats(times: List[float], name: str) -> dict:
        arr = np.array(times)
        return {
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_median": float(np.median(arr)),
            f"{name}_p95": float(np.percentile(arr, 95)),
            f"{name}_min": float(np.min(arr)),
            f"{name}_max": float(np.max(arr)),
        }

    results = {
        "scenario": scenario_name,
        "num_images": len(images),
        "warmup": warmup,
        "device": f"cuda:{device}" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
    }
    results.update(stats(enhance_times, "T_enhance_ms"))
    results.update(stats(detect_times, "T_detect_ms"))
    results.update(stats(total_times, "T_total_ms"))

    # Save
    json_path = os.path.join(output_dir, "latency.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n[LATENCY] === {scenario_name} ===")
    if enhancer:
        print(f"  T_enhance: {results['T_enhance_ms_mean']:.2f} ± {results['T_enhance_ms_std']:.2f} ms "
              f"(p95: {results['T_enhance_ms_p95']:.2f})")
    else:
        print(f"  T_enhance: 0.00 ms (no enhancement)")
    print(f"  T_detect:  {results['T_detect_ms_mean']:.2f} ± {results['T_detect_ms_std']:.2f} ms "
          f"(p95: {results['T_detect_ms_p95']:.2f})")
    print(f"  T_total:   {results['T_total_ms_mean']:.2f} ± {results['T_total_ms_std']:.2f} ms "
          f"(p95: {results['T_total_ms_p95']:.2f})")

    return results
