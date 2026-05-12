"""
FLOPs/GFLOPs computation for YOLO + enhancer models.

Reports model complexity as GFLOPs at fixed input size (640×640).
Positioned as "complexity metric", not real-time claim.
"""

import os
import json
import torch
import numpy as np
from typing import Optional


def compute_yolo_flops(
    weights_path: str,
    imgsz: int = 640,
) -> dict:
    """Compute FLOPs for YOLO model.

    Uses Ultralytics built-in model.info() which reports GFLOPs.

    Args:
        weights_path: Path to YOLO weights
        imgsz: Input size

    Returns:
        Dict with FLOPs info
    """
    from ultralytics import YOLO

    model = YOLO(weights_path)
    info = model.info(imgsz=imgsz)

    # model.info() returns (layers, params, gradients, flops)
    results = {
        "model": "YOLOv11n",
        "input_size": f"{imgsz}x{imgsz}",
        "params": None,
        "gflops": None,
    }

    # Try to get params count
    try:
        if hasattr(model.model, 'parameters'):
            results["params"] = sum(p.numel() for p in model.model.parameters())
            results["params_m"] = results["params"] / 1e6
    except Exception:
        pass

    # Primary: use thop
    try:
        from thop import profile
        dummy = torch.randn(1, 3, imgsz, imgsz).to(next(model.model.parameters()).device)
        flops, params = profile(model.model, inputs=(dummy,), verbose=False)
        results["flops"] = flops
        results["gflops"] = flops / 1e9
        results["params"] = params
        results["params_m"] = params / 1e6
    except Exception as e:
        print(f"[FLOPs] thop failed (non-critical): {e}")
        # Fallback: known value for YOLOv11n at 640x640
        results["gflops"] = 6.5  # Approximate known value from Ultralytics docs
        results["note"] = "approximate (thop failed)"

    return results


def compute_enhancer_flops(
    enhancer_name: str,
    model: torch.nn.Module,
    imgsz: int = 640,
    device: str = None,
) -> dict:
    """Compute FLOPs for an enhancer model.

    Args:
        enhancer_name: Name of enhancer
        model: PyTorch model
        imgsz: Input size
        device: Device (auto-detected if None)

    Returns:
        Dict with FLOPs info
    """
    # Hardcoded fallback values (measured from RESULT notebooks on Kaggle GPU)
    # Digunakan jika thop/fvcore tidak bisa menghitung (e.g. custom ops)
    KNOWN_GFLOPS = {
        "HVI_CIDNet":       15.4,   # measured
        "RetinexFormer":    56.8,   # measured
        "LYT_Net":          14.9,   # measured
    }

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "model": enhancer_name,
        "input_size": f"{imgsz}x{imgsz}",
        "params": sum(p.numel() for p in model.parameters()),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }

    # Primary: thop
    try:
        from thop import profile
        dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        results["flops"] = flops
        results["gflops"] = flops / 1e9
        print(f"[FLOPs] {enhancer_name}: {results['gflops']:.2f} GFLOPs (via thop)")
        return results
    except Exception as e:
        print(f"[FLOPs] thop failed for {enhancer_name}: {e}")

    # Secondary: fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
        fca = FlopCountAnalysis(model, dummy)
        results["flops"] = fca.total()
        results["gflops"] = fca.total() / 1e9
        print(f"[FLOPs] {enhancer_name}: {results['gflops']:.2f} GFLOPs (via fvcore)")
        return results
    except Exception as e2:
        print(f"[FLOPs] fvcore also failed: {e2}")

    # Tertiary: hardcoded known values
    for key, val in KNOWN_GFLOPS.items():
        if key.lower() in enhancer_name.lower():
            print(f"[FLOPs] Using hardcoded GFLOPs for {enhancer_name}: {val} GFLOPs")
            results["gflops"] = val
            results["note"] = f"hardcoded (thop+fvcore failed). Known value for {key}."
            return results

    # Ultimate fallback: None (will be handled upstream)
    print(f"[FLOPs] ⚠ Could not compute GFLOPs for {enhancer_name}. Returning None.")
    results["gflops"] = None
    results["note"] = "computation failed (all methods)"
    return results


def compute_all_flops(
    yolo_weights: str,
    output_dir: str,
    scenario_name: str,
    enhancer_model: Optional[torch.nn.Module] = None,
    enhancer_name: Optional[str] = None,
    imgsz: int = 640,
    device: str = None,
    force: bool = False,
    t_enhance_ms: Optional[float] = None,
) -> dict:
    """Compute FLOPs for entire pipeline (enhancer + YOLO).

    Skips computation if flops.json already exists (unless force=True).

    Args:
        yolo_weights: Path to YOLO weights
        output_dir: Output directory
        scenario_name: Scenario name
        enhancer_model: PyTorch enhancer model (None for S1)
        enhancer_name: Name of enhancer
        imgsz: Input size
        device: Device (auto-detected if None)
        force: If True, recompute even if results exist
        t_enhance_ms: Optional measured T_enhance mean (ms) from latency notebook.
                      Dipakai untuk inject nilai ke flops.json jika enhancer tidak
                      bisa di-profile ulang (Kaggle-only scenario).

    Returns:
        Combined FLOPs dict
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- Skip logic ---
    json_path = os.path.join(output_dir, "flops.json")
    if not force and os.path.exists(json_path):
        print(f"\n[SKIP] FLOPs results already exist for {scenario_name}")
        print(f"  Loaded from: {json_path}")
        print(f"  \u2192 To recompute, pass force=True")
        with open(json_path, "r") as f:
            cached = json.load(f)
        print(f"  Total: {cached.get('total_gflops', 'N/A')} GFLOPs")
        return cached

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "scenario": scenario_name,
        "input_size": f"{imgsz}x{imgsz}",
    }

    # YOLO FLOPs
    print(f"[FLOPs] Computing YOLO FLOPs...")
    yolo_flops = compute_yolo_flops(yolo_weights, imgsz)
    results["yolo"] = yolo_flops

    # Enhancer FLOPs
    if enhancer_model is not None and enhancer_name:
        print(f"[FLOPs] Computing {enhancer_name} FLOPs...")
        enh_flops = compute_enhancer_flops(enhancer_name, enhancer_model, imgsz, device)
        results["enhancer"] = enh_flops

        yolo_g = yolo_flops.get("gflops", 0) or 0
        enh_g = enh_flops.get("gflops") or 0   # None-safe
        results["total_gflops"] = yolo_g + enh_g
    else:
        # S1 (no enhancer) atau notebook yang tidak pass enhancer_model
        results["enhancer"] = {"model": enhancer_name or "None", "gflops": 0}
        results["total_gflops"] = yolo_flops.get("gflops", 0)

    # Inject T_enhance_ms jika disediakan (dari pengukuran notebook lama)
    if t_enhance_ms is not None:
        results["t_enhance_ms_measured"] = t_enhance_ms
        print(f"[FLOPs] T_enhance (measured): {t_enhance_ms:.2f} ms")

    # Save
    json_path = os.path.join(output_dir, "flops.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[FLOPs] === {scenario_name} ===")
    print(f"  YOLO:      {yolo_flops.get('gflops', 'N/A')} GFLOPs")
    if enhancer_model:
        print(f"  Enhancer:  {results['enhancer'].get('gflops', 'N/A')} GFLOPs")
    print(f"  Total:     {results['total_gflops']} GFLOPs")
    if t_enhance_ms is not None:
        print(f"  T_enhance: {t_enhance_ms:.2f} ms (measured)")

    return results
