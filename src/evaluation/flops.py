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
    # Access from model directly
    results = {
        "model": "YOLO11n",
        "input_size": f"{imgsz}x{imgsz}",
        "params": None,
        "gflops": None,
    }

    # Try to get FLOPs from model info
    try:
        # Ultralytics stores this after info() call
        if hasattr(model.model, 'info'):
            results["params"] = sum(p.numel() for p in model.model.parameters())
            results["params_m"] = results["params"] / 1e6
    except Exception:
        pass

    # Alternative: use thop
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
        # Fallback: known value for YOLO11n
        results["gflops"] = 6.5  # Approximate known value
        results["note"] = "approximate (thop failed)"

    return results


def compute_enhancer_flops(
    enhancer_name: str,
    model: torch.nn.Module,
    imgsz: int = 640,
    device: str = "cuda",
) -> dict:
    """Compute FLOPs for an enhancer model.

    Args:
        enhancer_name: Name of enhancer
        model: PyTorch model
        imgsz: Input size
        device: Device

    Returns:
        Dict with FLOPs info
    """
    results = {
        "model": enhancer_name,
        "input_size": f"{imgsz}x{imgsz}",
        "params": sum(p.numel() for p in model.parameters()),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }

    try:
        from thop import profile
        dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        results["flops"] = flops
        results["gflops"] = flops / 1e9
    except Exception as e:
        print(f"[FLOPs] thop failed for {enhancer_name}: {e}")
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
            fca = FlopCountAnalysis(model, dummy)
            results["flops"] = fca.total()
            results["gflops"] = fca.total() / 1e9
        except Exception as e2:
            print(f"[FLOPs] fvcore also failed: {e2}")
            results["gflops"] = None
            results["note"] = "computation failed"

    return results


def compute_all_flops(
    yolo_weights: str,
    output_dir: str,
    scenario_name: str,
    enhancer_model: Optional[torch.nn.Module] = None,
    enhancer_name: Optional[str] = None,
    imgsz: int = 640,
    device: str = "cuda",
) -> dict:
    """Compute FLOPs for entire pipeline (enhancer + YOLO).

    Args:
        yolo_weights: Path to YOLO weights
        output_dir: Output directory
        scenario_name: Scenario name
        enhancer_model: PyTorch enhancer model (None for S1)
        enhancer_name: Name of enhancer
        imgsz: Input size
        device: Device

    Returns:
        Combined FLOPs dict
    """
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

        # Total
        yolo_g = yolo_flops.get("gflops", 0) or 0
        enh_g = enh_flops.get("gflops", 0) or 0
        results["total_gflops"] = yolo_g + enh_g
    else:
        results["enhancer"] = {"model": "None", "gflops": 0}
        results["total_gflops"] = yolo_flops.get("gflops", 0)

    # Save
    json_path = os.path.join(output_dir, "flops.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[FLOPs] === {scenario_name} ===")
    print(f"  YOLO:      {yolo_flops.get('gflops', 'N/A')} GFLOPs")
    if enhancer_model:
        print(f"  Enhancer:  {results['enhancer'].get('gflops', 'N/A')} GFLOPs")
    print(f"  Total:     {results['total_gflops']} GFLOPs")

    return results
