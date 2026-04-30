"""
Configuration loader — merges base.yaml + paths.yaml + scenario yaml.

Handles:
- Auto-detection of Colab vs Local environment
- Variable interpolation (${drive_root}, ${project_root})
- Quick test mode override
- Config snapshot saving for reproducibility
"""

import os
import re
import copy
import json
import yaml
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _is_kaggle() -> bool:
    """Detect if running on Kaggle."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def _interpolate_vars(data: Any, variables: Dict[str, str]) -> Any:
    """Recursively interpolate ${var} placeholders in strings."""
    if isinstance(data, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        def replacer(match):
            key = match.group(1)
            return variables.get(key, match.group(0))
        # Multiple passes for nested references
        for _ in range(3):
            new_data = pattern.sub(replacer, data)
            if new_data == data:
                break
            data = new_data
        return data
    elif isinstance(data, dict):
        return {k: _interpolate_vars(v, variables) for k, v in data.items()}
    elif isinstance(data, list):
        return [_interpolate_vars(item, variables) for item in data]
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict. Override wins on conflict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    scenario: str = "s2_hvi_cidnet",
    config_dir: Optional[str] = None,
    quick_test: bool = False,
    epochs: Optional[int] = None,
) -> dict:
    """Load and merge configuration for a scenario.

    Args:
        scenario: Scenario name (s1_raw, s2_hvi_cidnet, s3_retinexformer, s4_lyt_net)
        config_dir: Path to configs/ directory. Auto-detected if None.
        quick_test: Override to quick test mode (1 epoch).
        epochs: Optional override for YOLO training epochs.

    Returns:
        Merged configuration dictionary with resolved paths.
    """
    # Find config directory
    if config_dir is None:
        # Try relative to this file, then CWD
        this_dir = Path(__file__).parent.parent
        config_dir = str(this_dir / "configs")
        if not os.path.exists(config_dir):
            config_dir = "configs"

    # Load base configs
    base = load_yaml(os.path.join(config_dir, "base.yaml"))
    paths = load_yaml(os.path.join(config_dir, "paths.yaml"))

    # Load scenario config
    scenario_file = os.path.join(config_dir, f"{scenario}.yaml")
    scenario_cfg = load_yaml(scenario_file) if os.path.exists(scenario_file) else {}

    # Determine environment
    is_kaggle = _is_kaggle()
    is_colab = _is_colab() and not is_kaggle
    if is_kaggle:
        env_key = "kaggle"
    elif is_colab:
        env_key = "colab"
    else:
        env_key = "local"
    env_paths = paths.get(env_key, {})

    # Build interpolation variables
    variables = {
        "drive_root": env_paths.get("drive_root", ""),
        "project_root": env_paths.get("project_root", ""),
    }

    # Resolve paths with variable interpolation
    resolved_paths = _interpolate_vars(env_paths, variables)

    # Merge: base + paths + scenario
    config = _deep_merge(base, {"paths": resolved_paths})
    config = _deep_merge(config, {"paths_meta": {
        k: v for k, v in paths.items()
        if k not in ("colab", "local")
    }})
    config = _deep_merge(config, scenario_cfg)

    # Backward-compatibility aliases for older notebook/script keys
    p = config.setdefault("paths", {})
    data = p.get("data", {})
    exdark_meta = config.get("paths_meta", {}).get("exdark", {})

    if "output_root" not in p:
        p["output_root"] = p.get("drive_root") or p.get("project_root")

    if "exdark_root" not in p:
        p["exdark_root"] = data.get("exdark_original")

    if "exdark_structure" not in p:
        p["exdark_structure"] = {
            "images": exdark_meta.get("images_dir", "Dataset"),
            "groundtruth": exdark_meta.get("groundtruth_dir", "Groundtruth"),
            "classlist": exdark_meta.get("classlist_file", "Groundtruth/imageclasslist.txt"),
        }

    # Add environment info
    config["environment"] = {
        "is_colab": is_colab,
        "is_kaggle": is_kaggle,
        "env_key": env_key,
        "platform": platform.system(),
    }

    # Manual Epochs Override
    if epochs is not None:
        config["yolo"]["epochs"] = epochs
        print(f"[CONFIG] Manual epochs override: {epochs}")

    # Quick test override
    if quick_test or config.get("quick_test", False):
        config["quick_test"] = True
        config["yolo"]["epochs"] = config.get("quick_test_epochs", 1)
        config["yolo"]["batch"] = config.get("quick_test_batch", 8)
        print("[CONFIG] Quick test mode active: epochs=1, batch=8")

    return config



def get_data_paths(config: dict) -> dict:
    """Extract resolved data paths from config.

    Returns dict with keys: exdark_original, output_root, model_cache
    Scenario-specific paths (prepared/, scenarios/) are computed in notebooks
    from output_root.
    """
    p = config.get("paths", {})
    data = p.get("data", {})
    return {
        "exdark_original": data.get("exdark_original", ""),
        "output_root": p.get("output_root", ""),
        "model_cache": p.get("model_cache", ""),
    }


def save_config_snapshot(config: dict, output_dir: str) -> str:
    """Save a frozen copy of the config for reproducibility.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save snapshot

    Returns:
        Path to saved snapshot file
    """
    os.makedirs(output_dir, exist_ok=True)
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")

    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[CONFIG] Snapshot saved: {snapshot_path}")
    return snapshot_path


def save_environment_info(output_dir: str) -> str:
    """Save environment info (GPU, CUDA, PyTorch) for reproducibility.

    pip freeze is written asynchronously in a background thread to avoid
    blocking the notebook (it can take 30-60 s on Colab).
    """
    import torch
    import threading

    os.makedirs(output_dir, exist_ok=True)

    gpu_memory_gb = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        gpu_memory_gb = round(total_mem / 1e9, 2)

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": gpu_memory_gb,
    }

    # Save JSON immediately (fast)
    info_path = os.path.join(output_dir, "system_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Save pip freeze asynchronously — avoids blocking the notebook
    def _write_pip_freeze(out_dir: str) -> None:
        try:
            freeze = subprocess.check_output(["pip", "freeze"], text=True, timeout=120)
            freeze_path = os.path.join(out_dir, "requirements_frozen.txt")
            with open(freeze_path, "w") as f:
                f.write(freeze)
        except Exception:
            pass

    t = threading.Thread(target=_write_pip_freeze, args=(output_dir,), daemon=True)
    t.start()

    print(f"[ENV] System info saved: {info_path}")
    return info_path
