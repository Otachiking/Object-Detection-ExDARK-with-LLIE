"""File I/O helpers for the project."""

import os
import json
import yaml
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def load_json(path: str) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> str:
    """Save data to JSON file. Returns saved path."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return path


def load_yaml(path: str) -> dict:
    """Load YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> str:
    """Save dict to YAML file. Returns saved path."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    return path


def read_text(path: str) -> str:
    """Read text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(text: str, path: str) -> str:
    """Write text to file. Returns saved path."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def count_files(directory: str, extensions: Optional[List[str]] = None, verbose: bool = False) -> int:
    """Count files in directory (recursively) with optional extension filter."""
    if verbose:
        print(f"  Counting files in {directory}...")
    count = 0
    for root, _, files in os.walk(directory):
        for f in files:
            if extensions is None or any(f.lower().endswith(ext) for ext in extensions):
                count += 1
    if verbose:
        print(f"  Found {count} files")
    return count


def list_files(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
) -> List[str]:
    """List files in directory with optional extension filter.

    Returns absolute paths sorted alphabetically.
    """
    result = []
    if recursive:
        for root, _, files in os.walk(directory):
            for f in files:
                if extensions is None or any(f.lower().endswith(ext) for ext in extensions):
                    result.append(os.path.join(root, f))
    else:
        for f in os.listdir(directory):
            fp = os.path.join(directory, f)
            if os.path.isfile(fp):
                if extensions is None or any(f.lower().endswith(ext) for ext in extensions):
                    result.append(fp)
    return sorted(result)


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]


def list_images(directory: str, recursive: bool = True) -> List[str]:
    """List image files in directory."""
    return list_files(directory, extensions=IMAGE_EXTENSIONS, recursive=recursive)


def compute_md5(path: str) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def safe_copy(src: str, dst: str, overwrite: bool = False) -> bool:
    """Copy file with overwrite control. Returns True if copied."""
    if os.path.exists(dst) and not overwrite:
        return False
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True


def get_size_mb(path: str, verbose: bool = False) -> float:
    """Get file/directory size in MB."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    if verbose:
        print(f"  Calculating size of {path}...")
    total = 0
    all_files = []
    for root, _, files in os.walk(path):
        for f in files:
            all_files.append(os.path.join(root, f))
    iterator = tqdm(all_files, desc="  Scanning size", unit="file") if verbose else all_files
    for fp in iterator:
        total += os.path.getsize(fp)
    return total / (1024 * 1024)


def patch_dataset_yaml_path(yaml_path: str) -> None:
    """Ensure dataset.yaml 'path:' is an absolute path valid on the current OS.

    Problem: dataset.yaml generated on Windows (or by old code) may contain a
    Windows absolute path like 'C:\\\\CODE\\\\...' which is invalid on Linux/Colab.
    Ultralytics then prepends its datasets dir, creating a garbage path and
    raising FileNotFoundError.

    Resolution: rewrite 'path' to the absolute directory containing the yaml.
    This is always the correct value: it means "the dataset root is right here,
    next to dataset.yaml", which is exactly how our directory structure is laid out.

    Also handles the case where 'path: "."' was used — correct but YOLO resolves
    '.' relative to CWD rather than the yaml file in some versions, so we make it
    explicit with an absolute path.

    Args:
        yaml_path: Absolute path to the dataset.yaml file.
    """
    if not os.path.isfile(yaml_path):
        return

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            return

        # The correct path is always the directory containing dataset.yaml
        correct_path = str(Path(yaml_path).parent.resolve())

        current_path = str(data.get("path", "."))

        # Needs patching when:
        #   - it is the relative sentinel "."
        #   - it is NOT already equal to the correct absolute path
        #   - it is a Windows-style path (backslash or drive letter like "C:")
        is_windows_path = "\\" in current_path or (
            len(current_path) >= 2 and current_path[1] == ":"
        )
        already_correct = (
            Path(current_path).is_absolute()
            and not is_windows_path
            and current_path == correct_path
        )

        if already_correct:
            print(f"[YAML] dataset.yaml path OK: {current_path!r}")
            return

        print(f"[YAML] Patching dataset.yaml path:")
        print(f"  File:   {yaml_path}")
        print(f"  Before: {current_path!r}")
        print(f"  After:  {correct_path!r}")
        data["path"] = correct_path
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"  [OK] Patched.")

    except Exception as exc:
        print(f"[WARN] patch_dataset_yaml_path failed for {yaml_path}: {exc}")
