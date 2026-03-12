"""
Google Drive sync utilities for cross-platform result persistence.

Provides two download strategies:

1. **download_weights_from_gdrive()** — Downloads ONLY ``best.pt``
   from a per-scenario Drive folder.  Accepts either a **folder URL**
   (navigates to ``runs/weights/best.pt`` inside a temp dir) or a
   **direct file URL** (downloads the single file).

2. **restore_scenario_from_gdrive()** — Legacy: downloads an *entire*
   shared folder.  Kept for backward compatibility but NOT recommended
   (wastes bandwidth on enhanced images).

Usage in notebook cell::

    from src.utils.gdrive_sync import download_weights_from_gdrive

    download_weights_from_gdrive(
        folder_url="https://drive.google.com/drive/folders/<ID>",
        scenario_name="S1_Raw",
        output_root=OUTPUT_ROOT,
    )
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def _ensure_gdown():
    """Import gdown, installing it first if necessary."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("[SYNC] Installing gdown …")
        subprocess.check_call(["pip", "install", "-q", "gdown"])
        import gdown
        return gdown


def _is_file_url(url: str) -> bool:
    """True if *url* points to a single Google Drive **file** (not a folder)."""
    return bool(re.search(r"/file/d/|[?&]id=|/uc\?", url))


# ─── Weights-only download (recommended) ─────────────────────────────────────

def download_weights_from_gdrive(
    folder_url: str,
    scenario_name: str,
    output_root: str,
    quiet: bool = False,
) -> dict:
    """Download **only best.pt** from a scenario Drive folder or file link.

    Accepts two kinds of URL:

    * **Folder URL** – ``https://drive.google.com/drive/folders/<ID>``
      The folder is downloaded to a temp dir; only ``best.pt`` is kept.
    * **File URL** – ``https://drive.google.com/file/d/<ID>/view``
      The single file is downloaded directly (~50 MB, fastest).

    Args:
        folder_url: Google Drive folder or file URL.
        scenario_name: e.g. ``"S1_Raw"``.
        output_root: Working directory (``/kaggle/working``).
        quiet: Suppress progress bars.

    Returns:
        ``{"success": True/False, "path": "<best.pt path>"}``
    """
    gdown = _ensure_gdown()

    url = folder_url.strip()
    if not url.startswith("http"):
        url = f"https://drive.google.com/drive/folders/{url}"

    target_dir = os.path.join(
        output_root, "scenarios", scenario_name, "runs", "weights",
    )
    target_path = os.path.join(target_dir, "best.pt")

    if os.path.exists(target_path):
        sz = os.path.getsize(target_path) / 1e6
        print(f"  ✓ {scenario_name}: best.pt sudah ada ({sz:.0f} MB) → skip")
        return {"success": True, "path": target_path}

    os.makedirs(target_dir, exist_ok=True)

    # ── Strategy A: direct file link → download single file ──────────
    if _is_file_url(url):
        print(f"  {scenario_name}: Downloading best.pt (direct link) …")
        try:
            out = gdown.download(url, output=target_path, quiet=quiet, fuzzy=True)
            if out and os.path.isfile(target_path):
                sz = os.path.getsize(target_path) / 1e6
                print(f"  ✓ {scenario_name}: best.pt downloaded ({sz:.0f} MB)")
                return {"success": True, "path": target_path}
        except Exception as e:
            print(f"  ✗ {scenario_name}: download failed — {e}")
            return {"success": False, "error": str(e)}

    # ── Strategy B: folder link → temp download, extract best.pt ─────
    print(f"  {scenario_name}: Downloading folder to temp dir …")
    print(f"    (hanya best.pt yang disimpan, sisanya dihapus)")

    with tempfile.TemporaryDirectory(prefix="gdrive_") as tmp:
        try:
            gdown.download_folder(
                url=url, output=tmp, quiet=quiet, use_cookies=False,
                remaining_ok=True,
            )
        except Exception as e:
            print(f"  ✗ {scenario_name}: folder download failed — {e}")
            print("    → Pastikan folder di-share 'Anyone with the link'")
            return {"success": False, "error": str(e)}

        # Walk temp dir seeking best.pt
        found = None
        for root, _dirs, files in os.walk(tmp):
            if "best.pt" in files:
                found = os.path.join(root, "best.pt")
                break

        if found:
            shutil.copy2(found, target_path)
            sz = os.path.getsize(target_path) / 1e6
            print(f"  ✓ {scenario_name}: best.pt restored ({sz:.0f} MB)")
            return {"success": True, "path": target_path}

        print(f"  ✗ {scenario_name}: best.pt tidak ditemukan dalam folder")
        print("    → Pastikan folder berisi runs/weights/best.pt")
        return {"success": False, "error": "best.pt not found"}


# ─── Legacy: full-folder download (not recommended) ──────────────────────────

def restore_scenario_from_gdrive(
    gdrive_url: str,
    output_root: str,
    quiet: bool = False,
) -> dict:
    """Download an entire shared Drive folder.  **Deprecated** — prefer
    :func:`download_weights_from_gdrive` to avoid downloading enhanced
    images unnecessarily.
    """
    gdown = _ensure_gdown()

    url = gdrive_url.strip()
    if not url.startswith("http"):
        url = f"https://drive.google.com/drive/folders/{url}"

    os.makedirs(output_root, exist_ok=True)

    print(f"[SYNC] Downloading ENTIRE folder from Google Drive …")
    print(f"  URL    : {url}")
    print(f"  Target : {output_root}")

    try:
        gdown.download_folder(
            url=url, output=output_root, quiet=quiet, use_cookies=False,
        )
    except Exception as e:
        print(f"[SYNC] ✗ Download failed: {e}")
        return {"success": False, "error": str(e)}

    restored: dict = {"success": True, "files": []}
    scenarios = ["S1_Raw", "S2_HVI_CIDNet", "S3_RetinexFormer", "S4_LYT_Net"]
    for sc in scenarios:
        best_pt = os.path.join(output_root, "scenarios", sc, "runs", "weights", "best.pt")
        if os.path.exists(best_pt):
            restored["files"].append(best_pt)
            print(f"  ✓ {sc}: best.pt found")

    direct_best = os.path.join(output_root, "runs", "weights", "best.pt")
    if os.path.exists(direct_best):
        restored["files"].append(direct_best)
    return restored


# ─── Zip for download / backup ────────────────────────────────────────────────

def zip_scenario_results(
    scenario_dir: str,
    output_path: str | None = None,
) -> str:
    """Create a lightweight zip of scenario results for download/backup.

    Includes trained weights, evaluation outputs, and config snapshots.
    Excludes large intermediate files (training batches, augmented images, etc.).

    Args:
        scenario_dir: Path to scenario directory (e.g. ``scenarios/S1_Raw``).
        output_path: Output zip path. Defaults to ``<scenario_dir>.zip``.

    Returns:
        Path to the created zip file.
    """
    if output_path is None:
        output_path = scenario_dir.rstrip("/\\") + ".zip"

    # Files to always include (relative to scenario_dir)
    include_exact = {
        "runs/weights/best.pt",
        "runs/weights/last.pt",
        "runs/results.csv",
        "runs/args.yaml",
        "runs/config_snapshot.yaml",
        "runs/system_info.json",
        "requirements_frozen.txt",
    }
    # Subdirectories to include entirely
    include_dirs = {"evaluation"}
    # Extensions to include from root
    include_exts = {".json", ".yaml", ".yml", ".txt", ".csv"}

    n_files = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(scenario_dir):
            rel_root = os.path.relpath(root, scenario_dir).replace("\\", "/")
            for fname in files:
                rel_path = (
                    fname if rel_root == "." else f"{rel_root}/{fname}"
                )
                should_include = (
                    rel_path in include_exact
                    or any(rel_path.startswith(d + "/") for d in include_dirs)
                    or (rel_root == "." and Path(fname).suffix in include_exts)
                )
                if should_include:
                    zf.write(os.path.join(root, fname), rel_path)
                    n_files += 1

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[ZIP] Created: {output_path}")
    print(f"  {n_files} files, {size_mb:.1f} MB")
    return output_path
