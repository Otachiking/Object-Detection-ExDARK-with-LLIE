"""
Google Drive sync utilities for cross-platform result persistence.

Allows downloading previous scenario results from a shared Google Drive
folder so that Kaggle (or any fresh environment) can skip already-completed
phases (data prep, training, evaluation).

Usage in notebook cell::

    from src.utils.gdrive_sync import restore_scenario_from_gdrive
    restore_scenario_from_gdrive(
        gdrive_url="https://drive.google.com/drive/folders/<FOLDER_ID>",
        output_root=OUTPUT_ROOT,
    )

Weight files and evaluation outputs are downloaded into the same directory
structure expected by the skip logic in each fase.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path


# ─── Download from Drive ──────────────────────────────────────────────────────

def restore_scenario_from_gdrive(
    gdrive_url: str,
    output_root: str,
    quiet: bool = False,
) -> dict:
    """Download previous scenario results from Google Drive via gdown.

    The shared Drive folder should mirror the output directory structure::

        <shared_folder>/
            prepared/          (splits, labels, ExDark_yolo)
            scenarios/
                S1_Raw/        (runs/, evaluation/)
                S2_HVI_CIDNet/
                ...

    Alternatively, share a *scenario-level* folder (e.g. ``S1_Raw/``) and set
    ``output_root`` to ``<kaggle_working>/scenarios/S1_Raw``.

    Args:
        gdrive_url: Google Drive folder URL **or** plain folder ID.
        output_root: Local directory to download into (e.g. ``/kaggle/working``).
        quiet: Suppress gdown progress bars.

    Returns:
        Dict with ``success`` flag and list of restored weight files.
    """
    try:
        import gdown
    except ImportError:
        print("[SYNC] Installing gdown ...")
        subprocess.check_call(["pip", "install", "-q", "gdown"])
        import gdown

    # Accept both full URL and bare folder ID
    url = gdrive_url.strip()
    if not url.startswith("http"):
        url = f"https://drive.google.com/drive/folders/{url}"

    os.makedirs(output_root, exist_ok=True)

    print(f"[SYNC] Downloading from Google Drive …")
    print(f"  URL    : {url}")
    print(f"  Target : {output_root}")

    try:
        gdown.download_folder(
            url=url,
            output=output_root,
            quiet=quiet,
            use_cookies=False,
        )
    except Exception as e:
        print(f"[SYNC] ✗ Download failed: {e}")
        print("  → Pastikan folder di-share sebagai 'Anyone with the link can view'")
        return {"success": False, "error": str(e)}

    # ── Report what was restored ─────────────────────────────────────────
    restored: dict = {"success": True, "files": []}

    scenarios = ["S1_Raw", "S2_HVI_CIDNet", "S3_RetinexFormer", "S4_LYT_Net"]
    for sc in scenarios:
        best_pt = os.path.join(output_root, "scenarios", sc, "runs", "weights", "best.pt")
        eval_dir = os.path.join(output_root, "scenarios", sc, "evaluation")
        if os.path.exists(best_pt):
            restored["files"].append(best_pt)
            print(f"  ✓ {sc}: best.pt found → training akan ter-skip")
        if os.path.isdir(eval_dir) and os.listdir(eval_dir):
            n = len(os.listdir(eval_dir))
            print(f"  ✓ {sc}: evaluation/ ({n} files)")

    prepared = os.path.join(output_root, "prepared")
    if os.path.isdir(prepared) and os.listdir(prepared):
        print(f"  ✓ prepared/ ditemukan → data preparation akan ter-skip")

    # Also check if best.pt is directly in output_root (scenario-level share)
    direct_best = os.path.join(output_root, "runs", "weights", "best.pt")
    if os.path.exists(direct_best):
        restored["files"].append(direct_best)
        print(f"  ✓ best.pt found at root level → training akan ter-skip")

    if not restored["files"]:
        print("[SYNC] ⚠ Tidak ditemukan best.pt setelah download")
        print("  → Pastikan folder yang di-share berisi struktur output yang benar")

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
