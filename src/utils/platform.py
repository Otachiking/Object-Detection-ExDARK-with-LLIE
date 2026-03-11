"""
Platform utilities for multi-environment support (Colab / Kaggle / Local).

Key helpers
-----------
stage_kaggle_weights(weights_input_dir, model_cache_dir)
    Copies pre-uploaded LLIE weights from Kaggle Dataset input to the exact
    paths + filenames expected by each enhancer. Run once per session before
    any enhancer is instantiated.

push_enhanced_to_kaggle(enhanced_dir, slug, ...)
    Creates or versions a Kaggle Dataset from the enhanced images directory,
    so enhanced images persist beyond the current session.

detect_platform()
    Returns 'kaggle', 'colab', or 'local'.

Weight filename mapping (Kaggle input → model_cache expected filename)
-----------------------------------------------------------------------
HVI-CIDNet:
  /kaggle/input/llie-model-cache/hvi_cidnet_LOL_v1.pth
  → model_cache/HVI-CIDNet/weights/pytorch_model.bin

RetinexFormer:
  /kaggle/input/llie-model-cache/retinexformer_LOL_v1.pth
  → model_cache/Retinexformer/weights/LOL_v1.pth

LYT-Net:
  /kaggle/input/llie-model-cache/lyt_net_lol.pth
  → model_cache/LYT-Net/weights/lyt_net_lol.pth  (already correct name)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# ─── Weight staging map ───────────────────────────────────────────────────────
# Keys match the enhancer_name as returned by scenario config ("hvi_cidnet" etc.)
# Each entry: source filename in Kaggle Dataset  →  (cache_subdir, target_filename)

KAGGLE_WEIGHT_MAP: dict[str, dict] = {
    "hvi_cidnet": {
        "source_filename": "hvi_cidnet_LOL_v1.pth",
        "cache_subdir":    "HVI-CIDNet",
        "target_filename": "pytorch_model.bin",       # ← expected by hvi_cidnet.py
    },
    "retinexformer": {
        "source_filename": "retinexformer_LOL_v1.pth",
        "cache_subdir":    "Retinexformer",
        "target_filename": "LOL_v1.pth",              # ← expected by retinexformer.py
    },
    "lyt_net": {
        "source_filename": "lyt_net_lol.pth",
        "cache_subdir":    "LYT-Net",
        "target_filename": "lyt_net_lol.pth",         # ← already correct
    },
}


# ─── Platform detection ───────────────────────────────────────────────────────

def detect_platform() -> str:
    """Returns 'kaggle', 'colab', or 'local'."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if os.path.exists("/content"):
        return "colab"
    return "local"


def resolve_kaggle_exdark(search_root: str = "/kaggle/input") -> str | None:
    """Auto-detect the ExDark dataset root inside /kaggle/input.

    Searches for imageclasslist.txt which is uniquely inside ExDark's Groundtruth/
    directory. Works regardless of how the dataset was mounted (slug-based or
    via Upload button which uses /kaggle/input/datasets/).

    Returns the directory that contains both Dataset/ and Groundtruth/ folders,
    or None if not found.
    """
    import glob as _glob

    # Search up to 3 levels deep for imageclasslist.txt
    candidates = _glob.glob(
        os.path.join(search_root, "**/imageclasslist.txt"), recursive=True
    )
    for classlist_path in sorted(candidates):
        # classlist is at: <exdark_root>/Groundtruth/imageclasslist.txt
        exdark_root = str(Path(classlist_path).parent.parent)
        # Validate expected structure
        if (
            os.path.isdir(os.path.join(exdark_root, "Dataset"))
            and os.path.isdir(os.path.join(exdark_root, "Groundtruth"))
        ):
            return exdark_root
    return None


def resolve_kaggle_llie_input(search_root: str = "/kaggle/input") -> str | None:
    """Auto-detect the LLIE weights input directory inside /kaggle/input.

    Looks for any of the known weight filenames uploaded to Kaggle.
    Works whether they're in /kaggle/input/llie-model-cache/ or /kaggle/input/datasets/.

    Returns the directory containing the weight files, or None.
    """
    import glob as _glob

    known_files = [
        "hvi_cidnet_LOL_v1.pth",
        "retinexformer_LOL_v1.pth",
        "lyt_net_lol.pth",
        "pytorch_model.bin",
        "LOL_v1.pth",
    ]
    for fname in known_files:
        matches = _glob.glob(os.path.join(search_root, "**", fname), recursive=True)
        if matches:
            return str(Path(matches[0]).parent)
    return None


# ─── Weight staging ───────────────────────────────────────────────────────────

def stage_kaggle_weights(
    weights_input_dir: str,
    model_cache_dir: str,
    enhancer_key: str | None = None,
) -> None:
    """Copy LLIE weights from Kaggle Dataset input to model_cache with correct names.

    Called from the '0.4 · Kaggle Weight Staging' notebook cell.
    Safe to call multiple times — skips files that are already in place.

    Args:
        weights_input_dir: Path to the Kaggle Dataset input, e.g.
                           /kaggle/input/llie-model-cache
        model_cache_dir:   model_cache root path resolved by config, e.g.
                           /kaggle/working/model_cache
        enhancer_key:      If given, only stage this one enhancer's weight.
                           Otherwise, stage all three.
    """
    src_root = Path(weights_input_dir)
    if not src_root.exists():
        print(f"[WARN] Kaggle weight input not found: {weights_input_dir}")
        print("       Enhancer weights will be downloaded at runtime instead.")
        return

    entries = (
        {enhancer_key: KAGGLE_WEIGHT_MAP[enhancer_key]}
        if enhancer_key and enhancer_key in KAGGLE_WEIGHT_MAP
        else KAGGLE_WEIGHT_MAP
    )

    print("[Kaggle Weight Staging]")
    staged = 0
    for key, mapping in entries.items():
        src_file = src_root / mapping["source_filename"]
        dst_dir  = Path(model_cache_dir) / mapping["cache_subdir"] / "weights"
        dst_file = dst_dir / mapping["target_filename"]

        if dst_file.exists():
            print(f"  [SKIP] {key}: {mapping['target_filename']} already in cache")
            continue

        if not src_file.exists():
            print(f"  [MISS] {key}: {mapping['source_filename']} not in {weights_input_dir}")
            print(f"         → Will fall back to internet download")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        size_mb = dst_file.stat().st_size / (1024 ** 2)
        print(f"  [OK]  {key}: {src_file.name} → .../{mapping['cache_subdir']}/weights/{mapping['target_filename']}  ({size_mb:.1f} MB)")
        staged += 1

    if staged:
        print(f"  Staged {staged} weight file(s) → {model_cache_dir}")
    else:
        print("  All weights already in cache — nothing to copy")


# ─── Enhanced dataset: Kaggle input check ────────────────────────────────────

def get_kaggle_enhanced_input(enhancer_name: str) -> str | None:
    """Return path to pre-uploaded Kaggle enhanced dataset, or None if not available.

    The expected slug for the Kaggle Dataset is:
        exdark-enhanced-{enhancer_name_slug}
    e.g.:  exdark-enhanced-hvi-cidnet
           exdark-enhanced-retinexformer
           exdark-enhanced-lyt-net

    If that dataset is added as an Input in the Kaggle notebook, its images/
    folder will be at /kaggle/input/exdark-enhanced-{slug}/images/
    """
    slug       = enhancer_name.lower().replace("_", "-")
    input_path = Path(f"/kaggle/input/exdark-enhanced-{slug}")
    if input_path.exists() and (input_path / "images").is_dir():
        return str(input_path)
    return None


def setup_enhanced_from_kaggle(
    kaggle_input_path: str,
    enhanced_dir: str,
    yolo_dir: str,
) -> bool:
    """Symlink pre-enhanced images + regenerate dataset.yaml so training can proceed.

    If a pre-uploaded Kaggle Dataset with enhanced images is available, creates
    symlinks from the writable enhanced_dir to the read-only Kaggle input, then
    generates dataset.yaml pointing to those paths.

    Returns True if setup succeeded (enhancement step can be skipped).
    """
    from src.data.build_yolo_dataset import generate_enhanced_dataset_yaml

    kaggle_path  = Path(kaggle_input_path)
    enhanced_path = Path(enhanced_dir)

    try:
        for split in ["train", "val", "test"]:
            src_images = kaggle_path / "images" / split
            dst_images = enhanced_path / "images" / split
            if not src_images.is_dir():
                print(f"  [WARN] {split} images not found in Kaggle input: {src_images}")
                return False
            if dst_images.exists():
                continue  # already set up
            dst_images.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(src_images), str(dst_images))

        # Use labels from original YOLO dir (same as enhance_dataset does)
        for split in ["train", "val", "test"]:
            src_labels = Path(yolo_dir) / "labels" / split
            dst_labels = enhanced_path / "labels" / split
            if src_labels.is_dir() and not dst_labels.exists():
                dst_labels.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(src_labels), str(dst_labels))

        # Generate fresh dataset.yaml with absolute paths (required for Ultralytics)
        output_yaml = os.path.join(enhanced_dir, "dataset.yaml")
        generate_enhanced_dataset_yaml(
            enhanced_images_dir=enhanced_dir,
            yolo_labels_dir=yolo_dir,
            output_yaml_path=output_yaml,
        )
        print(f"  [OK] Enhanced images symlinked from: {kaggle_input_path}")
        return True

    except Exception as exc:
        print(f"  [ERROR] Could not set up pre-enhanced dataset: {exc}")
        return False


# ─── Push enhanced dataset to Kaggle ─────────────────────────────────────────

def push_enhanced_to_kaggle(
    enhanced_dir: str,
    enhancer_name: str,
    username: str = "otachiking",
    update: bool = False,
) -> None:
    """Push enhanced images directory to Kaggle as a new Dataset (or new version).

    Creates a dataset with slug: exdark-enhanced-{enhancer_name_slug}
    e.g.: otachiking/exdark-enhanced-hvi-cidnet

    On subsequent runs, add this dataset as an Input in the Kaggle notebook
    and enhancement will be skipped automatically.

    Note: Unlike adding to exdark-dataset (which would require re-uploading
    all 1.4 GB every version), a separate per-enhancer dataset is much smaller
    (~200–400 MB) and only contains the derived enhanced images.

    Args:
        enhanced_dir:   Path to enhanced/ directory with images/ and labels/
        enhancer_name:  Enhancer key, e.g. 'hvi_cidnet'
        username:       Kaggle username
        update:         If True, creates a new version of an existing dataset.
                        If False, creates a brand new dataset.
    """
    slug       = f"exdark-enhanced-{enhancer_name.lower().replace('_', '-')}"
    title      = f"ExDark Enhanced — {enhancer_name.replace('_', ' ').title()}"
    src_path   = Path(enhanced_dir)

    if not src_path.exists():
        print(f"[ERROR] Enhanced directory not found: {enhanced_dir}")
        print("        Run Fase 2 (Enhancement) first.")
        return

    # Check if there are actually images to push
    total_images = sum(
        len([f for f in (src_path / "images" / s).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for s in ["train", "val", "test"]
        if (src_path / "images" / s).is_dir()
    )
    if total_images == 0:
        print(f"[ERROR] No images found in {enhanced_dir}/images/")
        return

    print(f"[Kaggle Push] Enhanced dataset")
    print(f"  Source    : {enhanced_dir}  ({total_images} images)")
    print(f"  Target    : {username}/{slug}")

    # Write metadata into the enhanced_dir
    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": f"ExDark {total_images} images enhanced with {enhancer_name.replace('_', ' ')}",
    }
    meta_path = src_path / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if update:
        cmd = ["kaggle", "datasets", "version",
               "-p", str(src_path), "-m", f"Enhanced images — {enhancer_name}",
               "-r", "zip", "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create",
               "-p", str(src_path),
               "-r", "zip", "--dir-mode", "zip"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [OK]  https://www.kaggle.com/datasets/{username}/{slug}")
        print()
        print("  Next run: add this dataset as Input in your Kaggle notebook,")
        print(f"  enhancement will be skipped automatically.")
    else:
        stderr = result.stderr.strip()
        print(f"  [ERROR] {stderr}")
        if "already exists" in stderr.lower() or "conflict" in stderr.lower():
            print("  → Re-run with update=True to create a new version")
