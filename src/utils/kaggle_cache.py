"""
Push all scenario results (prepared data, enhanced images, trained weights,
evaluation outputs) as a single Kaggle Dataset: exdark-scenario-cache.

This script runs ONLY on Kaggle (it uses the Kaggle CLI internally).
The created dataset can be added as Input to skip Fase 1-6 in future sessions.

Usage (in a notebook cell on Kaggle)::

    from src.utils.kaggle_cache import push_scenario_cache, restore_scenario_cache

    # After training is done — push:
    push_scenario_cache(output_root="/kaggle/working")

    # On next session — restore:
    restore_scenario_cache(output_root="/kaggle/working")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


CACHE_SLUG = "exdark-scenario-cache"
SCENARIOS = ["S1_Raw", "S2_HVI_CIDNet", "S3_RetinexFormer", "S4_LYT_Net"]


# ─── Push ─────────────────────────────────────────────────────────────────────

def push_scenario_cache(
    output_root: str = "/kaggle/working",
    username: str | None = None,
    update: bool = False,
) -> None:
    """Create or update the exdark-scenario-cache Kaggle Dataset.

    Includes:
    - prepared/  (splits, YOLO labels, YOLO dir with dataset.yaml — images
      are symlinked, so we copy the yaml + labels + split files only)
    - scenarios/{name}/runs/weights/best.pt  (trained weights)
    - scenarios/{name}/evaluation/*          (all JSON + figures)
    - scenarios/{name}/enhanced/dataset.yaml + images/ (for S2-S4)

    Args:
        output_root: Working directory that contains prepared/ and scenarios/.
        username: Kaggle username. Auto-detected if not provided.
        update: True to update an existing dataset (new version).
    """
    if username is None:
        username = _detect_kaggle_username()

    staging = os.path.join(output_root, "_scenario_cache_staging")
    if os.path.exists(staging):
        shutil.rmtree(staging)

    print(f"[CACHE] Staging scenario results → {staging}")

    # ── prepared/ (splits + labels + yaml, NO raw images to save space) ───
    prepared_src = os.path.join(output_root, "prepared")
    if os.path.isdir(prepared_src):
        _stage_prepared(prepared_src, os.path.join(staging, "prepared"))

    # ── each scenario ────────────────────────────────────────────────────
    for sn in SCENARIOS:
        sc_dir = os.path.join(output_root, "scenarios", sn)
        if not os.path.isdir(sc_dir):
            print(f"  {sn}: not found, skipping")
            continue

        dst_sc = os.path.join(staging, "scenarios", sn)

        # weights
        for wname in ("best.pt", "last.pt"):
            w_src = os.path.join(sc_dir, "runs", "weights", wname)
            if os.path.isfile(w_src):
                w_dst = os.path.join(dst_sc, "runs", "weights", wname)
                os.makedirs(os.path.dirname(w_dst), exist_ok=True)
                shutil.copy2(w_src, w_dst)

        # results.csv + args.yaml from runs/
        for fname in ("results.csv", "args.yaml", "config_snapshot.yaml"):
            f_src = os.path.join(sc_dir, "runs", fname)
            if os.path.isfile(f_src):
                f_dst = os.path.join(dst_sc, "runs", fname)
                os.makedirs(os.path.dirname(f_dst), exist_ok=True)
                shutil.copy2(f_src, f_dst)

        # evaluation/
        eval_src = os.path.join(sc_dir, "evaluation")
        if os.path.isdir(eval_src):
            eval_dst = os.path.join(dst_sc, "evaluation")
            shutil.copytree(eval_src, eval_dst, dirs_exist_ok=True)

        # enhanced/ (S2-S4 only) — full images + dataset.yaml
        enh_src = os.path.join(sc_dir, "enhanced")
        if os.path.isdir(enh_src):
            enh_dst = os.path.join(dst_sc, "enhanced")
            shutil.copytree(enh_src, enh_dst, dirs_exist_ok=True)

        _sz = _dir_size_mb(dst_sc)
        print(f"  {sn}: staged ({_sz:.0f} MB)")

    # ── Write dataset-metadata.json ──────────────────────────────────────
    meta = {
        "title": "ExDark Scenario Cache",
        "id": f"{username}/{CACHE_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = os.path.join(staging, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_mb = _dir_size_mb(staging)
    print(f"\n[CACHE] Total staging: {total_mb:.0f} MB")

    # ── Upload ──────────────────────────────────────────────────────────
    cmd = ["kaggle", "datasets"]
    cmd += ["version", "-m", "Auto-updated scenario cache"] if update else ["create"]
    cmd += ["-p", staging, "--dir-mode", "zip"]

    print(f"[CACHE] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        if "already exists" in (result.stderr or ""):
            print("[CACHE] Dataset exists. Retrying with --version ...")
            cmd2 = ["kaggle", "datasets", "version", "-m", "Auto-updated",
                    "-p", staging, "--dir-mode", "zip"]
            subprocess.run(cmd2, check=True)
    else:
        print(f"[CACHE] ✓ Pushed as: {username}/{CACHE_SLUG}")

    # Cleanup
    shutil.rmtree(staging, ignore_errors=True)


def _stage_prepared(src: str, dst: str) -> None:
    """Copy prepared/ selectively — splits + labels + yaml, skip raw images."""
    os.makedirs(dst, exist_ok=True)

    # splits/
    splits_src = os.path.join(src, "splits")
    if os.path.isdir(splits_src):
        shutil.copytree(splits_src, os.path.join(dst, "splits"), dirs_exist_ok=True)

    # ExDark_yolo_labels/
    labels_src = os.path.join(src, "ExDark_yolo_labels")
    if os.path.isdir(labels_src):
        shutil.copytree(labels_src, os.path.join(dst, "ExDark_yolo_labels"),
                        dirs_exist_ok=True)

    # ExDark_yolo/ — copy ONLY dataset.yaml + labels/, skip images/ (too large)
    yolo_src = os.path.join(src, "ExDark_yolo")
    if os.path.isdir(yolo_src):
        yolo_dst = os.path.join(dst, "ExDark_yolo")
        os.makedirs(yolo_dst, exist_ok=True)
        yaml_f = os.path.join(yolo_src, "dataset.yaml")
        if os.path.isfile(yaml_f):
            shutil.copy2(yaml_f, os.path.join(yolo_dst, "dataset.yaml"))
        labels_d = os.path.join(yolo_src, "labels")
        if os.path.isdir(labels_d):
            shutil.copytree(labels_d, os.path.join(yolo_dst, "labels"),
                            dirs_exist_ok=True)

    sz = _dir_size_mb(dst)
    print(f"  prepared/: staged ({sz:.0f} MB)")


# ─── Restore ──────────────────────────────────────────────────────────────────

def restore_scenario_cache(
    output_root: str = "/kaggle/working",
    cache_input: str | None = None,
) -> dict:
    """Restore scenario results from a mounted Kaggle Dataset.

    Creates symlinks (or copies where needed) from the Kaggle input directory
    to the expected output structure. After restore, the skip logic in each
    notebook fase will find existing files and skip processing.

    Args:
        output_root: Working directory (e.g. /kaggle/working).
        cache_input: Path to the mounted cache dataset. Auto-detected if None.

    Returns:
        Dict with restored scenario names and file counts.
    """
    if cache_input is None:
        cache_input = _find_cache_input()

    if cache_input is None:
        print("[CACHE] ⚠ exdark-scenario-cache not found in /kaggle/input/")
        print("  → Tambahkan dataset 'exdark-scenario-cache' sebagai Input")
        return {"restored": []}

    print(f"[CACHE] Restoring from: {cache_input}")
    restored = []

    # ── prepared/ (symlink where possible, copy yaml) ────────────────────
    prep_src = os.path.join(cache_input, "prepared")
    prep_dst = os.path.join(output_root, "prepared")
    if os.path.isdir(prep_src) and not os.path.exists(prep_dst):
        # Symlink the whole prepared dir
        os.symlink(prep_src, prep_dst)
        print(f"  ✓ prepared/ → symlinked")

    # ── scenarios ────────────────────────────────────────────────────────
    for sn in SCENARIOS:
        sc_src = os.path.join(cache_input, "scenarios", sn)
        if not os.path.isdir(sc_src):
            continue

        sc_dst = os.path.join(output_root, "scenarios", sn)

        # Weights — need to be in writable location (YOLO may write to
        # the same directory), so COPY them
        for wname in ("best.pt", "last.pt"):
            w_src = os.path.join(sc_src, "runs", "weights", wname)
            w_dst = os.path.join(sc_dst, "runs", "weights", wname)
            if os.path.isfile(w_src) and not os.path.exists(w_dst):
                os.makedirs(os.path.dirname(w_dst), exist_ok=True)
                shutil.copy2(w_src, w_dst)

        # runs/ metadata (results.csv, args.yaml)
        for fname in ("results.csv", "args.yaml", "config_snapshot.yaml"):
            f_src = os.path.join(sc_src, "runs", fname)
            f_dst = os.path.join(sc_dst, "runs", fname)
            if os.path.isfile(f_src) and not os.path.exists(f_dst):
                os.makedirs(os.path.dirname(f_dst), exist_ok=True)
                shutil.copy2(f_src, f_dst)

        # evaluation/ — symlink (read-only is fine)
        eval_src = os.path.join(sc_src, "evaluation")
        eval_dst = os.path.join(sc_dst, "evaluation")
        if os.path.isdir(eval_src) and not os.path.exists(eval_dst):
            os.makedirs(sc_dst, exist_ok=True)
            os.symlink(eval_src, eval_dst)

        # enhanced/ — symlink
        enh_src = os.path.join(sc_src, "enhanced")
        enh_dst = os.path.join(sc_dst, "enhanced")
        if os.path.isdir(enh_src) and not os.path.exists(enh_dst):
            os.makedirs(sc_dst, exist_ok=True)
            os.symlink(enh_src, enh_dst)

        # Report
        has_best = os.path.exists(os.path.join(sc_dst, "runs", "weights", "best.pt"))
        has_eval = os.path.exists(os.path.join(sc_dst, "evaluation"))
        has_enh  = os.path.exists(os.path.join(sc_dst, "enhanced"))
        parts = []
        if has_best: parts.append("best.pt")
        if has_eval: parts.append("evaluation")
        if has_enh:  parts.append("enhanced")
        if parts:
            restored.append(sn)
            print(f"  ✓ {sn}: {', '.join(parts)}")

    if restored:
        print(f"\n[CACHE] ✓ Restored {len(restored)} scenarios: {restored}")
        print("  → Fase yang sudah ada akan otomatis ter-skip")
    else:
        print("[CACHE] ⚠ Tidak ada scenario results ditemukan di cache")

    return {"restored": restored}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_cache_input() -> str | None:
    """Auto-detect the scenario cache dataset in /kaggle/input/."""
    import glob

    # Try exact slug first
    exact = f"/kaggle/input/{CACHE_SLUG}"
    if os.path.isdir(exact) and (
        os.path.isdir(os.path.join(exact, "scenarios")) or
        os.path.isdir(os.path.join(exact, "prepared"))
    ):
        return exact

    # Search for scenarios/ directory anywhere in /kaggle/input/
    for p in glob.glob("/kaggle/input/*/scenarios"):
        if os.path.isdir(p):
            return os.path.dirname(p)

    return None


def _detect_kaggle_username() -> str:
    """Get Kaggle username from environment or CLI."""
    # From Kaggle kernel env
    user = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USER_SECRETS_TOKEN_USERNAME")
    if user:
        return user
    # Fallback: try kaggle.json
    cfg_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            return json.load(f).get("username", "otachiking")
    return "otachiking"


def _dir_size_mb(path: str) -> float:
    """Total size of directory in MB."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / 1e6
