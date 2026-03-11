#!/usr/bin/env python3
"""
upload_kaggle_datasets.py
=========================
Upload ExDark dataset & LLIE model weights ke Kaggle Datasets via CLI.

Jalankan sekali saja — setelah itu dataset permanen di Kaggle dan bisa
di-mount sebagai Input di setiap notebook.

Prerequisite
------------
1. pip install kaggle gdown
2. Buat API token di https://www.kaggle.com/settings → "Create New Token"
   Simpan file ke:  C:\\Users\\<username>\\.kaggle\\kaggle.json   (Windows)
                    ~/.kaggle/kaggle.json                        (Linux/Mac)

Usage
-----
    python scripts/upload_kaggle_datasets.py                   # upload keduanya
    python scripts/upload_kaggle_datasets.py --only exdark     # hanya ExDark
    python scripts/upload_kaggle_datasets.py --only llie       # hanya LLIE weights
    python scripts/upload_kaggle_datasets.py --keep-staging    # jangan hapus staging
    python scripts/upload_kaggle_datasets.py --update          # update dataset yang sudah ada

Hasil di Kaggle
---------------
- https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/exdark-dataset
  → /kaggle/input/exdark-dataset/Dataset/...
  → /kaggle/input/exdark-dataset/Groundtruth/...

- https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/llie-model-cache
  → /kaggle/input/llie-model-cache/hvi_cidnet_LOL_v1.pth
  → /kaggle/input/llie-model-cache/retinexformer_LOL_v1.pth
  → /kaggle/input/llie-model-cache/lyt_net_lol.pth   (auto-downloaded)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

KAGGLE_USERNAME = "otachiking"

# Paths — relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STAGING_ROOT = PROJECT_ROOT / "_kaggle_staging"

# ExDark source
EXDARK_SOURCE = Path(r"c:\CODE\KULIAH\TA\DATASET\exdark-dataset")
EXDARK_ZIP    = Path(r"c:\CODE\KULIAH\TA\DATASET\exdark-dataset.zip")

# LLIE weights source
LLIE_WEIGHTS_DIR = PROJECT_ROOT / "llie-weights"

# LYT-Net .pth download (kode expects lyt_net_lol.pth, user hanya punya .h5)
LYTNET_GDRIVE_ID = "1GeEkasO2ubFi847pzrxfQ1fB3Y9NuhZ1"
LYTNET_WEIGHT_NAME = "lyt_net_lol.pth"

# Dataset slugs
EXDARK_SLUG = "exdark-dataset"
LLIE_SLUG   = "llie-model-cache"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def run_cmd(cmd: list[str], *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a command, print it, and return the result."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def check_tool(name: str, install_hint: str) -> bool:
    """Check if a CLI tool is available."""
    result = subprocess.run(
        ["where" if os.name == "nt" else "which", name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] '{name}' tidak ditemukan.")
        print(f"        Install: {install_hint}")
        return False
    return True


def check_kaggle_auth() -> bool:
    """Verify Kaggle API credentials exist (env var or kaggle.json)."""
    # Option A: KAGGLE_API_TOKEN env var (new token format, CLI >= 1.8.0)
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("  [OK] Kaggle credentials: KAGGLE_API_TOKEN env var detected")
        return True

    # Option B: KAGGLE_USERNAME + KAGGLE_KEY env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("  [OK] Kaggle credentials: KAGGLE_USERNAME + KAGGLE_KEY env vars detected")
        return True

    # Option C: Legacy kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        if os.name != "nt":
            os.chmod(kaggle_json, 0o600)
        print(f"  [OK] Kaggle credentials: {kaggle_json}")
        return True

    print(f"\n[ERROR] Kaggle credentials tidak ditemukan.")
    print("        Opsi 1 (Recommended): set env var KAGGLE_API_TOKEN=<token>")
    print("                              (Generate di kaggle.com/settings → API Tokens)")
    print("        Opsi 2 (Legacy):      Download kaggle.json ke ~/.kaggle/kaggle.json")
    return False


def write_metadata(staging_dir: Path, slug: str, title: str, *, subtitle: str = "") -> None:
    """Write dataset-metadata.json for Kaggle CLI."""
    metadata = {
        "title": title,
        "id": f"{KAGGLE_USERNAME}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    if subtitle:
        metadata["subtitle"] = subtitle

    meta_path = staging_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  [OK] Metadata: {meta_path}")


def upload_dataset(staging_dir: Path, *, update: bool = False) -> None:
    """Upload staging directory as a Kaggle Dataset."""
    if update:
        cmd = ["kaggle", "datasets", "version",
               "-p", str(staging_dir),
               "-m", "Update dataset",
               "-r", "zip", "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create",
               "-p", str(staging_dir),
               "-r", "zip", "--dir-mode", "zip"]
    run_cmd(cmd)


# ─── ExDark Dataset ──────────────────────────────────────────────────────────

def stage_exdark(staging_dir: Path) -> bool:
    """Stage ExDark dataset for Kaggle upload.

    Strategy: jika ada .zip, upload langsung dari zip (lebih cepat).
    Jika tidak, copy folder Dataset/ dan Groundtruth/.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Option A: Use existing zip (much faster for 7K images)
    if EXDARK_ZIP.exists():
        print(f"\n[ExDark] Ditemukan zip: {EXDARK_ZIP}")
        print("[ExDark] Akan upload via zip — ini JAUH lebih cepat.")

        # Kaggle datasets create needs dataset-metadata.json INSIDE the folder
        # But with zip, we extract structure. Let's use folder approach with zip.
        # Actually: kaggle expects either raw files or we point it to the folder.
        # Best approach: copy zip into staging + metadata, then use --dir-mode zip
        # Actually kaggle CLI uploads the folder contents. For a zip source,
        # it's more efficient to just extract and use the folder.

        # Simpler: just point to the extracted folder if it exists
        if EXDARK_SOURCE.exists():
            print(f"[ExDark] Menggunakan folder yang sudah ada: {EXDARK_SOURCE}")
            return _stage_exdark_from_folder(staging_dir)
        else:
            print("[ExDark] Folder sumber tidak ada, extract dari zip...")
            import zipfile
            with zipfile.ZipFile(EXDARK_ZIP, "r") as zf:
                zf.extractall(staging_dir)
            return True

    # Option B: Copy from extracted folder
    if EXDARK_SOURCE.exists():
        return _stage_exdark_from_folder(staging_dir)

    print(f"\n[ERROR] ExDark dataset tidak ditemukan!")
    print(f"        Expected zip : {EXDARK_ZIP}")
    print(f"        Expected dir : {EXDARK_SOURCE}")
    return False


def _stage_exdark_from_folder(staging_dir: Path) -> bool:
    """Copy Dataset/ and Groundtruth/ via directory junctions (Windows) or symlinks."""
    for subdir in ["Dataset", "Groundtruth"]:
        src = EXDARK_SOURCE / subdir
        dst = staging_dir / subdir
        if dst.exists():
            print(f"  [SKIP] {dst} sudah ada")
            continue
        if not src.exists():
            print(f"  [ERROR] {src} tidak ditemukan!")
            return False

        # Use junction/symlink to avoid copying 7K files
        if os.name == "nt":
            # Windows: directory junction (no admin rights needed)
            run_cmd(["cmd", "/c", "mklink", "/J", str(dst), str(src)])
        else:
            dst.symlink_to(src)
        print(f"  [OK] Linked: {dst} → {src}")

    return True


# ─── LLIE Model Weights ──────────────────────────────────────────────────────

def stage_llie_weights(staging_dir: Path) -> bool:
    """Stage LLIE model weights for Kaggle upload.

    Upload as-is dari llie-weights/ folder, plus download LYT-Net .pth
    yang belum ada (user hanya punya versi .h5 yang tidak kompatibel).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    if not LLIE_WEIGHTS_DIR.exists():
        print(f"\n[ERROR] Folder llie-weights/ tidak ditemukan: {LLIE_WEIGHTS_DIR}")
        return False

    success = True

    # Copy existing weight files (only the ones that are actually usable)
    usable_files = {
        "hvi_cidnet_LOL_v1.pth":       "HVI-CIDNet (LOL v1)",
        "retinexformer_LOL_v1.pth":    "RetinexFormer (LOL v1)",
    }

    for filename, desc in usable_files.items():
        src = LLIE_WEIGHTS_DIR / filename
        dst = staging_dir / filename
        if dst.exists():
            print(f"  [SKIP] {filename} sudah ada di staging")
            continue
        if not src.exists():
            print(f"  [WARN] {filename} ({desc}) tidak ditemukan di {LLIE_WEIGHTS_DIR}")
            success = False
            continue
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"  [OK] {filename} ({desc}) — {size_mb:.1f} MB")

    # Download LYT-Net .pth via gdown (user hanya punya .h5 yang incompatible)
    lytnet_dst = staging_dir / LYTNET_WEIGHT_NAME
    if lytnet_dst.exists():
        print(f"  [SKIP] {LYTNET_WEIGHT_NAME} sudah ada di staging")
    else:
        print(f"\n  [DOWNLOAD] LYT-Net weight (.pth) dari Google Drive...")
        print(f"  File .h5 yang ada TIDAK kompatibel (Keras format, kode pakai PyTorch)")
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={LYTNET_GDRIVE_ID}"
            gdown.download(url, str(lytnet_dst), quiet=False)
            if lytnet_dst.exists():
                size_mb = lytnet_dst.stat().st_size / (1024 * 1024)
                print(f"  [OK] {LYTNET_WEIGHT_NAME} — {size_mb:.1f} MB")
            else:
                print(f"  [ERROR] Download gagal — file tidak ada setelah gdown")
                success = False
        except ImportError:
            print("  [ERROR] gdown tidak terinstall. Jalankan: pip install gdown")
            success = False
        except Exception as e:
            print(f"  [ERROR] Download gagal: {e}")
            success = False

    # Skip incompatible files with explanation
    skipped = {
        "lyt_net_LOL_v1.h5": "Format Keras .h5, kode pakai torch.load() → INCOMPATIBLE",
        "PSNR_24.74.pth":    "Bukan weight LYT-Net (model tidak diketahui)",
    }
    for filename, reason in skipped.items():
        if (LLIE_WEIGHTS_DIR / filename).exists():
            print(f"  [SKIP] {filename} — {reason}")

    return success


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upload ExDark & LLIE weights ke Kaggle Datasets via CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only", choices=["exdark", "llie"],
        help="Upload hanya satu dataset (default: keduanya)",
    )
    parser.add_argument(
        "--keep-staging", action="store_true",
        help="Jangan hapus folder staging setelah upload",
    )
    parser.add_argument(
        "--update", action="store_true",
        help="Update dataset yang sudah ada (bukan create baru)",
    )
    parser.add_argument(
        "--stage-only", action="store_true",
        help="Hanya staging, jangan upload (untuk debug/cek)",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  Kaggle Dataset Uploader")
    print("  ExDark + LLIE Model Weights")
    print("=" * 64)

    # ── Pre-flight checks ──
    print("\n[1/5] Pre-flight checks...")
    ok = True
    ok = check_tool("kaggle", "pip install kaggle") and ok
    ok = check_kaggle_auth() and ok

    if not ok:
        print("\n[ABORT] Pre-flight gagal. Fix errors di atas, lalu jalankan ulang.")
        sys.exit(1)

    do_exdark = args.only is None or args.only == "exdark"
    do_llie   = args.only is None or args.only == "llie"

    # ── Stage ExDark ──
    if do_exdark:
        print(f"\n[2/5] Staging ExDark dataset...")
        exdark_staging = STAGING_ROOT / EXDARK_SLUG
        if not stage_exdark(exdark_staging):
            print("[ABORT] Staging ExDark gagal.")
            sys.exit(1)
        write_metadata(
            exdark_staging,
            slug=EXDARK_SLUG,
            title="ExDark Dataset",
            subtitle="Exclusively Dark Image Dataset for object detection (7K images, 12 classes)",
        )
    else:
        print("\n[2/5] Skip ExDark (--only llie)")

    # ── Stage LLIE Weights ──
    if do_llie:
        print(f"\n[3/5] Staging LLIE model weights...")
        llie_staging = STAGING_ROOT / LLIE_SLUG
        if not stage_llie_weights(llie_staging):
            print("[WARN] Beberapa weight gagal di-stage. Lanjut upload yang berhasil.")
        write_metadata(
            llie_staging,
            slug=LLIE_SLUG,
            title="LLIE Model Cache (Pretrained Weights)",
            subtitle="HVI-CIDNet, RetinexFormer, LYT-Net pretrained on LOL v1",
        )
    else:
        print("\n[3/5] Skip LLIE weights (--only exdark)")

    if args.stage_only:
        print(f"\n[DONE] Staging selesai di: {STAGING_ROOT}")
        print("       Jalankan ulang tanpa --stage-only untuk upload.")
        return

    # ── Upload ──
    if do_exdark:
        print(f"\n[4/5] Uploading ExDark dataset ke Kaggle...")
        print(f"       Slug: {KAGGLE_USERNAME}/{EXDARK_SLUG}")
        print(f"       Ini bisa 10–30 menit (7K images)...")
        upload_dataset(STAGING_ROOT / EXDARK_SLUG, update=args.update)
        print(f"  [OK] https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/{EXDARK_SLUG}")
    else:
        print("\n[4/5] Skip upload ExDark")

    if do_llie:
        print(f"\n[5/5] Uploading LLIE weights ke Kaggle...")
        print(f"       Slug: {KAGGLE_USERNAME}/{LLIE_SLUG}")
        upload_dataset(STAGING_ROOT / LLIE_SLUG, update=args.update)
        print(f"  [OK] https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/{LLIE_SLUG}")
    else:
        print("\n[5/5] Skip upload LLIE weights")

    # ── Cleanup ──
    if not args.keep_staging and STAGING_ROOT.exists():
        print(f"\n[CLEANUP] Menghapus staging folder: {STAGING_ROOT}")
        # Remove junctions first (Windows), then the rest
        for item in STAGING_ROOT.rglob("*"):
            if item.is_dir() and _is_junction(item):
                run_cmd(["cmd", "/c", "rmdir", str(item)], check=False)
        shutil.rmtree(STAGING_ROOT, ignore_errors=True)
        print("  [OK] Staging folder dihapus")
    elif args.keep_staging:
        print(f"\n[INFO] Staging folder tetap di: {STAGING_ROOT}")

    # ── Summary ──
    print("\n" + "=" * 64)
    print("  SELESAI!")
    print("=" * 64)
    print(f"""
Langkah selanjutnya di Kaggle Notebook:
1. Buat notebook baru di kaggle.com
2. Klik 'Add Data' → cari '{EXDARK_SLUG}' & '{LLIE_SLUG}'
3. Tambahkan keduanya sebagai Input
4. Dataset tersedia di:
   /kaggle/input/{EXDARK_SLUG}/Dataset/...
   /kaggle/input/{EXDARK_SLUG}/Groundtruth/...
   /kaggle/input/{LLIE_SLUG}/hvi_cidnet_LOL_v1.pth
   /kaggle/input/{LLIE_SLUG}/retinexformer_LOL_v1.pth
   /kaggle/input/{LLIE_SLUG}/{LYTNET_WEIGHT_NAME}
""")


def _is_junction(path: Path) -> bool:
    """Check if a path is a Windows junction / reparse point."""
    if os.name != "nt":
        return path.is_symlink()
    try:
        import ctypes
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except Exception:
        return False


if __name__ == "__main__":
    main()
