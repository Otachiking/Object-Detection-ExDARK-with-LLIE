"""
Run Fase 1 (Data Preparation) locally on your laptop.

This is MUCH faster than running on Colab because it avoids Google Drive I/O
latency. After running, upload the output folder to Google Drive.

Fase 1.1: Parse official ExDark splits (train/val/test)
Fase 1.2: Convert ExDark annotations → YOLO format
Fase 1.3: Build YOLO dataset (resize 640px + organize)
Fase 1.4: Validate dataset integrity

Usage:
    cd TA-IQBAL-ObjectDetectionExDARKwithLLIE
    python scripts/run_local_fase1.py

Output:
    outputs/
    ├── splits/              → train.txt, val.txt, test.txt, manifest.txt
    ├── ExDark_yolo_labels/  → 7363 YOLO .txt label files
    └── ExDark_yolo/         → images/{train,val,test}/ + labels/{train,val,test}/ + dataset.yaml

After completion, upload these 3 folders to Google Drive:
    outputs/splits/           →  TA-IQBAL/splits/
    outputs/ExDark_yolo_labels/ →  TA-IQBAL/ExDark_yolo_labels/
    outputs/ExDark_yolo/      →  TA-IQBAL/ExDark_yolo/
"""

import os
import sys
import time

# --- Resolve paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)  # Parent of scripts/

# Add repo root so we can import src.*
sys.path.insert(0, REPO_DIR)

# --- Configuration ---
# ExDark dataset lives here (adjust if different)
EXDARK_ROOT = os.path.join(REPO_DIR, "..", "DATASET", "ExDark - REPO")
EXDARK_ROOT = os.path.normpath(EXDARK_ROOT)

# Output goes here (mirrors what Colab expects on Drive)
OUTPUT_ROOT = os.path.join(REPO_DIR, "outputs")

# ExDark subdirectories
IMAGES_DIR = os.path.join(EXDARK_ROOT, "Dataset")
GT_DIR = os.path.join(EXDARK_ROOT, "Groundtruth")
CLASSLIST_FILE = os.path.join(GT_DIR, "imageclasslist.txt")

# Output subdirectories
SPLITS_DIR = os.path.join(OUTPUT_ROOT, "splits")
LABELS_DIR = os.path.join(OUTPUT_ROOT, "ExDark_yolo_labels")
YOLO_DIR = os.path.join(OUTPUT_ROOT, "ExDark_yolo")

# YOLO image resize target
TARGET_SIZE = 640


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_elapsed(start: float, label: str) -> None:
    elapsed = time.time() - start
    m, s = divmod(elapsed, 60)
    print(f"\n⏱  {label}: {int(m)}m {s:.1f}s")


def main():
    total_start = time.time()

    print_header("Fase 1 — Local Data Preparation")
    print(f"  Repo:        {REPO_DIR}")
    print(f"  ExDark root: {EXDARK_ROOT}")
    print(f"  Output:      {OUTPUT_ROOT}")
    print(f"  Target size: {TARGET_SIZE}px")

    # Sanity checks
    assert os.path.isdir(EXDARK_ROOT), f"ExDark root not found: {EXDARK_ROOT}"
    assert os.path.isdir(IMAGES_DIR), f"Images dir not found: {IMAGES_DIR}"
    assert os.path.isdir(GT_DIR), f"Groundtruth dir not found: {GT_DIR}"
    assert os.path.isfile(CLASSLIST_FILE), f"imageclasslist.txt not found: {CLASSLIST_FILE}"

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ==========================================
    # Fase 1.1: Parse Official Splits
    # ==========================================
    print_header("Fase 1.1: Parse Official Split")
    t0 = time.time()

    from src.data.split_dataset import parse_split_file
    splits = parse_split_file(CLASSLIST_FILE, SPLITS_DIR)

    print(f"  Train: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}")
    print(f"  Total: {splits['train'] + splits['val'] + splits['test']}")
    print_elapsed(t0, "Fase 1.1")

    # ==========================================
    # Fase 1.2: Convert Annotations → YOLO
    # ==========================================
    print_header("Fase 1.2: Convert ExDark → YOLO Labels")
    t0 = time.time()

    from src.data.convert_exdark import convert_exdark_to_yolo
    convert_stats = convert_exdark_to_yolo(
        exdark_images_dir=IMAGES_DIR,
        exdark_gt_dir=GT_DIR,
        output_labels_dir=LABELS_DIR,
    )

    print(f"\n  Images: {convert_stats['total_images']}")
    print(f"  Labels: {convert_stats['total_labels']}")
    print(f"  Objects: {convert_stats['total_objects']}")
    print(f"  Skipped: {convert_stats.get('skipped', 0)}")
    print(f"  Failed: {convert_stats['failed']}")
    print_elapsed(t0, "Fase 1.2")

    # ==========================================
    # Fase 1.3: Build YOLO Dataset
    # ==========================================
    print_header("Fase 1.3: Build YOLO Dataset Structure")
    t0 = time.time()

    from src.data.build_yolo_dataset import build_yolo_dataset
    build_stats = build_yolo_dataset(
        exdark_images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        splits_dir=SPLITS_DIR,
        output_dir=YOLO_DIR,
        target_size=TARGET_SIZE,
    )

    total_processed = sum(s.get("processed", 0) for s in build_stats["splits"].values())
    total_skipped = sum(s.get("skipped", 0) for s in build_stats["splits"].values())
    print(f"\n  Processed: {total_processed} | Skipped: {total_skipped}")
    print(f"  Errors: {len(build_stats.get('errors', []))}")
    print_elapsed(t0, "Fase 1.3")

    # ==========================================
    # Fase 1.4: Validate Dataset
    # ==========================================
    print_header("Fase 1.4: Validate YOLO Dataset")
    t0 = time.time()

    from src.data.validate_dataset import validate_yolo_dataset
    val_results = validate_yolo_dataset(YOLO_DIR)

    if val_results["valid"]:
        summary = val_results.get("summary", {})
        print(f"\n  ✓ Dataset validation PASSED")
        print(f"    Total images:  {summary.get('total_images', 0)}")
        print(f"    Total labels:  {summary.get('total_labels', 0)}")
        print(f"    Total objects: {summary.get('total_objects', 0)}")
    else:
        print(f"\n  ✗ Validation FAILED — check output above")
    print_elapsed(t0, "Fase 1.4")

    # ==========================================
    # Summary
    # ==========================================
    print_header("ALL DONE — Local Fase 1 Complete")
    print_elapsed(total_start, "Total time")

    print(f"""
  Output files:
    {SPLITS_DIR}
    {LABELS_DIR}
    {YOLO_DIR}

  Next steps — upload to Google Drive:
    1. Upload  outputs/splits/            →  TA-IQBAL/splits/
    2. Upload  outputs/ExDark_yolo_labels/ →  TA-IQBAL/ExDark_yolo_labels/
    3. Upload  outputs/ExDark_yolo/        →  TA-IQBAL/ExDark_yolo/

  Then on Colab, run Cell 0.1 to pull latest code.
  Fase 1.1-1.4 will auto-skip (data already on Drive).
  Continue with Fase 2+ (needs T4 GPU).
""")


if __name__ == "__main__":
    main()
