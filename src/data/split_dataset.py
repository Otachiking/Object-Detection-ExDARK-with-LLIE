"""
Parse imageclasslist.txt from ExDark dataset to produce fixed train/val/test splits.

Official ExDark split (column 5):
  1 = Training (3000 images, 250/class)
  2 = Validation (1800 images, 150/class)
  3 = Testing (2563 images)

Output: {split_dir}/train.txt, val.txt, test.txt
Each line: <filename> <class_id_1indexed> <class_folder>
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple


# ExDark 1-indexed class ID → folder name
CLASS_ID_TO_FOLDER = {
    1: "Bicycle", 2: "Boat", 3: "Bottle", 4: "Bus",
    5: "Car", 6: "Cat", 7: "Chair", 8: "Cup",
    9: "Dog", 10: "Motorbike", 11: "People", 12: "Table",
}

SPLIT_ID_TO_NAME = {1: "train", 2: "val", 3: "test"}

EXPECTED_COUNTS = {"train": 3000, "val": 1800, "test": 2563}


def parse_imageclasslist(classlist_path: str) -> List[dict]:
    """Parse imageclasslist.txt into structured records.

    Args:
        classlist_path: Path to imageclasslist.txt

    Returns:
        List of dicts with keys: filename, class_id, light, indoor_outdoor, split
    """
    records = []

    with open(classlist_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Skip header line
        if line.startswith("Name") or line.startswith("#") or line.startswith("%"):
            continue

        parts = line.split()
        if len(parts) < 5:
            print(f"[WARN] Skipping malformed line {i+1}: {line}")
            continue

        filename = parts[0]
        class_id = int(parts[1])       # 1-12
        light = int(parts[2])           # 1-10
        indoor_outdoor = int(parts[3])  # 1=indoor, 2=outdoor
        split_id = int(parts[4])        # 1=train, 2=val, 3=test

        # Validate
        if class_id not in CLASS_ID_TO_FOLDER:
            print(f"[WARN] Unknown class_id {class_id} at line {i+1}")
            continue
        if split_id not in SPLIT_ID_TO_NAME:
            print(f"[WARN] Unknown split_id {split_id} at line {i+1}")
            continue

        records.append({
            "filename": filename,
            "class_id": class_id,
            "class_folder": CLASS_ID_TO_FOLDER[class_id],
            "light": light,
            "indoor_outdoor": indoor_outdoor,
            "split": SPLIT_ID_TO_NAME[split_id],
        })

    return records


def generate_split_files(
    classlist_path: str,
    output_dir: str,
    verify: bool = True,
) -> Dict[str, List[dict]]:
    """Parse imageclasslist.txt and write split files.

    Args:
        classlist_path: Path to imageclasslist.txt
        output_dir: Directory to write train.txt, val.txt, test.txt
        verify: If True, verify counts match expected

    Returns:
        Dict mapping split name to list of records
    """
    print(f"[SPLIT] Parsing: {classlist_path}")
    records = parse_imageclasslist(classlist_path)
    print(f"[SPLIT] Total records parsed: {len(records)}")

    # Group by split
    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        splits[r["split"]].append(r)

    # Verify counts
    for split_name, expected in EXPECTED_COUNTS.items():
        actual = len(splits[split_name])
        status = "✓" if actual == expected else "✗ MISMATCH"
        print(f"[SPLIT] {split_name}: {actual} images (expected {expected}) {status}")

        if verify and actual != expected:
            print(f"[WARN] {split_name} count mismatch: got {actual}, expected {expected}")

    # Write split files
    os.makedirs(output_dir, exist_ok=True)

    for split_name, recs in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                # Format: filename class_folder
                f.write(f"{r['filename']}\t{r['class_folder']}\n")
        print(f"[SPLIT] Written: {out_path} ({len(recs)} entries)")

    # Also write full manifest with all metadata
    manifest_path = os.path.join(output_dir, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("filename\tclass_folder\tsplit\tlight\tindoor_outdoor\n")
        for r in records:
            f.write(f"{r['filename']}\t{r['class_folder']}\t{r['split']}\t{r['light']}\t{r['indoor_outdoor']}\n")
    print(f"[SPLIT] Full manifest: {manifest_path}")

    return splits


def load_split_file(split_path: str) -> List[Tuple[str, str]]:
    """Load a split file and return list of (filename, class_folder) tuples."""
    entries = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    return entries


def parse_split_file(
    classlist_path: str,
    output_dir: str,
    verify: bool = True,
) -> Dict[str, int]:
    """Convenience wrapper: parse classlist, write splits, return counts.

    Args:
        classlist_path: Path to imageclasslist.txt
        output_dir: Directory to write train.txt, val.txt, test.txt
        verify: If True, verify counts match expected

    Returns:
        Dict with keys 'train', 'val', 'test' mapped to image counts.
    """
    splits = generate_split_files(classlist_path, output_dir, verify=verify)
    return {k: len(v) for k, v in splits.items()}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python split_dataset.py <imageclasslist.txt> <output_dir>")
        sys.exit(1)
    generate_split_files(sys.argv[1], sys.argv[2])
