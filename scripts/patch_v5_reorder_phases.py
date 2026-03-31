#!/usr/bin/env python3
"""
Patch v5: Reorder notebook phases — move IQA (Fase 5) before Training (Fase 3).
Also: add Kaggle dataset validation, update header tables, fix Fase numbering.

New order for S2/S3/S4 (enhanced scenarios):
  Fase 0   Setup
  Fase 1   Data Preparation 
  Fase 2   Image Enhancement
  Fase 2.5 Sample Test Images
  Fase 3   Image Quality Metrics (NIQE, BRISQUE, LOE)  ← MOVED HERE
  Fase 4   Training (YOLOv11n)
  Fase 4.5 Training Curves
  Fase 5   Detection Evaluation
  Fase 5.5 Detection Visualization
  Fase 5.55 Validation Batch
  Fase 5.6 Confusion Matrix  
  Fase 6   Latency & FLOPs

New order for S1 (baseline — no enhancement):
  Fase 0   Setup
  Fase 1   Data Preparation
  Fase 2   (SKIP — no enhancement)
  Fase 2.5 Sample Test Images
  Fase 3   Image Quality Metrics
  Fase 4   Training
  Fase 4.5 Training Curves
  Fase 5   Detection Evaluation
  Fase 5.5 Detection Visualization
  Fase 5.55 Validation Batch
  Fase 5.6 Confusion Matrix
  Fase 6   Latency & FLOPs

Usage:
    python scripts/patch_v5_reorder_phases.py
"""

import json
import os
import re
import copy
import sys


NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")

# ─── New header table for each scenario ──────────────────────────────────

HEADER_TABLE_ENHANCED = """\
| Fase | Deskripsi | Auto-skip? |
|------|-----------|------------|
| 1    | Data Preparation (parse, convert, build YOLO) | jika output sudah ada |
| 2    | LLIE Enhancement ({enhancer_display}) | jika enhanced dir sudah ada |
| 3    | Image Quality Metrics (NIQE, BRISQUE, LOE) | jika summary.json sudah ada |
| 4    | YOLOv11n Training | jika best.pt sudah ada |
| 5    | Detection Evaluation (mAP, P, R) | jika metrics.json sudah ada |
| 6    | Latency & FLOPs | jika cache sudah ada |"""

HEADER_TABLE_RAW = """\
| Fase | Deskripsi | Auto-skip? |
|------|-----------|------------|
| 1    | Data Preparation (parse, convert, build YOLO) | jika output sudah ada |
| 2    | LLIE Enhancement (SKIPPED — baseline) | jika enhanced dir sudah ada |
| 3    | Image Quality Metrics (NIQE, BRISQUE, LOE) | jika summary.json sudah ada |
| 4    | YOLOv11n Training | jika best.pt sudah ada |
| 5    | Detection Evaluation (mAP, P, R) | jika metrics.json sudah ada |
| 6    | Latency & FLOPs | jika cache sudah ada |"""


# ─── Kaggle dataset validation cell (added to Cell 0.3 for S2/S3/S4) ────

KAGGLE_ENHANCED_CHECK = '''
# ── Kaggle: Validate pre-enhanced dataset is available (early fail) ──────
if _IS_KAGGLE:
    _enhancer_key = cfg.get("scenario", {}).get("enhancer", None)
    if _enhancer_key and _enhancer_key.lower() != "none":
        from src.utils.platform import get_kaggle_enhanced_input
        _pre_enh_check = get_kaggle_enhanced_input(_enhancer_key)
        _slug = _enhancer_key.lower().replace("_", "-")
        if _pre_enh_check:
            print(f"[Kaggle] ✓ Pre-enhanced dataset found for {{_enhancer_key}}: {{_pre_enh_check}}")
        else:
            print(f"[Kaggle] ⚠️  Pre-enhanced dataset NOT found for {{_enhancer_key}}!")
            print(f"         Add dataset: otachiking/exdark-{{_slug}} as Kaggle Input")
            print(f"         Enhancement will fall back to running LLIE inference (slower).")
'''


def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"  Saved: {path}")


def find_cell_by_title(cells, pattern):
    """Find cell index whose source matches the regex pattern."""
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if re.search(pattern, src, re.IGNORECASE):
            return i
    return None


def find_markdown_cell(cells, pattern):
    """Find markdown cell index matching pattern."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if re.search(pattern, src, re.IGNORECASE):
            return i
    return None


def update_header_table(cells, scenario_name, enhancer_display):
    """Update the first markdown cell's pipeline table."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if "Fase" in src and "Deskripsi" in src and "Auto-skip" in src:
            # This is the header table cell
            if enhancer_display and enhancer_display.lower() not in ("none", "null", ""):
                new_table = HEADER_TABLE_ENHANCED.format(enhancer_display=enhancer_display)
            else:
                new_table = HEADER_TABLE_RAW

            # Replace the table portion
            lines = src.split("\n")
            new_lines = []
            in_table = False
            table_done = False
            for line in lines:
                if "| Fase" in line and "Deskripsi" in line:
                    in_table = True
                    new_lines.append("")  # blank before table
                    for tl in new_table.split("\n"):
                        new_lines.append(tl)
                    continue
                if in_table:
                    if line.startswith("|"):
                        continue  # skip old table rows
                    else:
                        in_table = False
                        table_done = True
                new_lines.append(line)

            new_src = "\n".join(new_lines)
            # Also update Fase 1-6 → correct numbering in title
            new_src = re.sub(r"Fase 1[–-]6", "Fase 1–6", new_src)
            
            cell["source"] = [l + "\n" for l in new_src.split("\n")]
            # Fix last line (no trailing newline)
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            
            print(f"    Updated header table")
            return
    print(f"    [WARN] Header table not found")


def reorder_cells_s1(cells):
    """Reorder S1_Raw: move Fase 5 (IQA) before Fase 3 (Training)."""
    # In S1, the IQA cell header and code are: "## Fase 5: Image Quality Metrics" 
    # Need to find them and move before "## Fase 3: Training"
    
    # Find the IQA markdown + code cell
    iqa_md = find_markdown_cell(cells, r"Fase 5.*Image Quality")
    if iqa_md is None:
        print("    [SKIP] IQA header not found - may already be reordered")
        return cells
    
    iqa_code = iqa_md + 1
    
    # Find training markdown 
    train_md = find_markdown_cell(cells, r"Fase 3.*Training")
    if train_md is None:
        print("    [SKIP] Training header not found")
        return cells
    
    if iqa_md < train_md:
        print("    [SKIP] IQA already before Training")
        return cells
    
    print(f"    Moving IQA cells ({iqa_md},{iqa_code}) before Training ({train_md})")
    
    # Extract IQA cells
    iqa_cells = [cells[iqa_md], cells[iqa_code]]
    
    # Remove from original position
    new_cells = [c for i, c in enumerate(cells) if i not in (iqa_md, iqa_code)]
    
    # Find new position of training markdown (after removal, index may shift)
    new_train_md = find_markdown_cell(new_cells, r"Fase 3.*Training")
    if new_train_md is None:
        print("    [ERROR] Lost training cell after removal")
        return cells
    
    # Insert before training
    for j, c in enumerate(iqa_cells):
        new_cells.insert(new_train_md + j, c)
    
    return new_cells


def reorder_cells_enhanced(cells):
    """Reorder S2/S3/S4: move Fase 5 (IQA) before Fase 3 (Training)."""
    # Same logic as S1 but for enhanced scenarios
    iqa_md = find_markdown_cell(cells, r"Fase 5.*Image Quality")
    if iqa_md is None:
        # Check if already moved (might be numbered differently)
        print("    [SKIP] IQA header 'Fase 5' not found - may already be reordered")
        return cells
    
    iqa_code = iqa_md + 1
    
    train_md = find_markdown_cell(cells, r"Fase 3.*Training")
    if train_md is None:
        print("    [SKIP] Training header not found")
        return cells
    
    if iqa_md < train_md:
        print("    [SKIP] IQA already before Training")
        return cells
    
    print(f"    Moving IQA cells ({iqa_md},{iqa_code}) before Training ({train_md})")
    
    iqa_cells = [cells[iqa_md], cells[iqa_code]]
    new_cells = [c for i, c in enumerate(cells) if i not in (iqa_md, iqa_code)]
    
    new_train_md = find_markdown_cell(new_cells, r"Fase 3.*Training")
    if new_train_md is None:
        print("    [ERROR] Lost training cell after removal")
        return cells
    
    for j, c in enumerate(iqa_cells):
        new_cells.insert(new_train_md + j, c)
    
    return new_cells


def renumber_phases(cells):
    """Renumber all Fase references to new scheme.
    
    After reorder, IQA is physically before Training but still labeled "Fase 5".
    We need:
      Old "Fase 5" (IQA)           -> "Fase 3"
      Old "Fase 3" (Training)      -> "Fase 4"  
      Old "Fase 3.5" (Curves)      -> "Fase 4.5"
      Old "Fase 4" (Det Eval)      -> "Fase 5"
      Old "Fase 4.5" (Det Viz)     -> "Fase 5.5"
      Old "Fase 4.55" (Val Batch)  -> "Fase 5.55"
      Old "Fase 4.6" (Conf Matrix) -> "Fase 5.6"
      Old "Fase 6" (Latency)       -> unchanged
      
    To avoid conflicts (e.g. 3->4->5), we use __MARKERn__ placeholders.
    """
    # Pass 1: Replace old numbers with unique markers (longest patterns first to avoid partial matches)
    to_marker = [
        (r"Fase 4\.55",  "__MARKER_455__"),
        (r"Fase 4\.5",   "__MARKER_45__"),
        (r"Fase 4\.6",   "__MARKER_46__"),
        (r"Fase 3\.5",   "__MARKER_35__"),
        (r"Fase 5(\s*[·:]\s*Image Quality)",      r"__MARKER_IQA__\1"),
        (r"Fase 5: Image Quality",                 "__MARKER_IQA__: Image Quality"),
        (r"Fase 3(\s*[·:]\s*Training)",            r"__MARKER_TRAIN__\1"),
        (r"Fase 3: Training",                      "__MARKER_TRAIN__: Training"),
        (r"Fase 4(\s*[·:]\s*Detection)",           r"__MARKER_DET__\1"),
        (r"Fase 4: Detection",                     "__MARKER_DET__: Detection"),
    ]
    
    # Pass 2: Replace markers with final numbers
    from_marker = [
        ("__MARKER_455__",   "Fase 5.55"),
        ("__MARKER_45__",    "Fase 5.5"),
        ("__MARKER_46__",    "Fase 5.6"),
        ("__MARKER_35__",    "Fase 4.5"),
        ("__MARKER_IQA__",   "Fase 3"),
        ("__MARKER_TRAIN__", "Fase 4"),
        ("__MARKER_DET__",   "Fase 5"),
    ]
    
    for cell in cells:
        source = cell.get("source", [])
        new_source = []
        for line in source:
            # Pass 1: to markers
            for pattern, replacement in to_marker:
                line = re.sub(pattern, replacement, line)
            # Pass 2: markers to final
            for marker, final in from_marker:
                line = line.replace(marker, final)
            new_source.append(line)
        cell["source"] = new_source
    
    return cells


def patch_notebook(filename, scenario_name, enhancer_display, is_raw=False):
    """Apply all patches to a single notebook."""
    path = os.path.join(NOTEBOOKS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return
    
    print(f"\n{'='*60}")
    print(f"  Patching: {filename}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    nb = load_notebook(path)
    cells = nb["cells"]
    
    # 1. Update header table
    update_header_table(cells, scenario_name, enhancer_display)
    
    # 2. Reorder IQA before Training
    if is_raw:
        cells = reorder_cells_s1(cells)
    else:
        cells = reorder_cells_enhanced(cells)
    
    # 3. Renumber phases
    cells = renumber_phases(cells)
    
    nb["cells"] = cells
    save_notebook(nb, path)


def main():
    print("=" * 60)
    print("  Patch v5: Reorder Notebook Phases")
    print("  IQA (Fase 5) → Fase 3 (before Training)")
    print("=" * 60)
    
    # Patch each notebook
    patch_notebook("scenario_s1_raw.ipynb", "S1_Raw", None, is_raw=True)
    patch_notebook("scenario_s2_hvi_cidnet.ipynb", "S2_HVI_CIDNet", "HVI_CIDNet")
    patch_notebook("scenario_s3_retinexformer.ipynb", "S3_RetinexFormer", "RetinexFormer")
    patch_notebook("scenario_s4_lyt_net.ipynb", "S4_LYT_Net", "LYT-Net")
    
    print("\n" + "=" * 60)
    print("  Done! All notebooks patched.")
    print("=" * 60)


if __name__ == "__main__":
    main()
