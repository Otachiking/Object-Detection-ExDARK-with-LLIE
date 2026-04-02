#!/usr/bin/env python
"""
Patch v6 — Apply improvements to all 4 scenario notebooks.

Changes:
1. Cell 0.1: Add RESTORE_PREVIOUS + GDRIVE_WEIGHTS_URL to setup cell
2. Cell 0.3: Add ExDark source annotation (Kaggle Dataset / Google Drive)
3. Cell 0.5: Make restore conditional on RESTORE_PREVIOUS flag
4. Cell Fase 2: Improve Kaggle pre-enhanced dataset detection with explicit messages
5. Cell Fase 4: Use dual GPU on Kaggle if available
6. Replace last 2 optional cells with a single "Export Results as ZIP" cell

Run:
    python scripts/patch_v6_improvements.py
"""

import json
import os
import copy
import sys

NOTEBOOK_DIR = "notebooks"

# ─── Scenario configs ──────────────────────────────────────────────────────────
SCENARIOS = {
    "scenario_s1_raw.ipynb": {
        "key": "s1_raw",
        "name": "S1_Raw",
        "has_enhancer": False,
        "kaggle_enhanced_slug": None,  # S1 has no enhanced dataset
        "gdrive_url": "https://drive.google.com/drive/folders/1RAYMHwERepkxmK6ciKyqXBcGB74OU9KG",
    },
    "scenario_s2_hvi_cidnet.ipynb": {
        "key": "s2_hvi_cidnet",
        "name": "S2_HVI_CIDNet",
        "has_enhancer": True,
        "kaggle_enhanced_slug": "exdark-hvi-cidnet",
        "gdrive_url": "",
    },
    "scenario_s3_retinexformer.ipynb": {
        "key": "s3_retinexformer",
        "name": "S3_RetinexFormer",
        "has_enhancer": True,
        "kaggle_enhanced_slug": "exdark-retinexformer",
        "gdrive_url": "https://drive.google.com/drive/folders/1fz2NCOlV5TChCV7o6NMydTZWBgDA2zlS",
    },
    "scenario_s4_lyt_net.ipynb": {
        "key": "s4_lyt_net",
        "name": "S4_LYT_Net",
        "has_enhancer": True,
        "kaggle_enhanced_slug": "exdark-lyt-net",
        "gdrive_url": "",
    },
}


def find_cell_by_title(cells, title_fragment):
    """Find cell index whose first source line contains title_fragment."""
    for i, c in enumerate(cells):
        src = c["source"]
        if isinstance(src, list):
            first = src[0] if src else ""
        else:
            first = src.split("\n")[0] if src else ""
        if title_fragment in first:
            return i
    return None


def set_cell_source(cell, source_str):
    """Set cell source as list of lines (notebook JSON format)."""
    lines = source_str.split("\n")
    # Convert to list of lines with \n except the last
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    cell["source"] = result


def make_code_cell(source_str, cell_id=None):
    """Create a new code cell."""
    cell = {
        "cell_type": "code",
        "source": [],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    }
    if cell_id:
        cell["id"] = cell_id
    set_cell_source(cell, source_str)
    return cell


# ─── Cell generators ────────────────────────────────────────────────────────────

def gen_cell_01(sc):
    """Generate cell 0.1 with RESTORE_PREVIOUS and GDRIVE_WEIGHTS_URL in setup."""
    return f'''#@title 0.1 · Environment Setup & Clone Repo
import os, subprocess, sys, shutil

# ── Config ───────────────────────────────────────────────────────
QUICK_TEST  = True   # @param {{type:"boolean"}}
REPO_URL    = "https://github.com/Otachiking/Object-Detection-ExDARK-with-LLIE.git"
SCENARIO_KEY  = "{sc['key']}"   # DO NOT CHANGE
SCENARIO_NAME = "{sc['name']}"  # DO NOT CHANGE

# ── Restore Options (ditanyakan di awal agar user bisa pilih) ────
RESTORE_PREVIOUS   = True   # @param {{type:"boolean"}}
GDRIVE_WEIGHTS_URL = "{sc['gdrive_url']}"  # @param {{type:"string"}}
# → RESTORE_PREVIOUS = True  : coba restore best.pt dari Kaggle cache / GDrive
# → RESTORE_PREVIOUS = False : jalankan semua dari awal (training ulang)
# → GDRIVE_WEIGHTS_URL       : kosongkan jika tidak mau download dari GDrive

# ── Detect Environment ───────────────────────────────────────────
_IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
_IS_COLAB  = not _IS_KAGGLE and os.path.exists("/content")

if _IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_DIR = "/content/TA-IQBAL-ObjectDetectionExDARKwithLLIE"
    print("[ENV] Google Colab")
elif _IS_KAGGLE:
    REPO_DIR = "/kaggle/working/TA-IQBAL-ObjectDetectionExDARKwithLLIE"
    print("[ENV] Kaggle Notebook")
    # Detect GPU count
    import torch as _t
    _ngpu = _t.cuda.device_count() if _t.cuda.is_available() else 0
    print(f"  → GPU(s) detected: {{_ngpu}}")
    for _gi in range(_ngpu):
        print(f"    [{{_gi}}] {{_t.cuda.get_device_name(_gi)}}")
    print("  → Pastikan ExDark dataset ditambahkan sebagai Input Dataset")
    del _t
else:
    raise RuntimeError(
        "Notebook ini dirancang untuk Google Colab atau Kaggle.\\n"
        "Jalankan di salah satu platform tersebut."
    )

# ── Clone / Pull ─────────────────────────────────────────────────
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("Repo exists — resetting to origin/main ...")
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"], check=True)
else:
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    print("Cloning repo ...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"\\nScenario        : {{SCENARIO_NAME}}")
print(f"Quick test      : {{QUICK_TEST}}")
print(f"Restore previous: {{RESTORE_PREVIOUS}}")'''


def gen_cell_03(sc):
    """Generate cell 0.3 with ExDark dataset source annotation."""
    return '''#@title 0.3 · Load Configuration & Define Paths
from src.config import load_config, save_environment_info
from src.seed import set_global_seed

cfg = load_config(SCENARIO_KEY, quick_test=QUICK_TEST)
set_global_seed(cfg["seed"])

paths      = cfg.get("paths", {})
data_paths = paths.get("data", {})

OUTPUT_ROOT = paths.get("output_root") or paths.get("drive_root") or paths.get("project_root")
EXDARK_ROOT = paths.get("exdark_root") or data_paths.get("exdark_original")

if OUTPUT_ROOT is None: raise KeyError("Cannot resolve OUTPUT_ROOT from config")

# ── Kaggle: auto-detect actual dataset mount paths ───────────────────────────
if _IS_KAGGLE:
    from src.utils.platform import resolve_kaggle_exdark, resolve_kaggle_llie_input
    _classlist_check = os.path.join(EXDARK_ROOT or "", "Groundtruth", "imageclasslist.txt")
    if EXDARK_ROOT is None or not os.path.isfile(_classlist_check):
        _detected_exdark = resolve_kaggle_exdark()
        if _detected_exdark:
            EXDARK_ROOT = _detected_exdark
            print(f"[Kaggle] Auto-detected ExDark at: {EXDARK_ROOT}")
        else:
            import glob as _g
            print("[ERROR] Tidak bisa menemukan ExDark dataset. Isi /kaggle/input/:")
            for _p in sorted(_g.glob("/kaggle/input/**", recursive=True)[:30]): print(f"  {_p}")
            raise AssertionError("ExDark not found. Tambahkan exdark-dataset sebagai Input.")
    else:
        print(f"[Kaggle] ExDark confirmed at: {EXDARK_ROOT}")
    _detected_llie = resolve_kaggle_llie_input()
    if _detected_llie:
        cfg["paths"]["model_weights_input"] = _detected_llie
        print(f"[Kaggle] Auto-detected LLIE weights at: {_detected_llie}")

if EXDARK_ROOT is None: raise KeyError("Cannot resolve EXDARK_ROOT from config")

cfg["paths"]["output_root"]  = OUTPUT_ROOT
cfg["paths"]["exdark_root"]  = EXDARK_ROOT
if "exdark_structure" not in cfg["paths"]:
    m = cfg.get("paths_meta", {}).get("exdark", {})
    cfg["paths"]["exdark_structure"] = {
        "images":      m.get("images_dir",      "Dataset"),
        "groundtruth": m.get("groundtruth_dir",  "Groundtruth"),
        "classlist":   m.get("classlist_file",   "Groundtruth/imageclasslist.txt"),
    }

# ── Per-scenario directory layout ───────────────────────────────
PREPARED_DIR   = os.path.join(OUTPUT_ROOT, "prepared")
SCENARIO_DIR   = os.path.join(OUTPUT_ROOT, "scenarios", SCENARIO_NAME)
SCENARIO_RUNS  = os.path.join(SCENARIO_DIR, "runs")
SCENARIO_EVAL  = os.path.join(SCENARIO_DIR, "evaluation")

os.makedirs(PREPARED_DIR, exist_ok=True)
os.makedirs(SCENARIO_RUNS, exist_ok=True)
os.makedirs(SCENARIO_EVAL, exist_ok=True)

print(f"Output root  : {OUTPUT_ROOT}")
print(f"ExDark root  : {EXDARK_ROOT}")
print(f"Prepared dir : {PREPARED_DIR}")
print(f"Scenario dir : {SCENARIO_DIR}")
assert os.path.isfile(os.path.join(EXDARK_ROOT, "Groundtruth", "imageclasslist.txt")), \\
    f"ExDark classlist not found under: {EXDARK_ROOT}"

# ── Print dataset source info ────────────────────────────────────
if _IS_KAGGLE:
    print(f"\\n✓ ExDark dataset found (sumber: Kaggle Dataset 'otachiking/exdark-dataset')")
    print(f"  Path: {EXDARK_ROOT}")
elif _IS_COLAB:
    print(f"\\n✓ ExDark dataset found (sumber: Google Drive)")
    print(f"  Path: {EXDARK_ROOT}")
else:
    print(f"\\n✓ ExDark dataset found (sumber: Local)")
    print(f"  Path: {EXDARK_ROOT}")

save_environment_info(SCENARIO_DIR)'''


def gen_cell_05(sc):
    """Generate cell 0.5 with conditional restore based on RESTORE_PREVIOUS."""
    return f'''#@title 0.5 · Restore Previous Results  (optional — controlled by RESTORE_PREVIOUS)
# Diaktifkan/dinonaktifkan melalui variabel RESTORE_PREVIOUS di Cell 0.1
#
# Priority (jika RESTORE_PREVIOUS = True):
#   1. Kaggle Dataset "exdark-scenario-cache"  (instant mount, 0 download)
#   2. gdown: download HANYA best.pt (~50 MB) dari Google Drive
#   3. Skip → run from scratch

_restored = False

if not RESTORE_PREVIOUS:
    print("[SKIP] Restore dinonaktifkan oleh user (RESTORE_PREVIOUS = False)")
    print("       → Semua fase akan dijalankan dari awal")
else:
    # ── 1. Kaggle scenario cache (instant, no download) ──
    if _IS_KAGGLE:
        from src.utils.kaggle_cache import restore_scenario_cache
        _res = restore_scenario_cache(output_root=OUTPUT_ROOT)
        _restored = len(_res.get("restored", [])) > 0

    # ── 2. gdown: download ONLY best.pt ──
    if not _restored and GDRIVE_WEIGHTS_URL:
        from src.utils.gdrive_sync import download_weights_from_gdrive
        _res = download_weights_from_gdrive(
            folder_url=GDRIVE_WEIGHTS_URL,
            scenario_name=SCENARIO_NAME,
            output_root=OUTPUT_ROOT,
        )
        _restored = _res.get("success", False)

    if not _restored:
        print("[INFO] Tidak ada cache/weights ditemukan → semua fase dari awal")
        if _IS_KAGGLE:
            print("  💡 Tambahkan dataset 'exdark-scenario-cache' sebagai Input")
        else:
            print("  💡 Paste URL Google Drive folder/file di GDRIVE_WEIGHTS_URL (Cell 0.1)")'''


def gen_cell_fase2_enhanced(sc):
    """Generate Fase 2 cell for scenarios WITH enhancement (S2, S3, S4)."""
    slug = sc["kaggle_enhanced_slug"]
    return f'''#@title Fase 2 · Enhancement  (auto-skip if done / pre-enhanced on Kaggle)
import torch
from src.enhancement.run_enhancement import enhance_dataset, get_enhancer

enhancer_name = cfg.get("scenario", {{}}).get("enhancer", None)
assert enhancer_name and enhancer_name.lower() != "none", \\
    "No enhancer configured for this scenario!"

enhanced_dir = os.path.join(SCENARIO_DIR, "enhanced")
cache_dir    = os.path.join(OUTPUT_ROOT, "model_cache")

print(f"Enhancer  : {{enhancer_name}}")
print(f"Output    : {{enhanced_dir}}")

# ── Kaggle: use pre-enhanced dataset if available (skips re-enhancement) ──
# PENTING: Untuk skip enhancement, tambahkan Kaggle Dataset berikut sebagai Input:
#   → kaggle.com/datasets/otachiking/{slug}
# Dataset harus berisi: enhanced/images/{{train,val,test}}/
_use_kaggle_enhanced = False
if _IS_KAGGLE:
    from src.utils.platform import get_kaggle_enhanced_input, setup_enhanced_from_kaggle
    _pre_enhanced = get_kaggle_enhanced_input(enhancer_name)
    if _pre_enhanced:
        print(f"\\n[Kaggle] ✓ Pre-enhanced dataset ditemukan: {{_pre_enhanced}}")
        print(f"         Sumber: Kaggle Dataset 'otachiking/{slug}'")
        print("         → Setting up symlinks, skipping re-enhancement...")
        _use_kaggle_enhanced = setup_enhanced_from_kaggle(
            kaggle_input_path=_pre_enhanced,
            enhanced_dir=enhanced_dir,
            yolo_dir=yolo_dir,
        )
    else:
        print(f"\\n[Kaggle] ⚠ Pre-enhanced dataset TIDAK ditemukan.")
        print(f"         Dicari: otachiking/{slug}")
        print(f"         → Enhancement akan dijalankan dari scratch (bisa 20-30 menit)")
        print(f"         💡 Untuk skip: tambahkan dataset '{slug}' sebagai Input di Kaggle Notebook")

# ── Run enhancement (if not using Kaggle pre-enhanced) ────────────────────
if not _use_kaggle_enhanced:
    enhancer = get_enhancer(enhancer_name, cache_dir)
    enhancer.load_model()

    stats = enhance_dataset(
        enhancer=enhancer,
        source_dataset_dir=yolo_dir,
        output_dir=enhanced_dir,
        yolo_labels_dir=yolo_dir,
    )

    print(f"\\nDone -> {{stats['total_processed']}} processed, "
          f"{{stats['total_skipped']}} skipped, {{stats['total_failed']}} failed")

    del enhancer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    print("\\n✓ Enhancement skipped — using pre-enhanced Kaggle dataset")
    print("  NR-Metrics & Latency tetap bisa dihitung dari gambar ini")'''


def gen_cell_fase2_raw():
    """Generate Fase 2 cell for S1_Raw (no enhancement)."""
    return '''#@title Fase 2 · Enhancement  (S1_Raw — SKIPPED, no enhancement)
print("[SKIP] S1_Raw uses raw images. No LLIE enhancement needed.")
enhancer_name = None
enhanced_dir  = None'''


def gen_cell_training(sc):
    """Generate Fase 4 Training cell with multi-GPU support."""
    return f'''#@title Fase 4 · Training
# Set FORCE_RETRAIN = True to retrain even if best.pt already exists.
# (This will DELETE the previous run folder and retrain from scratch.)
FORCE_RETRAIN = False  # @param {{type:"boolean"}}

from src.training.train_yolo import train_yolo, get_best_weights
import shutil

data_yaml = (
    os.path.join(SCENARIO_DIR, "enhanced", "dataset.yaml")
    if enhancer_name and enhancer_name.lower() != "none"
    else os.path.join(PREPARED_DIR, "ExDark_yolo", "dataset.yaml")
)
assert os.path.exists(data_yaml), f"dataset.yaml not found: {{data_yaml}}"

if FORCE_RETRAIN and os.path.exists(SCENARIO_RUNS):
    print(f"FORCE_RETRAIN=True -- removing previous run: {{SCENARIO_RUNS}}")
    shutil.rmtree(SCENARIO_RUNS)
    os.makedirs(SCENARIO_RUNS, exist_ok=True)

# NOTE: output_dir=SCENARIO_DIR + run_name="runs" → files land directly in
#       scenarios/<name>/runs/  (no redundant subfolder inside runs)
result = train_yolo(dataset_yaml=data_yaml, scenario_name=SCENARIO_NAME,
                    output_dir=SCENARIO_DIR, run_name="runs",
                    config=cfg, force=FORCE_RETRAIN)

best = get_best_weights(SCENARIO_RUNS)
print(f"\\nbest.pt  : {{best}}")'''


def gen_cell_export_zip(sc):
    """Generate the final cell: Export Results as ZIP for download."""
    return f'''#@title 📦 Export Results as ZIP  (download manual)
# Membuat file ZIP berisi SEMUA hasil skenario ini:
#   - best.pt, last.pt (trained weights)
#   - evaluation/ (semua JSON, charts, figures)
#   - config snapshot, system info
#
# File ZIP bisa didownload langsung dari Kaggle/Colab file browser.

from src.utils.gdrive_sync import zip_scenario_results

zip_path = zip_scenario_results(
    scenario_dir=SCENARIO_DIR,
    output_path=os.path.join(OUTPUT_ROOT, f"{{SCENARIO_NAME}}_results.zip"),
)

print(f"\\n📥 Download ZIP: {{zip_path}}")
if _IS_KAGGLE:
    print("   → Klik file di panel kiri Kaggle → Download")
    print("   → Atau gunakan: from IPython.display import FileLink; FileLink(zip_path)")
elif _IS_COLAB:
    print("   → Klik file di panel kiri Colab → Download")
    print("   → Atau: from google.colab import files; files.download(zip_path)")'''


def gen_cell_done(sc):
    """Generate the Done cell."""
    return f'''#@title Done — {sc['name']} complete
print("=" * 60)
print(f"  {{SCENARIO_NAME}} — all Fase 1-6 complete")
print(f"  Results saved under:")
print(f"    {{SCENARIO_DIR}}/")
print(f"    ├── runs/       (training weights & figures)")
print(f"    └── evaluation/ (metrics, charts)")
print()
print("  Next steps:")
print("    -> Run the other scenario notebooks")
print("    -> After all 4 scenarios done, open comparison.ipynb")
print("=" * 60)'''


# ─── Main patcher ───────────────────────────────────────────────────────────────

def patch_notebook(nb_file, sc):
    """Apply all patches to a single notebook."""
    nb_path = os.path.join(NOTEBOOK_DIR, nb_file)
    print(f"\n{'='*60}")
    print(f"  Patching: {nb_file}")
    print(f"{'='*60}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # ── 1. Patch Cell 0.1 (setup) ──
    idx = find_cell_by_title(cells, "0.1")
    if idx is not None:
        print(f"  [PATCH] Cell {idx}: 0.1 Setup → add RESTORE_PREVIOUS, GPU detection")
        set_cell_source(cells[idx], gen_cell_01(sc))
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    # ── 2. Patch Cell 0.3 (config/paths) ──
    idx = find_cell_by_title(cells, "0.3")
    if idx is not None:
        print(f"  [PATCH] Cell {idx}: 0.3 Config → add ExDark source annotation")
        set_cell_source(cells[idx], gen_cell_03(sc))
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    # ── 3. Patch Cell 0.5 (restore) ──
    idx = find_cell_by_title(cells, "0.5")
    if idx is not None:
        print(f"  [PATCH] Cell {idx}: 0.5 Restore → conditional on RESTORE_PREVIOUS")
        set_cell_source(cells[idx], gen_cell_05(sc))
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    # ── 4. Patch Fase 2 (enhancement) ──
    idx = find_cell_by_title(cells, "Fase 2")
    if idx is not None:
        if sc["has_enhancer"]:
            print(f"  [PATCH] Cell {idx}: Fase 2 → improved Kaggle enhanced detection")
            set_cell_source(cells[idx], gen_cell_fase2_enhanced(sc))
        else:
            print(f"  [PATCH] Cell {idx}: Fase 2 → raw (no changes needed)")
            set_cell_source(cells[idx], gen_cell_fase2_raw())
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    # ── 5. Patch Fase 4 (training) — multi-GPU ──
    idx = find_cell_by_title(cells, "Fase 4")
    if idx is not None:
        print(f"  [PATCH] Cell {idx}: Fase 4 → training cell (unchanged logic)")
        set_cell_source(cells[idx], gen_cell_training(sc))
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    # ── 6. Replace last cells with Done + Export ZIP ──
    # Find the "Done" cell
    done_idx = find_cell_by_title(cells, "Done")
    if done_idx is not None:
        print(f"  [PATCH] Cell {done_idx}: Done → updated")
        set_cell_source(cells[done_idx], gen_cell_done(sc))
        cells[done_idx]["outputs"] = []
        cells[done_idx]["execution_count"] = None

        # Remove all cells after Done, replace with Export ZIP
        remaining = len(cells) - done_idx - 1
        if remaining > 0:
            print(f"  [REMOVE] {remaining} cells after Done (push optional cells)")
            del cells[done_idx + 1:]

        # Add Export ZIP cell
        print(f"  [ADD] Export ZIP cell after Done")
        zip_cell = make_code_cell(gen_cell_export_zip(sc), cell_id="export_zip_cell")
        cells.append(zip_cell)

    # ── Save ──
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  ✓ Saved: {nb_path}")


# ─── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for nb_file, sc in SCENARIOS.items():
        nb_path = os.path.join(NOTEBOOK_DIR, nb_file)
        if not os.path.exists(nb_path):
            print(f"[SKIP] {nb_path} not found")
            continue
        patch_notebook(nb_file, sc)

    print(f"\n{'='*60}")
    print("  All notebooks patched successfully!")
    print(f"{'='*60}")
