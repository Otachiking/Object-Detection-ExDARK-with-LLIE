"""
Generate the 5 research notebooks:
  notebooks/
    scenario_s1_raw.ipynb
    scenario_s2_hvi_cidnet.ipynb
    scenario_s3_retinexformer.ipynb
    scenario_s4_lyt_net.ipynb
    comparison.ipynb

Run from repo root:
    python scripts/generate_notebooks.py
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NB_DIR = REPO_ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

def code(src: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"cellView": "form"},
        "outputs": [],
        "source": src.lstrip("\n"),
    }

def md(src: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src.lstrip("\n"),
    }

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "colab": {"name": None, "provenance": []},
}

def save_nb(cells: list, name: str):
    nb = {"cells": cells, "metadata": METADATA.copy(), "nbformat": 4, "nbformat_minor": 5}
    nb["metadata"]["colab"]["name"] = name
    path = NB_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Written: {path}")


# ─────────────────────────────────────────
# Shared cell templates
# ─────────────────────────────────────────

SETUP_CELL = """\
#@title 0.1 · Environment Setup & Clone Repo
import os, subprocess, sys, shutil

QUICK_TEST  = False  # @param {{type:"boolean"}}
REPO_URL    = "https://github.com/Otachiking/Object-Detection-ExDARK-with-LLIE.git"
SCENARIO_KEY  = "{scenario_key}"
SCENARIO_NAME = "{scenario_name}"

RESTORE_PREVIOUS   = True   # @param {{type:"boolean"}}
GDRIVE_WEIGHTS_URL = ""     # @param {{type:"string"}}

_IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
_IS_COLAB  = not _IS_KAGGLE and os.path.exists("/content")

if _IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_DIR = "/content/TA-IQBAL-ObjectDetectionExDARKwithLLIE"
elif _IS_KAGGLE:
    REPO_DIR = "/kaggle/working/TA-IQBAL-ObjectDetectionExDARKwithLLIE"
else:
    raise RuntimeError("Run this on Kaggle or Colab")

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"], check=True)
else:
    if os.path.exists(REPO_DIR): shutil.rmtree(REPO_DIR)
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
"""

INSTALL_CELL = """\
#@title 0.2 · Install Dependencies
!pip install -q ultralytics pyiqa thop fvcore scipy pandas pyyaml seaborn tqdm gdown huggingface_hub
"""

CONFIG_CELL = """\
#@title 0.3 · Load Configuration & Define Paths
from src.config import load_config, save_environment_info
from src.seed import set_global_seed

cfg = load_config(SCENARIO_KEY, quick_test=QUICK_TEST)
set_global_seed(cfg["seed"])

OUTPUT_ROOT = cfg.get("paths", {}).get("output_root") or cfg.get("paths", {}).get("project_root")
EXDARK_ROOT = cfg.get("paths", {}).get("exdark_root") or cfg.get("paths", {}).get("data", {}).get("exdark_original")

# Auto-detect for Kaggle if needed
if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
    from src.utils.platform import resolve_kaggle_exdark, resolve_kaggle_llie_input
    _classlist_check = os.path.join(EXDARK_ROOT or "", "Groundtruth", "imageclasslist.txt")
    if not EXDARK_ROOT or not os.path.isfile(_classlist_check):
        _detected = resolve_kaggle_exdark()
        if _detected:
            EXDARK_ROOT = _detected
    llie_in = resolve_kaggle_llie_input()
    if llie_in:
        cfg["paths"]["model_weights_input"] = llie_in

print(f"Output root : {OUTPUT_ROOT}")
print(f"ExDark root : {EXDARK_ROOT}")

PREPARED_DIR   = os.path.join(OUTPUT_ROOT, "prepared")
SCENARIO_DIR   = os.path.join(OUTPUT_ROOT, "scenarios", SCENARIO_NAME)
SCENARIO_RUNS  = os.path.join(SCENARIO_DIR, "runs")
SCENARIO_EVAL  = os.path.join(SCENARIO_DIR, "evaluation")

os.makedirs(PREPARED_DIR, exist_ok=True)
os.makedirs(SCENARIO_RUNS, exist_ok=True)
os.makedirs(SCENARIO_EVAL, exist_ok=True)
save_environment_info(SCENARIO_DIR)
"""

FASE1_CELL = """\
#@title Fase 1 · Data Preparation (auto-skip if done)
from src.data.split_dataset     import parse_split_file
from src.data.convert_exdark    import convert_exdark_to_yolo
from src.data.build_yolo_dataset import build_yolo_dataset

img_dir        = os.path.join(EXDARK_ROOT, "Dataset")
gt_dir         = os.path.join(EXDARK_ROOT, "Groundtruth")
split_output   = os.path.join(PREPARED_DIR, "splits")
labels_dir     = os.path.join(PREPARED_DIR, "ExDark_yolo_labels")
yolo_dir       = os.path.join(PREPARED_DIR, "ExDark_yolo")

splits = parse_split_file(os.path.join(gt_dir, "imageclasslist.txt"), split_output)
stats = convert_exdark_to_yolo(img_dir, gt_dir, labels_dir)
build_stats = build_yolo_dataset(img_dir, labels_dir, split_output, yolo_dir, target_size=cfg["yolo"]["imgsz"])
"""

def fase2_cell_for(key):
    if key == "s1_raw":
        return """\
#@title Fase 2 · Enhancement (SKIPPED)
print("[SKIP] S1_Raw uses raw images. No LLIE enhancement needed.")
enhancer_name = None
enhanced_dir = None
"""
    return """\
#@title Fase 2 · Image Enhancement
import torch
from src.enhancement.run_enhancement import enhance_dataset, get_enhancer

enhancer_name = cfg.get("scenario", {}).get("enhancer")
enhanced_dir = os.path.join(SCENARIO_DIR, "enhanced")
cache_dir = os.path.join(OUTPUT_ROOT, "model_cache")

# Modularity: enhancement handles Kaggle skips internally or we just run it
enhancer = get_enhancer(enhancer_name, cache_dir)
enhancer.load_model()
stats = enhance_dataset(enhancer=enhancer, source_dataset_dir=yolo_dir, output_dir=enhanced_dir, yolo_labels_dir=yolo_dir)
"""

FASE2_5_CELL = """\
#@title Fase 2.5 · Sample Test Images Preview
from src.evaluation.visualize import plot_sample_images
plot_sample_images(
    raw_test_dir=os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test"),
    enh_test_dir=os.path.join(SCENARIO_DIR, "enhanced", "images", "test") if enhancer_name else None,
    output_dir=SCENARIO_EVAL,
    scenario_name=SCENARIO_NAME,
    enhancer_name=enhancer_name
)
"""

FASE3_CELL = """\
#@title Fase 3 · Image Quality Metrics
# DEFAULT FORCE TO TRUE based on requirements
FORCE_EVALUATION = True # @param {{type:"boolean"}}
from src.evaluation.nr_metrics import compute_nr_metrics

raw_test_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")
test_dir = os.path.join(SCENARIO_DIR, "enhanced", "images", "test") if enhancer_name else raw_test_dir

nr = compute_nr_metrics(images_dir=test_dir, output_dir=SCENARIO_EVAL, scenario_name=SCENARIO_NAME, force=FORCE_EVALUATION)
print(f"NIQE: {nr.get('niqe_mean')} | BRISQUE: {nr.get('brisque_mean')} | LOE: {nr.get('loe_mean')}")
"""

FASE4_CELL = """\
#@title Fase 4 · Training
# DEFAULT FORCE TO TRUE (best.pt ditimpa selalu kecuali skip manual)
FORCE_RETRAIN = True # @param {{type:"boolean"}}
from src.training.train_yolo import train_yolo, get_best_weights

data_yaml = os.path.join(SCENARIO_DIR, "enhanced", "dataset.yaml") if enhancer_name else os.path.join(PREPARED_DIR, "ExDark_yolo", "dataset.yaml")

result = train_yolo(dataset_yaml=data_yaml, scenario_name=SCENARIO_NAME, output_dir=SCENARIO_DIR, run_name="runs", config=cfg, force=FORCE_RETRAIN)
best_pt = get_best_weights(SCENARIO_RUNS)
"""

FASE4_5_CELL = """\
#@title Fase 4.5 · Training Curves & Figures
from src.evaluation.visualize import plot_training_curves
plot_training_curves(run_dir_train=SCENARIO_RUNS, output_dir=SCENARIO_EVAL, scenario_name=SCENARIO_NAME)
"""

FASE5_CELL = """\
#@title Fase 5 · Detection Evaluation
# Evaluasi otomatis retrain jika FORCE_EVALUATION True
from src.evaluation.eval_yolo import evaluate_yolo

data_yaml = os.path.join(SCENARIO_DIR, "enhanced", "dataset.yaml") if enhancer_name else os.path.join(PREPARED_DIR, "ExDark_yolo", "dataset.yaml")
best_pt = get_best_weights(SCENARIO_RUNS)

results = evaluate_yolo(weights_path=best_pt, dataset_yaml=data_yaml, output_dir=SCENARIO_EVAL, scenario_name=SCENARIO_NAME, force=FORCE_EVALUATION)
print(results.get("overall", {}))
"""

FASE5_5_CELL = """\
#@title Fase 5.5 · Detection Visualization (GT vs Prediction)
from src.evaluation.visualize import plot_detection_results

test_img_dir = os.path.join(SCENARIO_DIR, "enhanced", "images", "test") if enhancer_name else os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")
test_lbl_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "labels", "test")

plot_detection_results(
    weights_path=best_pt,
    test_img_dir=test_img_dir,
    test_lbl_dir=test_lbl_dir,
    output_dir=SCENARIO_EVAL,
    scenario_name=SCENARIO_NAME
)
"""

FASE6_CELL = """\
#@title Fase 6 · Latency & FLOPs
from src.evaluation.latency import measure_latency
from src.evaluation.flops import compute_all_flops

raw_test_dir = os.path.join(PREPARED_DIR, "ExDark_yolo", "images", "test")

lat = measure_latency(yolo_weights=best_pt, output_dir=SCENARIO_EVAL, scenario_name=SCENARIO_NAME, test_images_dir=raw_test_dir)
flops = compute_all_flops(yolo_weights=best_pt, output_dir=SCENARIO_EVAL, scenario_name=SCENARIO_NAME)
"""

DONE_CELL = """\
#@title Done
print("Notebook finished running all modular phases.")
"""

def build_scenario_notebook(key: str, name: str):
    cells = [
        md(f"# Scenario {name}\nModular Pipeline architecture", "md-h"),
        code(SETUP_CELL.format(scenario_key=key, scenario_name=name), "c-setup"),
        code(INSTALL_CELL, "c-install"),
        code(CONFIG_CELL, "c-config"),
        code(FASE1_CELL, "c-f1"),
        code(fase2_cell_for(key), "c-f2"),
        code(FASE2_5_CELL, "c-f2.5"),
        code(FASE3_CELL, "c-f3"),
        code(FASE4_CELL, "c-f4"),
        code(FASE4_5_CELL, "c-f4.5"),
        code(FASE5_CELL, "c-f5"),
        code(FASE5_5_CELL, "c-f5.5"),
        code(FASE6_CELL, "c-f6"),
        code(DONE_CELL, "c-done")
    ]
    save_nb(cells, f"scenario_{key}.ipynb")


# Build them all
if __name__ == "__main__":
    SCENARIOS = [
        ("s1_raw",          "S1_Raw"),
        ("s2_hvi_cidnet",   "S2_HVI_CIDNet"),
        ("s3_retinexformer","S3_RetinexFormer"),
        ("s4_lyt_net",      "S4_LYT_Net"),
    ]
    for k, n in SCENARIOS:
        build_scenario_notebook(k, n)
    print("Notebooks strictly regenerated in modular fashion!")
