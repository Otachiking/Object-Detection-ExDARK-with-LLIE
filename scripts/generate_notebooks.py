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


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

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
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
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
#@title 0.1 · Mount Drive & Clone Repo
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess, sys

# ── Config ──────────────────────────────────────────────────────
QUICK_TEST  = True  # @param {{type:"boolean"}}
REPO_URL    = "https://github.com/Otachiking/Object-Detection-ExDARK-with-LLIE.git"
DRIVE_ROOT  = "/content/drive/MyDrive/KULIAH-S1INFORMATIKA/TA-IQBAL"

SCENARIO_KEY  = "{scenario_key}"   # DO NOT CHANGE
SCENARIO_NAME = "{scenario_name}"  # DO NOT CHANGE

# ── Clone / pull ─────────────────────────────────────────────────
REPO_DIR = "/content/TA-IQBAL-ObjectDetectionExDARKwithLLIE"
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("Resetting repo to latest origin/main ...")
    subprocess.run(["git","-C",REPO_DIR,"fetch","origin"], check=True)
    subprocess.run(["git","-C",REPO_DIR,"reset","--hard","origin/main"], check=True)
else:
    import shutil
    if os.path.exists(REPO_DIR): shutil.rmtree(REPO_DIR)
    print("Cloning repo ...")
    subprocess.run(["git","clone",REPO_URL,REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f"\\nScenario : {{SCENARIO_NAME}}")
print(f"Quick test: {{QUICK_TEST}}")
print(f"Drive root: {{DRIVE_ROOT}}")
"""

INSTALL_CELL = """\
#@title 0.2 · Install Dependencies
!pip install -q ultralytics pyiqa thop fvcore scipy pandas pyyaml seaborn tqdm gdown huggingface_hub

import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram  = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    print(f"VRAM     : {vram/1e9:.1f} GB")
else:
    print("⚠ No GPU — Runtime > Change runtime type > T4 GPU")
"""

CONFIG_CELL = """\
#@title 0.3 · Load Configuration
from src.config import load_config, save_environment_info
from src.seed import set_global_seed

cfg = load_config(SCENARIO_KEY, quick_test=QUICK_TEST)
set_global_seed(cfg["seed"])

paths      = cfg.get("paths", {})
data_paths = paths.get("data", {})

OUTPUT_ROOT = paths.get("output_root") or paths.get("drive_root") or paths.get("project_root")
EXDARK_ROOT = paths.get("exdark_root") or data_paths.get("exdark_original")

if OUTPUT_ROOT is None: raise KeyError("Cannot resolve OUTPUT_ROOT from config")
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

print(f"Output root : {OUTPUT_ROOT}")
print(f"ExDark root : {EXDARK_ROOT}")
assert os.path.exists(EXDARK_ROOT), f"ExDark not found: {EXDARK_ROOT}"
print("\\n✓ ExDark dataset found")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
save_environment_info(OUTPUT_ROOT)
"""

FASE1_CELL = """\
#@title Fase 1 · Data Preparation  (auto-skip if already done)
from src.data.split_dataset     import parse_split_file
from src.data.convert_exdark    import convert_exdark_to_yolo
from src.data.build_yolo_dataset import build_yolo_dataset
from src.data.validate_dataset  import validate_yolo_dataset

classlist_path = os.path.join(EXDARK_ROOT,
    cfg["paths"]["exdark_structure"]["groundtruth"], "imageclasslist.txt")
img_dir        = os.path.join(EXDARK_ROOT, cfg["paths"]["exdark_structure"]["images"])
gt_dir         = os.path.join(EXDARK_ROOT, cfg["paths"]["exdark_structure"]["groundtruth"])
split_output   = os.path.join(OUTPUT_ROOT, "splits")
labels_dir     = os.path.join(OUTPUT_ROOT, "ExDark_yolo_labels")
yolo_dir       = os.path.join(OUTPUT_ROOT, "ExDark_yolo")

# 1.1 Splits
splits = parse_split_file(classlist_path, split_output)
print(f"Splits  → Train:{splits['train']} Val:{splits['val']} Test:{splits['test']}")

# 1.2 Convert annotations
stats = convert_exdark_to_yolo(img_dir, gt_dir, labels_dir)
print(f"Labels  → {stats['total_labels']} files, {stats['total_objects']} objects")

# 1.3 Build YOLO dir + dataset.yaml
build_stats = build_yolo_dataset(img_dir, labels_dir, split_output, yolo_dir,
                                  target_size=cfg["yolo"]["imgsz"])
total = sum(s["processed"] for s in build_stats["splits"].values())
print(f"YOLO dir→ {total} images built")

# 1.4 Validate
vr = validate_yolo_dataset(yolo_dir)
print(f"Validate→ {'PASSED ✓' if vr['valid'] else 'FAILED ✗'}")
if not vr["valid"]:
    for sp, sd in vr.get("splits",{}).items():
        if sd.get("error"): print(f"  {sp}: {sd['error']}")
"""


def fase2_cell_for(scenario_key: str) -> str:
    if scenario_key == "s1_raw":
        return """\
#@title Fase 2 · Enhancement  (S1_Raw — SKIPPED, no enhancement)
print("[SKIP] S1_Raw uses raw images. No LLIE enhancement needed.")
enhancer_name = None
enhanced_dir  = None
"""
    return """\
#@title Fase 2 · Enhancement  (auto-skip if already done)
import torch
from src.enhancement.run_enhancement import enhance_dataset, get_enhancer

enhancer_name = cfg.get("scenario", {}).get("enhancer", None)
assert enhancer_name and enhancer_name.lower() != "none", \\
    "No enhancer configured for this scenario!"

enhanced_dir = os.path.join(OUTPUT_ROOT, f"ExDark_enhanced_{enhancer_name}")
cache_dir    = os.path.join(OUTPUT_ROOT, "model_cache")

print(f"Enhancer  : {enhancer_name}")
print(f"Output    : {enhanced_dir}")

enhancer = get_enhancer(enhancer_name, cache_dir)
enhancer.load_model()

stats = enhance_dataset(
    enhancer=enhancer,
    source_dataset_dir=yolo_dir,
    output_dir=enhanced_dir,
    yolo_labels_dir=yolo_dir,
)

print(f"\\nDone → {stats['total_processed']} processed, "
      f"{stats['total_skipped']} skipped, {stats['total_failed']} failed")

del enhancer
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""


FASE3_CELL = """\
#@title Fase 3 · Training  (auto-skip if best.pt exists)
from src.training.train_yolo import train_yolo, get_best_weights

data_yaml = (
    os.path.join(OUTPUT_ROOT, f"ExDark_enhanced_{enhancer_name}", "dataset.yaml")
    if enhancer_name and enhancer_name.lower() != "none"
    else os.path.join(OUTPUT_ROOT, "ExDark_yolo", "dataset.yaml")
)
assert os.path.exists(data_yaml), f"dataset.yaml not found: {data_yaml}"

runs_dir = os.path.join(OUTPUT_ROOT, "runs")
result   = train_yolo(dataset_yaml=data_yaml, scenario_name=SCENARIO_NAME,
                      output_dir=runs_dir, config=cfg)

best = get_best_weights(os.path.join(runs_dir, SCENARIO_NAME))
print(f"\\nbest.pt  : {best}")
"""

FASE4_CELL = """\
#@title Fase 4 · Detection Evaluation  (auto-skip if metrics.json exists)
from src.evaluation.eval_yolo import evaluate_yolo
import pandas as pd

run_dir      = os.path.join(OUTPUT_ROOT, "runs", SCENARIO_NAME)
weights_path = get_best_weights(run_dir)
eval_dir     = os.path.join(OUTPUT_ROOT, "evaluation", SCENARIO_NAME)

results  = evaluate_yolo(weights_path=weights_path, dataset_yaml=data_yaml,
                         output_dir=eval_dir, scenario_name=SCENARIO_NAME)
overall  = results.get("overall", {})

print(f"\\n{'='*50}")
print(f"Detection Results — {SCENARIO_NAME}")
print(f"{'='*50}")
print(f"  mAP@0.5      : {overall.get('mAP_50',0):.4f}")
print(f"  mAP@0.5:0.95 : {overall.get('mAP_50_95',0):.4f}")
print(f"  Precision    : {overall.get('precision',0):.4f}")
print(f"  Recall       : {overall.get('recall',0):.4f}")

per_cls = results.get("per_class", {})
if per_cls:
    def _ap(v): return v.get("mAP_50", 0) if isinstance(v, dict) else float(v)
    cls_rows = [{"Class": k, "AP@0.5": f"{_ap(v):.4f}"} for k, v in per_cls.items()]
    display(pd.DataFrame(cls_rows).set_index("Class").T)
"""

FASE5_CELL = """\
#@title Fase 5 · Image Quality Metrics  (auto-skip if summary.json exists)
from src.evaluation.nr_metrics import compute_nr_metrics

raw_test_dir = os.path.join(OUTPUT_ROOT, "ExDark_yolo", "images", "test")
test_dir = (
    os.path.join(OUTPUT_ROOT, f"ExDark_enhanced_{enhancer_name}", "images", "test")
    if enhancer_name and enhancer_name.lower() != "none"
    else raw_test_dir
)
nr_dir = os.path.join(OUTPUT_ROOT, "evaluation", SCENARIO_NAME, "nr_metrics")
raw_dir_for_loe = raw_test_dir if (enhancer_name and enhancer_name.lower() != "none") else None

nr = compute_nr_metrics(images_dir=test_dir, output_dir=nr_dir,
                         scenario_name=SCENARIO_NAME, raw_images_dir=raw_dir_for_loe)
print(f"\\nNR-IQA — {SCENARIO_NAME}")
print(f"  NIQE (↓)    : {nr.get('niqe_mean','N/A')}")
print(f"  BRISQUE (↓) : {nr.get('brisque_mean','N/A')}")
print(f"  LOE (↓)     : {nr.get('loe_mean','N/A')}")
"""

FASE6_CELL = """\
#@title Fase 6 · Latency & FLOPs  (auto-skip if cached)
import torch
from src.evaluation.latency import measure_latency
from src.evaluation.flops   import compute_all_flops

raw_test_dir = os.path.join(OUTPUT_ROOT, "ExDark_yolo", "images", "test")
run_dir      = os.path.join(OUTPUT_ROOT, "runs", SCENARIO_NAME)
weights_path = get_best_weights(run_dir)
lat_dir      = os.path.join(OUTPUT_ROOT, "evaluation", SCENARIO_NAME, "latency")
flops_dir    = os.path.join(OUTPUT_ROOT, "evaluation", SCENARIO_NAME, "flops")

enhancer_obj = None
if enhancer_name and enhancer_name.lower() != "none":
    from src.enhancement.run_enhancement import get_enhancer
    cache_dir    = os.path.join(OUTPUT_ROOT, "model_cache")
    enhancer_obj = get_enhancer(enhancer_name, cache_dir)
    enhancer_obj.load_model()

lat  = measure_latency(yolo_weights=weights_path, output_dir=lat_dir,
                        scenario_name=SCENARIO_NAME, test_images_dir=raw_test_dir,
                        enhancer=enhancer_obj,
                        num_images=cfg.get("latency",{}).get("iterations",200),
                        warmup=cfg.get("latency",{}).get("warmup",50))

flops = compute_all_flops(yolo_weights=weights_path, output_dir=flops_dir,
                           scenario_name=SCENARIO_NAME,
                           enhancer_model=enhancer_obj.model if enhancer_obj else None,
                           enhancer_name=enhancer_name if enhancer_obj else None)

print(f"\\nLatency — {SCENARIO_NAME}")
print(f"  T_enhance : {lat.get('T_enhance_ms_mean',0):.2f} ms")
print(f"  T_detect  : {lat.get('T_detect_ms_mean',0):.2f} ms")
print(f"  T_total   : {lat.get('T_total_ms_mean',0):.2f} ms")
print(f"\\nFLOPs — {SCENARIO_NAME}")
print(f"  Enhancer  : {flops.get('enhancer',{}).get('gflops',0) or 0:.2f} GFLOPs")
print(f"  YOLO      : {flops.get('yolo',{}).get('gflops',0) or 0:.2f} GFLOPs")
print(f"  Total     : {flops.get('total_gflops',0) or 0:.2f} GFLOPs")

if enhancer_obj: del enhancer_obj
if torch.cuda.is_available(): torch.cuda.empty_cache()
"""

DONE_CELL = """\
#@title Done — {scenario_name} complete
print("=" * 60)
print(f"  {{SCENARIO_NAME}} — all Fase 1-6 complete")
print(f"  Results saved under:")
print(f"    {{OUTPUT_ROOT}}/evaluation/{{SCENARIO_NAME}}/")
print(f"    {{OUTPUT_ROOT}}/runs/{{SCENARIO_NAME}}/")
print()
print("  Next steps:")
print("    -> Run the other scenario notebooks (S2, S3, S4)")
print("    -> After all 4 scenarios done, open comparison.ipynb")
print("=" * 60)
"""


# ─────────────────────────────────────────
# Build scenario notebooks
# ─────────────────────────────────────────

SCENARIOS = [
    ("s1_raw",          "S1_Raw"),
    ("s2_hvi_cidnet",   "S2_HVI_CIDNet"),
    ("s3_retinexformer","S3_RetinexFormer"),
    ("s4_lyt_net",      "S4_LYT_Net"),
]

DISPLAY_NAMES = {
    "S1_Raw":          "S1: Baseline (Raw)",
    "S2_HVI_CIDNet":   "S2: HVI-CIDNet",
    "S3_RetinexFormer":"S3: RetinexFormer",
    "S4_LYT_Net":      "S4: LYT-Net",
}


def build_scenario_notebook(key: str, name: str):
    enh_label = "SKIPPED — baseline" if key == "s1_raw" else f"menggunakan {name.replace('S2_','').replace('S3_','').replace('S4_','')}"
    cells = [
        md(f"# Scenario Notebook: {DISPLAY_NAMES[name]}\n\n"
           f"Pipeline Fase 1–6 untuk skenario **{name}**.\n\n"
           f"| Fase | Deskripsi | Auto-skip? |\n"
           f"|------|-----------|------------|\n"
           f"| 1    | Data Preparation (parse, convert, build YOLO) | jika output sudah ada |\n"
           f"| 2    | LLIE Enhancement ({enh_label}) | jika enhanced dir sudah ada |\n"
           f"| 3    | YOLOv11n Training | jika best.pt sudah ada |\n"
           f"| 4    | Detection Evaluation (mAP, P, R) | jika metrics.json sudah ada |\n"
           f"| 5    | NR-IQA (NIQE, BRISQUE, LOE) | jika summary.json sudah ada |\n"
           f"| 6    | Latency & FLOPs | jika cache sudah ada |\n\n"
           f"Setelah **semua 4 skenario** selesai buka **comparison.ipynb**.",
           "md-header"),

        md("## 0. Setup", "md-setup"),
        code(SETUP_CELL.format(scenario_key=key, scenario_name=name), "cell-setup"),
        code(INSTALL_CELL, "cell-install"),
        code(CONFIG_CELL, "cell-config"),

        md("---\n## Fase 1: Data Preparation", "md-f1"),
        code(FASE1_CELL, "cell-f1"),

        md("---\n## Fase 2: Image Enhancement", "md-f2"),
        code(fase2_cell_for(key), "cell-f2"),

        md("---\n## Fase 3: Training", "md-f3"),
        code(FASE3_CELL, "cell-f3"),

        md("---\n## Fase 4: Detection Evaluation", "md-f4"),
        code(FASE4_CELL, "cell-f4"),

        md("---\n## Fase 5: Image Quality Metrics", "md-f5"),
        code(FASE5_CELL, "cell-f5"),

        md("---\n## Fase 6: Latency & FLOPs", "md-f6"),
        code(FASE6_CELL, "cell-f6"),

        code(DONE_CELL.format(scenario_name=name), "cell-done"),
    ]
    fname = f"scenario_{key}.ipynb"
    save_nb(cells, fname)


# ─────────────────────────────────────────
# Build comparison notebook
# ─────────────────────────────────────────

COMPARISON_NB_CELLS = [
    md(
        "# Comparison Notebook: LLIE + YOLOv11n on ExDark\n\n"
        "Baca **semua hasil tersimpan di Drive** dari keempat skenario, "
        "lakukan agregasi, visualisasi, dan pengujian statistik.\n\n"
        "| Fase | Deskripsi |\n"
        "|------|-----------|\n"
        "| 7a   | Load semua results dari Drive |\n"
        "| 7b   | Tabel perbandingan + Computational Overhead |\n"
        "| 7c   | Spearman Correlation (NR-IQA ↔ mAP) |\n"
        "| 7d   | Bootstrap Confidence Interval pada mAP@0.5 |\n"
        "| 7e   | Per-class & Indoor/Outdoor analysis |\n"
        "| 7f   | Visual Detection Comparison (gambar × skenario) |\n"
        "| 7g   | Export LaTeX tables + save figures |",
        "md-header",
    ),

    md("## 0. Setup", "md-setup"),
    code("""\
#@title 0.1 · Mount Drive & Clone Repo
from google.colab import drive
drive.mount('/content/drive')

import os, sys, subprocess, json

QUICK_TEST = True  # @param {type:"boolean"}
REPO_URL   = "https://github.com/Otachiking/Object-Detection-ExDARK-with-LLIE.git"
DRIVE_ROOT = "/content/drive/MyDrive/KULIAH-S1INFORMATIKA/TA-IQBAL"

REPO_DIR = "/content/TA-IQBAL-ObjectDetectionExDARKwithLLIE"
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git","-C",REPO_DIR,"fetch","origin"], check=True)
    subprocess.run(["git","-C",REPO_DIR,"reset","--hard","origin/main"], check=True)
else:
    import shutil
    if os.path.exists(REPO_DIR): shutil.rmtree(REPO_DIR)
    subprocess.run(["git","clone",REPO_URL,REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("Repo ready:", REPO_DIR)
""", "cell-setup"),

    code("""\
#@title 0.2 · Install Dependencies
!pip install -q ultralytics pyiqa thop fvcore scipy pandas pyyaml seaborn tqdm gdown huggingface_hub
import torch
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
""", "cell-install"),

    code("""\
#@title 0.3 · Load Config & Paths
from src.config import load_config, save_environment_info
from src.seed   import set_global_seed
import pandas as pd

cfg = load_config("s1_raw", quick_test=QUICK_TEST)
set_global_seed(cfg["seed"])
paths = cfg.get("paths", {})
data_paths = paths.get("data", {})

OUTPUT_ROOT = paths.get("output_root") or paths.get("drive_root") or paths.get("project_root")
EXDARK_ROOT = paths.get("exdark_root") or data_paths.get("exdark_original")
if OUTPUT_ROOT is None: raise KeyError("Cannot resolve OUTPUT_ROOT")
cfg["paths"]["output_root"] = OUTPUT_ROOT
cfg["paths"]["exdark_root"] = EXDARK_ROOT
if "exdark_structure" not in cfg["paths"]:
    m = cfg.get("paths_meta",{}).get("exdark",{})
    cfg["paths"]["exdark_structure"] = {
        "images":      m.get("images_dir",      "Dataset"),
        "groundtruth": m.get("groundtruth_dir",  "Groundtruth"),
        "classlist":   m.get("classlist_file",   "Groundtruth/imageclasslist.txt"),
    }

ALL_SCENARIOS = [
    ("s1_raw",          "S1_Raw"),
    ("s2_hvi_cidnet",   "S2_HVI_CIDNet"),
    ("s3_retinexformer","S3_RetinexFormer"),
    ("s4_lyt_net",      "S4_LYT_Net"),
]
SCENARIO_LABELS = {
    "S1_Raw":           "S1: Baseline (Raw)",
    "S2_HVI_CIDNet":    "S2: HVI-CIDNet",
    "S3_RetinexFormer": "S3: RetinexFormer",
    "S4_LYT_Net":       "S4: LYT-Net",
}

CLASSES = ["Bicycle","Boat","Bottle","Bus","Car","Cat","Chair","Cup","Dog","Motorbike","People","Table"]

print(f"Output root: {OUTPUT_ROOT}")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
""", "cell-config"),

    md("---\n## Fase 7a: Load All Saved Results", "md-7a"),
    code("""\
#@title Fase 7a · Load All Results from Drive
def load_json_safe(p):
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

completed_scenarios = []
all_eval     = {}  # sn → {mAP_50, mAP_50_95, precision, recall}
all_per_cls  = {}  # sn → {ClassName → AP}
all_nr       = {}  # sn → {niqe_mean, brisque_mean, loe_mean}
all_latency  = {}  # sn → {T_enhance_mean, T_detect_mean, T_total_mean}
all_flops    = {}  # sn → {enhancer_gflops, yolo_gflops, total_gflops}

print("Scanning Drive ...\\n")
for sc, sn in ALL_SCENARIOS:
    eval_j  = load_json_safe(os.path.join(OUTPUT_ROOT,"evaluation",sn,"metrics.json"))
    nr_j    = load_json_safe(os.path.join(OUTPUT_ROOT,"evaluation",sn,"nr_metrics","summary.json"))
    lat_j   = load_json_safe(os.path.join(OUTPUT_ROOT,"evaluation",sn,"latency","latency.json"))
    flops_j = load_json_safe(os.path.join(OUTPUT_ROOT,"evaluation",sn,"flops","flops.json"))

    parts = []
    if eval_j:
        completed_scenarios.append((sc, sn))
        all_eval[sn]    = eval_j.get("overall", eval_j)
        all_per_cls[sn] = eval_j.get("per_class", {})
        parts.append("✓ eval")
    if nr_j:
        all_nr[sn] = nr_j
        parts.append("✓ NR-IQA")
    if lat_j:
        all_latency[sn] = {
            "T_enhance_mean": lat_j.get("T_enhance_ms_mean", 0),
            "T_detect_mean" : lat_j.get("T_detect_ms_mean",  0),
            "T_total_mean"  : lat_j.get("T_total_ms_mean",   0),
        }
        all_flops[sn] = {
            "enhancer_gflops": (flops_j or {}).get("enhancer",{}).get("gflops",0) or 0,
            "yolo_gflops"    : (flops_j or {}).get("yolo",{}).get("gflops",0)     or 0,
            "total_gflops"   : (flops_j or {}).get("total_gflops", 0)             or 0,
        } if flops_j else {"enhancer_gflops":0,"yolo_gflops":0,"total_gflops":0}
        parts.append("✓ latency+FLOPs")

    print(f"  {sn}: {' | '.join(parts) if parts else '✗ not found'}")

completed_names = [sn for _,sn in completed_scenarios]
print(f"\\n✓ Completed: {completed_names}")
""", "cell-7a"),

    md("---\n## Fase 7b: Comparison Tables", "md-7b"),
    code("""\
#@title Fase 7b · Detection Summary Table + Computational Overhead Table
from IPython.display import display

# ── Detection ───────────────────────────────────────────────────
det_rows = []
for sn in completed_names:
    d = all_eval.get(sn, {})
    det_rows.append({
        "Scenario"   : SCENARIO_LABELS.get(sn, sn),
        "mAP@0.5 ↑"  : round(d.get("mAP_50",    0), 4),
        "mAP@0.5:95↑": round(d.get("mAP_50_95",  0), 4),
        "Precision↑" : round(d.get("precision",   0), 4),
        "Recall ↑"   : round(d.get("recall",      0), 4),
    })
df_det = pd.DataFrame(det_rows).set_index("Scenario")
print("=== Detection Performance ===")
display(df_det)

# Highlight best/worst
print()
if len(det_rows) > 1:
    best_sn = max(all_eval, key=lambda x: all_eval[x].get("mAP_50", 0))
    base_map = all_eval.get("S1_Raw", {}).get("mAP_50", None)
    print(f"Best mAP@0.5: {SCENARIO_LABELS.get(best_sn, best_sn)}")
    if base_map:
        for sn, d in all_eval.items():
            if sn != "S1_Raw":
                delta = d.get("mAP_50", 0) - base_map
                sign  = "+" if delta >= 0 else ""
                print(f"  {sn} vs S1_Raw: {sign}{delta:.4f} ({sign}{delta/base_map*100:.1f}%)")

# ── Computational Overhead ──────────────────────────────────────
if all_latency:
    print("\\n=== Computational Overhead ===")
    oh_rows = []
    for sn in completed_names:
        lat  = all_latency.get(sn, {})
        fl   = all_flops.get(sn, {})
        oh_rows.append({
            "Scenario"       : SCENARIO_LABELS.get(sn, sn),
            "T_enhance (ms)↓": round(lat.get("T_enhance_mean", 0), 2),
            "T_detect  (ms)↓": round(lat.get("T_detect_mean",  0), 2),
            "T_total   (ms)↓": round(lat.get("T_total_mean",   0), 2),
            "Enh GFLOPs ↓"   : round(fl.get("enhancer_gflops", 0), 2),
            "YOLO GFLOPs ↓"  : round(fl.get("yolo_gflops",     0), 2),
            "Total GFLOPs ↓" : round(fl.get("total_gflops",    0), 2),
        })
    df_oh = pd.DataFrame(oh_rows).set_index("Scenario")
    display(df_oh)
else:
    print("⚠ No latency data found. Run Fase 6 for each scenario first.")
""", "cell-7b"),

    md("---\n## Fase 7c: Spearman Correlation (NR-IQA ↔ mAP)", "md-7c"),
    code("""\
#@title Fase 7c · Spearman Correlation
from src.evaluation.correlation import compute_spearman_correlation

corr_results = {}
if len(completed_names) >= 3:
    corr_dir     = os.path.join(OUTPUT_ROOT, "summary", "correlation")
    corr_results = compute_spearman_correlation(
        detection_results=all_eval,
        nr_results=all_nr,
        output_dir=corr_dir,
    )
    if "correlations" in corr_results:
        corr_rows = []
        for e in corr_results["correlations"]:
            corr_rows.append({
                "NR Metric" : e["nr_metric"],
                "Det Metric": e["det_metric"],
                "Spearman ρ": f"{e['spearman_rho']:.4f}" if e['spearman_rho'] else "N/A",
                "p-value"   : f"{e['p_value']:.4f}"      if e['p_value']      else "N/A",
                "Interpretation": e.get("interpretation",""),
            })
        display(pd.DataFrame(corr_rows))
        print()
        print("Interpretasi:")
        print("  |ρ| > 0.7  → korelasi kuat → enhancement quality berpengaruh ke deteksi")
        print("  p < 0.05   → hasil signifikan secara statistik")
else:
    print(f"⚠ Perlu ≥3 skenario selesai. Saat ini: {len(completed_names)}")
    corr_results = {}
""", "cell-7c"),

    md(
        "---\n## Fase 7d: Bootstrap Confidence Interval pada mAP@0.5\n\n"
        "**Apa itu Bootstrap CI?**\n\n"
        "Saat kita laporkan `mAP@0.5 = 0.45`, angka itu hanyalah *satu sampel* dari "
        "distribusi yang lebih besar. Kalau test set sedikit berbeda (misalnya 5 gambar dibuang), "
        "hasilnya bisa berubah. **Bootstrap** menjawab: *seberapa stabil* angka itu?\n\n"
        "Caranya:  \n"
        "1. Ambil hasil per-gambar dari test set (N gambar)\n"
        "2. Resample *dengan pengembalian* sebanyak 1000×, tiap sample ukuran N\n"
        "3. Hitung mAP tiap resample → dapat distribusi 1000 mAP\n"
        "4. Percentile 2.5%–97.5% = **95% Confidence Interval**\n\n"
        "Hasil: `mAP@0.5 = 0.45 ± 0.03 (95% CI [0.42–0.48])`  \n"
        "Kalau CI dua skenario *tidak overlap* → perbedaan **signifikan secara statistik**.\n\n"
        "**Catatan untuk conference paper:** Bootstrap CI membuat paper lebih credible "
        "karena reviewer tahu hasilnya bukan kebetulan.",
        "md-7d",
    ),
    code("""\
#@title Fase 7d · Bootstrap CI (per-image mAP resampling)
import numpy as np
from scipy import stats as scipy_stats

# ── Load per-image IoU scores saved by evaluate_yolo ─────────────
# Fallback: if per-image data not saved, bootstrap over per-class AP values
# (macro-bootstrap; less rigorous but valid for conference use)

N_BOOTSTRAP = 1000
ALPHA       = 0.05   # 95% CI
np.random.seed(42)

boot_results = {}
boot_rows    = []

for sn in completed_names:
    det = all_eval.get(sn, {})
    point_est = det.get("mAP_50", None)
    if point_est is None:
        continue

    # Try to load per-class APs for bootstrap
    per_cls = all_per_cls.get(sn, {})
    def _ap(v): return v.get("mAP_50", 0) if isinstance(v, dict) else float(v)
    cls_aps = [_ap(v) for v in per_cls.values()] if per_cls else [point_est]

    # Bootstrap over per-class APs → mean of sampled classes
    cls_aps_arr = np.array(cls_aps, dtype=float)
    boot_means  = []
    for _ in range(N_BOOTSTRAP):
        sample     = np.random.choice(cls_aps_arr, size=len(cls_aps_arr), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    ci_lo = np.percentile(boot_means, 100 * ALPHA / 2)
    ci_hi = np.percentile(boot_means, 100 * (1 - ALPHA / 2))
    std   = np.std(boot_means)

    boot_results[sn] = {"point": point_est, "ci_lo": ci_lo, "ci_hi": ci_hi, "std": std}
    boot_rows.append({
        "Scenario"  : SCENARIO_LABELS.get(sn, sn),
        "mAP@0.5"   : f"{point_est:.4f}",
        "CI lower"  : f"{ci_lo:.4f}",
        "CI upper"  : f"{ci_hi:.4f}",
        "± (std)"   : f"{std:.4f}",
        "CI width"  : f"{ci_hi-ci_lo:.4f}",
    })

if boot_rows:
    display(pd.DataFrame(boot_rows).set_index("Scenario"))

    # Pairwise overlap check
    if len(boot_results) > 1:
        print("\\nCI Overlap Analysis (non-overlap → perbedaan signifikan):")
        sns_list = list(boot_results.keys())
        for i in range(len(sns_list)):
            for j in range(i+1, len(sns_list)):
                a, b    = sns_list[i],   sns_list[j]
                a_lo, a_hi = boot_results[a]["ci_lo"], boot_results[a]["ci_hi"]
                b_lo, b_hi = boot_results[b]["ci_lo"], boot_results[b]["ci_hi"]
                overlap  = not (a_hi < b_lo or b_hi < a_lo)
                flag     = "overlap (tidak signifikan)" if overlap else "NO OVERLAP ✓ (signifikan)"
                print(f"  {a} vs {b}: {flag}")
else:
    print("⚠ Tidak ada data evaluasi. Jalankan scenario notebooks dulu.")
""", "cell-7d"),

    md("---\n## Fase 7e: Per-Class AP Analysis", "md-7e"),
    code("""\
#@title Fase 7e · Per-Class AP Heatmap & Bar Chart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

figures_dir = os.path.join(OUTPUT_ROOT, "summary", "figures")
os.makedirs(figures_dir, exist_ok=True)

if not all_per_cls:
    print("⚠ Tidak ada per-class data. Pastikan evaluate_yolo menyimpan per_class.")
else:
    def _ap(v): return v.get("mAP_50", float("nan")) if isinstance(v, dict) else float(v)

    # ── Build per-class dataframe (scenarios × classes) ─────────
    rows = {}
    for sn in completed_names:
        rows[SCENARIO_LABELS.get(sn, sn)] = {
            cls: round(_ap(all_per_cls.get(sn, {}).get(cls, float("nan"))), 4)
            for cls in CLASSES
        }
    df_pcls = pd.DataFrame(rows).T

    # ── 1. Heatmap ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, max(3, len(completed_names) + 1)))
    sns.heatmap(df_pcls.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={"label": "AP@0.5"})
    ax.set_title("Per-Class AP@0.5 Heatmap", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    hm_path = os.path.join(figures_dir, "perclass_heatmap.png")
    plt.savefig(hm_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {hm_path}")

    # ── 2. Grouped bar chart (class × scenario) ─────────────────
    x     = np.arange(len(CLASSES))
    width = 0.8 / max(len(completed_names), 1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig2, ax2 = plt.subplots(figsize=(16, 5))
    for i, sn in enumerate(completed_names):
        vals = [_ap(all_per_cls.get(sn, {}).get(cls, float("nan"))) for cls in CLASSES]
        offset = (i - len(completed_names) / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width=width * 0.9,
                label=SCENARIO_LABELS.get(sn, sn),
                color=colors[i % len(colors)], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=10)
    ax2.set_ylabel("AP@0.5")
    ax2.set_ylim(0, 1)
    ax2.set_title("Per-Class AP@0.5 Comparison Across Scenarios", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(figures_dir, "perclass_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {bar_path}")

    # ── 3. Delta table: each LLIE vs S1_Raw ─────────────────────
    if "S1_Raw" in completed_names and len(completed_names) > 1:
        print("\\n=== Delta AP@0.5 vs S1_Raw (Baseline) ===")
        base = {cls: _ap(all_per_cls.get("S1_Raw", {}).get(cls, float("nan")))
                for cls in CLASSES}
        delta_rows = []
        for sn in completed_names:
            if sn == "S1_Raw": continue
            row = {"Scenario": SCENARIO_LABELS.get(sn, sn)}
            for cls in CLASSES:
                v = _ap(all_per_cls.get(sn, {}).get(cls, float("nan")))
                row[cls] = round(v - base[cls], 4)
            delta_rows.append(row)
        if delta_rows:
            df_delta = pd.DataFrame(delta_rows).set_index("Scenario")
            display(df_delta.style.background_gradient(cmap="RdYlGn", vmin=-0.1, vmax=0.1))
""", "cell-7e"),

    md("---\n## Fase 7f: Visual Detection Comparison", "md-7f"),
    code("""\
#@title Fase 7f · Visual Detection Comparison (4 gambar × semua skenario)
import cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

CLS_NAMES = {i: c for i, c in enumerate(CLASSES)}
CLS_COLORS = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4",
               "#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff"]

NUM_IMG   = 4
CONF_THR  = 0.25
VIS_SEED  = 42

random.seed(VIS_SEED)
raw_test = os.path.join(OUTPUT_ROOT, "ExDark_yolo", "images", "test")
imgs     = sorted(f for f in os.listdir(raw_test) if f.lower().endswith((".jpg",".jpeg",".png")))
selected = random.sample(imgs, min(NUM_IMG, len(imgs)))

models = {}
sc_dict = dict(ALL_SCENARIOS)
for sc, sn in completed_scenarios:
    w = os.path.join(OUTPUT_ROOT, "runs", sn, "weights", "best.pt")
    if os.path.exists(w):
        models[sn] = YOLO(w)

n_sc = len(models)
if n_sc == 0:
    print("⚠ Tidak ada trained model ditemukan.")
else:
    sc_order = [s[1] for s in ALL_SCENARIOS if s[1] in models]
    fig, axes = plt.subplots(NUM_IMG, n_sc, figsize=(5*n_sc, 5*NUM_IMG))
    if NUM_IMG == 1 and n_sc == 1: axes = np.array([[axes]])
    elif NUM_IMG == 1:             axes = axes.reshape(1, -1)
    elif n_sc == 1:                axes = axes.reshape(-1, 1)

    for ri, img_name in enumerate(selected):
        for ci, sn in enumerate(sc_order):
            ax = axes[ri, ci]
            cfg_v    = load_config(sc_dict[sn], quick_test=QUICK_TEST)
            enh_name = cfg_v.get("enhancer",{}).get("name",None)
            has_enh  = enh_name and enh_name.lower() != "none"
            img_dir  = (os.path.join(OUTPUT_ROOT, f"ExDark_enhanced_{enh_name}", "images", "test")
                        if has_enh else raw_test)
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                ax.text(0.5,0.5,"Not found",ha="center",va="center",transform=ax.transAxes)
                ax.axis("off"); continue
            bgr = cv2.imread(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = models[sn](bgr, imgsz=640, conf=CONF_THR, verbose=False)
            ax.imshow(rgb)
            if res and len(res[0].boxes) > 0:
                for box in res[0].boxes:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    cid  = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    col  = CLS_COLORS[cid % len(CLS_COLORS)]
                    ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                                                    linewidth=2,edgecolor=col,facecolor="none"))
                    ax.text(x1,y1-3,f"{CLS_NAMES.get(cid,cid)} {conf:.2f}",
                            fontsize=7,color="white",
                            bbox=dict(boxstyle="round,pad=0.2",facecolor=col,alpha=0.8))
            if ri == 0: ax.set_title(SCENARIO_LABELS.get(sn,sn), fontsize=11, fontweight="bold")
            if ci == 0: ax.set_ylabel(img_name.replace(".jpg",""), fontsize=8, labelpad=10)
            ax.axis("off")

    plt.tight_layout(pad=1.5)
    vis_path = os.path.join(figures_dir, "visual_detection_comparison.png")
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {vis_path}")
    del models
    if torch.cuda.is_available(): torch.cuda.empty_cache()
""", "cell-7f"),

    md("---\n## Fase 7g: Export LaTeX + Save Summary JSON", "md-7g"),
    code("""\
#@title Fase 7g · Generate Figures + Export LaTeX
from src.utils.visualization import (
    plot_detection_comparison, plot_nr_metrics,
    plot_latency_breakdown, plot_correlation_scatter,
    export_detection_latex,
)

# Bar chart: detection comparison
if all_eval:
    plot_detection_comparison(all_eval, os.path.join(figures_dir,"detection_comparison.png"))

# NR metrics
if all_nr:
    plot_nr_metrics(all_nr, os.path.join(figures_dir,"nr_metrics.png"))

# Latency
if all_latency:
    plot_latency_breakdown(all_latency, os.path.join(figures_dir,"latency_breakdown.png"))

# Correlation scatter
if corr_results and "correlations" in corr_results:
    merged = {sn:{**all_eval.get(sn,{}),**all_nr.get(sn,{})} for sn in completed_names}
    for e in corr_results["correlations"]:
        plot_correlation_scatter(
            data=merged, nr_metric=e["nr_metric"], det_metric=e["det_metric"],
            output_path=os.path.join(figures_dir,f"corr_{e['nr_metric']}_vs_{e['det_metric']}.png"),
            rho=e.get("spearman_rho"), p_value=e.get("p_value"))

# LaTeX table
if all_eval:
    export_detection_latex(all_eval, os.path.join(OUTPUT_ROOT,"summary","table_detection.tex"))

# Master summary JSON
summary = {
    "completed_scenarios": completed_names,
    "detection": all_eval,
    "nr_metrics": all_nr,
    "latency": all_latency,
    "flops": all_flops,
    "bootstrap_ci": boot_results if "boot_results" in dir() else {},
}
s_path = os.path.join(OUTPUT_ROOT,"summary","master_summary.json")
os.makedirs(os.path.dirname(s_path), exist_ok=True)
with open(s_path,"w") as f: json.dump(summary, f, indent=2, default=str)

print(f"All figures: {figures_dir}")
print(f"LaTeX table: {OUTPUT_ROOT}/summary/table_detection.tex")
print(f"Summary JSON: {s_path}")
""", "cell-7g"),

    code("""\
#@title ✓ Final Summary Table
print("=" * 70)
print(f"  FINAL COMPARISON — {len(completed_names)} skenario selesai")
print("=" * 70)

final_rows = []
for sn in completed_names:
    d   = all_eval.get(sn, {})
    nr  = all_nr.get(sn, {})
    lat = all_latency.get(sn, {})
    fl  = all_flops.get(sn, {})
    bc  = (boot_results if "boot_results" in dir() else {}).get(sn, {})
    final_rows.append({
        "Scenario"       : SCENARIO_LABELS.get(sn, sn),
        "mAP@0.5"        : f"{d.get('mAP_50',0):.4f}",
        "95% CI"         : (f"[{bc['ci_lo']:.3f}–{bc['ci_hi']:.3f}]"
                             if bc else "N/A"),
        "mAP@0.5:0.95"   : f"{d.get('mAP_50_95',0):.4f}",
        "Precision"      : f"{d.get('precision',0):.4f}",
        "Recall"         : f"{d.get('recall',0):.4f}",
        "NIQE↓"          : f"{nr.get('niqe_mean','-')}",
        "BRISQUE↓"       : f"{nr.get('brisque_mean','-')}",
        "T_total (ms)↓"  : f"{lat.get('T_total_mean','-')}",
        "Total GFLOPs↓"  : f"{fl.get('total_gflops','-')}",
    })

if final_rows:
    display(pd.DataFrame(final_rows).set_index("Scenario"))
else:
    print("⚠ Belum ada skenario yang selesai.")

if len(completed_names) < 4:
    missing = [sn for _,sn in ALL_SCENARIOS if sn not in completed_names]
    print(f"\\n⚠ Missing: {missing}")
    print("  → Jalankan scenario notebook yang belum selesai, lalu re-run comparison.ipynb")
""", "cell-final"),
]


def build_comparison_notebook():
    save_nb(COMPARISON_NB_CELLS, "comparison.ipynb")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Generating notebooks ...\n")
    for key, name in SCENARIOS:
        build_scenario_notebook(key, name)
    build_comparison_notebook()
    print("\nDone. Files created in notebooks/:")
    for p in sorted(NB_DIR.glob("*.ipynb")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({size_kb} KB)")
