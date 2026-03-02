# Evaluasi Peningkatan Citra Berbasis LLIE untuk Deteksi Objek Low-Light pada ExDark dengan YOLOv11

## Overview

This repository implements an end-to-end pipeline to evaluate **Low-Light Image Enhancement (LLIE)** methods as preprocessing for object detection on the **ExDark** dataset using **YOLOv11n**.

### 4 Experimental Scenarios

| Scenario | Enhancement | Detector |
|----------|------------|----------|
| **S1** (Baseline) | None (raw) | YOLOv11n |
| **S2** (Proposed) | HVI-CIDNet | YOLOv11n |
| **S3** (Comparison) | RetinexFormer | YOLOv11n |
| **S4** (Comparison) | LYT-Net | YOLOv11n |

### Metrics

- **Detection**: mAP@0.5, mAP@0.5:0.95, Precision, Recall (overall + per-class)
- **Enhancement Quality (NR)**: NIQE, BRISQUE, LOE (on 1000 test samples)
- **Efficiency**: Inference Latency (ms/image), GFLOPs
- **Correlation**: Spearman rank correlation (NR metrics vs mAP)

## Dataset

- **ExDark** (Exclusively Dark): 7,363 low-light images, 12 object classes
- **Split**: Official (Train 3,000 / Val 1,800 / Test 2,563)
- Images resized to 640px (longest side) before enhancement for consistency

## Pretrained Weights (LLIE)

| Model | Weight | Source |
|-------|--------|--------|
| HVI-CIDNet | Generalization | [HuggingFace](https://huggingface.co/Fediory/HVI-CIDNet-Generalization) |
| RetinexFormer | LOL_v1.pth | [Google Drive](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV) |
| LYT-Net | PyTorch .pth | [Google Drive](https://drive.google.com/file/d/1GeEkasO2ubFi847pzrxfQ1fB3Y9NuhZ1) |

## Quick Start (Google Colab)

1. Upload ExDark to Google Drive: `My Drive/KULIAH-S1INFORMATIKA/TA-IQBAL/data/Exdark_original/`
2. Open `notebooks/master_pipeline.ipynb` in Google Colab
3. Run cells sequentially (each cell is idempotent and resume-safe)

## Environment

- Google Colab, NVIDIA Tesla T4 16 GB
- Python 3.10+, PyTorch 2.x, Ultralytics 8.3+
- Seed: 42 (global reproducibility)

## Project Structure

```
├── configs/          # YAML configs (base + per-scenario)
├── notebooks/        # Colab orchestrator notebook
├── src/
│   ├── data/         # ExDark → YOLO conversion, split, validation
│   ├── enhancers/    # LLIE model wrappers (plug & play)
│   ├── enhancement/  # Batch enhancement pipeline
│   ├── training/     # YOLOv11n training wrapper
│   ├── evaluation/   # Detection eval, NR metrics, latency, FLOPs, correlation
│   └── utils/        # I/O, timing, visualization, logging
└── scripts/          # CLI entry points
```

## License

This project is for academic research purposes (Tugas Akhir / Thesis).

## Citation

If you use this pipeline, please cite the underlying methods:
- HVI-CIDNet (CVPR 2025)
- RetinexFormer (ICCV 2023)
- LYT-Net (2024)
- ExDark (CVIU 2019)
- YOLOv11 / Ultralytics
