# Evaluation Report: Object Detection on ExDARK with LLIE (YOLOv11)

**Date:** 13 May 2026  
**Author:** Iqbal  
**Project:** Object Detection on ExDARK with LLIE (YOLOv11)

---

## 1. Training Configurations

| Parameter | Value |
| :--- | :--- |
| **Model** | `yolo11n.pt` (Pretrained) |
| **Input Size** | 640x640 |
| **Epochs** | 50 |
| **Batch Size** | 16 |
| **Optimizer** | AdamW |
| **Initial Learning Rate** | 0.005 |
| **Weight Decay** | 0.001 |
| **Validation Conf Threshold** | 0.001 |
| **Validation IoU Threshold** | 0.7 |

### Augmentations (Identical Across Scenarios)
- **Mosaic:** 1.0
- **Mixup:** 0.2
- **Erasing:** 0.2
- **HSV (H/S/V):** 0.015 / 0.7 / 0.4

---

## 2. Quantitative Results Across Scenarios

### 2.1 Spatial & No-Reference Image Quality Metrics
*Note: For NR-Metrics (NIQE, BRISQUE, LOE), lower values indicate better perceptual quality. For EPI, a value closer to 1.0 means edges are well-preserved.*

| Scenario | NIQE ↓ | BRISQUE ↓ | LOE ↓ | Shadow Area ↓ | Mean Lum. (80-180) | RMS Contrast | Noise σ ↓ | EPI ↑ | Edge Density |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1 (Raw)** | 5.1770 | 31.1362 | - | 66.10% | 33.69 | 35.17 | **4.515** | **1.0000** | 4.28% |
| **S2 (HVI-CIDNet)** | **3.9159** | 25.7430 | 6.2178 | **11.64%** | **114.59** | **61.02** | 7.427 | 0.2004 | 11.21% |
| **S3 (RetinexFormer)** | 3.9408 | **17.1748** | 6.8211 | 21.61% | 92.41 | 56.49 | 6.833 | 0.2396 | 9.25% |
| **S4 (LYT-Net)** | 4.3292 | 26.4130 | **4.4619** | 13.65% | 96.88 | 57.83 | 7.136 | 0.2696 | 9.14% |

### 2.2 YOLOv11 Detection Performance
*Object detection evaluation on the ExDARK test set using COCO metrics.*

| Scenario | mAP@0.5 ↑ | mAP@0.5:0.95 ↑ | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **S1 (Raw)** | **0.5576** | **0.3309** | **0.6312** | 0.5146 |
| **S2 (HVI-CIDNet)** | 0.5491 | 0.3253 | 0.6153 | **0.5173** |
| **S3 (RetinexFormer)** | 0.5361 | 0.3181 | 0.5927 | 0.5130 |
| **S4 (LYT-Net)** | 0.5365 | 0.3184 | 0.6165 | 0.5017 |

### 2.3 Computational Complexity & Latency
*Measured per image (batch size 1) at 640x640 resolution.*

| Scenario | Enhancer Latency (T_enhance) | YOLO Latency (T_detect) | Total Pipeline GFLOPs |
| :--- | :--- | :--- | :--- |
| **S1 (Raw)** | **0.00 ms** (None) | ~13.68 ms | **3.22** GFLOPs |
| **S2 (HVI-CIDNet)** | 100.53 ms | ~13.50 ms | 18.62 GFLOPs |
| **S3 (RetinexFormer)** | 356.29 ms | ~13.85 ms | 60.02 GFLOPs |
| **S4 (LYT-Net)** | 101.67 ms | ~14.07 ms | 18.12 GFLOPs |

---

## 3. Key Observations (Discussion Points)

1. **The Over-Enhancement Trade-off:** S2 (HVI-CIDNet) achieves the best visual illumination (Mean Lum: 114.59, lowest shadow area) but severely degrades original image gradients (EPI drops to 0.2004). Because YOLOv11 relies heavily on edge/gradient features rather than raw brightness, S1 (Raw) retains the highest mAP (0.5576) due to its perfectly preserved edges (EPI: 1.0).
2. **Computational Viability:** RetinexFormer (S3) introduces an extreme computational bottleneck (~356 ms latency and 60 GFLOPs), making it unviable for real-time edge applications despite producing low-artifact images (BRISQUE: 17.17). S1 remains the fastest (~13.68 ms end-to-end).

