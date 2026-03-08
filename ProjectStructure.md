# Struktur Google Drive

```
TA-IQBAL/                             # Drive root: .../MyDrive/KULIAH-S1INFORMATIKA/TA-IQBAL
├── data/
│   └── Exdark_original/              # Dataset ExDark (upload manual)
│       ├── Dataset/                  # Folder gambar low-light ExDark
│       │   ├── Bicycle/
│       │   ├── Boat/  ...  Table/
│       ├── Groundtruth/              # Folder anotasi bounding-box ExDark
│       │   ├── Bicycle/  ...  Table/
│       │   └── imageclasslist.txt    # Daftar split resmi (train=1/val=2/test=3)
│       └── README.md
│
├── model_cache/                      # Cache model LLIE (weights, repo clone)
│   ├── hvi_cidnet/
│   ├── retinexformer/
│   └── lyt_net/
│
├── prepared/                         # Output Fase 1 — shared untuk semua skenario
│   ├── ExDark_yolo/                  # Dataset format YOLO
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── dataset.yaml
│   ├── ExDark_yolo_labels/           # Label intermediate (hasil konversi anotasi)
│   └── splits/                       # Metadata pembagian dataset
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
└── scenarios/                        # Output per skenario — terisolasi
    ├── S1_Raw/
    │   ├── runs/                     # Output training Ultralytics
    │   │   └── S1_Raw/
    │   │       ├── weights/best.pt
    │   │       ├── weights/last.pt
    │   │       └── results.csv
    │   └── evaluation/               # Hasil evaluasi (semua flat di sini)
    │       ├── metrics.json          # mAP, precision, recall (overall + per-class)
    │       ├── metrics_per_class.csv
    │       ├── flops.json            # GFLOPs model
    │       ├── latency.json          # Latency inference (ms/image)
    │       ├── summary.json          # NR-IQA: NIQE, BRISQUE, LOE
    │       ├── training_curves.png   # Kurva loss & metrik
    │       ├── mAP_progression.png   # mAP@0.5 & mAP@0.5:0.95 per epoch
    │       ├── lr_schedule.png       # Learning rate schedule
    │       ├── detection_samples_gt_vs_pred.png
    │       ├── confusion_matrix.png
    │       ├── sample_test_images.png
    │       ├── config_snapshot.yaml
    │       └── system_info.json
    │
    ├── S2_HVI_CIDNet/
    │   ├── enhanced/                 # Gambar hasil enhancement HVI-CIDNet
    │   │   ├── images/{train,val,test}/
    │   │   ├── labels/{train,val,test}/  → symlink ke ExDark_yolo
    │   │   └── dataset.yaml
    │   ├── runs/                     # Output training
    │   │   └── S2_HVI_CIDNet/weights/best.pt
    │   └── evaluation/               # Hasil evaluasi (flat, sama struktur dengan S1)
    │
    ├── S3_RetinexFormer/             # Sama dengan S2
    │   ├── enhanced/
    │   ├── runs/
    │   └── evaluation/
    │
    └── S4_LYT_Net/                   # Sama dengan S2
        ├── enhanced/
        ├── runs/
        └── evaluation/
```

---

# Struktur GitHub Repo

```
Object-Detection-ExDARK-with-LLIE/
├── configs/                          # File konfigurasi YAML
│   ├── base.yaml                     # Konfigurasi umum (seed, model, training params)
│   ├── paths.yaml                    # Semua path (Colab & Local) + struktur ExDark
│   ├── s1_raw.yaml                   # Skenario 1 — baseline tanpa enhancement
│   ├── s2_hvi_cidnet.yaml            # Skenario 2 — HVI-CIDNet
│   ├── s3_retinexformer.yaml         # Skenario 3 — RetinexFormer
│   └── s4_lyt_net.yaml               # Skenario 4 — LYT-Net
│
├── notebooks/
│   ├── scenario_s1_raw.ipynb         # Notebook skenario S1 (Baseline Raw)
│   ├── scenario_s2_hvi_cidnet.ipynb  # Notebook skenario S2 (HVI-CIDNet)
│   ├── scenario_s3_retinexformer.ipynb
│   ├── scenario_s4_lyt_net.ipynb
│   ├── comparison.ipynb              # Cross-scenario comparison & visualisasi
│   └── master_pipeline.ipynb         # (Legacy) Pipeline gabungan
│
├── scripts/                          # Entry-point CLI & patcher tools
│   ├── prepare_data.py               # Fase 1: split, konversi, build
│   ├── enhance_dataset.py            # Fase 2: enhancement LLIE
│   ├── train.py                      # Fase 3: training YOLOv11n
│   ├── evaluate.py                   # Fase 4: evaluasi mAP
│   ├── measure_efficiency.py         # Fase 6: latency & FLOPs
│   ├── aggregate_results.py          # Fase 7: agregasi hasil
│   ├── generate_notebooks.py         # Generator notebook dari template
│   ├── patch_notebooks.py            # Patch v1: tambah visualisasi cells
│   ├── patch_v2.py                   # Patch v2: fix layout, force retrain
│   └── patch_v3_restructure.py       # Patch v3: restructure per-scenario
│
├── src/                              # Library utama
│   ├── __init__.py
│   ├── config.py                     # Config loader (base + paths + scenario)
│   ├── seed.py                       # Reproducibility (set seed global)
│   │
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── split_dataset.py          # Parse imageclasslist.txt → train/val/test
│   │   ├── convert_exdark.py         # Konversi anotasi ExDark → YOLO format
│   │   ├── build_yolo_dataset.py     # Salin gambar+label ke struktur YOLO
│   │   └── validate_dataset.py       # (Unused) Validasi integritas dataset YOLO
│   │
│   ├── enhancement/                  # Orchestrator enhancement
│   │   ├── __init__.py
│   │   └── run_enhancement.py        # get_enhancer() + enhance_dataset()
│   │
│   ├── enhancers/                    # Wrapper per metode LLIE
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseEnhancer (abstract)
│   │   ├── hvi_cidnet.py             # HVI-CIDNet (HuggingFace weights)
│   │   ├── retinexformer.py          # RetinexFormer
│   │   └── lyt_net.py                # LYT-Net
│   │
│   ├── evaluation/                   # Evaluasi & metrik
│   │   ├── __init__.py
│   │   ├── eval_yolo.py              # Evaluasi mAP via Ultralytics val()
│   │   ├── nr_metrics.py             # No-Reference IQA (NIQE, BRISQUE, LOE)
│   │   ├── correlation.py            # Korelasi metrik IQA vs mAP
│   │   ├── latency.py                # Pengukuran latency inference
│   │   └── flops.py                  # Pengukuran FLOPs model
│   │
│   ├── training/                     # Training YOLO
│   │   ├── __init__.py
│   │   └── train_yolo.py             # train_yolo() — Ultralytics training
│   │
│   └── utils/                        # Utilitas umum
│       ├── __init__.py
│       ├── io.py                     # File I/O helpers
│       ├── logger.py                 # Logging setup
│       ├── timer.py                  # Timer context manager
│       └── visualization.py          # Visualisasi hasil
│
├── ProjectStructure.md               # Dokumen ini
├── README.md                         # Deskripsi proyek
├── requirements.txt                  # Dependensi Python
└── .gitignore
```