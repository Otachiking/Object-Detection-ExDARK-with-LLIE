# Struktur Google Drive

```
TA-IQBAL/                             # Drive root: .../MyDrive/KULIAH-S1INFORMATIKA/TA-IQBAL
├── data/
│   └── Exdark_original/
│       ├── Dataset/                  # Folder gambar low-light ExDark
│       │   ├── Bicycle/              # Berisi file gambar (.png, .jpg, .JPEG, .bmp)
│       │   ├── Boat/
│       │   ├── Bottle/
│       │   ├── Bus/
│       │   ├── Car/
│       │   ├── Cat/
│       │   ├── Chair/
│       │   ├── Cup/
│       │   ├── Dog/
│       │   ├── Motorbike/
│       │   ├── People/
│       │   └── Table/
│       ├── Groundtruth/              # Folder anotasi bounding-box ExDark
│       │   ├── Bicycle/              # Berisi file anotasi (.txt)
│       │   ├── Boat/
│       │   ├── Bottle/
│       │   ├── Bus/
│       │   ├── Car/
│       │   ├── Cat/
│       │   ├── Chair/
│       │   ├── Cup/
│       │   ├── Dog/
│       │   ├── ExDark_Annno/
│       │   ├── Motorbike/
│       │   ├── People/
│       │   ├── Table/
│       │   ├── imageclasslist.txt    # Daftar split resmi (train=1/val=2/test=3)
│       │   ├── annotations.png
│       │   ├── ExDark_Annno.zip
│       │   ├── exdark1.png
│       │   └── README.md
│       ├── exdarkimg.gif
│       ├── README.md
│       └── Thumbnails.png
│
├── ExDark_yolo/                      # Output pipeline — dataset format YOLO
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── dataset.yaml                  # Konfigurasi dataset untuk Ultralytics
│
├── enhanced/                         # Output LLIE per skenario
│   ├── S2_HVI_CIDNet/               # Gambar hasil enhancement HVI-CIDNet
│   ├── S3_RetinexFormer/
│   └── S4_LYT_Net/
│
├── model_cache/                      # Cache model LLIE (weights, repo clone)
│
├── splits/                           # Metadata pembagian dataset
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   └── manifest.txt
│
├── runs/                             # Output training & evaluation YOLO per skenario
│
├── repo/                             # Clone GitHub repo (oleh Cell 0.1)
│   └── Object-Detection-ExDARK-with-LLIE/
│
├── requirements_frozen.txt           # Daftar dependensi library Python
└── system_info.json                  # Informasi sistem/lingkungan
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
│   └── master_pipeline.ipynb         # Notebook utama (dijalankan di Google Colab)
│
├── scripts/                          # Entry-point CLI (dipanggil dari notebook)
│   ├── prepare_data.py               # Fase 1: split, konversi, build, validasi
│   ├── enhance_dataset.py            # Fase 2: enhancement LLIE
│   ├── train.py                      # Fase 3: training YOLOv11n
│   ├── evaluate.py                   # Fase 4: evaluasi mAP
│   ├── measure_efficiency.py         # Fase 6: latency & FLOPs
│   └── aggregate_results.py          # Fase 7: agregasi hasil
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
│   │   └── validate_dataset.py       # Validasi integritas dataset YOLO
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
│   │   ├── nr_metrics.py             # No-Reference IQA (NIQE, BRISQUE, dll.)
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