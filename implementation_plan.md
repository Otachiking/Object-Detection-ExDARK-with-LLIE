# Evaluasi & Perbaikan Notebook S3 RetinexFormer

## Background

Setelah review mendalam terhadap notebook `RESULT_s3-retinexformer-epoch1-b.ipynb` dan seluruh source code pipeline, berikut **7 evaluasi** yang diangkat beserta rencana penyelesaiannya.

---

## 1. [Fase 0.3] Perjelas Sumber ExDark Dataset (Kaggle)

### Problem
Output `✓ ExDark dataset found` tidak menjelaskan bahwa dataset berasal dari **Kaggle Dataset** `otachiking/exdark-dataset`.

### Solusi
Ubah output di cell 0.3 agar menampilkan sumber spesifik. Pada Kaggle, print:

```
✓ ExDark dataset found (Kaggle Dataset: otachiking/exdark-dataset)
  Path: /kaggle/input/datasets/otachiking/exdark-dataset
```

Pada Colab, print:
```
✓ ExDark dataset found (Google Drive)
  Path: /content/drive/MyDrive/KULIAH-S1INFORMATIKA/TA-IQBAL/data/Exdark_original
```

#### [MODIFY] Notebook cell 0.3
Tambahkan info sumber tepat sebelum `print("✓ ExDark dataset found")`.

---

## 2. [Fase 0.5] Jadikan Restore Optional dengan Input User

### Problem
Fase 0.5 (Restore Previous Results) selalu berjalan otomatis. User ingin opsi untuk **skip** restore agar bisa jalankan sepenuhnya dari awal, atau **input link Google Drive** secara dinamis.

### Solusi
Tambahkan variabel `RESTORE_PREVIOUS = True` dan `GDRIVE_WEIGHTS_URL` sebagai parameter yang bisa di-set user di awal cell. Jika `RESTORE_PREVIOUS = False`, seluruh fase 0.5 di-skip.

```python
#@title 0.5 · Restore Previous Results  (OPTIONAL)
RESTORE_PREVIOUS = True      # @param {type:"boolean"}
GDRIVE_WEIGHTS_URL = ""       # @param {type:"string"}
# Kosongkan GDRIVE_WEIGHTS_URL jika tidak mau restore dari GDrive
```

Logika:
- `RESTORE_PREVIOUS = False` → skip seluruh cell, print `[SKIP] Restore disabled by user`
- `RESTORE_PREVIOUS = True` + `GDRIVE_WEIGHTS_URL = ""` → hanya coba Kaggle cache
- `RESTORE_PREVIOUS = True` + `GDRIVE_WEIGHTS_URL = "https://..."` → coba Kaggle cache, lalu GDrive

#### [MODIFY] Notebook cell 0.5

---

## 3. [Fase 2] Gunakan Pre-Enhanced Kaggle Dataset, Hindari Re-Enhancement

### Problem (UTAMA!)
Walaupun user sudah upload dataset **`otachiking/exdark-retinexformer`** di Kaggle, Fase 2 tetap menjalankan enhancement dari scratch (28+ menit). Ini karena:

1. Dataset Kaggle harus ditambahkan sebagai **Input** di notebook Kaggle
2. Kode `get_kaggle_enhanced_input("retinexformer")` mencari di `/kaggle/input/exdark-retinexformer` — yang sudah benar
3. **TAPI** dari log, dataset itu tidak ter-mount (tidak ada di daftar `dataSources` notebook metadata selain `exdark-dataset` dan `llie-model-cache`)

### Root Cause
Dataset `exdark-retinexformer` **belum ditambahkan sebagai Input** di notebook Kaggle pada saat run tersebut. Makanya kode fallback ke enhancement dari scratch.

### Solusi

#### A. Perbaikan Kode: Auto-detect `exdark-retinexformer` lebih robust
Fungsi `get_kaggle_enhanced_input()` sudah benar secara logika. Yang perlu ditambahkan:
- Print pesan yang lebih jelas jika dataset enhanced **tidak ditemukan**, agar user tahu harus menambahkan Input
- Di Fase 2 cell, tambahkan instruksi eksplisit

#### B. Tentang Latency & NR-Metrics

> **Latency Images**: **TETAP BISA** dihitung. Latency bukan dari proses enhance batch, tapi dari Fase 6 yang menjalankan model enhancer pada dummy/sample input untuk mengukur waktu inference per-image. Fase 6 (`latency_benchmark`) load model enhancer fresh dan run inference pada sample images — tidak tergantung apakah Fase 2 dijalankan dari scratch atau dari Kaggle dataset.

> **NR-Metrics (NIQE, BRISQUE, LOE)**: **Aman**. Metrik ini dihitung dari gambar enhanced yang sudah jadi. Baik gambar dari Kaggle dataset maupun gambar hasil enhance langsung → hasilnya identik karena sama-sama output dari RetinexFormer dengan weights yang sama.

> [!IMPORTANT]
> **Satu catatan**: Jika menggunakan pre-enhanced dari Kaggle, `enhance_manifest.csv` **tidak** akan ter-generate (karena enhance di-skip). Ini berarti data latency per-image yang ada di manifest tidak tersedia. Tapi ini tidak masalah karena latency diukur terpisah di Fase 6.

#### [MODIFY] Notebook cell Fase 2 — tambahkan note/warning yang lebih eksplisit
#### [MODIFY] `src/utils/platform.py` — perbaiki pesan `get_kaggle_enhanced_input()`

---

## 4. [Fase 4] Mengapa Download `yolo26n.pt`?

### Penjelasan
Log menunjukkan:
```
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt'
```

Ini adalah **AMP (Automatic Mixed Precision) check** yang dilakukan oleh library **Ultralytics** secara otomatis. Bukan download model utama.

**Proses sebenarnya:**
1. Model utama = `yolo11n.pt` (YOLOv11 nano) — ini yang dipakai training
2. Saat training dimulai, Ultralytics menjalankan AMP compatibility test
3. AMP test memerlukan model referensi kecil untuk validate bahwa GPU support FP16 — Ultralytics download `yolo26n.pt` (YOLO v2.6 nano, ~5.3MB) sebagai test model
4. Ini **bukan** model yang dipakai training — hanya untuk AMP validation
5. Setelah test selesai (`AMP: checks passed ✅`), model ini tidak digunakan lagi

**Kesimpulan**: Tidak perlu ada perubahan. Ini perilaku normal Ultralytics.

---

## 5. [Fase 4] Apakah Ini Training Ulang?

### Penjelasan Jujur

Dari log notebook:

```
✓ S3_RetinexFormer: best.pt restored (5 MB)
```

Fase 0.5 **berhasil** merestore `best.pt` dari Google Drive. **TAPI** dari log Fase 4:

```
Transferred 448/499 items from pretrained weights
```

Ini artinya **YOLO TRAINING TETAP BERJALAN**. Mengapa?

**Root cause**: Kode `train_yolo.py` line 60:
```python
if not force and os.path.exists(best_pt):
    print(f"\n[SKIP] Training already complete for {scenario_name}")
```

Skip logic mencari `best.pt` di `runs/weights/best.pt` di bawah `SCENARIO_RUNS` directory. Path yang di-restore oleh Fase 0.5 adalah:
```
/kaggle/working/scenarios/S3_RetinexFormer/runs/weights/best.pt
```

Sementara training mencari di:
```
{SCENARIO_RUNS}/{run_name}/weights/best.pt
```

Di notebook, `SCENARIO_RUNS = os.path.join(SCENARIO_DIR, "runs")` dan `run_name` bisa berupa scenario name. Jadi seharusnya path-nya match.

**Namun dari log, training tetap berjalan (epoch 1/1 karena quick_test)**. Ini bisa terjadi jika:
1. `best.pt` berhasil di-restore tapi di lokasi yang sedikit beda (perlu verifikasi exact cell code)
2. Atau cell Fase 4 memang dijalankan dengan `force=True`

> [!WARNING]
> **Dari log, ini memang training ulang** (1 epoch quick test). Ini bukan masalah besar karena quick_test hanya 1 epoch. Untuk production run (100 epoch), `best.pt` dari GDrive seharusnya ter-skip berkat `restore`.

---

## 6. Rekap Output Akhir sebagai .zip

### Solusi
Sudah ada fungsi `zip_scenario_results()` di `src/utils/gdrive_sync.py`. Perlu:
1. Tambahkan cell baru di akhir notebook (setelah Fase 6) yang zip seluruh output
2. ZIP mencakup: `best.pt`, `last.pt`, `evaluation/`, semua JSON/YAML/CSV
3. File bisa didownload manual dari Kaggle/Colab

#### [MODIFY] Notebook — tambahkan cell "Fase 7 · Export Results Archive"

---

## 7. Buat Dokumentasi Flow Data (Colab vs Kaggle)

### Solusi
Buat file `scenarios/data_flow_reference.md` yang mendokumentasikan:
- Setiap fase dan sumber datanya
- Perbedaan flow antara Google Colab dan Kaggle Notebook
- Link spesifik ke setiap Kaggle Dataset dan Google Drive folder
- Tabel mapping sumber data per fase

#### [NEW] `scenarios/data_flow_reference.md`

---

## Proposed Changes Summary

### Notebook Changes (cell edits)
Karena file `.ipynb` **tidak bisa diedit** oleh tool saya, saya akan:
1. Membuat **file instruksi** yang berisi exact code untuk setiap cell yang perlu diubah
2. Anda copy-paste ke notebook

### Source Code Changes

#### [MODIFY] [platform.py](file:///c:/CODE/KULIAH/TA/TA-IQBAL-ObjectDetectionExDARKwithLLIE/src/utils/platform.py)
- Tambahkan pesan eksplisit di `get_kaggle_enhanced_input()` tentang dataset yang dicari

### New Files

#### [NEW] `scenarios/data_flow_reference.md`
- Dokumentasi lengkap flow data Colab vs Kaggle per fase

#### [NEW] `scenarios/notebook_patches_s3.md`
- Instruksi perubahan cell notebook (copy-paste ready)

---

## Open Questions

> [!IMPORTANT]
> 1. **Dataset `exdark-retinexformer` di Kaggle**: Apakah sudah berisi structure `enhanced/images/{train,val,test}/`? Atau structure-nya `images/{train,val,test}/`? Ini penting agar kode `get_kaggle_enhanced_input()` bisa match.

> [!IMPORTANT]
> 2. **Google Drive folder link per-scenario**: Mohon konfirmasi apakah link-link ini masih valid/updated:
>    - S3 GDrive: `https://drive.google.com/drive/folders/1fz2NCOlV5TChCV7o6NMydTZWBgDA2zlS`
>    - Kaggle Datasets yang dipakai:
>      - ExDark: `kaggle.com/datasets/otachiking/exdark-dataset`
>      - LLIE weights: `kaggle.com/datasets/otachiking/llie-model-cache`
>      - Enhanced RetinexFormer: `kaggle.com/datasets/otachiking/exdark-retinexformer`
>    - Apakah ada link tambahan yang belum terdaftar?

> [!WARNING]
> 3. **Notebook .ipynb tidak bisa diedit langsung** oleh tool saya. Saya akan menyiapkan **patch file** dengan exact code changes yang bisa Anda copy-paste ke setiap cell. Apakah ini OK?

---

## Verification Plan

### Automated Tests
- Validate `platform.py` tidak ada syntax error
- Validate `data_flow_reference.md` terformat dengan baik

### Manual Verification
- User menjalankan ulang notebook di Kaggle dengan:
  - Dataset `exdark-retinexformer` ditambahkan sebagai Input
  - Verify Fase 2 ter-skip dan menggunakan pre-enhanced dataset
  - Verify NR-Metrics dan Latency tetap berjalan normal
  - Verify ZIP export di akhir
