### 8. EPI, MAM, dan EigenCAM — Verifikasi Rumus vs Kode Aktual

---

#### 8a. EPI — ✅ Kode BENAR, Rumus di Paper SALAH

**Perbandingan `rumus_metrics.md` vs `spatial_metrics.py`:**

| | `rumus_metrics.md` | `spatial_metrics.py` | Status |
|---|---|---|---|
| Metode ekstraksi edge | Canny (50, 150) | Canny (50, 150) | ✅ Sama |
| Formula EPI | IoU = intersection / union | IoU = intersection / union | ✅ Sama |

**Kesimpulan: Kode dan dokumentasi sudah konsisten. Yang SALAH adalah rumus yang ditulis di `paper.tex`** — kamu menulis cosine similarity gradient, padahal implementasinya adalah Jaccard IoU.

**Formula yang SALAH (ada di paper.tex sekarang — hapus ini):**

$$
\text{EPI}_{\text{salah}} = \frac{\sum_{x,y} G_e(x,y) \cdot G_o(x,y)}{\sqrt{\sum_{x,y} G_e^2(x,y) \cdot \sum_{x,y} G_o^2(x,y)}}
$$

**Formula yang BENAR (sesuai implementasi — pakai ini di paper):**

$$
\text{EPI} = \frac{|E_o \cap E_e|}{|E_o \cup E_e|}
$$

di mana $E_o$ dan $E_e$ adalah binary edge map dari gambar original dan enhanced, diekstrak menggunakan Canny operator dengan threshold $(T_{\text{low}}, T_{\text{high}}) = (50, 150)$.

**Code snippet (`spatial_metrics.py`):**
```python
edges_raw    = cv2.Canny(gray_raw, 50, 150)      # binary edge map raw
edges_enh    = cv2.Canny(gray.astype(np.uint8), 50, 150)
mask_raw     = edges_raw > 0
mask_enh     = edges_enh > 0
intersection = np.logical_and(mask_raw, mask_enh).sum()
union        = np.logical_or(mask_raw, mask_enh).sum()
epi          = float(intersection / union) if union > 0 else 1.0
```

**Interpretasi:**
- EPI = 1.000 → 100% piksel tepi di posisi yang sama (identik) ✅
- EPI = 0.200 → hanya 20% piksel tepi yang lokasinya sama → 80% tepi hilang/bergeser ❌

---

#### 8b. EPI-v2: Gradient Cosine Similarity (GCS) — Sudah diimplementasikan ✅

Ditambahkan ke `spatial_metrics.py` sebagai `epi_gcs`. Algoritma: **Gradient Cosine Similarity (GCS)** menggunakan Sobel operator (bukan Canny, karena Canny bersifat binary — tidak cocok untuk cosine similarity yang membutuhkan nilai kontinu).

**Formula:**

$$
\text{EPI\_GCS} = \frac{\sum_{i} G_o^{(i)} \cdot G_e^{(i)}}{\sqrt{\sum_{i} \left(G_o^{(i)}\right)^2 \cdot \sum_{i} \left(G_e^{(i)}\right)^2}}
$$

di mana $G_o^{(i)}$ dan $G_e^{(i)}$ adalah nilai gradient magnitude Sobel di piksel ke-$i$ (vektor yang telah di-flatten), dengan:

$$
G = \sqrt{G_x^2 + G_y^2}, \quad G_x = \text{Sobel}(I, dx=1),\ G_y = \text{Sobel}(I, dy=1)
$$

**Code snippet (`spatial_metrics.py` — fungsi `compute_epi_gcs`):**
```python
def compute_epi_gcs(img_raw, img_enhanced):
    gray_raw = cv2.cvtColor(img_raw,      cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_enh = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def sobel_magnitude(gray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx**2 + gy**2)

    Go = sobel_magnitude(gray_raw).flatten()
    Ge = sobel_magnitude(gray_enh).flatten()

    dot   = np.dot(Go, Ge)
    denom = np.sqrt(np.dot(Go, Go)) * np.sqrt(np.dot(Ge, Ge))
    return float(dot / denom) if denom > 1e-7 else 1.0
```

**Perbedaan EPI vs EPI_GCS:**

| | EPI (Canny + IoU) | EPI_GCS (Sobel + Cosine) |
|---|---|---|
| Tipe output gradien | Binary (0 atau 1) | Kontinu (float) |
| Mengukur | Apakah edge ada di lokasi yang sama? | Seberapa mirip kekuatan gradient-nya? |
| Sensitif terhadap | Pergeseran posisi tepi | Perubahan amplitudo gradient |
| Ideal untuk paper | ✅ Lebih intuitif & jelas | ✅ Lebih kaya secara numerik |

> **Rekomendasi paper:** Laporkan keduanya — EPI untuk narasi utama ("80% edge hilang"), EPI_GCS sebagai konfirmasi kuantitatif tambahan.

---

#### 8c. Layer N = Layer Ke-21 pada YOLOv11n

Dari kode `interpretability.py`:
```python
deep_idx = len(model.model.model) - 2   # YOLOv11n: 23 layers → index 21
hook_handle_n = model.model.model[deep_idx].register_forward_hook(...)
```

**YOLOv11n memiliki 23 layer** (index 0–22):
- **Layer 0** = Blok konvolusi pertama (backbone awal) → fitur edge/tekstur tingkat rendah
- **Layer 21** = Layer tepat sebelum `Detect` head → fitur semantik tertinggi (shape, context)

> **Tulis di Methodology:** *"Feature maps were captured via forward hooks at layer 0 (first convolutional block) and layer 21 (penultimate layer prior to the Detect head) of YOLOv11n."*

---

#### 8d. MAM (Mean Activation Map) — Formula & Implementasi

**Formula:**

$$
\text{MAM}(h, w) = \frac{1}{C} \sum_{c=1}^{C} A_c(h, w)
$$

di mana $A_c(h, w)$ adalah nilai aktivasi channel ke-$c$ di posisi spasial $(h, w)$, dan $C$ adalah jumlah channel.

**Normalisasi ke [0, 255] (untuk visualisasi):**

$$
\hat{A}(h, w) = \frac{\max\!\left(A(h,w),\ 0\right) - \min(A)}{\max(A) - \min(A) + \varepsilon} \times 255, \quad \varepsilon = 10^{-7}
$$

**Alpha blending ke gambar asli:**

$$
\text{Overlay} = \alpha \cdot I_{\text{RGB}} + (1 - \alpha) \cdot \text{Colormap}(\hat{A}), \quad \alpha = 0.5
$$

**Code snippet (`interpretability.py`):**
```python
act = torch.mean(tensor, dim=1).squeeze().cpu().numpy()  # [H, W]
act = np.maximum(act, 0)           # clamp negatif → 0 (ReLU-like)
act = act - np.min(act)
act = act / (np.max(act) + 1e-7)  # normalisasi [0, 1]
act_uint8 = np.uint8(255 * act)
heatmap = cv2.applyColorMap(cv2.resize(act_uint8, (640,640)), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(ori_rgb, 0.5, heatmap, 0.5, 0)
```

---

#### 8e. EigenCAM — Formula & Implementasi

**Algoritma: SVD-based Principal Component Projection**
Referensi: Muhammad & Yeasin, *EigenCAM: Class Activation Map using Principal Components*, IJCNN 2020.

**Step 1 — Reshape tensor aktivasi $\mathbf{A} \in \mathbb{R}^{C \times H \times W}$ ke matrix 2D:**

$$
\mathbf{A} \in \mathbb{R}^{C \times (H \cdot W)}
$$

**Step 2 — Singular Value Decomposition:**

$$
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top
$$

**Step 3 — Ambil PC1 (baris pertama $\mathbf{V}^\top$, arah variansi tertinggi):**

$$
\text{PC1} = \mathbf{V}^\top[0] \in \mathbb{R}^{H \cdot W}
$$

**Step 4 — Reshape, ambil nilai absolut, normalisasi:**

$$
S(h, w) = |\text{PC1}|.\text{reshape}(H, W)
$$

$$
\hat{S}(h, w) = \frac{S(h, w) - \min(S)}{\max(S) - \min(S) + \varepsilon} \times 255
$$

> **Mengapa `abs()`?** PC1 bisa bernilai negatif (signed projection). Nilai absolut memastikan seluruh rentang variansi tertangkap, berbeda dengan MAM yang meng-clamp ke 0.

**Code snippet (`interpretability.py`):**
```python
act     = tensor.squeeze(0).cpu().numpy()       # [C, H, W]
act_2d  = act.reshape(act.shape[0], -1)         # [C, H*W]
_, _, Vt = np.linalg.svd(act_2d, full_matrices=False)
pc1     = Vt[0]                                 # PC1: direction of max variance
saliency = np.abs(pc1.reshape(act.shape[1], act.shape[2]))
saliency = saliency - np.min(saliency)
saliency = saliency / (np.max(saliency) + 1e-7)
heatmap  = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
```

---

#### 8f. Perbandingan MAM vs EigenCAM

| Aspek | MAM | EigenCAM |
|---|---|---|
| **Formula inti** | $\frac{1}{C}\sum_c A_c(h,w)$ | $\mathbf{V}^\top[0]$ dari SVD |
| **Mengukur** | Rata-rata aktivasi per posisi | Arah variansi tertinggi bersama |
| **Penanganan negatif** | Clamp → 0 (`max(x,0)`) | Ambil abs (`abs(x)`) |
| **Dipakai di layer** | Layer 0 (edge) + Layer 21 (semantic) | Layer 21 saja (semantic) |
| **Referensi** | Teknik umum visualisasi DL | Muhammad & Yeasin, IJCNN 2020 |
| **Perlu gradient?** | ❌ Tidak | ❌ Tidak |
| **Class-agnostic?** | ✅ Ya | ✅ Ya |

**Analogi:**
- **MAM** = rata-rata suara semua anggota tim → mudah didominasi channel yang paling aktif
- **EigenCAM** = "tema utama" yang disetujui bersama seluruh channel → lebih robust secara statistik

---

### 9. Latency & Computational Cost Data Exists But Is Not Discussed

Your report has latency and GFLOPs data that tells a compelling additional story — **frame it as computational overhead, not real-time deployment** (since your study scope is empirical evaluation, not deployment):

- **RetinexFormer: 356 ms / image average, 106.36 GFLOPs** — the highest computational burden by a large margin
- **HVI-CIDNet: 100.53 ms / image, 50.83 GFLOPs** — moderate overhead
- **LYT-Net: 101.67 ms / image, 10.63 GFLOPs** — most computationally efficient LLIE
- **YOLOv11n (constant): 3.226 GFLOPs** — the detector itself is extremely lightweight

The key insight: **RetinexFormer's 106 GFLOPs is 33× the cost of YOLOv11n itself**, yet produces the worst mAP. Frame this as a cost-benefit analysis — high computational cost with negative accuracy return.

> 📝 **Note on T_detect**: The variation in T_detect across scenarios (12–15 ms) is due to OS scheduling and hardware fluctuations, NOT differences in model architecture (all use identical YOLOv11n, 3.226 GFLOPs). You may report it for completeness, but do NOT draw inference conclusions from it.

---
