# Review: Paper IEEE Xplore
## "Evaluation of HVI-CIDNet-based Image Enhancement for Low-Light Object Detection on ExDark with YOLOv11"
> **Reviewer Perspective**: Simulates feedback from a senior IEEE reviewer, a methodology critic, and a writing/language editor.
> **Updated**: Full numerical data from `REPORT_18MEI_26.txt` now incorporated.

---

## 🗂️ Actual Data Reference (from your `REPORT_18MEI_26.txt`)

Before diving into feedback, here are your real numbers — use these directly in the paper.

### Table 1: Spatial Quality Metrics per Scenario

| Metric | S1 Raw | S2 HVI-CIDNet | S3 RetinexFormer | S4 LYT-Net |
|---|---|---|---|---|
| Mean Luminance | 33.69 | **114.59** | 92.41 | 96.88 |
| Shadow Area (%) | 66.10% | 11.64% | 21.61% | 13.65% |
| RMS Contrast | 35.17 | 61.02 | 56.49 | 57.83 |
| Noise σ | **4.515** | 7.427 | 6.833 | 7.136 |
| EPI | **1.0000** | 0.2004 | 0.2396 | 0.2696 |
| Edge Density | 4.28% | 11.21% | 9.25% | 9.14% |
| NIQE ↓ | 5.177 | **3.916** | 3.941 | 4.329 |
| BRISQUE ↓ | 31.136 | 25.743 | **17.175** | 26.413 |
| LOE ↓ | N/A | 6.218 | 6.821 | **4.462** |

### Table 2: Detection Performance Comparison

| Scenario | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| S1 Raw (Baseline) | **0.5576** | **0.3309** | **0.6312** | 0.5146 |
| S2 HVI-CIDNet | 0.5491 | 0.3253 | 0.6153 | **0.5173** |
| S3 RetinexFormer | 0.5361 | 0.3181 | 0.5927 | 0.5130 |
| S4 LYT-Net | 0.5365 | 0.3184 | 0.6165 | 0.5017 |

### Table 3: System Latency & Computational Cost

| Scenario | T_enhance (ms) | T_detect (ms)* | GFLOPs (Enhancer) | GFLOPs (YOLO) |
|---|---|---|---|---|
| S1 Raw | 0.00 | 14.52 ± 10.53 | 0.00 | 3.226 |
| S2 HVI-CIDNet | 100.53 | 12.89 ± 9.97 | 50.83 | 3.226 |
| S3 RetinexFormer | **356.29** | 14.39 ± 11.91 | **106.36** | 3.226 |
| S4 LYT-Net | 101.67 | 15.61 ± 11.25 | **10.63** | 3.226 |

> \* `T_detect` values are hardware-dependent (CPU/GPU scheduling) and not indicative of model performance differences. The YOLO model architecture and GFLOPs are identical (3.226) across all scenarios. The key metric is `T_enhance` and Enhancer GFLOPs, which directly measure preprocessing overhead.

---

## 🔴 Critical Issues (Must Fix Before Submission)

### 1. Placeholders Not Removed — Paper Is Not Submittable

Both `paper.tex` (Indonesian) and `paper eng.tex` (English) still contain **7 placeholder figure comments** and 2 internal instruction blocks that must be removed:

```latex
% [PLACEHOLDER_FIGURE_1: ...]   ← These must go
% [PLACEHOLDER_FIGURE_2: ...]   ← These must go
% THE FOLLOWING IS THE LAYOUT FOR SUBSEQUENT CHAPTERS  ← Must go
```
And the corresponding `\begin{figure}...\end{figure}` blocks with actual images must be inserted. **Reviewer will reject immediately.**

---

### 2. Abstract Still Prospective — Must Report Actual Results

The English version abstract still reads:
> *"The results of this study **are expected to** provide empirical insights..."*

This is a fatal flaw. For an IEEE conference paper (completed research), the abstract must state what **was found**, not what is expected. You now have the numbers — use them.

**Draft replacement abstract:**
```
Image quality in low-light conditions represents a critical challenge for
object detection systems in computer vision. This paper presents a systematic
empirical evaluation of the impact of state-of-the-art Low-Light Image
Enhancement (LLIE) preprocessing—namely HVI-CIDNet, RetinexFormer, and
LYT-Net—on the performance of a YOLOv11n object detector trained and
evaluated on the Exclusively Dark (ExDark) dataset (7,363 images, 12 classes).
Contrary to the conventional assumption that visual enhancement aids detection,
our results demonstrate that raw low-light images (S1) consistently outperform
all enhanced variants, achieving mAP@0.5 of 55.76% versus 54.91%, 53.61%,
and 53.65% for HVI-CIDNet, RetinexFormer, and LYT-Net respectively. Spatial
analysis reveals that LLIE algorithms drastically reduce the Edge Preservation
Index (EPI) from 1.000 (raw) to as low as 0.200, while increasing noise levels
by up to 64.5%. EigenCAM and Mean Activation Map visualizations confirm that
LLIE preprocessing disorients YOLOv11's spatial attention away from object
semantic features toward background noise artifacts. These findings expose a
fundamental disconnect between human-centric perceptual optimization and
machine-vision-centric gradient preservation, guiding future research toward
task-driven enhancement architectures.
```

---

### 3. Self-Citation to Internal/Unpublished Documents

Three citations point to documents that cannot be verified by reviewers or readers:

```bibtex
\cite{report2026internal}  % Internal evaluation report — not public
\cite{proposal2026sempro}  % Seminar proposal — not public
\cite{metrics2026formulas} % Technical note — not public
```

**Action**: All methodology details (EPI formula, training parameters, dataset split) must be stated directly in the paper body, not cited to private documents. Remove these citations.

---

### 4. BibTeX Entries Missing Critical Fields

Most entries are missing `journal`, `volume`, `pages`, and `doi`. IEEEtran will generate malformed references.

**Worst offenders (minimum fields needed for IEEE submission):**

```bibtex
% Current — broken:
@article{khanam2024yolov11,
  author = {Khanam, Rahima and Hussain, Muhammad},
  title  = {YOLOv11: An Overview ...},
  year   = {2024},
}

% Should be (arXiv preprint):
@article{khanam2024yolov11,
  author        = {Khanam, Rahima and Hussain, Muhammad},
  title         = {{YOLOv11}: An Overview of the Key Architectural Enhancements},
  journal       = {arXiv preprint arXiv:2410.17725},
  year          = {2024},
  eprint        = {2410.17725},
  archivePrefix = {arXiv},
}
```

Check and complete: `wu2024llieeffect`, `yan2025hvi`, `brateanu2025lytnet`, `gong2025multiscale`, `wang2023yolov5lowlight`, `sapkota2025yoloreview`, `peng2024yolov5llie`, `darmawan2025vitra`, `akavaram2025flops`.

---

### 5. Irrelevant Entry in BibTeX

```bibtex
@article{oishi2021network,
  title = {A study on interconnection between local 5G networks and existing networks}
}
```
This is completely unrelated to your paper. Also, `gries_hazzan_cs` has no year and is a generic CS textbook not cited in the paper body. Remove both.

---

### 6. No Explicit Contributions List in Introduction

IEEE reviewers specifically look for a bulleted contributions paragraph. The current Introduction ends with a general description of the study — no "The main contributions of this paper are:" statement. Add one.

---

## 🟡 Major Issues (Significant Improvements Needed)

### 7. The Performance Gap Is Narrow — You Must Address This

This is the **most critical analytical weakness** a reviewer will attack. Looking at your actual numbers:

| | mAP@0.5 | Δ vs S1 |
|---|---|---|
| S1 Raw | 0.5576 | — |
| S2 HVI-CIDNet | 0.5491 | **−0.85%** |
| S3 RetinexFormer | 0.5361 | **−2.15%** |
| S4 LYT-Net | 0.5365 | **−2.11%** |

A reviewer will challenge: *"These differences are tiny (~1-2%). Are they statistically significant? Could this be due to random seed variance during training?"*

**Action**: You must address this directly. Either:
- (a) Run each scenario 3× with different seeds and report mean ± std, OR
- (b) Explicitly acknowledge in the paper that the magnitude is modest but the **direction is consistent across all 3 LLIE methods** and is corroborated by the **qualitative EigenCAM evidence** — which makes the pattern a reliable trend.

---

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

### 10. No Limitations Section

IEEE papers at this level are expected to acknowledge limitations. Place a "Limitations" paragraph at the end of the Conclusion (before Future Work), or as a final sub-section in Results. Add 2-3 sentences:
- Only YOLOv11n (nano variant) was evaluated; larger variants may respond differently
- LLIE models were pre-trained on LOLv1 benchmark, not fine-tuned on ExDark (potential domain gap)
- Statistical significance across multiple random seeds was not formally tested due to resource constraints
- Scope is empirical evaluation for academic analysis; real-time deployment optimization was outside the research objective

---

### 11. EigenCAM Citation Missing

EigenCAM is attributed only to your internal `\cite{metrics2026formulas}`. The actual EigenCAM paper is:
> Muhammad, M. B., & Yeasin, M. (2020). *Eigen-CAM: Class Activation Map using Principal Components*. IJCNN 2020.

Add this citation properly.

---

## 🟢 Minor Issues (Polish)

### 12. Inconsistent Dataset Name

- Body text uses `ExDARK` (capital R, K)
- Original paper (Loh & Chan, 2019) uses `ExDark`

Standardize to **ExDark** throughout.

### 13. Citation Style

Replace `\cite{a}, \cite{b}` with `\cite{a,b}` for cleaner IEEE output: `[1,2]` instead of `[1], [2]`.

### 14. "Layer N" Is Vague

In Methodology, "Layer N" is mentioned multiple times. Specify the exact layer index (e.g., `model.model[9]` or `backbone stage 3 output`) so reviewers can reproduce your interpretability results.

### 15. `\usepackage{hyperref}` May Conflict

In IEEEtran conference mode, `hyperref` can cause issues. Use:
```latex
\usepackage[hidelinks]{hyperref}
```
or remove it if links aren't needed in the final PDF.

### 16. Keyword Additions

Current keywords: `Low-Light Image Enhancement, Object Detection, HVI-CIDNet, YOLOv11, ExDark`

Add: `EigenCAM, Edge Preservation Index, Feature Degradation, Task-Driven Enhancement`

---

## ✅ Strengths (Preserve and Emphasize)

| Aspect | Comment |
|---|---|
| **Counter-intuitive finding** | "Raw beats enhanced" is a compelling, publishable insight |
| **Three-LLIE comparison** | Comparing 3 SOTA methods makes the conclusion more generalizable |
| **EigenCAM + MAM dual interpretability** | Rare combination in this type of study; strong methodological contribution |
| **EPI metric** | Original use of EPI for machine-vision impact is a genuine novelty |
| **Latency data** | The 356ms RetinexFormer overhead is a real-world impactful finding |
| **ExDark dataset** | Right choice; specific and well-cited |
| **Related Work structure** | Three well-organized subsections covering LLIE, YOLO evolution, and the machine-vision gap |

---

## 📋 Prioritized Checklist

```
CRITICAL — Do First:
[ ] 1. Rewrite abstract with actual results (use draft above)
[ ] 2. Insert all 7 figures (delete all PLACEHOLDERs and comments)
[ ] 3. Add explicit "Contributions" bullet list at end of Introduction
[ ] 4. Remove \cite{report2026internal}, \cite{proposal2026sempro}, \cite{metrics2026formulas}
[ ] 5. Move all methodology details (training params, dataset split) to paper body directly
[ ] 6. Complete BibTeX fields (journal/arXiv ID, volume, pages, DOI)
[ ] 7. Remove oishi2021network and gries_hazzan_cs entries

HIGH — Do Before Final Draft:
[ ] 8. Insert Table 1 (Spatial Metrics) and Table 2 (Detection Performance) in Results
[ ] 9. Add Table 3 (Latency + GFLOPs comparison) + 1 paragraph discussing computational overhead (NOT real-time framing)
[ ] 10. Define EPI formula in Section 3.4 — use Canny+IoU formulation (see Section 8 above)
[ ] 11. Add EigenCAM citation (Muhammad & Yeasin, IJCNN 2020)
[ ] 12. Address narrow performance gap explicitly — either add std/seeds or argue direction+EigenCAM
[ ] 13. Add Limitations paragraph to Conclusion

POLISH — Before Submission:
[ ] 14. Standardize "ExDark" (not "ExDARK") everywhere
[ ] 15. Replace \cite{a}, \cite{b} → \cite{a,b}
[ ] 16. Specify exact "Layer N" index in Methodology
[ ] 17. Fix \usepackage[hidelinks]{hyperref}
[ ] 18. Expand keyword list
```

---

## 🔮 Positioning & Conference Suggestions

**Title**: Consider rephrasing to signal the counter-intuitive finding:
> *"Does LLIE Help or Hurt? Empirical Evaluation of HVI-CIDNet Preprocessing for YOLOv11 Object Detection on ExDark"*

**Target Conferences**:
| Conference | Scope Match | Notes |
|---|---|---|
| **ICCEREC** (IEEE Indonesia) | ✅ High | Image processing + computer vision |
| **ICICSE** | ✅ High | Applied ML/CV |
| **IEEE Access** (journal) | ✅ High | Open access, broader reach |
| **ICIP** (IEEE Image Processing) | ⚠️ Medium | More competitive, higher bar |

**Page Limit**: Most IEEE conferences = 6 pages. With 7 figures + 3 tables, you will need to be concise. Plan your page budget before writing final draft.