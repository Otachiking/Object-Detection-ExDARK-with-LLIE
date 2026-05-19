import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude via Sobel (ksize=3). Returns float32 [H, W]."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def compute_epi_gcs(img_raw: np.ndarray, img_enhanced: np.ndarray) -> float:
    """EPI-v2: Gradient Cosine Similarity (GCS) using Sobel.

    Algorithm: Gradient Cosine Similarity (GCS)
    Formula:
                     sum(G_o * G_e)
        EPI_GCS = ---------------------------
                  sqrt(sum(G_o^2) * sum(G_e^2))

    Where G_o and G_e are Sobel gradient magnitude vectors (flattened)
    of the original and enhanced images respectively.

    Note: Complements EPI (Canny+IoU) — EPI measures binary edge location
    overlap, EPI_GCS measures continuous gradient magnitude similarity.

    Returns float in [0, 1]. Returns 1.0 if both images have zero gradients.
    """
    gray_raw = cv2.cvtColor(img_raw,      cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_enh = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY).astype(np.float32)

    Go = _sobel_magnitude(gray_raw).flatten()
    Ge = _sobel_magnitude(gray_enh).flatten()

    dot    = np.dot(Go, Ge)
    norm_o = np.sqrt(np.dot(Go, Go))
    norm_e = np.sqrt(np.dot(Ge, Ge))
    denom  = norm_o * norm_e

    return float(dot / denom) if denom > 1e-7 else 1.0


# ---------------------------------------------------------------------------
# Main metric class
# ---------------------------------------------------------------------------

class ImageQualityMetrics:
    """Evaluate spatial metrics for LLIE quality assessment (YOLO-oriented).

    Metrics:
        1. shadow_ratio  -- % pixels with intensity < 30
        2. luminance     -- mean brightness (0-255)
        3. rms_contrast  -- std dev of pixel intensities
        4. noise_sigma   -- high-freq residual after Gaussian blur
        5. edge_density  -- % edge pixels via Canny
        6. epi           -- Edge Preservation Index (Canny + Jaccard IoU)
        7. epi_gcs       -- EPI-v2: Gradient Cosine Similarity (Sobel)
    """

    @staticmethod
    def compute_metrics(image_path, raw_path=None):
        """Compute all spatial quality metrics for a single image.

        Args:
            image_path: Path to enhanced/evaluated image.
            raw_path:   Path to raw reference image (for EPI & EPI_GCS).
                        If None, both epi and epi_gcs are returned as None.

        Returns:
            dict with keys: shadow_ratio, luminance, rms_contrast,
                            noise_sigma, edge_density, epi, epi_gcs
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. SHADOW AREA RATIO
        shadow_ratio = float((gray < 30).mean() * 100)

        # 2. MEAN LUMINANCE
        luminance = float(np.mean(gray))

        # 3. RMS CONTRAST
        rms_contrast = float(np.std(gray))

        # 4. NOISE SIGMA — residual setelah Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_sigma = float(np.std(gray - blurred))

        # 5. EDGE DENSITY — % piksel tepi (Canny)
        edges_enh = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = float(np.count_nonzero(edges_enh) / edges_enh.size * 100)

        # 6. EPI (Canny + Jaccard IoU)  &  7. EPI_GCS (Sobel Cosine Similarity)
        epi     = None
        epi_gcs = None
        if raw_path is not None:
            raw_img = cv2.imread(str(raw_path))
            if raw_img is not None:
                gray_raw = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

                # --- EPI: |E_o ∩ E_e| / |E_o ∪ E_e|  (binary Canny masks) ---
                edges_raw    = cv2.Canny(gray_raw, 50, 150)
                mask_raw     = edges_raw > 0
                mask_enh     = edges_enh > 0
                intersection = np.logical_and(mask_raw, mask_enh).sum()
                union        = np.logical_or(mask_raw, mask_enh).sum()
                epi = float(intersection / union) if union > 0 else 1.0

                # --- EPI_GCS: sum(G_o*G_e) / sqrt(sum(G_o^2)*sum(G_e^2)) ---
                epi_gcs = compute_epi_gcs(raw_img, img)

        return {
            'shadow_ratio': shadow_ratio,
            'luminance':    luminance,
            'rms_contrast': rms_contrast,
            'noise_sigma':  noise_sigma,
            'edge_density': edge_density,
            'epi':          epi,
            'epi_gcs':      epi_gcs,
        }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_spatial_metrics(image_dir, label='Dataset', raw_dir=None):
    """Evaluate all images in directory and return summary dict + full DataFrame."""
    image_paths = (
        list(Path(image_dir).glob('*.jpg'))
        + list(Path(image_dir).glob('*.jpeg'))
        + list(Path(image_dir).glob('*.png'))
    )

    results = []
    for img_path in tqdm(image_paths, desc=f'Spatial Metrics [{label}]'):
        raw_path = Path(raw_dir) / img_path.name if raw_dir else None
        metrics  = ImageQualityMetrics.compute_metrics(img_path, raw_path=raw_path)
        if metrics:
            metrics['image'] = img_path.name
            results.append(metrics)

    if not results:
        return None, None

    df = pd.DataFrame(results)

    summary = {'dataset': label}
    for col in ['shadow_ratio', 'luminance', 'rms_contrast', 'noise_sigma', 'edge_density']:
        summary[col]          = float(df[col].mean())
        summary[f'{col}_std'] = float(df[col].std())

    # EPI & EPI_GCS — hanya ada jika raw_dir disediakan
    for key in ['epi', 'epi_gcs']:
        if key in df.columns:
            vals = df[key].dropna()
            summary[key]          = float(vals.mean()) if len(vals) > 0 else None
            summary[f'{key}_std'] = float(vals.std())  if len(vals) > 0 else None
        else:
            summary[key]          = None
            summary[f'{key}_std'] = None

    return summary, df


# ---------------------------------------------------------------------------
# Pretty-print assessment
# ---------------------------------------------------------------------------

def print_spatial_assessment(summary_dict):
    """Print interpretasi semua metrik LLIE dengan range ideal."""
    label = summary_dict.get('dataset', '?')
    print(f"\n🎯 SPATIAL METRICS ASSESSMENT — {label}")
    print("=" * 60)

    # Shadow Area (ideal: < 5%)
    sa = summary_dict.get('shadow_ratio')
    if sa is not None:
        if sa < 5:      status = "✅ SEDIKIT AREA GELAP"
        elif sa < 15:   status = "⚠️ MODERATE"
        else:           status = "❌ BANYAK AREA GELAP (enhancement kurang efektif)"
        print(f"  🌑 Shadow Area    : {sa:.2f}%  {status}")

    # Mean Luminance (ideal: 80–180)
    lum = summary_dict.get('luminance')
    if lum is not None:
        if 80 <= lum <= 180:  status = "✅ OPTIMAL"
        elif lum < 80:        status = "❌ TERLALU GELAP"
        else:                 status = "⚠️ TERLALU TERANG (overexposed)"
        print(f"  ☀️  Mean Lum.      : {lum:.2f}   {status}")

    # RMS Contrast (ideal: 30–70)
    rc = summary_dict.get('rms_contrast')
    if rc is not None:
        if 30 <= rc <= 70:  status = "✅ OPTIMAL"
        elif rc < 30:       status = "❌ TERLALU RENDAH (flat/pucat)"
        else:               status = "⚠️ TERLALU TINGGI (over-contrasted)"
        print(f"  🌓 RMS Contrast   : {rc:.2f}   {status}")

    # Noise Sigma (ideal: < 5)
    ns = summary_dict.get('noise_sigma')
    if ns is not None:
        if ns < 5:    status = "✅ BERSIH"
        elif ns < 10: status = "⚠️ MODERATE NOISE"
        else:         status = "❌ NOISE TINGGI (noise amplification)"
        print(f"  ✨ Noise σ        : {ns:.3f}  {status}")

    # EPI — Canny + Jaccard IoU (ideal: > 0.7)
    epi = summary_dict.get('epi')
    if epi is not None:
        if epi >= 0.7:    status = "✅ TEPI TERJAGA BAIK"
        elif epi >= 0.4:  status = "⚠️ MODERATE (sebagian tepi bergeser)"
        else:             status = "❌ TEPI BANYAK BERUBAH (risiko deteksi terganggu)"
        print(f"  🛡️  EPI   (Canny)  : {epi:.4f} {status}")
    else:
        print(f"  🛡️  EPI   (Canny)  : N/A  (raw reference tidak tersedia)")

    # EPI_GCS — Sobel Gradient Cosine Similarity (ideal: > 0.9)
    epi_gcs = summary_dict.get('epi_gcs')
    if epi_gcs is not None:
        if epi_gcs >= 0.9:    status = "✅ GRADIEN SANGAT MIRIP"
        elif epi_gcs >= 0.7:  status = "⚠️ MODERATE (pergeseran gradien signifikan)"
        else:                  status = "❌ GRADIEN BERBEDA JAUH (struktur visual berubah drastis)"
        print(f"  🛡️  EPI_GCS(Sobel) : {epi_gcs:.4f} {status}")
    else:
        print(f"  🛡️  EPI_GCS(Sobel) : N/A  (raw reference tidak tersedia)")

    # Edge Density (ideal: 3–20%)
    ed = summary_dict.get('edge_density')
    if ed is not None:
        if 3 <= ed <= 20:  status = "✅ OPTIMAL"
        elif ed < 3:       status = "❌ TERLALU RENDAH (over-smoothing?)"
        else:              status = "⚠️ TERLALU TINGGI (noise terdeteksi sebagai tepi?)"
        print(f"  📐 Edge Density   : {ed:.2f}%  {status}")


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def compute_and_show_spatial_metrics(test_dir, output_dir, scenario_name, raw_dir=None):
    """Compute, print assessment, and save CSV. No charts/plots."""
    summary, df_full = evaluate_spatial_metrics(test_dir, label=scenario_name, raw_dir=raw_dir)
    if summary is None:
        print(f"[SPATIAL] No images found in {test_dir}")
        return None

    print_spatial_assessment(summary)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'spatial_metrics_{scenario_name}.csv')
    df_full.to_csv(csv_path, index=False)
    print(f"\n💾 Saved spatial metrics to: {csv_path}")

    return summary
