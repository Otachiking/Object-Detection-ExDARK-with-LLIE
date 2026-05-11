import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os


class ImageQualityMetrics:
    """Evaluate 6 spatial metrics for LLIE quality assessment (YOLO-oriented)."""

    @staticmethod
    def compute_metrics(image_path, raw_path=None):
        """
        Compute 6 spatial quality metrics.

        Args:
            image_path: Path to enhanced/evaluated image.
            raw_path: Path to raw reference image (for EPI). If None, EPI = None.

        Returns:
            dict with keys: shadow_ratio, luminance, rms_contrast,
                            noise_sigma, edge_density, epi
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 1. SHADOW AREA RATIO — % piksel intensitas < 30
        shadow_ratio = float((gray < 30).mean() * 100)

        # 2. MEAN LUMINANCE — rata-rata kecerahan (0-255)
        luminance = float(np.mean(gray))

        # 3. RMS CONTRAST — std dev intensitas piksel
        rms_contrast = float(np.std(gray))

        # 4. NOISE SIGMA — residual setelah Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_sigma = float(np.std(gray - blurred))

        # 5. EDGE DENSITY — % piksel tepi (Canny), dalam persen
        edges_enh = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = float(np.count_nonzero(edges_enh) / edges_enh.size * 100)

        # 6. EPI — Edge Preservation Index (IoU edge maps raw vs enhanced)
        epi = None
        if raw_path is not None:
            raw_img = cv2.imread(str(raw_path))
            if raw_img is not None:
                gray_raw = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                edges_raw = cv2.Canny(gray_raw, 50, 150)
                mask_raw = edges_raw > 0
                mask_enh = edges_enh > 0
                intersection = np.logical_and(mask_raw, mask_enh).sum()
                union = np.logical_or(mask_raw, mask_enh).sum()
                epi = float(intersection / union) if union > 0 else 1.0

        return {
            'shadow_ratio': shadow_ratio,
            'luminance':    luminance,
            'rms_contrast': rms_contrast,
            'noise_sigma':  noise_sigma,
            'edge_density': edge_density,
            'epi':          epi,
        }


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
        metrics = ImageQualityMetrics.compute_metrics(img_path, raw_path=raw_path)
        if metrics:
            metrics['image'] = img_path.name
            results.append(metrics)

    if not results:
        return None, None

    df = pd.DataFrame(results)

    summary = {'dataset': label}
    for col in ['shadow_ratio', 'luminance', 'rms_contrast', 'noise_sigma', 'edge_density']:
        summary[col]            = float(df[col].mean())
        summary[f'{col}_std']   = float(df[col].std())

    # EPI — hanya ada jika raw_dir disediakan
    if 'epi' in df.columns:
        epi_vals = df['epi'].dropna()
        summary['epi']     = float(epi_vals.mean()) if len(epi_vals) > 0 else None
        summary['epi_std'] = float(epi_vals.std())  if len(epi_vals) > 0 else None
    else:
        summary['epi'] = None

    return summary, df


def print_spatial_assessment(summary_dict):
    """Print interpretasi 6 metrik LLIE dengan range ideal."""
    label = summary_dict.get('dataset', '?')
    print(f"\n\U0001f3af SPATIAL METRICS ASSESSMENT \u2014 {label}")
    print("=" * 60)

    # Shadow Area (ideal setelah enhancement: < 5%)
    sa = summary_dict.get('shadow_ratio')
    if sa is not None:
        if sa < 5:
            status = "\u2705 SEDIKIT AREA GELAP"
        elif sa < 15:
            status = "\u26a0\ufe0f MODERATE"
        else:
            status = "\u274c BANYAK AREA GELAP (enhancement kurang efektif)"
        print(f"  \U0001f311 Shadow Area   : {sa:.2f}%  {status}")

    # Mean Luminance (ideal: 80–180)
    lum = summary_dict.get('luminance')
    if lum is not None:
        if 80 <= lum <= 180:
            status = "\u2705 OPTIMAL"
        elif lum < 80:
            status = "\u274c TERLALU GELAP"
        else:
            status = "\u26a0\ufe0f TERLALU TERANG (overexposed)"
        print(f"  \u2600\ufe0f  Mean Lum.     : {lum:.2f}   {status}")

    # RMS Contrast (ideal: 30–70)
    rc = summary_dict.get('rms_contrast')
    if rc is not None:
        if 30 <= rc <= 70:
            status = "\u2705 OPTIMAL"
        elif rc < 30:
            status = "\u274c TERLALU RENDAH (flat/pucat)"
        else:
            status = "\u26a0\ufe0f TERLALU TINGGI (over-contrasted)"
        print(f"  \U0001f313 RMS Contrast  : {rc:.2f}   {status}")

    # Noise Sigma (ideal: < 5)
    ns = summary_dict.get('noise_sigma')
    if ns is not None:
        if ns < 5:
            status = "\u2705 BERSIH"
        elif ns < 10:
            status = "\u26a0\ufe0f MODERATE NOISE"
        else:
            status = "\u274c NOISE TINGGI (noise amplification)"
        print(f"  \u2728 Noise \u03c3       : {ns:.3f}  {status}")

    # EPI (ideal: > 0.5)
    epi = summary_dict.get('epi')
    if epi is not None:
        if epi >= 0.7:
            status = "\u2705 TEPI TERJAGA BAIK"
        elif epi >= 0.4:
            status = "\u26a0\ufe0f MODERATE (sebagian tepi bergeser)"
        else:
            status = "\u274c TEPI BANYAK BERUBAH (risiko deteksi terganggu)"
        print(f"  \U0001f6e1\ufe0f  EPI           : {epi:.4f} {status}")
    else:
        print(f"  \U0001f6e1\ufe0f  EPI           : N/A  (raw reference tidak tersedia)")

    # Edge Density (ideal: 3–20%)
    ed = summary_dict.get('edge_density')
    if ed is not None:
        if 3 <= ed <= 20:
            status = "\u2705 OPTIMAL"
        elif ed < 3:
            status = "\u274c TERLALU RENDAH (over-smoothing?)"
        else:
            status = "\u26a0\ufe0f TERLALU TINGGI (noise terdeteksi sebagai tepi?)"
        print(f"  \U0001f4d0 Edge Density  : {ed:.2f}%  {status}")


def compute_and_show_spatial_metrics(test_dir, output_dir, scenario_name, raw_dir=None):
    """Compute, print assessment, and save CSV. No charts/plots."""
    summary, df_full = evaluate_spatial_metrics(test_dir, label=scenario_name, raw_dir=raw_dir)
    if summary is None:
        print(f"[SPATIAL] No images found in {test_dir}")
        return None

    print_spatial_assessment(summary)

    # Save full per-image CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'spatial_metrics_{scenario_name}.csv')
    df_full.to_csv(csv_path, index=False)
    print(f"\n\U0001f4be Saved spatial metrics to: {csv_path}")

    return summary
