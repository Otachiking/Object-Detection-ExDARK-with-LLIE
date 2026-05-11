

# TRAINING PARAMETERS
epochs: 50
batch: 16
imgsz: 640
patience: 15
device: 0

# LEARNING RATE (Anti-overfitting)
lr0: 0.005        # Lower dari default
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# OPTIMIZER
optimizer: AdamW

# AUGMENTATION
hsv_h: 0.015      # Hue shift (minimal)
hsv_s: 0.7        # Saturation
hsv_v: 0.25       # Value (sesuai request)
degrees: 0.0      # No rotation
translate: 0.15   # Translation
scale: 0.7        # Scaling
shear: 0.0        # No shear
perspective: 0.0  # No perspective
flipud: 0.0       # No vertical flip
fliplr: 0.5       # Horizontal flip
mosaic: 1.0       # Mosaic augmentation
mixup: 0.2        # Cross-sample regularization
copy_paste: 0.1   # Edge-aware augmentation
auto_augment: randaugment
erasing: 0.2      # Random erasing

# COLOR
bgr: 0.0          # No channel shuffle (preserve LLIE colors if used)

# VALIDATION
val: true
plots: true
save: true

__________________________
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class ImageQualityMetrics:
    """Evaluate critical metrics for YOLO detection"""
    
    @staticmethod
    def compute_metrics(image_path):
        """Compute 4 critical metrics"""
        
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. GRADIENT MEAN (Edge Strength)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient)
        
        # 2. LAPLACIAN VARIANCE (Sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # 3. RMS CONTRAST
        rms_contrast = np.std(gray)
        
        # 4. EDGE DENSITY
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'gradient_mean': gradient_mean,
            'laplacian_var': laplacian_var,
            'rms_contrast': rms_contrast,
            'edge_density': edge_density
        }


def evaluate_dataset(image_dir, label='Dataset'):
    """Evaluate all images in directory"""
    
    image_paths = list(Path(image_dir).glob('*.jpg')) + \
                  list(Path(image_dir).glob('*.png'))
    
    results = []
    
    for img_path in tqdm(image_paths, desc=f'Evaluating {label}'):
        metrics = ImageQualityMetrics.compute_metrics(img_path)
        metrics['image'] = img_path.name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Summary statistics
    summary = {
        'dataset': label,
        'gradient_mean': df['gradient_mean'].mean(),
        'gradient_std': df['gradient_mean'].std(),
        'laplacian_var': df['laplacian_var'].mean(),
        'laplacian_std': df['laplacian_var'].std(),
        'rms_contrast': df['rms_contrast'].mean(),
        'rms_std': df['rms_contrast'].std(),
        'edge_density': df['edge_density'].mean(),
        'edge_density_std': df['edge_density'].std()
    }
    
    return summary, df


def compare_scenarios(raw_dir, hvi_dir, retinex_dir, lyt_dir):
    """Compare all preprocessing scenarios"""
    
    print("🔍 EVALUATING ALL SCENARIOS\n" + "="*60)
    
    # Evaluate each
    s1_summary, s1_df = evaluate_dataset(raw_dir, 'S1_Raw')
    s2_summary, s2_df = evaluate_dataset(hvi_dir, 'S2_HVI-CIDNet')
    s3_summary, s3_df = evaluate_dataset(retinex_dir, 'S3_RetinexFormer')
    s4_summary, s4_df = evaluate_dataset(lyt_dir, 'S4_LYT-Net')
    
    # Combine summaries
    comparison = pd.DataFrame([s1_summary, s2_summary, s3_summary, s4_summary])
    
    # Print comparison
    print("\n📊 METRIC COMPARISON")
    print("="*60)
    print(comparison.to_string(index=False))
    
    # Assess each scenario
    print("\n🎯 ASSESSMENT (vs Ideal Range)")
    print("="*60)
    
    for _, row in comparison.iterrows():
        print(f"\n📌 {row['dataset']}")
        
        # Gradient Mean (IDEAL: 10-30)
        gm = row['gradient_mean']
        if 10 <= gm <= 30:
            status = "✅ OPTIMAL"
        elif gm < 10:
            status = "❌ TOO LOW (weak edges)"
        else:
            status = "⚠️ TOO HIGH (noisy)"
        print(f"  Gradient Mean: {gm:.2f} {status}")
        
        # Laplacian Var (IDEAL: >300)
        lv = row['laplacian_var']
        if lv > 500:
            status = "✅ SHARP"
        elif lv > 300:
            status = "⚠️ MODERATE"
        else:
            status = "❌ BLURRY"
        print(f"  Laplacian Var: {lv:.2f} {status}")
        
        # RMS Contrast (IDEAL: 30-60)
        rc = row['rms_contrast']
        if 30 <= rc <= 60:
            status = "✅ OPTIMAL"
        elif rc < 30:
            status = "❌ TOO LOW"
        else:
            status = "⚠️ OVER-CONTRASTED"
        print(f"  RMS Contrast:  {rc:.2f} {status}")
        
        # Edge Density (IDEAL: 0.05-0.20)
        ed = row['edge_density']
        if 0.05 <= ed <= 0.20:
            status = "✅ OPTIMAL"
        elif ed < 0.05:
            status = "❌ TOO LOW"
        else:
            status = "⚠️ TOO NOISY"
        print(f"  Edge Density:  {ed:.4f} {status}")
    
    # Save to CSV
    comparison.to_csv('metrics_comparison.csv', index=False)
    print(f"\n💾 Saved to: metrics_comparison.csv")
    
    return comparison


# USAGE
if __name__ == "__main__":
    comparison = compare_scenarios(
        raw_dir='datasets/s1_raw/images/train',
        hvi_dir='datasets/s2_hvi/images/train',
        retinex_dir='datasets/s3_retinex/images/train',
        lyt_dir='datasets/s4_lyt/images/train'
    )
__________________
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_comparison(comparison_df):
    """Visualize metric comparison (4 metrics only)"""
    
    metrics = ['gradient_mean', 'laplacian_var', 'rms_contrast', 'edge_density']
    ideal_ranges = {
        'gradient_mean': (10, 30),
        'laplacian_var': (300, 1000),
        'rms_contrast': (30, 60),
        'edge_density': (0.05, 0.20)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Bar plot
        scenarios = comparison_df['dataset']
        values = comparison_df[metric]
        
        bars = ax.bar(scenarios, values, alpha=0.7, color=['#2ecc71', '#e74c3c', '#3498db', '#f39c12'])
        
        # Add ideal range (if applicable)
        if metric in ideal_ranges:
            min_val, max_val = ideal_ranges[metric]
            ax.axhline(min_val, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Min Ideal')
            ax.axhline(max_val, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Max Ideal')
            ax.fill_between(range(len(scenarios)), min_val, max_val, alpha=0.1, color='green')
        
        # Labels
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x labels
        ax.set_xticklabels(scenarios, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
    print("💾 Saved visualization to: metrics_comparison.png")
    plt.show()


# USAGE
visualize_comparison(comparison)