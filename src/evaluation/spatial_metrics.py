import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class ImageQualityMetrics:
    """Evaluate critical metrics for YOLO detection"""
    
    @staticmethod
    def compute_metrics(image_path):
        """Compute 4 critical metrics"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
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

def evaluate_spatial_metrics(image_dir, label='Dataset'):
    """Evaluate all images in directory and return summary df format"""
    image_paths = list(Path(image_dir).glob('*.jpg')) + \
                  list(Path(image_dir).glob('*.png'))
    
    results = []
    for img_path in tqdm(image_paths, desc=f'Evaluating Spatial Metrics for {label}'):
        metrics = ImageQualityMetrics.compute_metrics(img_path)
        if metrics:
            metrics['image'] = img_path.name
            results.append(metrics)
    
    if not results:
        return None, None
        
    df = pd.DataFrame(results)
    
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

def print_spatial_assessment(summary_dict):
    """Print assessment of the metrics based on ideal ranges"""
    print(f"\n🎯 SPATIAL METRICS ASSESSMENT for {summary_dict['dataset']} (vs Ideal Range)")
    print("="*60)
    
    # Gradient Mean (IDEAL: 10-30)
    gm = summary_dict['gradient_mean']
    if 10 <= gm <= 30:
        status = "✅ OPTIMAL"
    elif gm < 10:
        status = "❌ TOO LOW (weak edges)"
    else:
        status = "⚠️ TOO HIGH (noisy)"
    print(f"  Gradient Mean: {gm:.2f} {status}")
    
    # Laplacian Var (IDEAL: >300)
    lv = summary_dict['laplacian_var']
    if lv > 500:
        status = "✅ SHARP"
    elif lv > 300:
        status = "⚠️ MODERATE"
    else:
        status = "❌ BLURRY"
    print(f"  Laplacian Var: {lv:.2f} {status}")
    
    # RMS Contrast (IDEAL: 30-60)
    rc = summary_dict['rms_contrast']
    if 30 <= rc <= 60:
        status = "✅ OPTIMAL"
    elif rc < 30:
        status = "❌ TOO LOW"
    else:
        status = "⚠️ OVER-CONTRASTED"
    print(f"  RMS Contrast:  {rc:.2f} {status}")
    
    # Edge Density (IDEAL: 0.05-0.20)
    ed = summary_dict['edge_density']
    if 0.05 <= ed <= 0.20:
        status = "✅ OPTIMAL"
    elif ed < 0.05:
        status = "❌ TOO LOW"
    else:
        status = "⚠️ TOO NOISY"
    print(f"  Edge Density:  {ed:.4f} {status}")
    
def visualize_spatial_metrics(summary_dict, output_dir, scenario_name):
    """Visualize metric comparison (4 metrics only) for a single scenario"""
    metrics = ['gradient_mean', 'laplacian_var', 'rms_contrast', 'edge_density']
    ideal_ranges = {
        'gradient_mean': (10, 30),
        'laplacian_var': (300, 1000),
        'rms_contrast': (30, 60),
        'edge_density': (0.05, 0.20)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # Create a 1-row df for simple bar plotting
    df = pd.DataFrame([summary_dict])
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        val = df[metric].iloc[0]
        
        bars = ax.bar([scenario_name], [val], color='#3498db', alpha=0.8)
        
        # Add ideal range
        if metric in ideal_ranges:
            min_val, max_val = ideal_ranges[metric]
            ax.axhline(min_val, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Min Ideal')
            ax.axhline(max_val, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Max Ideal')
            ax.fill_between([-0.5, 0.5], min_val, max_val, alpha=0.1, color='green')
            
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
    
    title = f'Spatial Metrics Visualization - {scenario_name}'
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f'spatial_metrics_{scenario_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return out_path

def compute_and_show_spatial_metrics(test_dir, output_dir, scenario_name):
    summary, df_full = evaluate_spatial_metrics(test_dir, scenario_name)
    if summary:
        print_spatial_assessment(summary)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f'spatial_metrics_{scenario_name}.csv')
        df_full.to_csv(csv_path, index=False)
        print(f"\n💾 Saved full spatial metrics to: {csv_path}")
        
        # Visualize
        img_path = visualize_spatial_metrics(summary, output_dir, scenario_name)
        return summary, img_path
    
    return None, None
