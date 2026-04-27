import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def generate_and_plot_yolo_vision(weights_path, test_images_dir, output_dir, scenario_name, num_samples=2):
    """
    Generates YOLO feature map visualizations (What YOLO Sees)
    Uses Ultralytics' built-in `visualize=True`.
    """
    out_path_f56 = os.path.join(output_dir, "interpretability")
    os.makedirs(out_path_f56, exist_ok=True)
    
    # Get random test images
    all_images = glob.glob(os.path.join(test_images_dir, "*.jpg")) + glob.glob(os.path.join(test_images_dir, "*.png"))
    if not all_images:
        print(f"Batal FASE 5.6: Tidak ditemukan gambar di {test_images_dir}")
        return None
        
    random.shuffle(all_images)
    samples = all_images[:num_samples]
    
    model = YOLO(weights_path)
    
    generated_plots = []
    
    for i, img_path in enumerate(samples):
        # We save output to a temporary runs directory
        img_name = os.path.basename(img_path)
        run_name = f"viz_{scenario_name}_{i}"
        
        # Visualize=True saves intermediate feature maps into the runs/detect/ folder
        results = model.predict(source=img_path, visualize=True, save=False, name=run_name, project=out_path_f56, exist_ok=True)
        
        # Ultralytics saves visualizations inside: out_path_f56 / run_name / 
        viz_dir = os.path.join(out_path_f56, run_name)
        
        # Let's find the feature map images. 
        # Typically they are named like 'stage0_Conv_features.png', 'stage1_Conv_features.png', etc.
        feature_maps = glob.glob(os.path.join(viz_dir, "*_features.png"))
        
        if not feature_maps:
            print(f"[!] Warning: Gagal men-generate Feature map untuk {img_name}")
            continue
            
        # Urutkan berdasarkan stage number (e.g., stage0, stage1, stage20)
        feature_maps.sort(key=lambda x: int(os.path.basename(x).split('_')[0].replace('stage', '')) if 'stage' in os.path.basename(x) else 999)
        
        # We want to show the original image, Layer 0 (Edge Detection/Low-level features), 
        # and a Deep Layer (High-level abstract features)
        if len(feature_maps) >= 2:
            layer_0 = feature_maps[0]   # Early Layer (Edge Detector)
            layer_n = feature_maps[-1]  # Deep Layer (Semantics / Attention)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Original Image
            ori_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            axes[0].imshow(ori_img)
            axes[0].set_title(f"Original Image\n({img_name})")
            axes[0].axis('off')
            
            # 2. Layer 0 (Edge / Pattern Search)
            l0_img = cv2.imread(layer_0)
            if l0_img is not None:
                l0_img = cv2.cvtColor(l0_img, cv2.COLOR_BGR2RGB)
                axes[1].imshow(l0_img)
                layer0_name = os.path.basename(layer_0).split('_features')[0]
                axes[1].set_title(f"Early Layer (Edge & Texture)\n{layer0_name}")
            axes[1].axis('off')
            
            # 3. Layer N (Deep Semantic Attention)
            ln_img = cv2.imread(layer_n)
            if ln_img is not None:
                ln_img = cv2.cvtColor(ln_img, cv2.COLOR_BGR2RGB)
                axes[2].imshow(ln_img)
                layern_name = os.path.basename(layer_n).split('_features')[0]
                axes[2].set_title(f"Deep Layer (Semantic Concept)\n{layern_name}")
            axes[2].axis('off')
            
            plt.tight_layout()
            out_file = os.path.join(out_path_f56, f"viz_compare_{img_name}")
            plt.savefig(out_file, dpi=150)
            plt.close()
            
            generated_plots.append(out_file)
            
    print(f"✅ Selesai ekstrak 'What YOLO Sees'. Dimensi Edge & Semantic Features dibuat untuk {len(generated_plots)} gambar.")
    return generated_plots
