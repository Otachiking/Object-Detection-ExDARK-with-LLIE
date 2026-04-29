import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

def generate_and_plot_yolo_vision(weights_path, test_images_dir, output_dir, scenario_name, target_images=None, num_samples=None):
    """
    Generates YOLO feature map visualizations (What YOLO Sees)
    1. Grid Feature Maps (Bawaan Ultralytics)
    2. Single Heatmap (Mean Activation) via PyTorch Hooks
    """
    # 1. FIXED IMAGE TARGETS
    if target_images is None:
        target_images = ['2015_00402.jpg', '2015_00403.jpg', '2015_00523.jpg']
        
    out_path_f56 = os.path.join(output_dir, "interpretability")
    os.makedirs(out_path_f56, exist_ok=True)
    
    model = YOLO(weights_path)
    
    # 2. REGISTER PYTORCH HOOKS (Untuk menangkap sinyal dari otak YOLO)
    activations = {}
    def get_activation(name):
        def hook(mod, inp, out):
            # Beberapa layer mengembalikan tuple, kita ambil tensor pertamanya
            if isinstance(out, tuple):
                activations[name] = out[0].detach()
            else:
                activations[name] = out.detach()
        return hook
        
    # Memasang alat penyadap (hook) ke Stage 0 (Edge) dan Deep Stage (Semantic)
    hook_handle_0 = model.model.model[0].register_forward_hook(get_activation('layer0'))
    deep_idx = len(model.model.model) - 2 # Biasanya 1 layer sebelum deteksi akhir
    hook_handle_n = model.model.model[deep_idx].register_forward_hook(get_activation('layerN'))
    
    generated_plots = []
    
    for img_name in target_images:
        img_path = os.path.join(test_images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[!] Warning: Target gambar {img_name} tidak ditemukan di {test_images_dir}")
            continue
            
        run_name = f"viz_{scenario_name}_{img_name.split('.')[0]}"
        
        # Trik agar Heatmap presisi: Paksa gambar menjadi persegi (640x640) 
        # sehingga tidak ada distorsi akibat padding (letterbox)
        ori_img = cv2.imread(img_path)
        ori_img_640 = cv2.resize(ori_img, (640, 640))
        img_640_path = os.path.join(out_path_f56, f"temp_640_{img_name}")
        cv2.imwrite(img_640_path, ori_img_640)
        
        ori_img_rgb = cv2.cvtColor(ori_img_640, cv2.COLOR_BGR2RGB)
        
        # Eksekusi YOLO: Akan men-trigger Hook dan nge-save Grid kotak-kotak
        results = model.predict(source=img_640_path, visualize=True, save=False, name=run_name, project=out_path_f56, exist_ok=True)
        
        # 3. MENGHITUNG MEAN ACTIVATION HEATMAP
        heatmaps = {}
        for layer_name in ['layer0', 'layerN']:
            if layer_name in activations:
                tensor = activations[layer_name] # Ukuran: [Batch=1, Channel, Height, Width]
                
                # Rata-ratakan semua channel (meringkas puluhan kotak jadi 1 gambar)
                mean_act = torch.mean(tensor, dim=1).squeeze().cpu().numpy()
                
                # Normalisasi skor menjadi 0 - 255 (format gambar standar)
                mean_act = np.maximum(mean_act, 0)
                mean_act = mean_act - np.min(mean_act)
                mean_act = mean_act / (np.max(mean_act) + 1e-7)
                mean_act = np.uint8(255 * mean_act)
                
                # Warnai dengan efek thermal/panas (JET)
                heatmap_resized = cv2.resize(mean_act, (640, 640))
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                
                # Tumpuk (Overlay) ke gambar asli
                overlay = cv2.addWeighted(ori_img_rgb, 0.5, heatmap_color, 0.5, 0)
                heatmaps[layer_name] = overlay
                
        # 4. PLOT VISUALISASI LENGKAP (5 PANEL)
        viz_dir = os.path.join(out_path_f56, run_name)
        feature_maps = glob.glob(os.path.join(viz_dir, "*_features.png"))
        feature_maps.sort(key=lambda x: int(os.path.basename(x).split('_')[0].replace('stage', '')) if 'stage' in os.path.basename(x) else 999)
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        axes[0].imshow(ori_img_rgb)
        axes[0].set_title(f"Original Image\n{img_name}")
        axes[0].axis('off')
        
        # Layer 0 (Grid)
        if len(feature_maps) > 0:
            l0_grid = cv2.cvtColor(cv2.imread(feature_maps[0]), cv2.COLOR_BGR2RGB)
            axes[1].imshow(l0_grid)
            title0 = os.path.basename(feature_maps[0]).split('_features')[0]
            axes[1].set_title(f"Layer 0 Grid (All Channels)\n{title0}")
        axes[1].axis('off')
        
        # Layer 0 (Heatmap Overlay)
        if 'layer0' in heatmaps:
            axes[2].imshow(heatmaps['layer0'])
            axes[2].set_title("Layer 0 Heatmap (Edge/Noise)\nMean Activation")
        axes[2].axis('off')
        
        # Layer N (Grid)
        if len(feature_maps) > 1:
            ln_grid = cv2.cvtColor(cv2.imread(feature_maps[-1]), cv2.COLOR_BGR2RGB)
            axes[3].imshow(ln_grid)
            titlen = os.path.basename(feature_maps[-1]).split('_features')[0]
            axes[3].set_title(f"Layer N Grid (Deep)\n{titlen}")
        axes[3].axis('off')
        
        # Layer N (Heatmap Overlay)
        if 'layerN' in heatmaps:
            axes[4].imshow(heatmaps['layerN'])
            axes[4].set_title("Layer N Heatmap (Semantic)\nMean Activation")
        axes[4].axis('off')
        
        plt.tight_layout()
        out_file = os.path.join(out_path_f56, f"viz_compare_{img_name}")
        plt.savefig(out_file, dpi=150)
        plt.close()
        
        generated_plots.append(out_file)
        
        # Bersihkan file temp
        if os.path.exists(img_640_path):
            os.remove(img_640_path)
        
    # Copot alat penyadap agar memori tidak penuh
    hook_handle_0.remove()
    hook_handle_n.remove()
    
    print(f"✅ Selesai memproses interpretability untuk {len(generated_plots)} gambar TARGET (Fixed).")
    return generated_plots

