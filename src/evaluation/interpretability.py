import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from ultralytics import YOLO


def _compute_mean_activation(tensor: torch.Tensor, target_size=(640, 640)) -> np.ndarray:
    """
    Mean Activation Map dari tensor aktivasi [1, C, H, W].
    Rata-ratakan semua channel, normalisasi ke [0,255], resize, lalu colormap JET.
    Returns: overlay-ready RGB uint8 array (target_size).
    """
    act = torch.mean(tensor, dim=1).squeeze().cpu().numpy()  # [H, W]
    act = np.maximum(act, 0)
    act = act - np.min(act)
    act = act / (np.max(act) + 1e-7)
    act_uint8 = np.uint8(255 * act)
    heatmap = cv2.resize(act_uint8, target_size)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def _compute_eigencam(tensor: torch.Tensor, target_size=(640, 640)) -> np.ndarray:
    """
    EigenCAM via SVD (class-agnostic, tidak butuh gradient).
    Mengambil Principal Component pertama (PC1) dari activation matrix.

    Steps:
      1. Reshape [C, H*W]
      2. SVD → ambil V[:, 0] (PC1 direction)
      3. Project → A @ V[:, 0] → scalar saliency per spasial pos
      4. Reshape ke [H, W], normalisasi, colormap.

    Referensi: Muhammad et al., "EigenCAM: Class Activation Map using Principal
    Components", IJCNN 2020.
    """
    act = tensor.squeeze(0).cpu().numpy()  # [C, H, W]
    C, H, W = act.shape
    act_2d = act.reshape(C, H * W)         # [C, H*W]

    # SVD: U[C,k], S[k], Vt[k, H*W]
    # PC1 = Vt[0] — direction of max variance in activation space
    try:
        _, _, Vt = np.linalg.svd(act_2d, full_matrices=False)
        pc1 = Vt[0]                        # [H*W]
    except np.linalg.LinAlgError:
        pc1 = act_2d.mean(axis=0)          # fallback: mean channel

    saliency = pc1.reshape(H, W)

    # JANGAN clamp negatif — PC1 bisa dominan negatif.
    # Gunakan nilai absolut agar seluruh rentang signed projection terwakili.
    saliency = np.abs(saliency)
    saliency = saliency - np.min(saliency)
    saliency = saliency / (np.max(saliency) + 1e-7)
    saliency_uint8 = np.uint8(255 * saliency)

    heatmap = cv2.resize(saliency_uint8, target_size)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)


def _blend_overlay(base_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha=0.5) -> np.ndarray:
    """Alpha-blend heatmap ke gambar asli: alpha*base + (1-alpha)*heatmap."""
    return cv2.addWeighted(base_rgb, alpha, heatmap_rgb, 1 - alpha, 0)


def generate_and_plot_yolo_vision(
    weights_path,
    test_images_dir,
    output_dir,
    scenario_name,
    target_images=None,
    num_samples=None,
):
    """
    Generates YOLO feature map visualizations — 1 unified row per image (4 panels):
      [0] Original Image
      [1] Layer 0 — Mean Activation (Edge/Low-level)
      [2] Layer N — Mean Activation (Semantic/Deep)
      [3] Layer N — EigenCAM (PC1 Projection)

    Menggunakan PyTorch Forward Hooks untuk menangkap aktivasi tanpa gradient.
    EigenCAM via SVD: class-agnostic, cocok untuk multi-class detector.
    """
    if target_images is None:
        target_images = [
            '2015_00402.jpg', '2015_00403.jpg', '2015_00523.jpg',
            '2015_00448.jpg', '2015_01739.jpg',
        ]

    out_path = os.path.join(output_dir, "interpretability")
    os.makedirs(out_path, exist_ok=True)

    model = YOLO(weights_path)

    # --- Register PyTorch Forward Hooks ---
    activations = {}

    def get_activation(name):
        def hook(mod, inp, out):
            activations[name] = out[0].detach() if isinstance(out, tuple) else out.detach()
        return hook

    hook_handle_0 = model.model.model[0].register_forward_hook(get_activation('layer0'))
    deep_idx = len(model.model.model) - 2  # 1 layer sebelum detection head
    hook_handle_n = model.model.model[deep_idx].register_forward_hook(get_activation('layerN'))

    generated_plots = []
    n_imgs = len(target_images)

    # --- Buat figure tunggal: n_imgs baris × 4 kolom ---
    # Setiap baris = 1 gambar, 4 panel berdampingan
    NCOLS = 4
    fig = plt.figure(figsize=(NCOLS * 5.5, n_imgs * 5.5))
    fig.patch.set_facecolor('white')

    fig.suptitle(
        f"YOLO Interpretability — {scenario_name}\n"
        f"[Original]  [Layer 0: Mean Act. (Edge)]  "
        f"[Layer N: Mean Act. (Semantic)]  [Layer N: EigenCAM (PC1)]",
        fontsize=14,
        fontweight='bold',
        color='black',
        y=1.01,
    )

    gs = gridspec.GridSpec(
        n_imgs, NCOLS,
        figure=fig,
        wspace=0.04,
        hspace=0.14,
        left=0.01, right=0.99,
        top=0.97, bottom=0.01,
    )

    valid_count = 0

    for row_idx, img_name in enumerate(target_images):
        img_path = os.path.join(test_images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[!] Warning: {img_name} tidak ditemukan di {test_images_dir}")
            # Isi dengan placeholder kosong agar grid tidak rusak
            for col in range(NCOLS):
                ax = fig.add_subplot(gs[row_idx, col])
                ax.set_facecolor('#0d0d1a')
                ax.text(0.5, 0.5, "Not found", ha='center', va='center',
                        color='red', fontsize=10, transform=ax.transAxes)
                ax.axis('off')
            continue

        # --- Resize ke 640x640 agar hook map tidak terdistorsi letterbox ---
        ori_img = cv2.imread(img_path)
        ori_img_640 = cv2.resize(ori_img, (640, 640))
        tmp_path = os.path.join(out_path, f"_tmp_{img_name}")
        cv2.imwrite(tmp_path, ori_img_640)
        ori_rgb = cv2.cvtColor(ori_img_640, cv2.COLOR_BGR2RGB)

        # --- Trigger forward pass (hooks aktif otomatis) ---
        run_name = f"viz_{scenario_name}_{img_name.split('.')[0]}"
        model.predict(
            source=tmp_path,
            visualize=False,  # Tidak butuh grid internal lagi
            save=False,
            name=run_name,
            project=out_path,
            exist_ok=True,
            verbose=False,
        )

        # Bersihkan tmp
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # --- Hitung heatmaps ---
        PANEL_SIZE = (640, 640)

        # Panel 0: Original
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(ori_rgb)
        ax0.set_title(
            f"[{scenario_name}] {img_name}",
            fontsize=11, color='black', pad=4, fontweight='bold',
        )
        ax0.axis('off')
        _style_ax(ax0)

        # Panel 1: Layer 0 Mean Activation
        ax1 = fig.add_subplot(gs[row_idx, 1])
        if 'layer0' in activations:
            hmap0 = _compute_mean_activation(activations['layer0'], PANEL_SIZE)
            overlay0 = _blend_overlay(ori_rgb, hmap0)
            ax1.imshow(overlay0)
        else:
            ax1.set_facecolor('white')
            ax1.text(0.5, 0.5, "No activation", ha='center', va='center',
                     color='gray', transform=ax1.transAxes)
        ax1.set_title("Layer 0\nMean Activation (Edge)", fontsize=11, color='black', pad=4)
        ax1.axis('off')
        _style_ax(ax1)

        # Panel 2: Layer N Mean Activation
        ax2 = fig.add_subplot(gs[row_idx, 2])
        if 'layerN' in activations:
            hmapN = _compute_mean_activation(activations['layerN'], PANEL_SIZE)
            overlayN = _blend_overlay(ori_rgb, hmapN)
            ax2.imshow(overlayN)
        else:
            ax2.set_facecolor('white')
            ax2.text(0.5, 0.5, "No activation", ha='center', va='center',
                     color='gray', transform=ax2.transAxes)
        ax2.set_title("Layer N\nMean Activation (Semantic)", fontsize=11, color='black', pad=4)
        ax2.axis('off')
        _style_ax(ax2)

        # Panel 3: Layer N EigenCAM (PC1)
        ax3 = fig.add_subplot(gs[row_idx, 3])
        if 'layerN' in activations:
            eigen_map = _compute_eigencam(activations['layerN'], PANEL_SIZE)
            eigen_overlay = _blend_overlay(ori_rgb, eigen_map)
            ax3.imshow(eigen_overlay)
        else:
            ax3.set_facecolor('white')
            ax3.text(0.5, 0.5, "No activation", ha='center', va='center',
                     color='gray', transform=ax3.transAxes)
        ax3.set_title("Layer N\nEigenCAM (PC1 Projection)", fontsize=11, color='black', pad=4)
        ax3.axis('off')
        _style_ax(ax3)

        valid_count += 1

    # --- Simpan figure tunggal ---
    out_file = os.path.join(out_path, f"interpretability_grid_{scenario_name}.png")
    fig.savefig(out_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    generated_plots.append(out_file)

    # --- Copot hooks ---
    hook_handle_0.remove()
    hook_handle_n.remove()

    print(f"✅ Interpretability grid saved: {out_file}")
    print(f"   ({valid_count}/{len(target_images)} gambar berhasil diproses)")
    return generated_plots


def _style_ax(ax):
    """Styling terang untuk axis (white background, border abu-abu tipis)."""
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)
    ax.set_facecolor('white')
