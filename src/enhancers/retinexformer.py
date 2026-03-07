"""
RetinexFormer enhancer wrapper.

Uses LOL_v1.pth pretrained weights (validated on ExDark by the original authors).
Repo: https://github.com/caiyuanhao1998/Retinexformer
"""

import os
import sys
import cv2
import torch
import numpy as np
from typing import Optional

from src.enhancers.base import BaseEnhancer


class RetinexFormerEnhancer(BaseEnhancer):
    """RetinexFormer Low-Light Image Enhancement wrapper."""

    def __init__(self, cache_dir: str = "cache/Retinexformer"):
        super().__init__(name="RetinexFormer")
        self.cache_dir = cache_dir
        self.repo_dir = os.path.join(cache_dir, "repo")
        self.weight_path = None

    def _clone_repo(self) -> None:
        """Clone RetinexFormer repo if not exists."""
        if os.path.exists(os.path.join(self.repo_dir, "basicsr")):
            print(f"[RetinexFormer] Repo already exists: {self.repo_dir}")
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        print("[RetinexFormer] Cloning repository...")
        ret = os.system(f"git clone https://github.com/caiyuanhao1998/Retinexformer.git {self.repo_dir}")
        if ret != 0:
            raise RuntimeError("Failed to clone RetinexFormer repository")
        print("[RetinexFormer] Repository cloned successfully")

    def _download_weights(self) -> str:
        """Download LOL_v1 weights."""
        weights_dir = os.path.join(self.cache_dir, "weights")
        weight_file = os.path.join(weights_dir, "LOL_v1.pth")

        if os.path.exists(weight_file):
            print(f"[RetinexFormer] Weights already exist: {weight_file}")
            return weight_file

        os.makedirs(weights_dir, exist_ok=True)
        print("[RetinexFormer] Please download LOL_v1.pth manually from:")
        print("  https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV")
        print(f"  Save to: {weight_file}")

        # Try gdown for Google Drive download
        try:
            import gdown
            # LOL_v1.pth file ID (from RetinexFormer repo)
            url = "https://drive.google.com/uc?id=1sft9MU-fwtVH0ubNg-GPmGi1BVvIsFSi"
            gdown.download(url, weight_file, quiet=False)
            print(f"[RetinexFormer] Weights downloaded: {weight_file}")
        except Exception as e:
            print(f"[RetinexFormer] Auto-download failed: {e}")
            print("  Install gdown: pip install gdown")
            raise FileNotFoundError(f"Weights not found: {weight_file}")

        return weight_file

    def load_model(self, device: str = None) -> None:
        """Load RetinexFormer model with LOL_v1 weights."""
        from src.enhancers.base import _auto_device
        self.device = _auto_device(device)

        self._clone_repo()
        self.weight_path = self._download_weights()

        # Add repo to path
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        # RetinexFormer uses basicsr framework
        # Import the model architecture
        try:
            from basicsr.models.archs.Retinexformer_arch import Retinexformer
        except ImportError:
            try:
                # Alternative import path
                from basicsr.archs.Retinexformer_arch import Retinexformer
            except ImportError as e:
                raise ImportError(
                    f"Failed to import Retinexformer architecture. "
                    f"Check {self.repo_dir} structure. Error: {e}"
                )

        # Initialize model with default config for LOL
        self.model = Retinexformer(
            in_channels=3,
            out_channels=3,
            n_feat=40,
            stage=1,
            num_blocks=[1, 2, 2],
        )

        # Load weights
        checkpoint = torch.load(self.weight_path, map_location=device, weights_only=False)
        if "params" in checkpoint:
            self.model.load_state_dict(checkpoint["params"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(device)
        self.model.eval()
        self._loaded = True
        print(f"[RetinexFormer] Model loaded on {device}")

    @torch.no_grad()
    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Enhance a single low-light image."""
        assert self._loaded, "Model not loaded. Call load_model() first."

        h, w = img_bgr.shape[:2]

        # BGR → RGB, normalize, to tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Pad to multiple of 8 (RetinexFormer requirement)
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        # Inference
        output = self.model(img_tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

        # To numpy BGR
        output = output.squeeze(0).clamp(0, 1).cpu().numpy()
        output = (output * 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output
