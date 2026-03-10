"""
LYT-Net enhancer wrapper (PyTorch port).

Uses the PyTorch pretrained weights (LOL-v1 trained).
Repo: https://github.com/albrateanu/LYT-Net
Weights: PyTorch .pth from Google Drive

LYT-Net is extremely lightweight: ~45K params, 3.49 GFLOPs.
"""

import os
import sys
import cv2
import torch
import numpy as np
from typing import Optional

from src.enhancers.base import BaseEnhancer


class LYTNetEnhancer(BaseEnhancer):
    """LYT-Net Low-Light Image Enhancement wrapper (PyTorch)."""

    def __init__(self, cache_dir: str = "cache/LYT-Net"):
        super().__init__(name="LYT-Net")
        self.cache_dir = cache_dir
        self.repo_dir = os.path.join(cache_dir, "repo")
        self.weight_path = None

    def _clone_repo(self) -> None:
        """Clone LYT-Net repo if not exists."""
        if os.path.exists(os.path.join(self.repo_dir, "model")) or \
           os.path.exists(os.path.join(self.repo_dir, "models")):
            print(f"[LYT-Net] Repo already exists: {self.repo_dir}")
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        print("[LYT-Net] Cloning repository...")
        ret = os.system(f"git clone https://github.com/albrateanu/LYT-Net.git {self.repo_dir}")
        if ret != 0:
            raise RuntimeError("Failed to clone LYT-Net repository")
        print("[LYT-Net] Repository cloned successfully")

    def _download_weights(self) -> str:
        """Download PyTorch weights."""
        weights_dir = os.path.join(self.cache_dir, "weights")
        weight_file = os.path.join(weights_dir, "lyt_net_lol.pth")

        if os.path.exists(weight_file):
            print(f"[LYT-Net] Weights already exist: {weight_file}")
            return weight_file

        os.makedirs(weights_dir, exist_ok=True)
        print("[LYT-Net] Downloading PyTorch weights...")

        try:
            import gdown
            # PyTorch weights file ID
            url = "https://drive.google.com/uc?id=1GeEkasO2ubFi847pzrxfQ1fB3Y9NuhZ1"
            gdown.download(url, weight_file, quiet=False)
            print(f"[LYT-Net] Weights downloaded: {weight_file}")
        except Exception as e:
            print(f"[LYT-Net] Auto-download failed: {e}")
            print(f"  Download manually from: https://drive.google.com/file/d/1GeEkasO2ubFi847pzrxfQ1fB3Y9NuhZ1")
            print(f"  Save to: {weight_file}")
            raise FileNotFoundError(f"Weights not found: {weight_file}")

        return weight_file

    def load_model(self, device: str = None) -> None:
        """Load LYT-Net PyTorch model."""
        from src.enhancers.base import _auto_device
        self.device = _auto_device(device)

        self._clone_repo()
        self.weight_path = self._download_weights()

        # Add repo to path
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        # Import model architecture
        # LYT-Net PyTorch port structure may vary — try multiple import paths
        model = None
        import_errors = []

        # Try different import patterns based on repo structure
        for try_import in [
            lambda: __import__("model", fromlist=["LYTNet"]),
            lambda: __import__("models", fromlist=["LYTNet"]),
            lambda: __import__("models.LYTNet", fromlist=["LYTNet"]),
            lambda: __import__("pytorch.model", fromlist=["LYTNet"]),
        ]:
            try:
                mod = try_import()
                if hasattr(mod, "LYTNet"):
                    model = mod.LYTNet()
                    break
                elif hasattr(mod, "LYT_Net"):
                    model = mod.LYT_Net()
                    break
            except Exception as e:
                import_errors.append(str(e))
                continue

        if model is None:
            # Fallback: try to find and load model definition dynamically
            import glob
            py_files = glob.glob(os.path.join(self.repo_dir, "**/*.py"), recursive=True)
            model_files = [f for f in py_files if "model" in os.path.basename(f).lower()]
            raise ImportError(
                f"Failed to import LYT-Net model. "
                f"Tried multiple import paths. Errors: {import_errors}. "
                f"Found Python files with 'model': {model_files}"
            )

        # Load weights
        checkpoint = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        self.model = model
        self._loaded = True
        print(f"[LYT-Net] Model loaded on {self.device}")

    @torch.no_grad()
    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Enhance a single low-light image."""
        assert self._loaded, "Model not loaded. Call load_model() first."

        h, w = img_bgr.shape[:2]

        # BGR → RGB, normalize, to tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        # Pad to multiple of 4
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
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
